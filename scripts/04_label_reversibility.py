#!/usr/bin/env python
"""M3: measure reversibility R(s) by branch rollout, in parallel, without cooking the Mac.

For each perturbed episode we walk it forward, and every `stride` steps snapshot the world,
run `branches` independent continuations, and record the fraction that still succeed. That
fraction is R at that state, together with the ACT features observed there.

Three things this script is careful about, because it runs for hours on a laptop:

**Heat.** Workers are capped at the number of *performance* cores minus one, each pinned to
a single torch thread, and every worker checks `ThermalGovernor` between label points —
pausing while macOS reports serious thermal pressure and resuming when it drops. Total
paused time is recorded in the output.

**Interruption.** Each episode's labels are written as soon as that episode finishes, so
Ctrl-C or a closed lid costs at most one episode. Re-running skips episodes already on
disk; `--restart` forces a clean run.

**Honesty about cost.** The first episode is timed and the full projection is printed
before committing to the rest, so an overrun is visible in minutes rather than hours.

    python scripts/04_label_reversibility.py --episodes 40 --stride 25 --branches 8
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.envs.perturb import KINDS, sample_spec  # noqa: E402
from faact.eval.runner import PerturbationNeverFired, run_episode  # noqa: E402
from faact.labeling.branch_rollout import DEFAULT_BRANCH_NOISE, label_state  # noqa: E402
from faact.runtime.controller import fixed  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.thermal import (  # noqa: E402
    ThermalGovernor,
    limit_torch_threads,
    safe_worker_count,
    thermal_state,
)
from faact.utils import ARTIFACTS, timed, write_json  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
# Onset window, chosen from the task's own phase timings rather than guessed.
#
# Measured on 5 successful episodes: the right gripper first contacts the cube at median
# step 154, and the transfer completes at median step 278. The original window (40, 160)
# therefore fired almost entirely BEFORE the cube was grasped — which is why `grasp_slip`
# measured as a complete no-op (65% at every duration, identical to the unperturbed rate):
# forcing a gripper open when it holds nothing does nothing.
#
# (150, 280) puts the disturbance inside grasp-and-transport, which is where recoverability
# is actually in question.
ONSET_RANGE = (150, 280)
SHARD_DIR = ARTIFACTS / "reversibility_shards"

log = logging.getLogger("faact.label")


def label_config_key(job: dict) -> str:
    """Short identifier for the settings that determine an episode's labels."""
    return (
        f"{job['kind']}_s{job['stride']}_m{job['branches']}"
        f"_n{job['branch_noise']}_h{job['horizon']}_t{job['max_steps']}"
    )


def label_episode(job: dict) -> dict:
    """Label one episode. Runs in a worker process; returns a JSON-safe record.

    Structured as a pure function of `job` so the pool needs no shared state and a crashed
    worker loses only its own episode.
    """
    limit_torch_threads(1)

    seed = job["seed"]
    # The shard key includes every setting that changes the labels, not just the seed.
    # Keying on seed alone would silently reuse labels from a different --branches or
    # --branch-noise, which is the worst kind of stale cache: invisible and wrong.
    shard = SHARD_DIR / f"episode_{seed}_{label_config_key(job)}.json"
    if shard.exists() and not job["restart"]:
        return {"seed": seed, "cached": True, **json.loads(shard.read_text())}

    governor = ThermalGovernor(poll_seconds=job["poll_seconds"])
    policy = ACTWrapper(CHECKPOINT, device="cpu")
    env = make_env()
    started = time.perf_counter()

    try:
        rng = np.random.default_rng(seed)
        spec = sample_spec(job["kind"], rng, onset_range=ONSET_RANGE)

        # Walk the episode, pausing at every `stride`-th step to measure R there. The
        # callback fires before the action is taken, so features and R describe the same
        # state — a mismatch here would silently misalign every training pair.
        points: list[dict] = []
        executor = ChunkExecutor(policy, fixed(job["horizon"]))

        def on_step(t: int, obs: dict) -> None:
            if t % job["stride"] or t == 0:
                return
            governor.checkpoint()
            _, features = policy.predict_chunk(obs)
            r = label_state(
                env,
                policy,
                timestep=t,
                n_branches=job["branches"],
                branch_noise=job["branch_noise"],
                max_steps=job["max_steps"],
            )
            points.append(
                {
                    "timestep": t,
                    "reversibility": r,
                    "features": {k: np.asarray(v).tolist() for k, v in features.items()},
                }
            )

        result = run_episode(
            env,
            executor,
            seed=seed,
            perturb_spec=spec,
            max_steps=job["max_steps"],
            on_step=on_step,
        )
        record = {
            "seed": seed,
            "kind": job["kind"],
            "perturb": spec.to_dict(),
            "episode_success": result.success,
            "episode_steps": result.steps,
            "points": points,
            "seconds": round(time.perf_counter() - started, 1),
            "thermal": governor.report(),
        }
    except PerturbationNeverFired as exc:
        # The episode finished before its disturbance was due, so it is unperturbed and
        # carries no reversibility signal. Recorded as excluded rather than dropped.
        record = {
            "seed": seed,
            "kind": job["kind"],
            "excluded": "ended_before_onset",
            "detail": str(exc),
            "points": [],
            "seconds": round(time.perf_counter() - started, 1),
            "thermal": governor.report(),
        }
    finally:
        env.close()

    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(record))
    return {"cached": False, **record}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=2000, help="first episode seed")
    ap.add_argument("--stride", type=int, default=25, help="steps between label points")
    ap.add_argument("--branches", type=int, default=8, help="M continuations per state")
    ap.add_argument("--branch-noise", type=float, default=DEFAULT_BRANCH_NOISE)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--workers", type=int, default=None,
                    help="default: performance cores - 1, to leave the machine usable")
    ap.add_argument("--poll-seconds", type=float, default=15.0)
    ap.add_argument("--restart", action="store_true", help="ignore existing episode shards")
    ap.add_argument("--out", default="reversibility_labels.json")
    args = ap.parse_args()

    workers = safe_worker_count(args.workers)
    seeds = [args.seed + i for i in range(args.episodes)]
    # Perturbation kinds are assigned round-robin so the dataset is balanced across them
    # by construction rather than by chance.
    jobs = [
        {
            "seed": seed,
            "kind": args.kinds[i % len(args.kinds)],
            "stride": args.stride,
            "branches": args.branches,
            "branch_noise": args.branch_noise,
            "horizon": args.horizon,
            "max_steps": args.max_steps,
            "restart": args.restart,
            "poll_seconds": args.poll_seconds,
        }
        for i, seed in enumerate(seeds)
    ]

    approx_points = args.max_steps // args.stride
    print(
        f"labelling {len(jobs)} episodes x ~{approx_points} points x {args.branches} branches\n"
        f"workers={workers} (performance cores available: {safe_worker_count(reserve=0)})  "
        f"thermal state now: {thermal_state()}  shards: {SHARD_DIR}\n"
    )

    records = []
    with timed("reversibility labelling") as t:
        # 'spawn' avoids inheriting MuJoCo/torch state through fork, which is not fork-safe.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
            for i, rec in enumerate(pool.imap_unordered(label_episode, jobs), start=1):
                records.append(rec)
                done_pts = sum(len(r["points"]) for r in records)
                tag = "cached" if rec.get("cached") else f"{rec['seconds']:.0f}s"
                note = rec.get("excluded", f"{len(rec['points'])} pts")
                print(
                    f"  [{i:>3}/{len(jobs)}] seed={rec['seed']} {rec['kind']:<16} "
                    f"{note:<22} {tag:>7}   total_points={done_pts}",
                    flush=True,
                )
                if i == 1 and not rec.get("cached"):
                    projected = rec["seconds"] * len(jobs) / workers / 60
                    print(f"      -> projected total: ~{projected:.0f} min "
                          f"at {workers} workers\n", flush=True)

    labelled = [r for r in records if r["points"]]
    excluded = [r for r in records if r.get("excluded")]
    all_r = [p["reversibility"] for r in labelled for p in r["points"]]

    # The M3 sanity check: R must be lower after a disturbance than before it. If it is
    # not, the labels are not measuring recoverability and nothing downstream is valid.
    before, after = [], []
    for r in labelled:
        onset = r["perturb"]["onset_step"]
        for p in r["points"]:
            (before if p["timestep"] < onset else after).append(p["reversibility"])

    summary = {
        "gate": "M3",
        "n_episodes_requested": len(jobs),
        "n_episodes_labelled": len(labelled),
        "n_excluded_before_onset": len(excluded),
        "n_points": len(all_r),
        "stride": args.stride,
        "branches": args.branches,
        "branch_noise": args.branch_noise,
        "workers": workers,
        "seconds": round(t["seconds"], 1),
        "mean_R": float(np.mean(all_r)) if all_r else None,
        "mean_R_before_onset": float(np.mean(before)) if before else None,
        "mean_R_after_onset": float(np.mean(after)) if after else None,
        "n_before": len(before),
        "n_after": len(after),
        "frac_R_degenerate": (
            float(np.mean([r in (0.0, 1.0) for r in all_r])) if all_r else None
        ),
        "total_thermal_pause_seconds": round(
            sum(r.get("thermal", {}).get("total_paused_seconds", 0.0) for r in records), 1
        ),
        "max_thermal_state": max(
            (r.get("thermal", {}).get("max_state_seen", 0) for r in records), default=0
        ),
        "episodes": [{k: v for k, v in r.items() if k != "points"} for r in records],
    }
    out = write_json(args.out, summary)

    drop_ok = (
        summary["mean_R_before_onset"] is not None
        and summary["mean_R_after_onset"] is not None
        and summary["mean_R_after_onset"] < summary["mean_R_before_onset"]
    )
    print(
        f"\nMILESTONE: M3\n"
        f"GATE: {'PASS' if drop_ok else 'FAIL'} (mean R must drop after perturbation onset)\n"
        f"MEASURED: {summary['n_points']} labelled states over "
        f"{summary['n_episodes_labelled']} episodes "
        f"({summary['n_excluded_before_onset']} excluded: ended before onset)\n"
        f"          mean R before onset {summary['mean_R_before_onset']}  (n={len(before)})\n"
        f"          mean R after  onset {summary['mean_R_after_onset']}  (n={len(after)})\n"
        f"          fraction of R exactly 0 or 1: {summary['frac_R_degenerate']}\n"
        f"WALL CLOCK: {t['seconds'] / 60:.1f} min at {workers} workers, "
        f"thermal pauses {summary['total_thermal_pause_seconds']:.0f}s, "
        f"max thermal state {summary['max_thermal_state']}\n"
        f"FILES: {out}, {SHARD_DIR}/"
    )
    return 0 if drop_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
