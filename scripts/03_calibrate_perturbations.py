#!/usr/bin/env python
"""M2 acceptance gate: tune each perturbation kind until it lands in the headroom band.

A perturbation is only useful if it sits between two failure modes:

    too weak    -> success barely drops, there is nothing to recover from
    too strong  -> success hits 0%, and no controller can beat a floor of zero

The target band is 25-65% success under perturbation. That band *is* the experiment: it is
the only regime where "did reversibility-gated horizon control help?" is a measurable
question. v1 never entered it, because its perturbations were never implemented at all.

The script sweeps magnitudes per kind, measures success at each, and picks the magnitude
whose measured rate sits closest to the middle of the band. It reports the whole sweep, not
just the winner, so the choice is auditable rather than asserted.

    python scripts/03_calibrate_perturbations.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.envs.perturb import KINDS, sample_spec  # noqa: E402
from faact.eval.runner import PerturbationNeverFired, run_episode, success_rate  # noqa: E402
from faact.runtime.controller import fixed  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.utils import timed, write_json  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"

BAND = (0.25, 0.65)
BAND_MID = sum(BAND) / 2

# A perturbation must also *reduce* success by at least this much, in absolute percentage
# points, against the unperturbed reference on the same seeds.
#
# The band alone is not a sufficient test and this was caught the hard way: with an
# unperturbed reference of 65%, `grasp_slip` scored 60-65% at every duration — no effect
# whatsoever — and passed the 25-65% band check four times in a row. An absolute band
# cannot distinguish "calibrated" from "does nothing" when the baseline sits at its edge.
MIN_SUCCESS_DROP = 0.15

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

# Sweeps per kind, as (magnitude, duration) pairs. Units differ by kind — see
# faact/envs/perturb.py DEFAULTS. `duration=None` keeps that kind's default.
#
# grasp_slip sweeps duration rather than magnitude: the gripper is either forced fully open
# or it is not, so "how severe" is entirely "for how many steps".
SWEEPS: dict[str, list[tuple[float, int | None]]] = {
    "object_displace": [(m, None) for m in (0.02, 0.035, 0.05, 0.08)],  # metres
    "grasp_slip": [(1.0, d) for d in (2, 3, 6, 12)],                    # steps held open
    "actuation_noise": [(m, None) for m in (0.02, 0.05, 0.10, 0.20)],   # action std dev
    "occlusion": [(m, None) for m in (0.10, 0.25, 0.50, 0.90)],         # image area fraction
}


def evaluate(
    env, executor, kind: str, magnitude: float, duration: int | None, seeds: list[int],
    base_rate: float | None = None,
) -> dict:
    """Run one (kind, magnitude) cell and return its measured success rate.

    Episodes that finish before their onset step are excluded, not counted as perturbed —
    they never experienced the disturbance, so scoring them would inflate the rate with
    unperturbed successes. The exclusion count is reported so it is never invisible.
    """
    results, skipped = [], []
    for seed in seeds:
        # Spec RNG is seeded from the episode seed, so the same cell replays identically.
        rng = np.random.default_rng(seed)
        spec = sample_spec(
            kind, rng, onset_range=ONSET_RANGE, magnitude=magnitude, duration=duration
        )
        try:
            results.append(
                run_episode(
                    env, executor, seed=seed, perturb_spec=spec, max_steps=MAX_EPISODE_STEPS
                )
            )
        except PerturbationNeverFired as exc:
            skipped.append({"seed": seed, "steps": exc.steps, "onset": spec.onset_step})

    if not results:
        raise RuntimeError(
            f"every episode for {kind} mag={magnitude} ended before its onset; "
            f"ONSET_RANGE={ONSET_RANGE} is too late for this policy"
        )

    rate = success_rate(results)
    drop = None if base_rate is None else base_rate - rate
    return {
        "drop_vs_unperturbed": drop,
        "effective": bool(
            BAND[0] <= rate <= BAND[1] and drop is not None and drop >= MIN_SUCCESS_DROP
        ),
        "kind": kind,
        "magnitude": magnitude,
        "duration": duration,
        "success_rate": rate,
        "n_success": sum(r.success for r in results),
        "n_episodes": len(results),
        "n_skipped_before_onset": len(skipped),
        "skipped": skipped,
        "in_band": BAND[0] <= rate <= BAND[1],
        "mean_max_reward": float(np.mean([r.max_reward for r in results])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="perturbation_calibration.json")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.episodes)]
    policy = ACTWrapper(CHECKPOINT, device=args.device)
    # Perturbations are calibrated against ACT as published: full 100-step commitment.
    executor = ChunkExecutor(policy, fixed(100))
    env = make_env()

    # Unperturbed reference on the *same seeds*. Absolute rates are meaningless without
    # it: seeds 1000-1019 score 65% unperturbed while the full 500-seed set scores 83%.
    with timed("unperturbed reference") as t_ref:
        baseline = [run_episode(env, executor, seed=s, max_steps=MAX_EPISODE_STEPS) for s in seeds]
    base_rate = success_rate(baseline)
    print(f"unperturbed reference on these seeds: {base_rate:.0%} "
          f"({sum(r.success for r in baseline)}/{len(baseline)}) in {t_ref['seconds']:.0f}s\n")

    cells, chosen = [], {}
    with timed("perturbation sweep") as t:
        for kind in args.kinds:
            print(f"{kind}:")
            for magnitude, duration in SWEEPS[kind]:
                cell = evaluate(env, executor, kind, magnitude, duration, seeds, base_rate)
                cells.append(cell)
                skipped = cell["n_skipped_before_onset"]
                print(
                    f"  mag={magnitude:<6} dur={str(duration):<5} "
                    f"success={cell['n_success']:>2}/{cell['n_episodes']} "
                    f"= {cell['success_rate']:.0%}  mean_reward={cell['mean_max_reward']:.1f}"
                    f"  drop {cell['drop_vs_unperturbed']:+.0%}"
                    f"{f'  ({skipped} ended before onset)' if skipped else ''}"
                    f"  {'EFFECTIVE' if cell['effective'] else ''}",
                    flush=True,
                )

            # Prefer an in-band magnitude closest to the band midpoint; if none landed in
            # band, still record the closest so the failure is explicit rather than empty.
            kind_cells = [c for c in cells if c["kind"] == kind]
            effective = [c for c in kind_cells if c["effective"]]
            pool = effective or kind_cells
            best = min(pool, key=lambda c: abs(c["success_rate"] - BAND_MID))
            chosen[kind] = {**best, "any_in_band": bool(effective)}
            print(f"  -> chose mag={best['magnitude']} dur={best['duration']} "
                  f"at {best['success_rate']:.0%}"
                  f"{'' if in_band else '  (NO SETTING WAS EFFECTIVE)'}\n")
    env.close()

    all_in_band = all(v["any_in_band"] for v in chosen.values())
    payload = {
        "gate": "M2",
        "passed": all_in_band,
        "band": BAND,
        "min_success_drop": MIN_SUCCESS_DROP,
        "onset_range": ONSET_RANGE,
        "unperturbed_reference": {
            "success_rate": base_rate,
            "n_success": sum(r.success for r in baseline),
            "n_episodes": len(baseline),
        },
        "seeds": [seeds[0], seeds[-1]],
        "device": policy.device,
        "seconds": round(t["seconds"] + t_ref["seconds"], 1),
        "chosen": chosen,
        "sweep": cells,
    }
    out = write_json(args.out, payload)

    print(f"MILESTONE: M2\nGATE: {'PASS' if all_in_band else 'FAIL'} "
          f"(every kind needs a setting inside {BAND[0]:.0%}-{BAND[1]:.0%} AND dropping "
          f"success by >= {MIN_SUCCESS_DROP:.0%} vs unperturbed)")
    print(f"MEASURED: unperturbed {base_rate:.0%} on seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)}")
    for kind, c in chosen.items():
        print(f"          {kind:<17} mag={c['magnitude']:<6} dur={str(c['duration']):<5} "
              f"{c['n_success']}/{c['n_episodes']} = {c['success_rate']:.0%}"
              f"  drop {c['drop_vs_unperturbed']:+.0%}"
              f"{'' if c['any_in_band'] else '   NOT EFFECTIVE'}")
    print(f"WALL CLOCK: {payload['seconds']:.0f}s\nFILES: {out}")
    return 0 if all_in_band else 1


if __name__ == "__main__":
    raise SystemExit(main())
