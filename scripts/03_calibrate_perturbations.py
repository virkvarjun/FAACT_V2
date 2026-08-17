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
from faact.eval.runner import run_episode, success_rate  # noqa: E402
from faact.runtime.controller import fixed  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.utils import timed, write_json  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"

BAND = (0.25, 0.65)
BAND_MID = sum(BAND) / 2

# Onset window: after the arms have engaged the cube but before a successful transfer
# completes (unperturbed successes finish around step 230-320). Perturbing here disturbs
# an episode that was otherwise on track, which is the case reversibility is about.
ONSET_RANGE = (40, 160)

# Magnitude sweeps per kind. Units differ by kind — see faact/envs/perturb.py DEFAULTS.
SWEEPS: dict[str, list[float]] = {
    "object_displace": [0.02, 0.035, 0.05, 0.08],  # metres
    "grasp_slip": [1.0],                            # binary: gripper is forced open or not
    "actuation_noise": [0.02, 0.05, 0.10, 0.20],    # action-space std dev
    "occlusion": [0.10, 0.25, 0.50, 0.90],          # fraction of image area
}


def evaluate(env, executor, kind: str, magnitude: float, seeds: list[int]) -> dict:
    """Run one (kind, magnitude) cell and return its measured success rate."""
    results = []
    for seed in seeds:
        # Spec RNG is seeded from the episode seed, so the same cell replays identically.
        rng = np.random.default_rng(seed)
        spec = sample_spec(kind, rng, onset_range=ONSET_RANGE, magnitude=magnitude)
        results.append(
            run_episode(env, executor, seed=seed, perturb_spec=spec, max_steps=MAX_EPISODE_STEPS)
        )

    rate = success_rate(results)
    return {
        "kind": kind,
        "magnitude": magnitude,
        "success_rate": rate,
        "n_success": sum(r.success for r in results),
        "n_episodes": len(results),
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
            for magnitude in SWEEPS[kind]:
                cell = evaluate(env, executor, kind, magnitude, seeds)
                cells.append(cell)
                print(
                    f"  mag={magnitude:<6} success={cell['n_success']:>2}/{cell['n_episodes']} "
                    f"= {cell['success_rate']:.0%}  mean_reward={cell['mean_max_reward']:.1f}"
                    f"  {'IN BAND' if cell['in_band'] else ''}",
                    flush=True,
                )

            # Prefer an in-band magnitude closest to the band midpoint; if none landed in
            # band, still record the closest so the failure is explicit rather than empty.
            kind_cells = [c for c in cells if c["kind"] == kind]
            in_band = [c for c in kind_cells if c["in_band"]]
            pool = in_band or kind_cells
            best = min(pool, key=lambda c: abs(c["success_rate"] - BAND_MID))
            chosen[kind] = {**best, "any_in_band": bool(in_band)}
            print(f"  -> chose mag={best['magnitude']} at {best['success_rate']:.0%}"
                  f"{'' if in_band else '  (NO MAGNITUDE LANDED IN BAND)'}\n")
    env.close()

    all_in_band = all(v["any_in_band"] for v in chosen.values())
    payload = {
        "gate": "M2",
        "passed": all_in_band,
        "band": BAND,
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
          f"(every kind must have a magnitude in {BAND[0]:.0%}-{BAND[1]:.0%})")
    print(f"MEASURED: unperturbed {base_rate:.0%} on seeds {seeds[0]}-{seeds[-1]}, n={len(seeds)}")
    for kind, c in chosen.items():
        print(f"          {kind:<17} mag={c['magnitude']:<6} "
              f"{c['n_success']}/{c['n_episodes']} = {c['success_rate']:.0%}"
              f"{'' if c['any_in_band'] else '   OUT OF BAND'}")
    print(f"WALL CLOCK: {payload['seconds']:.0f}s\nFILES: {out}")
    return 0 if all_in_band else 1


if __name__ == "__main__":
    raise SystemExit(main())
