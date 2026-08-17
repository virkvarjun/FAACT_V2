#!/usr/bin/env python
"""M1 acceptance gate: unperturbed ACT success >= 60% over 20 episodes at fixed h=100.

This is the blocking gate for the whole project. A policy that cannot do the task
unperturbed leaves no headroom: v1's baseline was 0/20, and no intervention method can
show a lift over a floor of zero. If this prints FAIL, nothing downstream is meaningful.

    python scripts/02_eval_baseline.py --checkpoint lerobot/act_aloha_sim_transfer_cube_human
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.eval.runner import run_episode, success_rate  # noqa: E402
from faact.runtime.controller import fixed  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.utils import timed, write_json  # noqa: E402

GATE_SUCCESS_RATE = 0.60


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="lerobot/act_aloha_sim_transfer_cube_human")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=100, help="fixed commitment horizon")
    ap.add_argument("--seed", type=int, default=1000, help="first episode seed")
    ap.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="baseline_eval.json")
    args = ap.parse_args()

    policy = ACTWrapper(args.checkpoint, device=args.device)
    executor = ChunkExecutor(policy, fixed(args.horizon))
    env = make_env()

    results = []
    with timed("baseline eval") as t:
        for i in range(args.episodes):
            seed = args.seed + i
            ep_start = time.perf_counter()
            r = run_episode(env, executor, seed=seed, max_steps=args.max_steps)
            results.append(r)
            print(
                f"  ep {i + 1:>2}/{args.episodes} seed={seed} "
                f"success={int(r.success)} reward={r.max_reward:.0f} steps={r.steps:>3} "
                f"({time.perf_counter() - ep_start:.1f}s)",
                flush=True,
            )
    env.close()

    rate = success_rate(results)
    passed = rate >= GATE_SUCCESS_RATE

    payload = {
        "gate": "M1",
        "passed": passed,
        "success_rate": rate,
        "n_success": sum(r.success for r in results),
        "n_episodes": len(results),
        "checkpoint": args.checkpoint,
        "device": policy.device,
        "horizon": args.horizon,
        "first_seed": args.seed,
        "max_steps": args.max_steps,
        "seconds": round(t["seconds"], 1),
        "episodes": [r.to_record() for r in results],
    }
    out = write_json(args.out, payload)

    print(
        f"\nMILESTONE: M1\n"
        f"GATE: {'PASS' if passed else 'FAIL'} (threshold {GATE_SUCCESS_RATE:.0%})\n"
        f"MEASURED: success {payload['n_success']}/{payload['n_episodes']} = {rate:.0%}, "
        f"h={args.horizon}, seeds {args.seed}-{args.seed + args.episodes - 1}, "
        f"device={policy.device}, checkpoint={args.checkpoint}\n"
        f"WALL CLOCK: {t['seconds']:.1f}s ({t['seconds'] / max(1, len(results)):.1f}s/episode)\n"
        f"FILES: {out}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
