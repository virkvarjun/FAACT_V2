#!/usr/bin/env python
"""M1 verification: reproduce LeRobot's official ACT evaluation, seed for seed.

The build plan assumed ACT might need retraining. It does not — the HF checkpoint loads
against lerobot 0.3.2. But "it loads" is not evidence it is *correct*, and a subtly wrong
inference path (bad normalisation, wrong image layout, an off-by-one in chunk indexing)
would still produce plausible-looking episodes and quietly cap every downstream result.

So we verify against ground truth instead of against a threshold. The checkpoint repo ships
`eval_info.json`: 500 per-episode records from the official evaluation, each tagged with the
seed that produced it, starting at seed 1000. Our runner uses the same seeds, so we can
compare episode by episode rather than rate to rate.

Why this matters more than a success-rate gate: the official run scores **83% over 500
episodes but only 65% over the first 20**, because those seeds are a harder-than-average
draw. A 20-episode run measured against "80-90%" would look like a broken checkpoint when
it is in fact exact. Per-seed agreement is invariant to that.

    python scripts/01_verify_act_checkpoint.py --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.eval.runner import run_episode, success_rate  # noqa: E402
from faact.runtime.controller import fixed  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.utils import timed, write_json  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
# Seed of the official run's first episode; subsequent episodes increment by one.
OFFICIAL_FIRST_SEED = 1000
# Agreement below this means our inference path differs from the reference in a way that
# changes outcomes, not just in float noise.
GATE_AGREEMENT = 0.90


def load_official() -> dict[int, dict]:
    """Fetch the official per-episode results, keyed by seed."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CHECKPOINT, "eval_info.json")
    info = json.load(open(path))
    return {e["seed"]: e for e in info["per_episode"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=100, help="ACT as published commits all 100")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="act_verification.json")
    args = ap.parse_args()

    official = load_official()
    seeds = [OFFICIAL_FIRST_SEED + i for i in range(args.episodes)]
    missing = [s for s in seeds if s not in official]
    if missing:
        raise SystemExit(f"official eval has no record for seeds {missing[:5]}...")

    policy = ACTWrapper(CHECKPOINT, device=args.device)
    executor = ChunkExecutor(policy, fixed(args.horizon))
    env = make_env()

    rows, results = [], []
    with timed("verification eval") as t:
        for seed in seeds:
            r = run_episode(env, executor, seed=seed, max_steps=MAX_EPISODE_STEPS)
            results.append(r)
            o = official[seed]
            rows.append(
                {
                    "seed": seed,
                    "ours_success": bool(r.success),
                    "official_success": bool(o["success"]),
                    "ours_max_reward": r.max_reward,
                    "official_max_reward": o["max_reward"],
                    "agree": bool(r.success) == bool(o["success"]),
                }
            )
            print(
                f"  seed {seed}  ours={int(r.success)}({r.max_reward:.0f})  "
                f"official={int(o['success'])}({o['max_reward']:.0f})  "
                f"{'OK' if rows[-1]['agree'] else 'DIFFER'}",
                flush=True,
            )
    env.close()

    n = len(rows)
    n_agree = sum(r["agree"] for r in rows)
    agreement = n_agree / n
    ours = success_rate(results)
    theirs = sum(r["official_success"] for r in rows) / n
    passed = agreement >= GATE_AGREEMENT

    payload = {
        "gate": "M1-verify",
        "passed": passed,
        "agreement": agreement,
        "n_agree": n_agree,
        "n_episodes": n,
        "our_success_rate": ours,
        "official_success_rate_same_seeds": theirs,
        "official_success_rate_500_episodes": 0.83,
        "checkpoint": CHECKPOINT,
        "device": policy.device,
        "horizon": args.horizon,
        "seconds": round(t["seconds"], 1),
        "per_seed": rows,
    }
    out = write_json(args.out, payload)

    print(
        f"\nMILESTONE: M1-verify\n"
        f"GATE: {'PASS' if passed else 'FAIL'} (per-seed agreement >= {GATE_AGREEMENT:.0%})\n"
        f"MEASURED: agreement {n_agree}/{n} = {agreement:.0%}\n"
        f"          ours     {ours:.1%} on seeds {seeds[0]}-{seeds[-1]}\n"
        f"          official {theirs:.1%} on the same seeds "
        f"(83.0% over its full 500)\n"
        f"          device={policy.device}, h={args.horizon}\n"
        f"WALL CLOCK: {t['seconds']:.1f}s ({t['seconds'] / n:.1f}s/episode)\n"
        f"FILES: {out}"
    )
    if not passed:
        differ = [r["seed"] for r in rows if not r["agree"]]
        print(f"disagreeing seeds: {differ}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
