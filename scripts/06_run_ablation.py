#!/usr/bin/env python
"""M5: run all five horizon conditions on one fixed set of perturbed episodes.

Every condition sees the same seeds and the same perturbation specs; only the horizon
policy changes. Conditions are then aligned to the seeds that survived in all of them, so
the table compares one experiment rather than five slightly different ones.

Needs a trained reversibility head (script 05) for the gated conditions. Without one it
runs the fixed-horizon conditions and says so, rather than silently substituting a
constant — that is the kind of quiet degradation this repo exists to avoid.

    python scripts/06_run_ablation.py --episodes 30 --head artifacts/reversibility_head.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.envs.perturb import KINDS, sample_spec  # noqa: E402
from faact.eval.ablation import Condition, align_conditions, build_table, run_condition  # noqa: E402
from faact.eval.metrics import markdown_table  # noqa: E402
from faact.runtime.controller import H_MAX, H_MIN, fixed, reversibility_gated  # noqa: E402
from faact.thermal import ThermalGovernor, limit_torch_threads  # noqa: E402
from faact.utils import ARTIFACTS, timed, write_json  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
ONSET_RANGE = (40, 160)


def load_scorer(head_path: Path):
    """Load a trained head plus its standardiser, or raise with a pointer to script 05."""
    import torch

    from faact.labeling.dataset import Standardiser
    from faact.models.reversibility import ReversibilityHead, ScorerAdapter

    if not head_path.exists():
        raise FileNotFoundError(
            f"no reversibility head at {head_path}. Run scripts/05_train_reversibility.py "
            "first, or pass --skip-gated to run only the fixed-horizon conditions."
        )
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    head = ReversibilityHead(input_dim=blob["input_dim"])
    head.load_state_dict(blob["state_dict"])
    head.eval()
    std = Standardiser(mean=np.asarray(blob["mean"]), std=np.asarray(blob["std"]))
    return ScorerAdapter(head, std, blob["feature_keys"])


def build_conditions(scorer, gammas: list[float], h_min: int, fixed_hs: list[int]):
    """Ablation rows. Gated rows are omitted when no head is available.

    `h_min` is the floor the gated controller may shorten to, and it matters more than it
    looks: fixed h=5 measured 0/30 on this task, because ACT emits absolute joint targets
    and replanning that often — without temporal ensembling — makes each new chunk jump to
    a different target and the motion falls apart. A gated controller whose floor sits in
    that regime is penalised for reacting, which inverts the intent.
    """
    conditions = [
        Condition(f"fixed h={h}", (lambda h: lambda _seed: fixed(h))(h), "fixed baseline")
        for h in fixed_hs
    ]
    if scorer is not None:
        for gamma in gammas:
            conditions.append(
                Condition(
                    f"reversibility_gated (gamma={gamma}, h_min={h_min})",
                    (lambda g: lambda _seed: reversibility_gated(scorer, h_min=h_min, gamma=g))(gamma),
                    "the claim",
                )
            )
    return conditions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3000, help="first episode seed")
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--gammas", type=float, nargs="*", default=[1.0])
    ap.add_argument("--h-min", type=int, default=H_MIN,
                    help="floor for the gated horizon; 5 is the measured collapse regime")
    ap.add_argument("--fixed-hs", type=int, nargs="*", default=[100, 20, 5])
    ap.add_argument("--head", default=str(ARTIFACTS / "reversibility_head.pt"))
    ap.add_argument("--skip-gated", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    ap.add_argument("--out", default="ablation.json")
    args = ap.parse_args()

    limit_torch_threads(1)
    governor = ThermalGovernor()

    # One fixed episode set, built once and reused by every condition.
    episodes = []
    for i in range(args.episodes):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        kind = args.kinds[i % len(args.kinds)]
        episodes.append((seed, sample_spec(kind, rng, onset_range=ONSET_RANGE)))

    scorer = None if args.skip_gated else load_scorer(Path(args.head))
    conditions = build_conditions(scorer, args.gammas, args.h_min, args.fixed_hs)

    policy = ACTWrapper(CHECKPOINT, device="cpu")
    env = make_env()

    per_condition, exclusions = {}, {}
    with timed("ablation") as t:
        for cond in conditions:
            governor.checkpoint()
            with timed(cond.name) as ct:
                results, excluded = run_condition(
                    env, policy, cond, episodes, max_steps=args.max_steps
                )
            per_condition[cond.name] = results
            exclusions[cond.name] = excluded
            print(
                f"  {cond.name:<32} {sum(r.success for r in results)}/{len(results)} "
                f"({len(excluded)} excluded)  {ct['seconds']:.0f}s",
                flush=True,
            )
    env.close()

    aligned = align_conditions(per_condition)
    n_common = len(next(iter(aligned.values()))) if aligned else 0
    rows = build_table(aligned)
    table = markdown_table(rows)

    payload = {
        "gate": "M5",
        "n_episodes_common": n_common,
        "n_episodes_requested": args.episodes,
        "gammas": args.gammas,
        "h_min": args.h_min,
        "onset_range": ONSET_RANGE,
        "seeds": [args.seed, args.seed + args.episodes - 1],
        "gated_included": scorer is not None,
        "seconds": round(t["seconds"], 1),
        "thermal": governor.report(),
        "rows": rows,
        "exclusions": {k: v for k, v in exclusions.items()},
        # Per-episode traces, kept so figure 3 (committed horizon vs. onset) and the
        # side-by-side videos can be built without re-running the whole ablation.
        "episodes": {
            name: [r.to_record() for r in results] for name, results in aligned.items()
        },
    }
    out = write_json(args.out, payload)
    (ARTIFACTS / "ablation_table.md").write_text(table + "\n")

    print(f"\n{table}\n")
    print(
        f"MILESTONE: M5\n"
        f"GATE: {'PASS' if rows else 'FAIL'} (table produced with n stated)\n"
        f"MEASURED: n={n_common} common episodes, seeds {args.seed}-"
        f"{args.seed + args.episodes - 1}, gammas={args.gammas}, h_min={args.h_min}\n"
        f"          gated conditions {'included' if scorer else 'SKIPPED (no head)'}\n"
        f"WALL CLOCK: {t['seconds'] / 60:.1f} min\n"
        f"FILES: {out}, {ARTIFACTS / 'ablation_table.md'}"
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
