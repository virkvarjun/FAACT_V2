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
from faact.envs.perturb import KINDS, PerturbationSpec, sample_spec  # noqa: E402
from faact.eval.ablation import Condition, align_conditions, build_table, run_condition  # noqa: E402
from faact.eval.metrics import markdown_table  # noqa: E402
from faact.runtime.controller import H_MAX, H_MIN, fixed, oracle, reversibility_gated  # noqa: E402
from faact.thermal import ThermalGovernor, limit_torch_threads  # noqa: E402
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


def load_oracle_labels(shard_dir: Path) -> dict[int, dict]:
    """Load measured R per timestep for each labelled episode, keyed by seed.

    These are branch-rollout ground truth, so the oracle condition needs no estimator at
    all — which is the point. If the oracle also fails to beat a fixed horizon, then the
    limitation is the *lever*, not our ability to predict R.
    """
    labels: dict[int, dict] = {}
    for path in sorted(shard_dir.glob("episode_*.json")):
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a shard being written by a concurrent labelling job
        if not rec.get("points"):
            continue
        labels[int(rec["seed"])] = {
            "perturb": rec["perturb"],
            "R": {int(p["timestep"]): float(p["reversibility"]) for p in rec["points"]},
        }
    if not labels:
        raise FileNotFoundError(f"no labelled episodes in {shard_dir}")
    return labels


def load_oof_scorers(path: Path) -> dict[int, object]:
    """Map each episode seed to a scorer whose head never trained on that episode.

    The single head saved by script 05 is refit on all data, so scoring a labelled episode
    with it would be scoring the training set — which would flatter the learned controller
    exactly where it is being compared against the oracle.
    """
    import torch

    from faact.labeling.dataset import Standardiser
    from faact.models.reversibility import ReversibilityHead, ScorerAdapter

    blob = torch.load(path, map_location="cpu", weights_only=False)
    per_seed: dict[int, object] = {}
    for fold in blob["folds"]:
        head = ReversibilityHead(input_dim=blob["input_dim"])
        head.load_state_dict(fold["state_dict"])
        head.eval()
        scorer = ScorerAdapter(
            head, Standardiser(mean=np.asarray(fold["mean"]), std=np.asarray(fold["std"])),
            blob["feature_keys"],
        )
        for seed in fold["held_out_episodes"]:
            per_seed[int(seed)] = scorer
    return per_seed


def build_conditions(scorer, gammas: list[float], h_min: int, fixed_hs: list[int],
                     oracle_labels: dict[int, dict] | None = None,
                     oof_scorers: dict[int, object] | None = None):
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
    if oof_scorers:
        # Learned controller, scored per episode by a head that never saw it. Directly
        # comparable to the oracle rows on the same seeds.
        for gamma in gammas:
            conditions.append(
                Condition(
                    f"gated_oof (gamma={gamma}, h_min={h_min})",
                    (lambda g: lambda seed: reversibility_gated(
                        oof_scorers[seed], h_min=h_min, gamma=g))(gamma),
                    "the claim, evaluated out-of-fold",
                )
            )
    if oracle_labels:
        # Ceiling condition: the same controller fed measured R instead of a prediction.
        # Its gap to the gated rows is exactly the cost of estimation error.
        for gamma in gammas:
            conditions.append(
                Condition(
                    f"oracle_R (gamma={gamma}, h_min={h_min})",
                    (lambda g: lambda seed: oracle(
                        oracle_labels[seed]["R"], h_min=h_min, gamma=g))(gamma),
                    "ceiling: perfect reversibility knowledge",
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
    ap.add_argument("--oracle-shards", default=None,
                    help="label dir; adds the oracle condition and replays those episodes")
    ap.add_argument("--oof-heads", default=None,
                    help="per-fold heads from script 05; adds an uncontaminated gated row")
    ap.add_argument("--out", default="ablation.json")
    args = ap.parse_args()

    limit_torch_threads(1)
    governor = ThermalGovernor()

    # One fixed episode set, built once and reused by every condition.
    oracle_labels = load_oracle_labels(Path(args.oracle_shards)) if args.oracle_shards else {}
    if oracle_labels:
        # Replay the *labelled* episodes with their recorded specs. The oracle needs
        # ground-truth R for the very episodes being run, and re-sampling a spec here
        # would silently pair each episode with a different disturbance than it was
        # measured under.
        episodes = [(seed, PerturbationSpec.from_dict(rec["perturb"]))
                    for seed, rec in sorted(oracle_labels.items())][: args.episodes]
        print(f"replaying {len(episodes)} labelled episodes for the oracle condition")
    else:
        episodes = []
        for i in range(args.episodes):
            seed = args.seed + i
            rng = np.random.default_rng(seed)
            kind = args.kinds[i % len(args.kinds)]
            episodes.append((seed, sample_spec(kind, rng, onset_range=ONSET_RANGE)))

    scorer = None if args.skip_gated else load_scorer(Path(args.head))
    oof_scorers = load_oof_scorers(Path(args.oof_heads)) if args.oof_heads else {}
    conditions = build_conditions(scorer, args.gammas, args.h_min, args.fixed_hs,
                                  oracle_labels, oof_scorers)

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
        "seeds": [episodes[0][0], episodes[-1][0]] if episodes else [],
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
        f"MEASURED: n={n_common} common episodes, seeds {episodes[0][0]}-{episodes[-1][0]}, "
        f"gammas={args.gammas}, h_min={args.h_min}\n"
        f"          gated conditions {'included' if scorer else 'SKIPPED (no head)'}\n"
        f"WALL CLOCK: {t['seconds'] / 60:.1f} min\n"
        f"FILES: {out}, {ARTIFACTS / 'ablation_table.md'}"
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
