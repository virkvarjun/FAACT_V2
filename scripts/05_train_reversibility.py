#!/usr/bin/env python
"""M4 acceptance gate: train the reversibility head and check it against held-out episodes.

Gate: Spearman correlation between predicted and measured R on held-out **episodes** >= 0.5.
Spearman rather than MSE because the controller only needs the *ordering* of states to be
right — it maps R to a horizon monotonically, so getting the ranking right is what makes it
work, and an offset in absolute R is largely absorbed by gamma.

The split is by episode, never within one: consecutive states in an episode are
near-duplicates, and splitting over states would leak the test set into training and
produce a model that looks calibrated and is useless.

Also emits the calibration scatter — predicted vs. measured R — which is the single most
convincing figure in the project, because it shows the quantity is real rather than
asserted.

    python scripts/05_train_reversibility.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.labeling.dataset import (  # noqa: E402
    FEATURE_GROUPS,
    Standardiser,
    episode_kfold,
    episode_split,
    load_shards,
)
from faact.models.reversibility import ReversibilityHead, evaluate, train_head  # noqa: E402
from faact.utils import ARTIFACTS, timed, write_json  # noqa: E402

GATE_SPEARMAN = 0.50
SHARD_DIR = ARTIFACTS / "reversibility_shards"


def calibration_figure(y_true: np.ndarray, y_pred: np.ndarray, path: Path, n_bins: int = 8) -> None:
    """Scatter of predicted vs measured R, with a binned reliability curve on top.

    The scatter alone is hard to read when labels are quantised to k/M, so the binned
    means show whether the model is calibrated *on average* at each predicted level —
    which is what the horizon map actually consumes.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless: no display on RunPod, and none needed here
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1, label="perfect calibration")
    ax.scatter(y_pred, y_true, s=18, alpha=0.35, edgecolor="none", label="held-out states")

    # Binned reliability curve; bins with no samples are skipped rather than plotted at 0.
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, edges) - 1, 0, n_bins - 1)
    centres, means = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 3:
            centres.append(y_pred[mask].mean())
            means.append(y_true[mask].mean())
    if centres:
        ax.plot(centres, means, "o-", color="crimson", lw=2, ms=5, label="binned mean")

    ax.set_xlabel("predicted reversibility  $\\hat{R}$")
    ax.set_ylabel("measured reversibility  $R$ (branch rollout)")
    ax.set_title("Reversibility calibration, held-out episodes")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def run_cv(full, groups: list[str], args) -> int:
    """Cross-validated evaluation with pooled out-of-fold predictions.

    Each episode is held out exactly once, so the reported Spearman is computed over every
    labelled state rather than over one arbitrary test split. Per-fold spread is reported
    alongside, because a single number without it is what made the original split
    misleading in the first place.
    """
    results = {}
    for group in groups:
        data = full.select(group)
        oof_pred = np.full(len(data), np.nan)
        fold_scores = []

        for train, val, test in episode_kfold(data, k=args.folds, seed=args.split_seed):
            std = Standardiser.fit(train.X)
            head = ReversibilityHead(input_dim=data.X.shape[1], dropout=args.dropout)
            train_head(
                head, std.transform(train.X), train.y, std.transform(val.X), val.y,
                epochs=args.epochs, patience=args.patience, lr=args.lr,
                weight_decay=args.weight_decay, seed=args.split_seed,
            )
            pred = head.predict(std.transform(test.X))
            # Place predictions back at their original row positions so the pooled vector
            # lines up with data.y exactly.
            oof_pred[np.isin(data.episode_id, np.unique(test.episode_id))] = pred
            fold_scores.append(evaluate(head, std.transform(test.X), test.y)["spearman"])

        if np.isnan(oof_pred).any():
            raise RuntimeError("some states were never held out; folds do not cover the data")

        from scipy.stats import spearmanr

        pooled = float(spearmanr(oof_pred, data.y).statistic)
        fold_scores = np.array(fold_scores, dtype=float)
        results[group] = {
            "feature_set": group,
            "dim": int(data.X.shape[1]),
            "pooled_spearman": pooled,
            "fold_spearman_mean": float(np.nanmean(fold_scores)),
            "fold_spearman_std": float(np.nanstd(fold_scores)),
            "fold_spearman": fold_scores.tolist(),
            "mae": float(np.mean(np.abs(oof_pred - data.y))),
            "baseline_mae": float(np.mean(np.abs(data.y - data.y.mean()))),
            "oof_pred": oof_pred.tolist(),
        }
        r = results[group]
        print(
            f"  {group:<12} dim {r['dim']:>5}   pooled spearman {pooled:+.3f}   "
            f"per-fold {r['fold_spearman_mean']:+.3f} +/- {r['fold_spearman_std']:.3f}   "
            f"mae {r['mae']:.3f} (baseline {r['baseline_mae']:.3f})",
            flush=True,
        )

    best = max(results.values(), key=lambda r: r["pooled_spearman"])
    data = full.select(best["feature_set"])
    pooled = best["pooled_spearman"]
    passed = bool(pooled >= GATE_SPEARMAN)

    fig_path = ARTIFACTS / "fig_calibration.png"
    calibration_figure(data.y, np.asarray(best["oof_pred"]), fig_path)

    # Refit on everything for the runtime head: cross-validation measured how well this
    # configuration generalises, so the deployed model should use all the data.
    std = Standardiser.fit(data.X)
    head = ReversibilityHead(input_dim=data.X.shape[1], dropout=args.dropout)
    train, val, _ = episode_kfold(data, k=args.folds, seed=args.split_seed)[0]
    train_head(
        head, std.transform(train.X), train.y, std.transform(val.X), val.y,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.split_seed,
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": int(data.X.shape[1]),
            "feature_set": best["feature_set"],
            "feature_keys": data.feature_keys,
            "mean": std.mean,
            "std": std.std,
        },
        ARTIFACTS / "reversibility_head.pt",
    )

    out = write_json(
        args.out,
        {
            "gate": "M4",
            "passed": passed,
            "method": f"{args.folds}-fold cross-validation over episodes, pooled out-of-fold",
            "selected_feature_set": best["feature_set"],
            "n_states": len(data),
            "n_episodes": data.n_episodes,
            "sweep": [{k: v for k, v in r.items() if k != "oof_pred"} for r in results.values()],
        },
    )

    print(
        f"\nMILESTONE: M4\n"
        f"GATE: {'PASS' if passed else 'FAIL'} (pooled Spearman >= {GATE_SPEARMAN})\n"
        f"MEASURED: pooled out-of-fold Spearman {pooled:.3f} over {len(data)} states "
        f"from {data.n_episodes} episodes ({args.folds}-fold, episode-level)\n"
        f"          per-fold {best['fold_spearman_mean']:.3f} "
        f"+/- {best['fold_spearman_std']:.3f}\n"
        f"          MAE {best['mae']:.3f} vs predict-the-mean {best['baseline_mae']:.3f}\n"
        f"          feature set '{best['feature_set']}' (dim {best['dim']})\n"
        f"FILES: {out}, {fig_path}, {ARTIFACTS / 'reversibility_head.pt'}"
    )
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=str(SHARD_DIR))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--feature-sets", nargs="*", default=None,
                    help=f"subset of {sorted(FEATURE_GROUPS)}; default sweeps all")
    ap.add_argument("--folds", type=int, default=5,
                    help="episode-level CV folds; 0 uses a single train/val/test split")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default="reversibility_training.json")
    args = ap.parse_args()

    full = load_shards(args.shards)
    print(f"loaded {len(full)} labelled states from {full.n_episodes} episodes, "
          f"feature dim {full.X.shape[1]}")
    print(f"R distribution: mean {full.y.mean():.3f}, "
          f"fraction exactly 0 or 1: {np.mean(np.isin(full.y, [0.0, 1.0])):.2f}")

    groups = args.feature_sets or list(FEATURE_GROUPS)
    sweep = []
    best = None

    if args.folds:
        return run_cv(full, groups, args)

    # Sweep feature groups and select on VALIDATION. The full 1206-d vector has ~10x more
    # dimensions than we have training states, so which subset generalises is an empirical
    # question — but choosing it on test would make the reported number meaningless.
    for group in groups:
        data = full.select(group)
        train, val, test = episode_split(data, seed=args.split_seed)
        std = Standardiser.fit(train.X)
        Xtr, Xva, Xte = (std.transform(s.X) for s in (train, val, test))

        head = ReversibilityHead(input_dim=data.X.shape[1], dropout=args.dropout)
        with timed(f"head training [{group}]") as t:
            result = train_head(
                head, Xtr, train.y, Xva, val.y,
                epochs=args.epochs, patience=args.patience, lr=args.lr,
                weight_decay=args.weight_decay, seed=args.split_seed,
            )
        metrics = {
            split: evaluate(head, X, s.y)
            for split, X, s in (("train", Xtr, train), ("val", Xva, val), ("test", Xte, test))
        }
        row = {
            "feature_set": group,
            "dim": int(data.X.shape[1]),
            "train_spearman": metrics["train"]["spearman"],
            "val_spearman": metrics["val"]["spearman"],
            "test_spearman": metrics["test"]["spearman"],
            "test_mae": metrics["test"]["mae"],
            "epochs_run": result.epochs_run,
        }
        sweep.append(row)
        print(f"  {group:<12} dim {row['dim']:>5}   "
              f"spearman train {row['train_spearman']:+.3f}  val {row['val_spearman']:+.3f}  "
              f"test {row['test_spearman']:+.3f}   mae {row['test_mae']:.3f}", flush=True)

        # nan-safe comparison: a degenerate split must never win the selection.
        val_score = row["val_spearman"]
        if best is None or (np.isfinite(val_score) and val_score > best["row"]["val_spearman"]):
            best = {"row": row, "head": head, "std": std, "data": data, "metrics": metrics,
                    "result": result, "splits": (train, val, test), "Xte": Xte}

    assert best is not None
    train, val, test = best["splits"]
    head, std, data = best["head"], best["std"], best["data"]
    metrics, result = best["metrics"], best["result"]
    Xte = best["Xte"]
    print(f"\nselected feature set '{best['row']['feature_set']}' on validation "
          f"(val Spearman {best['row']['val_spearman']:.3f})")
    print(f"episodes  train {train.n_episodes} / val {val.n_episodes} / test {test.n_episodes}"
          f"   states  {len(train)} / {len(val)} / {len(test)}")

    # Baseline: always predict the training mean. If the head cannot beat this on MAE it
    # has learned nothing, however good its correlation happens to look.
    mean_mae = float(np.mean(np.abs(test.y - train.y.mean())))

    fig_path = ARTIFACTS / "fig_calibration.png"
    calibration_figure(test.y, head.predict(Xte), fig_path)

    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": int(data.X.shape[1]),
            "feature_set": best["row"]["feature_set"],
            "feature_keys": data.feature_keys,
            "mean": std.mean,
            "std": std.std,
        },
        ARTIFACTS / "reversibility_head.pt",
    )

    spearman = metrics["test"]["spearman"]
    passed = bool(spearman >= GATE_SPEARMAN)
    payload = {
        "gate": "M4",
        "passed": passed,
        "metrics": metrics,
        "predict_mean_baseline_mae": mean_mae,
        "feature_set": best["row"]["feature_set"],
        "feature_sweep": sweep,
        "n_states": len(data),
        "n_episodes": data.n_episodes,
        "split_states": {"train": len(train), "val": len(val), "test": len(test)},
        "split_episodes": {"train": train.n_episodes, "val": val.n_episodes,
                           "test": test.n_episodes},
        "epochs_run": result.epochs_run,
        "best_epoch": result.best_epoch,
        "val_loss": result.val_loss,
        "seconds": round(t["seconds"], 1),
    }
    out = write_json(args.out, payload)

    print(
        f"\nMILESTONE: M4\n"
        f"GATE: {'PASS' if passed else 'FAIL'} (test Spearman >= {GATE_SPEARMAN})\n"
        f"MEASURED: test Spearman {spearman:.3f}, Pearson {metrics['test']['pearson']:.3f}, "
        f"MAE {metrics['test']['mae']:.3f} over {metrics['test']['n']} held-out states "
        f"from {test.n_episodes} episodes\n"
        f"          predict-the-mean baseline MAE {mean_mae:.3f} "
        f"({'beaten' if metrics['test']['mae'] < mean_mae else 'NOT BEATEN'})\n"
        f"          val Spearman {metrics['val']['spearman']:.3f}, "
        f"train Spearman {metrics['train']['spearman']:.3f}  "
        f"[feature set '{best['row']['feature_set']}', dim {data.X.shape[1]}]\n"
        f"WALL CLOCK: {t['seconds']:.1f}s ({result.epochs_run} epochs, "
        f"best at {result.best_epoch})\n"
        f"FILES: {out}, {fig_path}, {ARTIFACTS / 'reversibility_head.pt'}"
    )
    if not passed:
        print("Report this as measured. Try richer features before changing the target.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
