"""Assemble the (features, R) training set, with episode-level splits.

The split rule is the whole point of this file: **an episode's states never straddle two
splits.** Consecutive states within an episode are enormously correlated — 25 steps apart,
from the same seed, the same perturbation, and often the same chunk — so a random split
over *states* puts near-duplicates of the test set into training. That does not merely
inflate the score; it produces a model that looks calibrated and is useless, which is a
worse outcome than an obviously bad one.

Standardisation is fit on the training split alone, for the same reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from faact.backbone.act_wrapper import concat_features


# Feature groups, so the head can be trained on a subset without re-extracting anything.
#
# This exists because of a measured problem: the full vector is 1206-d (two 512-d
# transformer poolings plus 182-d of action/state features), while 40 episodes at stride 25
# yield only ~300 training states. Ten times more dimensions than samples overfits hard —
# first dry run gave train Spearman 0.76 against test 0.26. Which group actually carries
# the signal is then an empirical question worth reporting rather than assuming.
FEATURE_GROUPS: dict[str, list[str] | None] = {
    # Cheap, low-dimensional, and available without any hook: 182-d.
    "cheap": [
        "feat_state",
        "feat_action_first",
        "feat_action_prefix_mean_10",
        "feat_action_prefix_flat_10",
    ],
    # ACT's internal representation: 1024-d.
    "transformer": ["feat_encoder_mean", "feat_decoder_mean"],
    # Smallest sensible set: where the arm is and what it is about to do. 28-d.
    "minimal": ["feat_state", "feat_action_first"],
    "all": None,
}


@dataclass
class ReversibilityDataset:
    """Flat arrays plus the episode id each row came from."""

    X: np.ndarray  # (n_states, n_features)
    y: np.ndarray  # (n_states,) reversibility in [0, 1]
    episode_id: np.ndarray  # (n_states,)
    timestep: np.ndarray  # (n_states,)
    onset: np.ndarray  # (n_states,) perturbation onset for the episode, for analysis
    kind: list[str]  # per-state perturbation kind
    feature_keys: list[str]
    # Column span of each feature key in X, so a group can be sliced out after loading.
    feature_slices: dict[str, slice] = None  # type: ignore[assignment]

    def select(self, group: str) -> "ReversibilityDataset":
        """Return a copy keeping only the columns of one feature group."""
        if group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {group!r}; have {sorted(FEATURE_GROUPS)}")
        keys = FEATURE_GROUPS[group]
        if keys is None:
            return self
        if self.feature_slices is None:
            raise RuntimeError("dataset has no feature_slices; rebuild it with load_shards()")

        missing = [k for k in keys if k not in self.feature_slices]
        if missing:
            raise KeyError(f"feature group {group!r} needs {missing}, absent from the labels")

        keys = [k for k in self.feature_keys if k in set(keys)]  # keep canonical order
        cols = np.concatenate([np.arange(self.feature_slices[k].start,
                                         self.feature_slices[k].stop) for k in keys])
        out = ReversibilityDataset(
            X=self.X[:, cols],
            y=self.y,
            episode_id=self.episode_id,
            timestep=self.timestep,
            onset=self.onset,
            kind=self.kind,
            feature_keys=keys,
            feature_slices=None,
        )
        # Recompute spans for the narrowed matrix so select() stays composable.
        start, spans = 0, {}
        for k in keys:
            width = self.feature_slices[k].stop - self.feature_slices[k].start
            spans[k] = slice(start, start + width)
            start += width
        out.feature_slices = spans
        return out

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_episodes(self) -> int:
        return int(np.unique(self.episode_id).size)

    def subset(self, mask: np.ndarray) -> "ReversibilityDataset":
        mask = np.asarray(mask)
        return ReversibilityDataset(
            X=self.X[mask],
            y=self.y[mask],
            episode_id=self.episode_id[mask],
            timestep=self.timestep[mask],
            onset=self.onset[mask],
            kind=[k for k, m in zip(self.kind, mask) if m],
            feature_keys=self.feature_keys,
            feature_slices=self.feature_slices,
        )


def load_shards(shard_dir: str | Path, feature_keys: list[str] | None = None) -> ReversibilityDataset:
    """Build a dataset from the per-episode JSON shards written by script 04."""
    shard_dir = Path(shard_dir)
    files = sorted(shard_dir.glob("episode_*.json"))
    if not files:
        raise FileNotFoundError(f"no episode shards in {shard_dir}")

    rows_X, rows_y, eps, ts, onsets, kinds = [], [], [], [], [], []
    last_feats: dict = {}
    for path in files:
        rec = json.loads(path.read_text())
        if not rec.get("points"):
            continue  # excluded episode (ended before onset) — carries no signal
        onset = rec["perturb"]["onset_step"]
        for point in rec["points"]:
            feats = {k: np.asarray(v, dtype=np.float32) for k, v in point["features"].items()}
            if feature_keys is None:
                feature_keys = sorted(feats)
            rows_X.append(concat_features(feats, feature_keys))
            last_feats = feats
            rows_y.append(float(point["reversibility"]))
            eps.append(int(rec["seed"]))
            ts.append(int(point["timestep"]))
            onsets.append(int(onset))
            kinds.append(rec["kind"])

    if not rows_X:
        raise ValueError(f"{len(files)} shards contained no labelled states")

    # Column span of each key, matching concat_features' sorted-key order.
    spans, start = {}, 0
    for key in feature_keys:
        width = int(np.asarray(last_feats[key]).size)
        spans[key] = slice(start, start + width)
        start += width

    return ReversibilityDataset(
        X=np.stack(rows_X).astype(np.float32),
        y=np.asarray(rows_y, dtype=np.float32),
        episode_id=np.asarray(eps),
        timestep=np.asarray(ts),
        onset=np.asarray(onsets),
        kind=kinds,
        feature_keys=list(feature_keys or []),
        feature_slices=spans,
    )


def episode_split(
    dataset: ReversibilityDataset,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> tuple[ReversibilityDataset, ReversibilityDataset, ReversibilityDataset]:
    """Split into train/val/test by **episode**, never within one.

    Episodes are shuffled and partitioned, so every state of an episode lands in exactly
    one split. Raises if any split would come out empty, rather than returning a degenerate
    set that produces a meaningless validation number.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions} = {sum(fractions)}")

    episodes = np.unique(dataset.episode_id)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)

    n = len(shuffled)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    groups = [shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]]

    names = ("train", "val", "test")
    empty = [name for name, g in zip(names, groups) if g.size == 0]
    if empty:
        raise ValueError(
            f"splits {empty} are empty with {n} episodes and fractions {fractions}; "
            "label more episodes or change the fractions"
        )

    return tuple(dataset.subset(np.isin(dataset.episode_id, g)) for g in groups)


def episode_kfold(
    dataset: ReversibilityDataset, k: int = 5, seed: int = 0
) -> list[tuple[ReversibilityDataset, ReversibilityDataset, ReversibilityDataset]]:
    """K-fold cross-validation over **episodes**, yielding (train, val, test) per fold.

    A single 24/8/8 split of 40 episodes turned out to be far too noisy to draw conclusions
    from — validation and test disagreed by 0.24 Spearman on the same model. K-fold fixes
    that without needing more data: every episode is tested exactly once, so the pooled
    out-of-fold estimate uses all 493 states instead of ~96.

    Fold i uses fold i as test and fold (i+1) % k as validation (for early stopping), with
    the remaining k-2 folds for training. Episodes never straddle folds, so the leakage
    rule is preserved.
    """
    if k < 3:
        raise ValueError(f"need k >= 3 to carve out a validation fold, got {k}")

    episodes = np.unique(dataset.episode_id)
    if len(episodes) < k:
        raise ValueError(f"{len(episodes)} episodes cannot fill {k} folds")

    shuffled = np.random.default_rng(seed).permutation(episodes)
    folds = np.array_split(shuffled, k)

    splits = []
    for i in range(k):
        test_eps = folds[i]
        val_eps = folds[(i + 1) % k]
        train_eps = np.concatenate([folds[j] for j in range(k) if j not in (i, (i + 1) % k)])
        splits.append(
            tuple(
                dataset.subset(np.isin(dataset.episode_id, eps))
                for eps in (train_eps, val_eps, test_eps)
            )
        )
    return splits


@dataclass
class Standardiser:
    """Zero-mean unit-variance scaling, fit on training data only."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardiser":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # Constant features would divide by zero; leave them at zero rather than blowing up.
        std[std < 1e-8] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)
