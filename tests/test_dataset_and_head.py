"""Tests for dataset assembly, episode-level splits, and the reversibility head.

The split tests are the important ones. States 25 steps apart in the same episode are
near-duplicates, so a random split over *states* leaks the test set into training and
produces a model that looks calibrated and is worthless — a worse failure than an obviously
bad score, because nothing flags it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from faact.labeling.dataset import (
    ReversibilityDataset,
    Standardiser,
    episode_split,
    load_shards,
)
from faact.models.reversibility import ReversibilityHead, evaluate, train_head


def make_dataset(n_episodes: int = 10, points_per_episode: int = 8, dim: int = 6):
    rng = np.random.default_rng(0)
    n = n_episodes * points_per_episode
    return ReversibilityDataset(
        X=rng.normal(size=(n, dim)).astype(np.float32),
        y=rng.uniform(size=n).astype(np.float32),
        episode_id=np.repeat(np.arange(n_episodes), points_per_episode),
        timestep=np.tile(np.arange(points_per_episode) * 25, n_episodes),
        onset=np.repeat(100, n),
        kind=["grasp_slip"] * n,
        feature_keys=[f"feat_{i}" for i in range(dim)],
    )


# -- episode-level splitting -------------------------------------------------------------


def test_no_episode_appears_in_two_splits():
    """The leak test the build spec requires."""
    train, val, test = episode_split(make_dataset(n_episodes=10))

    ids = [set(np.unique(s.episode_id).tolist()) for s in (train, val, test)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_split_preserves_every_state_exactly_once():
    data = make_dataset(n_episodes=10)
    train, val, test = episode_split(data)
    assert len(train) + len(val) + len(test) == len(data)


def test_all_states_of_an_episode_stay_together():
    data = make_dataset(n_episodes=10, points_per_episode=8)
    for split in episode_split(data):
        for ep in np.unique(split.episode_id):
            assert (split.episode_id == ep).sum() == 8


def test_split_is_deterministic_for_a_seed():
    data = make_dataset()
    a = episode_split(data, seed=1)[0].episode_id
    b = episode_split(data, seed=1)[0].episode_id
    np.testing.assert_array_equal(a, b)
    c = episode_split(data, seed=2)[0].episode_id
    assert not np.array_equal(np.unique(a), np.unique(c))


def test_split_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        episode_split(make_dataset(), fractions=(0.5, 0.2, 0.2))


def test_split_raises_rather_than_returning_an_empty_split():
    """Too few episodes must fail loudly, not yield a meaningless validation set."""
    with pytest.raises(ValueError, match="empty"):
        episode_split(make_dataset(n_episodes=2))


# -- standardiser --------------------------------------------------------------------------


def test_standardiser_normalises_training_data():
    X = np.random.default_rng(0).normal(loc=5.0, scale=3.0, size=(200, 4)).astype(np.float32)
    s = Standardiser.fit(X)
    Z = s.transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-5)


def test_standardiser_survives_a_constant_feature():
    """A constant column must not divide by zero and produce nan/inf downstream."""
    X = np.column_stack([np.ones(50), np.arange(50)]).astype(np.float32)
    Z = Standardiser.fit(X).transform(X)
    assert np.isfinite(Z).all()


# -- shard loading -------------------------------------------------------------------------


def test_load_shards_reads_points_and_skips_excluded_episodes(tmp_path):
    def shard(seed, points, excluded=False):
        rec = {
            "seed": seed,
            "kind": "grasp_slip",
            "perturb": {"onset_step": 100},
            "points": points,
        }
        if excluded:
            rec["excluded"] = "ended_before_onset"
        (tmp_path / f"episode_{seed}.json").write_text(json.dumps(rec))

    pts = [
        {"timestep": t, "reversibility": r, "features": {"feat_a": [1.0, 2.0], "feat_b": [3.0]}}
        for t, r in [(25, 1.0), (50, 0.5)]
    ]
    shard(2000, pts)
    shard(2001, pts)
    shard(2002, [], excluded=True)  # must be skipped, not counted as data

    data = load_shards(tmp_path)
    assert len(data) == 4 and data.n_episodes == 2
    assert data.X.shape == (4, 3)  # feat_a (2) + feat_b (1), in sorted key order
    assert data.feature_keys == ["feat_a", "feat_b"]
    np.testing.assert_array_equal(sorted(np.unique(data.episode_id)), [2000, 2001])


def test_load_shards_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no episode shards"):
        load_shards(tmp_path)


# -- head ------------------------------------------------------------------------------------


def test_head_outputs_probabilities_in_the_unit_interval():
    head = ReversibilityHead(input_dim=6)
    pred = head.predict(np.random.default_rng(0).normal(size=(20, 6)))
    assert pred.shape == (20,)
    assert np.all((pred >= 0) & (pred <= 1))


def test_head_learns_a_recoverable_signal():
    """On a synthetic task where R is a known function of the features, it must fit.

    This is a wiring test, not a claim about the real data: it proves the training loop,
    the soft-target BCE, and the sigmoid output actually work together.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - X[:, 1])))  # smooth target in (0, 1)

    head = ReversibilityHead(input_dim=6, dropout=0.0)
    train_head(head, X[:400], y[:400], X[400:500], y[400:500], epochs=150, patience=30)

    metrics = evaluate(head, X[500:], y[500:])
    assert metrics["spearman"] > 0.9, metrics
    assert metrics["mae"] < 0.1, metrics


def test_training_restores_the_best_validation_checkpoint():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4)).astype(np.float32)
    y = rng.uniform(size=200).astype(np.float32)  # pure noise: val loss will not improve

    head = ReversibilityHead(input_dim=4)
    result = train_head(head, X[:150], y[:150], X[150:], y[150:], epochs=100, patience=5)

    assert result.best_epoch <= result.epochs_run
    assert result.epochs_run < 100, "early stopping should have fired on unlearnable data"


def test_evaluate_flags_a_degenerate_target_instead_of_returning_nan():
    """All-identical labels make correlation undefined; that must be visible, not silent."""
    head = ReversibilityHead(input_dim=4)
    metrics = evaluate(head, np.zeros((10, 4), dtype=np.float32), np.ones(10, dtype=np.float32))
    assert metrics["degenerate"] is True


def test_scorer_adapter_bundles_standardisation_with_the_model():
    """The runtime must never feed raw features to a model trained on standardised ones."""
    from faact.models.reversibility import ScorerAdapter

    keys = ["feat_a", "feat_b"]
    X = np.random.default_rng(0).normal(loc=10.0, scale=2.0, size=(50, 3)).astype(np.float32)
    std = Standardiser.fit(X)
    head = ReversibilityHead(input_dim=3)

    scorer = ScorerAdapter(head, std, keys)
    r = scorer.predict_one({"feat_a": np.array([10.0, 11.0]), "feat_b": np.array([9.0])})
    assert 0.0 <= r <= 1.0
