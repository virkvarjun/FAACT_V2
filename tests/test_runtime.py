"""Unit tests for the chunk executor and the horizon policies.

No simulator and no trained policy: a fake predictor returns chunks whose values encode
(chunk_index, step_index), so a test can read an executed action and say exactly which
chunk it came from and how far into it. That is what makes commitment horizon — the
quantity this whole project controls — directly assertable.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.runtime.controller import (
    H_MAX,
    H_MIN,
    fixed,
    horizon_from_score,
    oracle,
    score_gated,
)
from faact.runtime.executor import ChunkExecutor

ACTION_DIM = 14
CHUNK_SIZE = 100


class FakePolicy:
    """Emits chunks tagged with their identity: action[i, 0] = chunk_id, [i, 1] = i."""

    chunk_size = CHUNK_SIZE

    def __init__(self) -> None:
        self.n_calls = 0

    def reset(self) -> None:
        self.n_calls = 0

    def predict_chunk(self, obs):
        chunk = np.zeros((CHUNK_SIZE, ACTION_DIM))
        chunk[:, 0] = self.n_calls
        chunk[:, 1] = np.arange(CHUNK_SIZE)
        features = {"feat_dummy": np.array([float(self.n_calls)])}
        self.n_calls += 1
        return chunk, features


def run(executor, n_steps: int) -> np.ndarray:
    """Step the executor for n_steps against a dummy observation."""
    return np.array([executor.act(t, obs={}) for t in range(n_steps)])


# -- horizon map -----------------------------------------------------------------------


def test_horizon_map_is_monotone_in_score():
    hs = [horizon_from_score(s) for s in np.linspace(0, 1, 21)]
    assert hs == sorted(hs)


def test_horizon_map_clips_to_bounds():
    assert horizon_from_score(0.0) == H_MIN
    assert horizon_from_score(1.0) == H_MAX
    assert horizon_from_score(0.001) == H_MIN, "must never return 0 — that would never step"


def test_horizon_map_gamma_is_more_conservative():
    """Higher gamma shortens the horizon at every score below 1."""
    for s in (0.3, 0.5, 0.7, 0.9):
        assert horizon_from_score(s, gamma=2.0) <= horizon_from_score(s, gamma=1.0)


def test_horizon_map_rejects_scores_outside_the_unit_interval():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="score"):
            horizon_from_score(bad)


# -- executor: commitment ---------------------------------------------------------------


def test_fixed_horizon_replans_on_exactly_that_period():
    ex = ChunkExecutor(FakePolicy(), fixed(20))
    ex.reset()
    actions = run(ex, 100)

    assert ex.n_replans == 5
    assert [r.timestep for r in ex.replans] == [0, 20, 40, 60, 80]
    # Each committed segment must come from the chunk predicted at its start...
    np.testing.assert_array_equal(actions[:, 0], np.repeat(np.arange(5), 20))
    # ...and walk that chunk from index 0 to 19, never past the commitment.
    np.testing.assert_array_equal(actions[:, 1], np.tile(np.arange(20), 5))


def test_h100_executes_a_whole_chunk_open_loop():
    """ACT as published: one prediction, 100 blind steps."""
    ex = ChunkExecutor(FakePolicy(), fixed(100))
    ex.reset()
    actions = run(ex, 100)

    assert ex.n_replans == 1
    assert (actions[:, 0] == 0).all()
    np.testing.assert_array_equal(actions[:, 1], np.arange(100))


def test_shorter_horizon_replans_more_often():
    counts = {}
    for h in (5, 20, 100):
        ex = ChunkExecutor(FakePolicy(), fixed(h))
        ex.reset()
        run(ex, 200)
        counts[h] = ex.n_replans
    assert counts[5] > counts[20] > counts[100]


def test_varying_horizon_tracks_the_score():
    """A scorer that collapses partway through should shorten commitments from then on."""
    scores = {"value": 1.0}
    ex = ChunkExecutor(FakePolicy(), score_gated(lambda t, f: scores["value"]))
    ex.reset()

    ex.act(0, {})  # commits h=100 at full reversibility
    assert ex.replans[-1].horizon == H_MAX

    scores["value"] = 0.1
    for t in range(100, 130):
        ex.act(t, {})
    assert ex.replans[-1].horizon == horizon_from_score(0.1)
    assert ex.mean_committed_horizon < H_MAX


def test_executor_rejects_an_out_of_range_horizon():
    ex = ChunkExecutor(FakePolicy(), lambda t, f, c: 0)
    ex.reset()
    with pytest.raises(ValueError, match="must be in"):
        ex.act(0, {})

    ex = ChunkExecutor(FakePolicy(), lambda t, f, c: CHUNK_SIZE + 1)
    ex.reset()
    with pytest.raises(ValueError, match="must be in"):
        ex.act(0, {})


def test_reset_clears_episode_state():
    ex = ChunkExecutor(FakePolicy(), fixed(20))
    ex.reset()
    run(ex, 60)
    assert ex.n_replans == 3

    ex.reset()
    assert ex.n_replans == 0
    assert np.isnan(ex.mean_committed_horizon)
    actions = run(ex, 5)
    assert (actions[:, 0] == 0).all(), "must start from a fresh chunk"


def test_features_are_recorded_at_each_replan():
    """The reversibility dataset is assembled from exactly these records."""
    ex = ChunkExecutor(FakePolicy(), fixed(25))
    ex.reset()
    run(ex, 100)
    assert [r.features["feat_dummy"][0] for r in ex.replans] == [0.0, 1.0, 2.0, 3.0]


# -- temporal ensembling ----------------------------------------------------------------


def test_temporal_ensembling_blends_overlapping_chunks():
    ex = ChunkExecutor(FakePolicy(), fixed(20), temporal_ensemble=True)
    ex.reset()
    actions = run(ex, 60)

    # After the second replan, the executed action mixes chunk 0 and chunk 1, so the
    # chunk-id channel lands strictly between them rather than on either.
    assert 0.0 < actions[25, 0] < 1.0


def test_temporal_ensembling_favours_the_freshest_chunk():
    ex = ChunkExecutor(FakePolicy(), fixed(20), temporal_ensemble=True, ensemble_coeff=0.1)
    ex.reset()
    actions = run(ex, 60)
    # Chunk id rises toward the newest chunk as older ones age out of the weighting.
    assert actions[25, 0] < actions[45, 0]


# -- oracle -----------------------------------------------------------------------------


def test_oracle_interpolates_between_labelled_timesteps():
    """Unlabelled steps must interpolate, not silently fall back to a constant."""
    fn = oracle({0: 1.0, 100: 0.0})
    assert fn(0, {}, np.zeros((100, 14))) == H_MAX
    assert fn(100, {}, np.zeros((100, 14))) == H_MIN
    mid = fn(50, {}, np.zeros((100, 14)))
    assert H_MIN < mid < H_MAX


def test_oracle_requires_labels():
    with pytest.raises(ValueError, match="at least one"):
        oracle({})


def test_history_is_not_retained_when_ensembling_is_off():
    """Without the ensembler nothing reads history, so retaining chunks is pure waste."""
    ex = ChunkExecutor(FakePolicy(), fixed(5))
    ex.reset()
    run(ex, 200)
    assert ex.n_replans == 40
    assert ex._history == []


def test_history_is_retained_when_ensembling_is_on():
    ex = ChunkExecutor(FakePolicy(), fixed(5), temporal_ensemble=True)
    ex.reset()
    run(ex, 50)
    assert ex._history, "the ensembler needs overlapping chunks"
