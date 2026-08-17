"""Tests for the ACT wrapper and the end-to-end episode loop.

Marked `policy` because they download and run the real ACT checkpoint. They are slow but
they are the only place the feature-hook contract is actually verified — and a hook that
silently returns nothing would poison the reversibility dataset without any other test
noticing.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.backbone.act_wrapper import ACTWrapper, concat_features
from faact.envs.make import ACTION_DIM, make_env
from faact.envs.perturb import PerturbationSpec
from faact.eval.runner import run_episode
from faact.runtime.controller import fixed
from faact.runtime.executor import ChunkExecutor

pytestmark = [pytest.mark.sim, pytest.mark.policy]

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"

EXPECTED_FEATURES = {
    "feat_encoder_mean",
    "feat_decoder_mean",
    "feat_state",
    "feat_action_first",
    "feat_action_prefix_mean_10",
    "feat_action_prefix_flat_10",
}


@pytest.fixture(scope="module")
def policy():
    return ACTWrapper(CHECKPOINT)


@pytest.fixture(scope="module")
def obs():
    env = make_env()
    try:
        o, _ = env.reset(seed=0)
        return o
    finally:
        env.close()


def test_predict_chunk_returns_a_full_chunk(policy, obs):
    chunk, _ = policy.predict_chunk(obs)
    assert chunk.shape == (policy.chunk_size, ACTION_DIM)
    assert np.isfinite(chunk).all()


def test_all_expected_features_are_produced(policy, obs):
    _, feats = policy.predict_chunk(obs)
    assert set(feats) == EXPECTED_FEATURES


def test_hooked_transformer_features_are_not_degenerate(policy, obs):
    """The check that matters: hooks must carry real activations, not zeros.

    A hook that never fires, or fires on the wrong module, tends to yield an all-zero or
    all-constant vector. Trained on that, the reversibility head predicts the mean and the
    failure looks like "features aren't informative" rather than like a bug.
    """
    _, feats = policy.predict_chunk(obs)
    for key in ("feat_encoder_mean", "feat_decoder_mean"):
        v = feats[key]
        assert v.ndim == 1 and v.size >= 64, f"{key} has suspicious shape {v.shape}"
        assert np.isfinite(v).all(), f"{key} contains non-finite values"
        assert np.abs(v).sum() > 0, f"{key} is all zeros — the hook did not fire"
        assert v.std() > 1e-6, f"{key} is constant — the hook captured the wrong tensor"


def test_features_respond_to_the_observation(policy, obs):
    """Different states must produce different features, or they carry no information."""
    _, a = policy.predict_chunk(obs)

    moved = {**obs, "agent_pos": obs["agent_pos"] + 0.15}
    _, b = policy.predict_chunk(moved)

    assert not np.allclose(a["feat_encoder_mean"], b["feat_encoder_mean"])
    assert not np.allclose(a["feat_decoder_mean"], b["feat_decoder_mean"])


def test_concat_features_is_stable_and_ordered(policy, obs):
    _, feats = policy.predict_chunk(obs)
    v1 = concat_features(feats)
    v2 = concat_features(feats)

    np.testing.assert_array_equal(v1, v2)
    assert v1.size == sum(np.asarray(f).size for f in feats.values())
    # Sorted key order, so a vector built today matches a model trained yesterday.
    np.testing.assert_array_equal(v1, concat_features(feats, sorted(feats)))


def test_concat_features_raises_on_a_missing_key():
    with pytest.raises(KeyError, match="missing feature keys"):
        concat_features({"feat_a": np.zeros(3)}, ["feat_a", "feat_b"])


def test_episode_runs_and_records_a_replan_trace(policy):
    """End-to-end smoke test: h=25 over 50 steps must replan exactly twice."""
    env = make_env()
    try:
        ex = ChunkExecutor(policy, fixed(25))
        r = run_episode(env, ex, seed=1001, max_steps=50)
    finally:
        env.close()

    assert r.steps == 50
    assert r.replan_steps == [0, 25]
    assert r.horizons == [25, 25]
    assert len(r.features) == 2 and set(r.features[0]) == EXPECTED_FEATURES
    assert r.perturb is None


def test_runner_refuses_an_episode_whose_perturbation_never_fired(policy):
    """A no-op perturbation must raise, not quietly return an unperturbed episode."""
    env = make_env()
    try:
        ex = ChunkExecutor(policy, fixed(50))
        spec = PerturbationSpec("grasp_slip", onset_step=300, magnitude=1.0, duration=3)
        with pytest.raises(RuntimeError, match="never fired"):
            run_episode(env, ex, seed=1001, perturb_spec=spec, max_steps=30)
    finally:
        env.close()
