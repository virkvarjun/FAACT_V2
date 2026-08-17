"""Simulator-level tests for snapshot / restore and the cube teleport.

Marked `sim` because they need MuJoCo. The determinism test here is the load-bearing one:
M3 estimates reversibility by restoring to a state M times and counting successes, so any
state leakage between branches silently corrupts every label in the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.envs.make import make_env
from faact.envs.perturb import Perturbation, PerturbationSpec
from faact.envs.state import (
    displace_free_body,
    free_joint_qpos_slice,
    physics_of,
    restore,
    snapshot,
)

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def env():
    e = make_env(seed=0)
    yield e
    e.close()


def rollout_actions(env, actions):
    """Step a fixed action sequence and collect the resulting agent positions."""
    return np.array([env.step(a)[0]["agent_pos"] for a in actions])


def test_free_joint_slice_points_at_a_7dof_free_body(env):
    sl = free_joint_qpos_slice(physics_of(env))
    assert sl.stop - sl.start == 7


def test_restore_from_the_same_snapshot_is_bitwise_reproducible(env):
    """Two branches from one snapshot, driven by identical actions, must agree exactly.

    Bitwise, not approximately: float drift here would mean the snapshot is missing part
    of the simulator state, and reversibility labels would inherit the error.
    """
    env.reset(seed=0)
    for _ in range(20):
        env.step(env.action_space.sample() * 0.1)

    state = snapshot(env)
    rng = np.random.default_rng(0)
    # Long enough to matter: M3 branches run to episode completion, and solver drift
    # compounds, so a 30-step check would pass on a snapshot that is still incomplete.
    actions = rng.uniform(-0.2, 0.2, size=(150, 14))

    restore(env, state)
    first = rollout_actions(env, actions)

    restore(env, state)
    second = rollout_actions(env, actions)

    np.testing.assert_array_equal(first, second)


def test_snapshot_is_an_owned_copy_not_a_live_view(env):
    """A snapshot taken before stepping must not change as the sim advances."""
    env.reset(seed=0)
    state = snapshot(env)
    before = state.copy()
    for _ in range(5):
        env.step(np.zeros(14))
    np.testing.assert_array_equal(state, before)


def test_restore_rejects_a_wrongly_shaped_state(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="does not match"):
        restore(env, np.zeros(3))


def test_displace_free_body_moves_the_cube_in_plane_only(env):
    env.reset(seed=0)
    physics = physics_of(env)
    sl = free_joint_qpos_slice(physics)
    before = np.array(physics.data.qpos[sl], copy=True)

    delta = np.array([0.03, -0.02])
    displace_free_body(env, delta)
    after = np.array(physics.data.qpos[sl], copy=True)

    np.testing.assert_allclose(after[:2] - before[:2], delta, atol=1e-9)
    assert after[2] == pytest.approx(before[2]), "height must not change"
    np.testing.assert_allclose(after[3:], before[3:], atol=1e-9)  # orientation untouched


def test_displace_free_body_rejects_a_bad_delta(env):
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        displace_free_body(env, np.array([0.03, 0.0, 0.0]))


def test_object_displace_hook_actually_changes_the_world(env):
    """The end-to-end check v1 was missing: does the perturbation change env state at all?"""
    env.reset(seed=0)
    physics = physics_of(env)
    sl = free_joint_qpos_slice(physics)

    spec = PerturbationSpec("object_displace", onset_step=3, magnitude=0.04, duration=1, seed=1)
    p = Perturbation(spec).bind(env, image_hw=(480, 640), gripper_idx=(6, 13))

    before = np.array(physics.data.qpos[sl][:2], copy=True)
    assert not p.on_env(env, t=2), "must not fire before onset"
    np.testing.assert_array_equal(physics.data.qpos[sl][:2], before)

    assert p.on_env(env, t=3), "must fire at onset"
    moved = np.linalg.norm(np.array(physics.data.qpos[sl][:2]) - before)
    assert moved == pytest.approx(0.04, abs=1e-6), "displacement magnitude must match the spec"
    assert p.fired
