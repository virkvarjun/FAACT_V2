"""Unit tests for the perturbation suite.

These run without MuJoCo: the action and observation hooks are pure numpy, and the env
hook is covered separately in tests/test_state.py.

What matters here is that every applier *demonstrably changes its input*. v1 shipped a
perturbation flag that silently did nothing, and no test would have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.envs.perturb import KINDS, Perturbation, PerturbationSpec, sample_spec


class FakeActionSpace:
    def __init__(self, dim: int = 14) -> None:
        self.low = -np.ones(dim)
        self.high = np.ones(dim)


class FakeEnv:
    def __init__(self, dim: int = 14) -> None:
        self.action_space = FakeActionSpace(dim)


def bound(spec: PerturbationSpec, image_hw: tuple[int, int] = (48, 64)) -> Perturbation:
    return Perturbation(spec).bind(FakeEnv(), image_hw=image_hw, gripper_idx=(6, 13))


# -- spec validation -------------------------------------------------------------------


def test_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown perturbation kind"):
        PerturbationSpec(kind="teleport_robot", onset_step=10, magnitude=1.0, duration=1)


def test_spec_rejects_nonpositive_duration():
    with pytest.raises(ValueError, match="duration"):
        PerturbationSpec(kind="grasp_slip", onset_step=10, magnitude=1.0, duration=0)


def test_active_window_is_half_open():
    spec = PerturbationSpec(kind="grasp_slip", onset_step=10, magnitude=1.0, duration=3)
    assert [t for t in range(15) if spec.is_active(t)] == [10, 11, 12]


def test_spec_roundtrips_through_dict():
    spec = PerturbationSpec("occlusion", onset_step=7, magnitude=0.2, duration=20, seed=3)
    assert PerturbationSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize("kind", KINDS)
def test_sample_spec_onset_lands_in_range(kind):
    rng = np.random.default_rng(0)
    onsets = [sample_spec(kind, rng, onset_range=(50, 150)).onset_step for _ in range(50)]
    assert all(50 <= o < 150 for o in onsets)
    assert len(set(onsets)) > 1, "onset should be randomised, not constant"


# -- action-level hooks ----------------------------------------------------------------


def test_grasp_slip_opens_grippers_and_leaves_arm_joints_alone():
    p = bound(PerturbationSpec("grasp_slip", onset_step=5, magnitude=1.0, duration=3))
    action = np.zeros(14)
    out = p.on_action(action, t=5)

    assert out[6] == pytest.approx(1.0) and out[13] == pytest.approx(1.0)
    arm = [i for i in range(14) if i not in (6, 13)]
    np.testing.assert_array_equal(out[arm], action[arm])


def test_action_hooks_are_inert_outside_the_window():
    p = bound(PerturbationSpec("grasp_slip", onset_step=5, magnitude=1.0, duration=3))
    action = np.zeros(14)
    np.testing.assert_array_equal(p.on_action(action, t=4), action)
    np.testing.assert_array_equal(p.on_action(action, t=8), action)
    assert not p.fired


def test_actuation_noise_perturbs_every_channel_and_clips_to_bounds():
    p = bound(PerturbationSpec("actuation_noise", onset_step=0, magnitude=0.5, duration=20))
    action = np.full(14, 0.9)
    out = p.on_action(action, t=0)

    assert np.all(out != action), "noise should touch every channel"
    assert np.all(out <= 1.0) and np.all(out >= -1.0), "must stay inside the action space"


def test_action_hook_does_not_mutate_the_callers_array():
    p = bound(PerturbationSpec("actuation_noise", onset_step=0, magnitude=0.5, duration=20))
    action = np.zeros(14)
    p.on_action(action, t=0)
    np.testing.assert_array_equal(action, np.zeros(14))


def test_same_spec_reproduces_the_same_noise():
    """Branch rollouts replay episodes from snapshots; the disturbance must replay too."""
    spec = PerturbationSpec("actuation_noise", onset_step=0, magnitude=0.5, duration=20, seed=7)
    a = bound(spec).on_action(np.zeros(14), t=0)
    b = bound(spec).on_action(np.zeros(14), t=0)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_noise():
    kw = dict(kind="actuation_noise", onset_step=0, magnitude=0.5, duration=20)
    a = bound(PerturbationSpec(seed=1, **kw)).on_action(np.zeros(14), t=0)
    b = bound(PerturbationSpec(seed=2, **kw)).on_action(np.zeros(14), t=0)
    assert not np.array_equal(a, b)


# -- observation-level hook ------------------------------------------------------------


def make_obs(h: int = 48, w: int = 64) -> dict:
    return {"pixels": {"top": np.full((h, w, 3), 255, dtype=np.uint8)}, "agent_pos": np.zeros(14)}


def test_occlusion_blacks_out_roughly_the_requested_area():
    p = bound(PerturbationSpec("occlusion", onset_step=0, magnitude=0.25, duration=20))
    out = p.on_obs(make_obs(), t=0)

    img = out["pixels"]["top"]
    black_fraction = float((img == 0).all(axis=-1).mean())
    # Square side is rounded to whole pixels, so allow a little slack around 0.25.
    assert 0.20 < black_fraction < 0.30


def test_occlusion_does_not_mutate_the_input_observation():
    p = bound(PerturbationSpec("occlusion", onset_step=0, magnitude=0.25, duration=20))
    obs = make_obs()
    p.on_obs(obs, t=0)
    assert (obs["pixels"]["top"] == 255).all()


def test_occlusion_raises_on_a_missing_camera():
    p = bound(PerturbationSpec("occlusion", onset_step=0, magnitude=0.25, duration=20))
    with pytest.raises(KeyError, match="pixels"):
        p.on_obs({"pixels": {"wrist": np.zeros((48, 64, 3))}}, t=0)


# -- bind contract ---------------------------------------------------------------------


def test_hooks_raise_before_bind():
    p = Perturbation(PerturbationSpec("grasp_slip", onset_step=0, magnitude=1.0, duration=3))
    with pytest.raises(RuntimeError, match="bind"):
        p.on_action(np.zeros(14), t=0)


def test_fired_reports_whether_the_perturbation_ever_touched_the_episode():
    p = bound(PerturbationSpec("grasp_slip", onset_step=5, magnitude=1.0, duration=3))
    assert not p.fired
    p.on_action(np.zeros(14), t=5)
    assert p.fired
