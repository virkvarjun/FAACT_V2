"""MuJoCo state snapshot / restore, and the free-body edit that `object_displace` needs.

Branch-rollout labelling (M3) stands or falls on this file: to estimate reversibility at
state `s` we restore to `s` M times and count how many continuations succeed. If restore
leaks *any* state between branches, every label is quietly wrong. So the accompanying unit
test asserts that two restores from one snapshot produce **bitwise-identical** action
sequences, and every accessor here raises rather than guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# dm_control joint type ids. Only free joints (6-DoF, 7 qpos entries) can be teleported.
_MJ_JNT_FREE = 0
_FREE_JOINT_NQ = 7  # 3 translation + 4 quaternion


def physics_of(env: Any) -> Any:
    """Dig the dm_control `physics` object out of a gym-aloha env.

    The chain is gymnasium wrappers -> `AlohaEnv` -> `dm_control.rl.control.Environment`
    -> `.physics`. Raises if any link is missing; a silent None here would produce
    perturbations that never fire.
    """
    inner = getattr(env, "unwrapped", env)
    dm_env = getattr(inner, "_env", None)
    if dm_env is None:
        raise AttributeError(f"{type(inner).__name__} has no `_env`; not a gym-aloha env?")
    physics = getattr(dm_env, "physics", None)
    if physics is None:
        raise AttributeError(f"{type(dm_env).__name__} has no `physics`")
    return physics


def _state_spec() -> Any:
    """The MuJoCo state fields a snapshot must cover to be replay-exact.

    `mjSTATE_INTEGRATION` = everything the integrator reads: time, qpos, qvel, act, ctrl,
    applied forces, mocap, userdata, **and the constraint solver's warm-start cache**.

    That last field is the reason we do not use dm_control's `physics.get_state()`, which
    covers only qpos/qvel/act. Restoring without the warm start leaves the solver iterating
    from a different starting guess, and two branches from one snapshot then diverge by
    ~1e-8 immediately and visibly later. Measured here: 45 floats vs 262.
    """
    import mujoco

    return mujoco.mjtState.mjSTATE_INTEGRATION


def _mj(env: Any) -> tuple[Any, Any]:
    """Raw (mjModel, mjData) pointers behind dm_control's wrappers."""
    physics = physics_of(env)
    return physics.model.ptr, physics.data.ptr


def snapshot(env: Any) -> np.ndarray:
    """Capture full integration state as an owned copy.

    This is *simulator* state only — it excludes wrapper bookkeeping such as the TimeLimit
    step counter, so callers doing branch rollouts track their own step budget.
    """
    import mujoco

    model, data = _mj(env)
    spec = _state_spec()
    buf = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
    mujoco.mj_getState(model, data, buf, spec)
    return buf


def restore(env: Any, state: np.ndarray) -> None:
    """Restore simulator state and re-derive dependent quantities.

    `forward()` is mandatory: setting the state writes the integrator's inputs but leaves
    derived data (body xpos, contacts, sensors) stale, so an observation read without it
    would describe the *old* state.
    """
    import mujoco

    model, data = _mj(env)
    spec = _state_spec()
    expected = mujoco.mj_stateSize(model, spec)
    state = np.ascontiguousarray(state, dtype=np.float64)
    if state.shape != (expected,):
        raise ValueError(f"state shape {state.shape} does not match physics state ({expected},)")
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)


def observe(env: Any) -> dict[str, Any]:
    """Re-derive the current observation *without stepping the simulator*.

    Needed at two points where the world changes outside `env.step`:
      - after `object_displace` teleports the cube, so the policy sees the moved cube
        rather than a stale frame from before the disturbance;
      - after `restore`, so a branch rollout starts from an observation of the state it
        was actually restored to.

    Reaches through the gym wrapper to the task's own observation function, so the result
    is byte-identical to what `env.step` would have returned (verified in tests).
    """
    inner = getattr(env, "unwrapped", env)
    raw = inner._env.task.get_observation(physics_of(env))
    return inner._format_raw_obs(raw)


def free_joint_qpos_slice(physics: Any) -> slice:
    """Locate the qpos entries of the scene's single free body (the red cube).

    Resolved by introspection rather than by name or by an assumed `qpos[-7:]`, and it
    raises unless there is exactly one free joint — if the model ever gains a second one,
    this fails loudly instead of teleporting the wrong thing.
    """
    model = physics.model
    free = [j for j in range(model.njnt) if model.jnt_type[j] == _MJ_JNT_FREE]
    if len(free) != 1:
        raise RuntimeError(f"expected exactly 1 free joint in the model, found {len(free)}")
    adr = int(model.jnt_qposadr[free[0]])
    return slice(adr, adr + _FREE_JOINT_NQ)


def displace_free_body(env: Any, delta_xy: np.ndarray) -> np.ndarray:
    """Teleport the cube in the ground plane by `delta_xy` metres. Returns the new xyz.

    Position only — orientation and velocity are untouched, so the cube is moved rather
    than launched. `forward()` propagates the edit before the next observation is read.
    """
    delta_xy = np.asarray(delta_xy, dtype=np.float64)
    if delta_xy.shape != (2,):
        raise ValueError(f"delta_xy must have shape (2,), got {delta_xy.shape}")

    physics = physics_of(env)
    sl = free_joint_qpos_slice(physics)
    qpos = physics.data.qpos
    qpos[sl.start : sl.start + 2] += delta_xy
    physics.forward()
    return np.array(qpos[sl.start : sl.start + 3], copy=True)
