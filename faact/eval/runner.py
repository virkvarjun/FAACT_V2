"""Episode runner — one place where env, perturbation, and executor meet.

Every experiment in the repo (baseline, perturbation calibration, branch rollouts,
ablation) goes through `run_episode`, so success is defined once and identically for all
of them. v1's numbers were not comparable across runs; this is the fix.

Hook order within a step matters, and is deliberate:

    1. perturbation.on_env(t)   — edit physics (teleport the cube)
    2. re-observe               — so the policy sees the edit, not a stale frame
    3. perturbation.on_obs(t)   — corrupt what the policy sees (occlusion)
    4. executor.act(t, obs)     — replan if the commitment expired, else continue
    5. perturbation.on_action(t)— corrupt what the policy does (slip, noise)
    6. env.step(action)

Getting 1-and-2 backwards would teleport the cube and then let the policy act on an image
of where it used to be — a disturbance the policy cannot even perceive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from faact.envs.make import CAMERA, GRIPPER_ACTION_IDX, MAX_EPISODE_STEPS, image_hw
from faact.envs.perturb import Perturbation, PerturbationSpec
from faact.envs.state import observe

# ALOHA transfer-cube grades 0-4; 4 means the left arm holds the transferred cube.
SUCCESS_REWARD = 4


class PerturbationNeverFired(RuntimeError):
    """A perturbed episode in which the disturbance never actually happened.

    Split into its own type because there are two very different causes and only one is a
    bug. An episode that *ended* before its onset step is legitimate data — the policy
    simply finished early — and a caller running a long sweep should count and exclude it,
    not crash. An episode that ran *past* the onset without firing is a real defect, and is
    raised as a plain RuntimeError instead.

    Either way the episode is unperturbed and must never be counted as perturbed. Quietly
    returning it is precisely what produced v1's 0/20.
    """

    def __init__(self, spec: PerturbationSpec, steps: int) -> None:
        super().__init__(
            f"episode ended after {steps} steps, before perturbation onset at "
            f"{spec.onset_step} ({spec.kind}) — it is unperturbed and must be excluded"
        )
        self.spec = spec
        self.steps = steps


@dataclass
class EpisodeResult:
    """Everything one episode produced. Serialisable, and the unit of every metric."""

    seed: int
    success: bool
    steps: int
    max_reward: float
    # Per-replan trace — the input to horizon metrics and to the reversibility dataset.
    replan_steps: list[int] = field(default_factory=list)
    horizons: list[int] = field(default_factory=list)
    features: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)
    # Perturbation ground truth. onset is known by construction, so lead time is exact.
    perturb: dict[str, Any] | None = None
    perturb_fired: bool = False
    frames: list[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def mean_horizon(self) -> float:
        return float(np.mean(self.horizons)) if self.horizons else float("nan")

    def to_record(self) -> dict[str, Any]:
        """JSON-safe summary. Drops features and frames — those go to .npz / video."""
        return {
            "seed": self.seed,
            "success": self.success,
            "steps": self.steps,
            "max_reward": self.max_reward,
            "n_replans": len(self.replan_steps),
            "mean_horizon": self.mean_horizon,
            "replan_steps": self.replan_steps,
            "horizons": self.horizons,
            "perturb": self.perturb,
            "perturb_fired": self.perturb_fired,
        }


def run_episode(
    env: Any,
    executor: Any,
    seed: int,
    perturb_spec: PerturbationSpec | None = None,
    max_steps: int = MAX_EPISODE_STEPS,
    record_frames: bool = False,
    on_step: Callable[[int, dict], None] | None = None,
) -> EpisodeResult:
    """Run one episode and return its result.

    `executor` is a `ChunkExecutor`; swapping its `horizon_fn` is what distinguishes every
    condition in the ablation. Nothing else about the loop changes between conditions.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    obs, _ = env.reset(seed=seed)
    executor.reset()

    perturbation: Perturbation | None = None
    if perturb_spec is not None:
        perturbation = Perturbation(perturb_spec, camera=CAMERA).bind(
            env, image_hw=image_hw(obs), gripper_idx=GRIPPER_ACTION_IDX
        )

    max_reward = 0.0
    success = False
    frames: list[np.ndarray] = []
    t = 0

    for t in range(max_steps):
        # 1-2. State-level disturbance, then re-observe so the policy can perceive it.
        if perturbation is not None and perturbation.on_env(env, t):
            obs = observe(env)

        # 3. Observation-level disturbance (occlusion).
        policy_obs = perturbation.on_obs(obs, t) if perturbation is not None else obs

        if record_frames:
            frames.append(np.asarray(policy_obs["pixels"][CAMERA]))
        if on_step is not None:
            on_step(t, policy_obs)

        # 4-5. Act, then apply action-level disturbance (slip, actuation noise).
        action = executor.act(t, policy_obs)
        if perturbation is not None:
            action = perturbation.on_action(action, t)

        # 6. Advance the world.
        obs, reward, terminated, truncated, _ = env.step(action)
        max_reward = max(max_reward, float(reward))
        if float(reward) >= SUCCESS_REWARD:
            success = True
        if terminated or truncated:
            break

    if perturbation is not None and not perturbation.fired:
        steps = t + 1
        if steps <= perturb_spec.onset_step:
            # Legitimate: the episode finished (usually succeeded) before the disturbance
            # was due. Recoverable by the caller, which should exclude and count it.
            raise PerturbationNeverFired(perturb_spec, steps)
        # Ran past the onset and still never fired — that is a defect in the applier.
        raise RuntimeError(
            f"perturbation {perturb_spec} never fired despite the episode reaching step "
            f"{steps}, past its onset at {perturb_spec.onset_step}. This is a bug in the "
            "applier, not an early finish."
        )

    return EpisodeResult(
        seed=seed,
        success=success,
        steps=t + 1,
        max_reward=max_reward,
        replan_steps=[r.timestep for r in executor.replans],
        horizons=[r.horizon for r in executor.replans],
        features=[r.features for r in executor.replans],
        perturb=perturb_spec.to_dict() if perturb_spec is not None else None,
        perturb_fired=perturbation.fired if perturbation is not None else False,
        frames=frames,
    )


def success_rate(results: list[EpisodeResult]) -> float:
    """Fraction of episodes that reached the success reward. Raises on an empty list."""
    if not results:
        raise ValueError("success_rate of zero episodes is undefined")
    return sum(r.success for r in results) / len(results)
