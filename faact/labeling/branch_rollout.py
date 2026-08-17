"""Branch-rollout labelling: measure reversibility R(s) directly from the simulator.

R(s) is defined operationally: **snapshot the world at state s, replan from scratch, run
to completion M times, and count how often the task still succeeds.** No model, no proxy —
the simulator answers the question by being replayed. This is what makes reversibility a
measured quantity rather than an asserted one, and it is the core primitive of the project.

    for each perturbed episode:
        for t in 0, S, 2S, ...:
            s = snapshot(env)
            wins = sum(rollout_to_completion(restore(s)) for _ in range(M))
            R[t] = wins / M

**The determinism problem, and why `branch_noise` exists.**

`faact.envs.state.restore` is bitwise exact (that is tested), and ACT is deterministic at
inference. So M branches from one snapshot would replay *identically*, every R would come
out as exactly 0.0 or 1.0, and the "probability" would be a coin that is welded to one
side. The head would then be trained on binary labels that no longer mean what R claims.

So each branch adds small Gaussian noise to its actions, seeded per branch. This makes R
the success probability under the policy's own execution variability — which is the
quantity the controller actually needs ("if I replan now, will this recover?"), and is a
stated modelling choice rather than a hidden default.

**Choosing the noise scale.** It has to be large enough to spread outcomes at ambiguous
states, and small enough that it does not itself change what the policy can do — otherwise
every R is biased downward and we would be measuring the noise, not the state. Measured on
20 unperturbed episodes (seeds 1000-1019), rolling out through this exact code path:

    branch_noise   unperturbed success
    0.0            13/20 = 65%   <- identical to the h=100 baseline, so the open-loop
    0.01           13/20 = 65%      rollout path is faithful
    0.02            8/20 = 40%   <- already destructive; would bias every label
    0.05            0/20 =  0%

Hence the 0.01 default. At `0.0` the labels are exactly binary (verified: R came out 1.00
and 0.00 on a healthy and a doomed state respectively), which is a legitimate cheaper mode
— binary labels under BCE still train a calibrated probability — but it makes M > 1
branches pure waste, so prefer M=1 if running that way.

**What a branch does *not* re-apply, and why.** Branches roll forward without re-applying
the perturbation, so R is defined as *recoverability of the state we are now in*, under
undisturbed future dynamics — "given where I am, can the policy still finish?" This is the
quantity the horizon controller needs, and for `object_displace`, `grasp_slip` and
`actuation_noise` it is also faithful: those disturbances act through the world, so their
effect is already inside the snapshot.

The exception worth stating plainly is `occlusion`, which is purely perceptual and leaves
no trace in physics. Labelling a state mid-occlusion therefore measures recoverability as
if the camera had just cleared, which will read slightly optimistic. It is a known
limitation of the definition, not an oversight.

**Budget.** Branches run open-loop at h=100, so a 250-step continuation costs ~3 policy
forward passes instead of 250. Measured on the dev box: MuJoCo runs at 84 steps/s while an
ACT forward is 76 ms, so a 250-step branch is 2.96 s of simulation against 0.23 s of
policy. **The simulator is the bottleneck by ~13x, and the GPU is almost irrelevant here** —
this scales with CPU cores, not with accelerator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from faact.envs.make import MAX_EPISODE_STEPS
from faact.envs.state import observe, restore, snapshot
from faact.eval.runner import SUCCESS_REWARD

log = logging.getLogger("faact.labeling")

# Std-dev of per-branch action noise, in normalised action units. See the module docstring
# for the sweep that picked this: 0.01 leaves unperturbed success at the 65% baseline,
# while 0.02 drops it to 40% and would bias every label downward.
DEFAULT_BRANCH_NOISE = 0.01


@dataclass
class LabelPoint:
    """One labelled state: features observed at `timestep`, with its measured R."""

    episode_id: int
    timestep: int
    reversibility: float
    n_branches: int
    features: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def rollout_to_completion(
    env: Any,
    policy: Any,
    start_step: int,
    branch_seed: int,
    branch_noise: float = DEFAULT_BRANCH_NOISE,
    max_steps: int = MAX_EPISODE_STEPS,
    horizon: int = 100,
) -> bool:
    """Run one branch from the *current* env state to completion. Returns success.

    Deliberately open-loop at `horizon`: replanning every step would cost ~250 forward
    passes per branch and the label is about whether recovery is possible at all, not
    about how cleverly we drive afterwards.

    Assumes the caller has already restored the snapshot; it does not reset the env.
    """
    rng = np.random.default_rng(branch_seed)
    policy.reset()

    obs = observe(env)
    chunk: np.ndarray | None = None
    chunk_start = 0

    for t in range(start_step, max_steps):
        if chunk is None or (t - chunk_start) >= horizon:
            chunk, _ = policy.predict_chunk(obs)
            chunk_start = t

        action = np.array(chunk[t - chunk_start], dtype=np.float64)
        if branch_noise > 0:
            action += rng.normal(0.0, branch_noise, size=action.shape)

        obs, reward, terminated, truncated, _ = env.step(action)
        if float(reward) >= SUCCESS_REWARD:
            return True
        if terminated or truncated:
            return False
    return False


def label_state(
    env: Any,
    policy: Any,
    timestep: int,
    n_branches: int,
    branch_noise: float = DEFAULT_BRANCH_NOISE,
    max_steps: int = MAX_EPISODE_STEPS,
    horizon: int = 100,
) -> float:
    """Measure R at the env's current state: fraction of M branches that still succeed.

    Snapshots once, then restores before every branch, so branches are independent and the
    env is left exactly where it started — the caller's episode can continue afterwards.
    """
    if n_branches < 1:
        raise ValueError(f"n_branches must be >= 1, got {n_branches}")

    state = snapshot(env)
    wins = 0
    for m in range(n_branches):
        restore(env, state)
        wins += rollout_to_completion(
            env,
            policy,
            start_step=timestep,
            # Branch seed mixes timestep and index so no two branches anywhere share noise.
            branch_seed=timestep * 10_000 + m,
            branch_noise=branch_noise,
            max_steps=max_steps,
            horizon=horizon,
        )

    # Leave the env as we found it, so labelling does not perturb the episode it measures.
    restore(env, state)
    return wins / n_branches
