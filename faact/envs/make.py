"""Env factory and the one place ALOHA-specific constants live.

Every other module takes these as arguments rather than importing task knowledge, so
swapping tasks later is a change to this file only.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

ENV_ID = "gym_aloha/AlohaTransferCube-v0"

# ALOHA action layout: 14 = 2 arms x (6 arm joints + 1 gripper). Verified against the
# model's joint list — arms are `vx300s_left/*` then `vx300s_right/*`.
ACTION_DIM = 14
GRIPPER_ACTION_IDX: tuple[int, ...] = (6, 13)

# The sim scene ships exactly one camera.
CAMERA = "top"

# Episode length used by every experiment in this repo. gym-aloha's own default is 400;
# we fix it here so baseline, perturbed, and branch-rollout episodes are all comparable.
MAX_EPISODE_STEPS = 400


def make_env(
    seed: int | None = None,
    obs_type: str = "pixels_agent_pos",
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> Any:
    """Build the ALOHA transfer-cube env, seeded if a seed is given.

    Picks a MuJoCo rendering backend if the user has not, rather than letting rendering
    fail deep inside an eval run. EGL is the headless-Linux (RunPod) choice; it does not
    exist on macOS, where GLFW is the working one. An explicit MUJOCO_GL always wins.
    """
    os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

    import gym_aloha  # noqa: F401  — import registers the env id with gymnasium
    import gymnasium as gym

    env = gym.make(ENV_ID, obs_type=obs_type, max_episode_steps=max_episode_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def image_hw(obs: dict[str, Any], camera: str = CAMERA) -> tuple[int, int]:
    """(height, width) of a camera image in an observation. Raises if the camera is absent."""
    pixels = obs.get("pixels")
    if not isinstance(pixels, dict) or camera not in pixels:
        raise KeyError(
            f"observation has no pixels[{camera!r}]; got "
            f"{sorted(pixels) if isinstance(pixels, dict) else type(pixels)}"
        )
    img = np.asarray(pixels[camera])
    if img.ndim != 3:
        raise ValueError(f"expected HxWxC image, got shape {img.shape}")
    return int(img.shape[0]), int(img.shape[1])
