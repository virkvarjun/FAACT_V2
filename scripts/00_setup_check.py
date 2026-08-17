#!/usr/bin/env python
"""M0 acceptance gate: prove the environment actually works before trusting any run.

Every check either PASSes with a measured detail or FAILs with the exception that caused
it. Nothing is skipped quietly. Exit code is 0 only if every required check passed.

    python scripts/00_setup_check.py                  # on the GPU box (CUDA required)
    python scripts/00_setup_check.py --allow-no-cuda  # on the CPU dev machine

Known blockers this gate is designed to surface early, from the predecessor project:
  - headless MuJoCo needs MUJOCO_GL=egl plus libegl1/libgles2/libglvnd0
  - torchcodec needs FFmpeg *shared libraries*, not just the CLI binary
  - gym-aloha speaks the Gym v0.26 API, bridged by shimmy[gym-v26]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Import faact from the repo root without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.utils import ARTIFACTS, timed, write_json  # noqa: E402


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def run(name: str, fn: Callable[[], str], required: bool = True) -> Check:
    """Run one check, converting any exception into a FAIL with its traceback tail."""
    try:
        return Check(name, True, fn(), required)
    except Exception as exc:  # noqa: BLE001 — a gate reports failures, it does not raise
        tb = traceback.format_exc().strip().splitlines()[-1]
        return Check(name, False, f"{type(exc).__name__}: {exc} | {tb}", required)


# -- individual checks -----------------------------------------------------------------


def check_torch() -> str:
    import torch

    return f"torch {torch.__version__}"


def check_cuda() -> str:
    """Report the accelerator by name. Required on the GPU box, optional on the dev Mac."""
    import torch

    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)} (n={torch.cuda.device_count()})"
    if torch.backends.mps.is_available():
        return "no cuda; mps available (dev machine — do not report numbers from here)"
    return "no cuda; cpu only"


def check_imports() -> str:
    import importlib.metadata as md

    import gym_aloha  # noqa: F401
    import gymnasium  # noqa: F401
    import lerobot  # noqa: F401
    import mujoco  # noqa: F401

    pkgs = ["lerobot", "gym-aloha", "mujoco", "gymnasium", "numpy"]
    return ", ".join(f"{p} {md.version(p)}" for p in pkgs)


def check_env_steps() -> str:
    """Create the task env and step it, confirming the observation contract we rely on."""
    import numpy as np

    from faact.envs.make import ACTION_DIM, CAMERA, make_env, image_hw

    env = make_env(seed=0)
    try:
        obs, _ = env.reset(seed=0)
        h, w = image_hw(obs)
        if obs["agent_pos"].shape != (ACTION_DIM,):
            raise ValueError(f"agent_pos shape {obs['agent_pos'].shape} != ({ACTION_DIM},)")

        for _ in range(10):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(ACTION_DIM))
        return f"10 steps ok; pixels[{CAMERA}]={h}x{w}, agent_pos={obs['agent_pos'].shape[0]}"
    finally:
        env.close()


def check_headless_render() -> str:
    """Render one frame and assert it is not black.

    A black frame is the classic headless-MuJoCo failure: rendering "succeeds" and returns
    zeros. std > 1.0 is the cheapest test that pixels actually contain a scene.
    """
    import imageio.v3 as iio
    import numpy as np

    from faact.envs.make import CAMERA, make_env

    env = make_env(seed=0)
    try:
        obs, _ = env.reset(seed=0)
        frame = np.asarray(obs["pixels"][CAMERA])
        std = float(frame.std())
        if std <= 1.0:
            raise ValueError(f"frame std {std:.3f} <= 1.0 — rendering produced a blank image")

        out = ARTIFACTS / "setup_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out, frame)
        return f"{frame.shape} std={std:.1f} MUJOCO_GL={os.environ.get('MUJOCO_GL')} -> {out}"
    finally:
        env.close()


FFMPEG_REMEDY = (
    "torchcodec could not load its FFmpeg bindings. The dataset stores camera streams as "
    "video, so this blocks ACT training. Install FFmpeg *shared libraries* — a static CLI "
    "binary on PATH is not enough:\n"
    "    Ubuntu/RunPod: apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev\n"
    "    macOS:         brew install ffmpeg, then export DYLD_LIBRARY_PATH=/opt/homebrew/lib\n"
    "                   (torchcodec's @rpath resolves against the Python prefix, not brew's "
    "libdir, so the install alone is not enough)\n"
    "lerobot's 'pyav' backend is not an alternative: it routes through "
    "torchvision.io.VideoReader, which current torchvision no longer ships."
)


def check_dataset_decode() -> str:
    """Load one episode of the demo dataset and decode frames.

    This is really a torchcodec/FFmpeg check: decoding fails loudly here rather than 40
    minutes into a training run.
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human", episodes=[0])
    image_keys = [k for k in ds.features if k.startswith("observation.images")]
    if not image_keys:
        raise KeyError(f"no image features found; have {sorted(ds.features)}")

    key = image_keys[0]
    shapes = set()
    try:
        for i in range(5):
            arr = np.asarray(ds[i][key])
            if arr.size == 0:
                raise ValueError(f"decoded an empty frame at index {i}")
            shapes.add(tuple(arr.shape))
    except OSError as exc:
        # torchcodec surfaces a missing libav* as a dlopen OSError, deep in the decode
        # call. Translate it into the fix rather than a 40-line ctypes traceback.
        raise RuntimeError(FFMPEG_REMEDY) from exc
    return f"{ds.num_episodes} ep / {ds.num_frames} frames; decoded 5x {key} {shapes.pop()}"


# -- driver ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="downgrade the CUDA check to optional (dev machines only)",
    )
    ap.add_argument(
        "--skip-dataset",
        action="store_true",
        help="skip the ~70MB dataset download / decode check",
    )
    args = ap.parse_args()

    with timed("setup check") as t:
        checks = [
            run("torch", check_torch),
            run("accelerator", check_cuda, required=not args.allow_no_cuda),
            run("imports", check_imports),
            run("env 10 steps", check_env_steps),
            run("headless render", check_headless_render),
        ]
        if not args.skip_dataset:
            checks.append(run("dataset decode", check_dataset_decode))

    width = max(len(c.name) for c in checks)
    for c in checks:
        status = "PASS" if c.ok else ("FAIL" if c.required else "WARN")
        print(f"[{status}] {c.name:<{width}}  {c.detail}")

    # The accelerator check is informational when --allow-no-cuda: it still prints its
    # measured detail, so a CPU-only run can never be mistaken for a GPU one.
    if args.allow_no_cuda:
        print("\nNOTE: --allow-no-cuda. This machine is for development only; "
              "no reported result may come from here.")

    failed = [c.name for c in checks if c.required and not c.ok]
    write_json(
        "setup_check.json",
        {
            "passed": not failed,
            "seconds": round(t["seconds"], 1),
            "checks": [{"name": c.name, "ok": c.ok, "required": c.required, "detail": c.detail}
                       for c in checks],
        },
    )

    print(f"\n{'PASS' if not failed else 'FAIL'} — "
          f"{sum(c.ok for c in checks)}/{len(checks)} checks ok in {t['seconds']:.1f}s")
    if failed:
        print(f"failing required checks: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
