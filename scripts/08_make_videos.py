#!/usr/bin/env python
"""Side-by-side videos: same seed, same perturbation, two horizon policies.

One clear save is worth more than a montage, so this renders matched pairs rather than a
highlight reel: identical initial state, identical disturbance, and the only difference is
how long the controller commits to each chunk.

Episodes are chosen by *outcome divergence* — pairs where one policy succeeds and the other
fails — because a pair where both succeed shows nothing. If no such pair exists, the script
says so instead of quietly rendering a pair that makes no point.

    python scripts/08_make_videos.py --episodes 12 --gamma 1.0 --h-min 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.backbone.act_wrapper import ACTWrapper  # noqa: E402
from faact.envs.make import CAMERA, MAX_EPISODE_STEPS, make_env  # noqa: E402
from faact.envs.perturb import KINDS, sample_spec  # noqa: E402
from faact.eval.runner import PerturbationNeverFired, run_episode  # noqa: E402
from faact.runtime.controller import fixed, reversibility_gated  # noqa: E402
from faact.runtime.executor import ChunkExecutor  # noqa: E402
from faact.thermal import limit_torch_threads  # noqa: E402
from faact.utils import ARTIFACTS, timed  # noqa: E402

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
# Onset window, chosen from the task's own phase timings rather than guessed.
#
# Measured on 5 successful episodes: the right gripper first contacts the cube at median
# step 154, and the transfer completes at median step 278. The original window (40, 160)
# therefore fired almost entirely BEFORE the cube was grasped — which is why `grasp_slip`
# measured as a complete no-op (65% at every duration, identical to the unperturbed rate):
# forcing a gripper open when it holds nothing does nothing.
#
# (150, 280) puts the disturbance inside grasp-and-transport, which is where recoverability
# is actually in question.
ONSET_RANGE = (150, 280)


def label_frames(frames: list[np.ndarray], onset: int, tag: str, success: bool) -> list[np.ndarray]:
    """Burn a caption bar and an onset flash into a frame sequence.

    Rendered into the pixels rather than added as overlay tracks so the video is
    self-describing wherever it ends up — a slide, a site, a Slack message.
    """
    from PIL import Image, ImageDraw

    out = []
    for t, frame in enumerate(frames):
        img = Image.fromarray(np.asarray(frame).copy())
        draw = ImageDraw.Draw(img)
        colour = (60, 200, 90) if success else (220, 70, 70)
        draw.rectangle([0, 0, img.width, 28], fill=(20, 20, 20))
        draw.text((8, 8), f"{tag}   t={t:>3}   {'SUCCESS' if success else 'FAILURE'}",
                  fill=colour)
        # Flash a border for a few steps at the disturbance so the eye catches it.
        if onset <= t < onset + 12:
            draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(255, 170, 0), width=6)
        out.append(np.asarray(img))
    return out


def stack_side_by_side(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    """Pair frames horizontally, holding the last frame of whichever ends first.

    Holding rather than truncating keeps both runs visible to the end — truncating would
    hide the outcome of the longer episode, which is usually the failure.
    """
    n = max(len(left), len(right))
    pad = lambda seq: seq + [seq[-1]] * (n - len(seq))  # noqa: E731
    return [np.concatenate([a, b], axis=1) for a, b in zip(pad(left), pad(right))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS))
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--h-min", type=int, default=20)
    ap.add_argument("--baseline-h", type=int, default=100)
    ap.add_argument("--max-pairs", type=int, default=3)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--head", default=str(ARTIFACTS / "reversibility_head.pt"))
    ap.add_argument("--compare-fixed", type=int, nargs=2, default=None,
                    metavar=("H_A", "H_B"),
                    help="film two fixed horizons against each other instead of fixed vs gated")
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "videos"))
    args = ap.parse_args()

    import imageio.v3 as iio

    limit_torch_threads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the ablation's head loader rather than duplicating it, so the two cannot
    # drift apart. The module name starts with a digit, so it needs import_module.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module

    scorer = None if args.compare_fixed else import_module("06_run_ablation").load_scorer(Path(args.head))

    policy = ACTWrapper(CHECKPOINT, device="cpu")
    env = make_env()

    # Either compare two fixed horizons (--compare-fixed A B) or a fixed baseline against
    # the gated controller. The two-fixed mode exists to film the measured cliff: h=5 scores
    # 0/40 while h=40 scores 49%, and that is the project's clearest single result.
    if args.compare_fixed:
        a, b = args.compare_fixed
        conditions = {
            f"fixed h={a}": (lambda v: lambda: fixed(v))(a),
            f"fixed h={b}": (lambda v: lambda: fixed(v))(b),
        }
    else:
        conditions = {
            f"fixed h={args.baseline_h}": lambda: fixed(args.baseline_h),
            f"reversibility-gated (gamma={args.gamma})":
                lambda: reversibility_gated(scorer, h_min=args.h_min, gamma=args.gamma),
        }

    pairs, written = [], 0
    with timed("video rendering") as t:
        for i in range(args.episodes):
            seed = args.seed + i
            rng = np.random.default_rng(seed)
            spec = sample_spec(args.kinds[i % len(args.kinds)], rng, onset_range=ONSET_RANGE)

            runs = {}
            for name, factory in conditions.items():
                try:
                    runs[name] = run_episode(
                        env, ChunkExecutor(policy, factory()), seed=seed,
                        perturb_spec=spec, max_steps=MAX_EPISODE_STEPS, record_frames=True,
                    )
                except PerturbationNeverFired:
                    runs = {}
                    break
            if not runs:
                continue

            names = list(conditions)
            a, b = runs[names[0]], runs[names[1]]
            diverged = a.success != b.success
            pairs.append((seed, spec.kind, a.success, b.success, diverged))
            print(f"  seed {seed} {spec.kind:<16} "
                  f"{names[0]}={int(a.success)}  {names[1]}={int(b.success)}"
                  f"{'   <- diverged' if diverged else ''}", flush=True)

            if diverged and written < args.max_pairs:
                frames = stack_side_by_side(
                    label_frames(a.frames, spec.onset_step, names[0], a.success),
                    label_frames(b.frames, spec.onset_step, names[1], b.success),
                )
                tag = "cliff" if args.compare_fixed else "gated"
                path = out_dir / f"{tag}_seed{seed}_{spec.kind}.mp4"
                iio.imwrite(path, np.stack(frames), fps=args.fps, codec="libx264")
                written += 1
                print(f"      -> wrote {path}", flush=True)
    env.close()

    n_div = sum(p[4] for p in pairs)
    print(f"\nMEASURED: {len(pairs)} matched pairs, {n_div} diverged in outcome, "
          f"{written} videos written to {out_dir}")
    print(f"WALL CLOCK: {t['seconds'] / 60:.1f} min")
    if written == 0:
        print("No pair diverged, so no video would show a difference. "
              "Nothing written — rerun with more --episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
