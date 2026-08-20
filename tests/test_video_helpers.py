"""Tests for the side-by-side video helpers.

Pure array manipulation, so no simulator and no rendering. What matters is that pairing
never truncates the longer run — truncating would hide the outcome of whichever episode
lasted longer, which is usually the failure the video exists to show.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
videos = importlib.import_module("08_make_videos")


def frames(n: int, value: int, h: int = 32, w: int = 48) -> list[np.ndarray]:
    return [np.full((h, w, 3), value, dtype=np.uint8) for _ in range(n)]


def test_pairing_holds_the_shorter_run_rather_than_truncating():
    out = videos.stack_side_by_side(frames(5, 10), frames(9, 20))
    assert len(out) == 9, "must run to the length of the longer episode"
    assert out[0].shape == (32, 96, 3)


def test_held_frames_are_dimmed_and_marked_not_silently_frozen():
    """A frozen panel must read as "this episode finished", not as a glitch.

    Plain duplication looked like the simulation had hung, which is exactly how a reader
    first interpreted it. Held frames are therefore darkened and captioned.
    """
    left = frames(3, 200, h=80, w=120)
    out = videos.stack_side_by_side(left, frames(6, 20, h=80, w=120))

    live = out[2][:, :120]
    held = out[5][:, :120]
    assert held.mean() < live.mean() * 0.8, "held frames must be visibly dimmed"
    assert held.max() > 0, "dimming must not black the frame out entirely"


def test_held_frames_derive_from_the_last_live_frame():
    """Dimmed, but still the same picture — not a blank or a different frame."""
    left = frames(3, 200, h=80, w=120)
    out = videos.stack_side_by_side(left, frames(6, 20, h=80, w=120))
    held = out[5][:, :120]
    # 200 * 0.55 = 110; allow room for the caption text pixels.
    assert 90 < np.median(held) < 130


def test_equal_length_runs_are_paired_one_to_one():
    out = videos.stack_side_by_side(frames(4, 10), frames(4, 20))
    assert len(out) == 4
    assert (out[0][:, :48] == 10).all() and (out[0][:, 48:] == 20).all()


def test_labelling_preserves_frame_shape_and_count():
    src = frames(6, 128)
    out = videos.label_frames(src, onset=2, tag="fixed h=100", success=True)
    assert len(out) == len(src)
    assert out[0].shape == src[0].shape


def test_labelling_does_not_mutate_the_input_frames():
    src = frames(4, 128)
    videos.label_frames(src, onset=1, tag="x", success=False)
    assert all((f == 128).all() for f in src)


def test_onset_flash_marks_only_the_disturbance_window():
    """The border must appear at onset and clear afterwards, or it marks nothing.

    Sampled well below the 28px caption bar, at the left edge where the 6px border sits.
    """
    out = videos.label_frames(frames(30, 128, h=160, w=200), onset=10, tag="x", success=True)
    edge = lambda f: tuple(int(v) for v in f[100, 0])  # noqa: E731

    assert edge(out[5]) == (128, 128, 128), "no border before onset"
    assert edge(out[12]) == (255, 170, 0), "border during the disturbance window"
    assert edge(out[25]) == (128, 128, 128), "border cleared after the window"
