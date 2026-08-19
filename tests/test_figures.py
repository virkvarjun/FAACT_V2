"""Tests for figure logic.

Appearance is not testable and not worth testing; the derived quantity is. `point_of_no_return`
is a claim the talk makes explicitly, so it gets pinned down here.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.viz.figures import plot_ablation_bars, plot_reversibility_trace, point_of_no_return

T = np.arange(0, 400, 25)


def test_no_point_of_no_return_when_reversibility_stays_high():
    assert point_of_no_return(T, np.full(len(T), 0.9)) is None


def test_a_recovered_dip_is_not_a_point_of_no_return():
    """The distinction the whole idea rests on: a stumble the policy recovers from."""
    r = np.array([0.9, 0.9, 0.2, 0.2, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert point_of_no_return(T, r) is None


def test_permanent_collapse_is_reported_at_its_start():
    r = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert point_of_no_return(T, r) == T[6]


def test_a_late_dip_that_never_recovers_counts():
    r = np.concatenate([np.full(len(T) - 2, 0.9), [0.1, 0.0]])
    assert point_of_no_return(T, r) == T[-2]


def test_an_episode_doomed_from_the_start_reports_the_first_step():
    assert point_of_no_return(T, np.zeros(len(T))) == T[0]


def test_threshold_is_respected():
    r = np.concatenate([np.full(8, 0.9), np.full(8, 0.4)])
    assert point_of_no_return(T, r, threshold=0.5) == T[8]
    assert point_of_no_return(T, r, threshold=0.3) is None


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="must match"):
        point_of_no_return(T, np.zeros(3))


def test_empty_input_has_no_point_of_no_return():
    assert point_of_no_return(np.array([]), np.array([])) is None


def test_reversibility_trace_writes_png_and_svg(tmp_path):
    r = np.concatenate([np.full(8, 0.9), np.full(8, 0.1)])
    out = tmp_path / "trace.png"
    ponr = plot_reversibility_trace(T, r, onset=150, path=out)

    assert out.exists() and out.with_suffix(".svg").exists()
    assert ponr == T[8]


def test_ablation_bars_render_every_condition(tmp_path):
    rows = [
        {"condition": "fixed h=100", "success_rate": 0.4, "n_success": 8, "n_episodes": 20},
        {"condition": "reversibility_gated", "success_rate": 0.6, "n_success": 12,
         "n_episodes": 20},
    ]
    out = tmp_path / "ablation.png"
    plot_ablation_bars(rows, out)
    assert out.exists() and out.with_suffix(".svg").exists()


def test_horizon_curve_renders_with_adaptive_points(tmp_path):
    """The project's central figure: fixed-horizon curve with controllers overlaid."""
    from faact.viz.figures import plot_horizon_curve

    out = tmp_path / "curve.png"
    plot_horizon_curve(
        np.array([5, 20, 40, 60, 80, 100]),
        np.array([0.0, 0.33, 0.40, 0.43, 0.45, 0.47]),
        out,
        operating_band=(41, 56),
    )
    assert out.exists() and out.with_suffix(".svg").exists()


def test_horizon_curve_sorts_unordered_input(tmp_path):
    """Conditions arrive in table order, not horizon order; the line must not zig-zag."""
    from faact.viz.figures import plot_horizon_curve

    out = tmp_path / "curve2.png"
    plot_horizon_curve(np.array([100, 5, 40]), np.array([0.47, 0.0, 0.40]), out)
    assert out.exists()
