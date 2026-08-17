"""Metrics tests on hand-checked toy episodes.

Every expected value here is worked out by hand in the test, so a metric that drifts gets
caught by arithmetic rather than by a plausible-looking number in the ablation table.
"""

from __future__ import annotations

import numpy as np
import pytest

from faact.eval.metrics import (
    false_alarm_rate,
    interventions_per_episode,
    lead_time,
    markdown_table,
    mean_committed_horizon,
    mean_lead_time,
    success_rate,
    summarise,
)
from faact.eval.runner import EpisodeResult


def episode(
    success: bool,
    replan_steps: list[int],
    horizons: list[int],
    onset: int | None = None,
    seed: int = 0,
    max_reward: float = 4.0,
) -> EpisodeResult:
    perturb = None
    if onset is not None:
        perturb = {"kind": "grasp_slip", "onset_step": onset, "magnitude": 1.0,
                   "duration": 3, "seed": 0}
    return EpisodeResult(
        seed=seed,
        success=success,
        steps=400,
        max_reward=max_reward if success else 2.0,
        replan_steps=replan_steps,
        horizons=horizons,
        perturb=perturb,
        perturb_fired=onset is not None,
    )


# -- basics ------------------------------------------------------------------------------


def test_success_rate_counts_successes():
    eps = [episode(True, [0], [100]), episode(False, [0], [100]), episode(True, [0], [100])]
    assert success_rate(eps) == pytest.approx(2 / 3)


def test_success_rate_raises_on_empty():
    with pytest.raises(ValueError, match="zero episodes"):
        success_rate([])


def test_interventions_per_episode_averages_replan_counts():
    # 4 replans and 2 replans -> mean 3.0
    eps = [episode(True, [0, 20, 40, 60], [20] * 4), episode(True, [0, 50], [50, 50])]
    assert interventions_per_episode(eps) == pytest.approx(3.0)


def test_mean_committed_horizon_pools_over_replans_not_episodes():
    """One long commitment must not outweigh four short ones just by being its own episode."""
    eps = [episode(True, [0], [100]), episode(True, [0, 10, 20, 30], [10, 10, 10, 10])]
    # Pooled over 5 replans: (100 + 10*4) / 5 = 28. Per-episode averaging would give 55.
    assert mean_committed_horizon(eps) == pytest.approx(28.0)


def test_mean_committed_horizon_raises_when_nothing_replanned():
    with pytest.raises(ValueError, match="no replans"):
        mean_committed_horizon([episode(True, [], [])])


# -- lead time ---------------------------------------------------------------------------


def test_lead_time_measures_steps_from_onset_to_first_short_commitment():
    # Onset 100; horizons 100,100,20 at steps 0,100,200 -> first reaction at t=200 -> +100.
    ep = episode(False, [0, 100, 200], [100, 100, 20], onset=100)
    assert lead_time(ep) == pytest.approx(100.0)


def test_lead_time_is_negative_when_the_controller_was_already_cautious():
    """Shortening before onset is prior caution, not prescience — reported, not clipped."""
    ep = episode(False, [0, 50, 150], [100, 20, 20], onset=100)
    assert lead_time(ep) == pytest.approx(-50.0)


def test_lead_time_is_none_when_the_controller_never_reacted():
    ep = episode(True, [0, 100, 200], [100, 100, 100], onset=100)
    assert lead_time(ep) is None


def test_lead_time_is_none_for_unperturbed_episodes():
    assert lead_time(episode(True, [0], [20])) is None


def test_mean_lead_time_reports_coverage_alongside_the_mean():
    """A good mean over few episodes is not a good result; coverage must come with it."""
    eps = [
        episode(False, [0, 100, 110], [100, 100, 10], onset=100),  # reacted at +10
        episode(False, [0, 100, 130], [100, 100, 10], onset=100),  # reacted at +30
        episode(True, [0, 100], [100, 100], onset=100),            # never reacted
        episode(True, [0], [100]),                                 # unperturbed
    ]
    mean, n_reacted, n_perturbed = mean_lead_time(eps)
    assert mean == pytest.approx(20.0)
    assert (n_reacted, n_perturbed) == (2, 3)


def test_mean_lead_time_is_nan_when_nothing_ever_reacted():
    mean, n_reacted, n_perturbed = mean_lead_time([episode(True, [0], [100], onset=50)])
    assert np.isnan(mean)
    assert (n_reacted, n_perturbed) == (0, 1)


# -- false alarms -------------------------------------------------------------------------


def test_false_alarm_rate_counts_shortening_on_successful_episodes():
    eps = [
        episode(True, [0, 100], [100, 10]),   # succeeded but got nervous -> alarm
        episode(True, [0], [100]),            # succeeded, stayed committed -> no alarm
        episode(False, [0], [10]),            # failed -> not counted either way
    ]
    assert false_alarm_rate(eps) == pytest.approx(0.5)


def test_false_alarm_rate_penalises_an_always_cautious_controller():
    """The counterweight to lead time: commit h=5 always and this goes to 100%."""
    eps = [episode(True, [0, 5, 10], [5, 5, 5]) for _ in range(4)]
    assert false_alarm_rate(eps) == pytest.approx(1.0)


def test_false_alarm_rate_is_nan_without_successes():
    assert np.isnan(false_alarm_rate([episode(False, [0], [10])]))


# -- reporting ----------------------------------------------------------------------------


def test_summarise_produces_every_ablation_field():
    eps = [
        episode(True, [0, 100], [100, 100], onset=50),
        episode(False, [0, 100, 120], [100, 100, 10], onset=100),
    ]
    row = summarise(eps, label="fixed h=100")

    assert row["condition"] == "fixed h=100"
    assert row["n_episodes"] == 2
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["interventions_per_episode"] == pytest.approx(2.5)
    assert row["mean_committed_horizon"] == pytest.approx((100 + 100 + 100 + 100 + 10) / 5)
    assert row["mean_lead_time"] == pytest.approx(20.0)
    assert (row["n_reacted"], row["n_perturbed"]) == (1, 2)


def test_markdown_table_renders_a_row_per_condition():
    rows = [summarise([episode(True, [0], [100], onset=50)], label=c)
            for c in ("fixed h=100", "reversibility_gated")]
    table = markdown_table(rows)

    assert table.count("\n") == 3  # header + separator + 2 rows
    assert "fixed h=100" in table and "reversibility_gated" in table
    assert "—" in table, "an unreacted lead time must render as a dash, not as nan"
