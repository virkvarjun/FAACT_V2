"""Tests for ablation bookkeeping.

The alignment logic is what keeps the five conditions comparable. Shortening the horizon
changes when episodes terminate, so exclusions genuinely differ between conditions — and
scoring conditions on different episode subsets would compare five different experiments.
"""

from __future__ import annotations

from faact.eval.ablation import align_conditions, build_table
from faact.eval.runner import EpisodeResult


def ep(seed: int, success: bool = True) -> EpisodeResult:
    return EpisodeResult(
        seed=seed,
        success=success,
        steps=400,
        max_reward=4.0 if success else 2.0,
        replan_steps=[0, 100],
        horizons=[100, 100],
        perturb={"kind": "grasp_slip", "onset_step": 50, "magnitude": 1.0,
                 "duration": 3, "seed": 0},
        perturb_fired=True,
    )


def test_alignment_keeps_only_seeds_present_in_every_condition():
    per_condition = {
        "fixed h=100": [ep(1), ep(2), ep(3)],
        "fixed h=20": [ep(1), ep(3)],        # seed 2 was excluded here
        "reversibility": [ep(1), ep(2), ep(3)],
    }
    aligned = align_conditions(per_condition)

    assert {len(v) for v in aligned.values()} == {2}
    for results in aligned.values():
        assert sorted(r.seed for r in results) == [1, 3]


def test_alignment_is_a_noop_when_conditions_already_agree():
    per_condition = {"a": [ep(1), ep(2)], "b": [ep(1), ep(2)]}
    aligned = align_conditions(per_condition)
    assert {len(v) for v in aligned.values()} == {2}


def test_alignment_handles_no_common_seeds():
    aligned = align_conditions({"a": [ep(1)], "b": [ep(2)]})
    assert all(len(v) == 0 for v in aligned.values())


def test_alignment_of_nothing_is_empty():
    assert align_conditions({}) == {}


def test_build_table_emits_one_row_per_condition_with_n():
    rows = build_table({"fixed h=100": [ep(1), ep(2, False)], "fixed h=20": [ep(1), ep(2)]})
    assert [r["condition"] for r in rows] == ["fixed h=100", "fixed h=20"]
    assert rows[0]["success_rate"] == 0.5 and rows[1]["success_rate"] == 1.0
    assert all(r["n_episodes"] == 2 for r in rows)
