"""Tests for isotonic calibration.

The load-bearing property is that calibration changes VALUES without ever *inverting* an
ordering — the controller consumes R̂ as a value, while Spearman (which we report) sees only
rank. Note it is not exactly rank-preserving: isotonic regression can merge distinct
predictions into ties, and ties shift Spearman a little. The tests pin down what is actually
true rather than the tidier claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from faact.models.calibration import CalibratedScorer, IsotonicCalibrator


def test_calibration_never_inverts_an_ordering():
    """The exact guarantee: p_i < p_j implies cal_i <= cal_j, for every pair."""
    rng = np.random.default_rng(0)
    pred = rng.uniform(size=200)
    target = np.clip(pred * 0.5 + 0.25 + rng.normal(0, 0.05, 200), 0, 1)
    cal = IsotonicCalibrator().fit(pred, target).predict(pred)

    order = np.argsort(pred)
    assert np.all(np.diff(cal[order]) >= -1e-12), "calibration inverted a pair"


def test_calibration_barely_moves_rank_correlation():
    """Ties introduced by pooling shift Spearman slightly; it must not collapse."""
    rng = np.random.default_rng(0)
    pred = rng.uniform(size=300)
    target = np.clip(pred * 0.5 + 0.25 + rng.normal(0, 0.05, 300), 0, 1)

    before = spearmanr(pred, target).statistic
    after = spearmanr(IsotonicCalibrator().fit(pred, target).predict(pred), target).statistic
    assert after == pytest.approx(before, abs=0.05), (before, after)


def test_calibration_removes_a_systematic_bias():
    """The measured failure mode: well-ranked predictions that are systematically low."""
    rng = np.random.default_rng(1)
    target = rng.uniform(size=400)
    pred = np.clip(target * 0.5, 0, 1)  # correct order, badly wrong values

    before = np.mean(np.abs(pred - target))
    after = np.mean(np.abs(IsotonicCalibrator().fit(pred, target).predict(pred) - target))
    assert after < before / 2, (before, after)


def test_output_is_monotone_non_decreasing():
    rng = np.random.default_rng(2)
    pred, target = rng.uniform(size=200), rng.uniform(size=200)
    cal = IsotonicCalibrator().fit(pred, target)

    grid = np.linspace(0, 1, 50)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-12)


def test_output_stays_in_the_unit_interval():
    rng = np.random.default_rng(3)
    cal = IsotonicCalibrator().fit(rng.uniform(size=100), rng.uniform(size=100))
    out = cal.predict(np.array([-5.0, 0.0, 0.5, 1.0, 5.0]))
    assert np.all(out >= 0) and np.all(out <= 1)


def test_saturated_targets_are_handled():
    """Our real labels are 84% exactly 0 or 1, so this is the actual regime."""
    rng = np.random.default_rng(4)
    pred = rng.uniform(size=200)
    target = (pred > 0.5).astype(float)
    out = IsotonicCalibrator().fit(pred, target).predict(pred)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0 and out.max() <= 1


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit must be called"):
        IsotonicCalibrator().predict(np.array([0.5]))


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="must match"):
        IsotonicCalibrator().fit(np.zeros(5), np.zeros(4))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="zero samples"):
        IsotonicCalibrator().fit(np.array([]), np.array([]))


def test_single_distinct_prediction_does_not_crash():
    cal = IsotonicCalibrator().fit(np.full(10, 0.3), np.linspace(0, 1, 10))
    assert 0.0 <= float(cal.predict(np.array([0.3]))[0]) <= 1.0


def test_calibrated_scorer_wraps_predict_one():
    class Raw:
        def predict_one(self, features):
            return 0.2

    rng = np.random.default_rng(5)
    pred = rng.uniform(size=200)
    cal = IsotonicCalibrator().fit(pred, np.clip(pred + 0.3, 0, 1))
    scorer = CalibratedScorer(Raw(), cal)

    value = scorer.predict_one({})
    assert 0.0 <= value <= 1.0
    assert value > 0.2, "calibration should lift a systematically low estimate"
