"""Isotonic calibration for the reversibility estimate.

The horizon map consumes R̂ as a **value**, not a rank: `h = clip(h_min, h_max, h_max·R̂^γ)`.
So an estimator can rank states well and still drive the controller badly if its values are
biased — and ours is. The measured calibration curve (artifacts/fig_calibration.png) sits
above the diagonal through the middle of the range: predicted 0.2 corresponds to a measured
0.35, predicted 0.5 to 0.6. Those are exactly the states where the horizon decision is live.

Isotonic regression fixes the values without ever inverting the ranking, because the fitted
map is monotone non-decreasing. Note the precise claim: it never reorders a pair, but it
*can* merge distinct predictions into ties, and ties do move Spearman slightly. So
calibration is close to rank-preserving rather than exactly so — worth stating, because the
reported metric is a rank correlation and it is not strictly invariant here.

Implemented with pool-adjacent-violators rather than pulling in scikit-learn for one
function; the algorithm is short and the dependency is not.
"""

from __future__ import annotations

import numpy as np


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the weighted monotone non-decreasing least-squares fit.

    Walks left to right maintaining a stack of blocks; whenever a new block would violate
    monotonicity against the previous one, the two are merged into their weighted mean and
    the check repeats backwards.
    """
    values, weights, counts = [], [], []
    for value, weight in zip(y, w):
        values.append(float(value))
        weights.append(float(weight))
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2, c2 = values.pop(), weights.pop(), counts.pop()
            v1, w1, c1 = values.pop(), weights.pop(), counts.pop()
            merged_w = w1 + w2
            values.append((v1 * w1 + v2 * w2) / merged_w)
            weights.append(merged_w)
            counts.append(c1 + c2)
    return np.repeat(np.asarray(values), np.asarray(counts))


class IsotonicCalibrator:
    """Monotone map from predicted R̂ to calibrated R̂, fitted on out-of-fold predictions.

    Must be fitted on *out-of-fold* predictions. Fitting on in-sample predictions would
    learn the model's memorisation of its training set and calibrate to a fiction.
    """

    def __init__(self) -> None:
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, pred: np.ndarray, target: np.ndarray) -> "IsotonicCalibrator":
        pred = np.asarray(pred, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()
        if pred.shape != target.shape:
            raise ValueError(f"pred {pred.shape} and target {target.shape} must match")
        if pred.size == 0:
            raise ValueError("cannot calibrate on zero samples")

        order = np.argsort(pred, kind="mergesort")
        x = pred[order]
        fitted = _pava(target[order], np.ones_like(target))

        # Collapse duplicate x values so np.interp gets a strictly increasing grid.
        keep = np.concatenate([np.diff(x) > 0, [True]])
        self.x_, self.y_ = x[keep], fitted[keep]
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        """Apply the fitted map, clipped to [0, 1] and flat outside the fitted range."""
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicCalibrator.fit must be called before predict")
        p = np.asarray(pred, dtype=np.float64)
        if self.x_.size == 1:  # degenerate fit: one distinct prediction
            return np.clip(np.full(p.shape, self.y_[0]), 0.0, 1.0)
        return np.clip(np.interp(p, self.x_, self.y_), 0.0, 1.0)


class CalibratedScorer:
    """Wraps a scorer so the controller sees calibrated values instead of raw ones."""

    def __init__(self, scorer, calibrator: IsotonicCalibrator) -> None:
        self.scorer = scorer
        self.calibrator = calibrator

    def predict_one(self, features: dict) -> float:
        return float(self.calibrator.predict(np.array([self.scorer.predict_one(features)]))[0])
