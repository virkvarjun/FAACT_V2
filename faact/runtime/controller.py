"""Horizon policies — the five conditions of the M5 ablation, as interchangeable functions.

Each returns a `HorizonFn` for `ChunkExecutor`, so the ablation varies exactly one thing.

The claim under test lives in `horizon_from_score`: commit in proportion to how recoverable
the state is. High reversibility means a mistake is still fixable, so commit long and enjoy
the smoothness of chunking; low reversibility means the point of no return is close, so
commit short and keep the option to react.

    h = clip(h_min, h_max, round(h_max * R^gamma))

`gamma > 1` makes the controller more conservative — it shortens the horizon sooner as R
falls. gamma is a reported hyperparameter, not a tuned-until-pretty one.

**Measured caveat, and it is the project's main result.** Success against horizon is a cliff
and a plateau, not a ramp: h=5 scores 0/40, h=10 scores 3%, h=20 reaches 30%, and everything
from 20 to 100 is statistically indistinguishable (all p >= 0.09).

The cliff is a *stall*, not the thrashing one might expect. An ACT chunk begins at the current
pose, so executing only its first few actions and replanning from a barely-changed state
re-emits a near-identical chunk; the arm creeps forward at a fraction of normal speed and runs
out of time. Measured net joint displacement over an episode: 0.16 at h=5 against 2.02 at
h=100, identical across all five seeds tested, with h=10 bimodal right at the threshold.

The consequence for gating: every controller operates inside the plateau, where horizon does
not change success at all. It can only lose by approaching the cliff and cannot gain by moving
within the flat — which is why an oracle fed ground-truth R does no better than a fixed
horizon. See the README.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Floor for a gated horizon. NOT 5, which is where the map's range naturally ends: fixed
# h=5 measured 0/40 because the arm stalls there — net joint displacement 0.16 against 2.02
# at h=100 (see the module docstring). A controller whose floor sits in that regime is
# punished for reacting, which inverts the intent of gating.
H_MIN = 20
H_MAX = 100

# The absolute lower bound the map will accept, kept only so the collapse regime can be
# measured deliberately (see the horizon sweep) rather than wandered into by default.
H_FLOOR = 5

# A scorer maps (timestep, features) to a scalar in [0, 1].
Scorer = Callable[[int, dict], float]


def horizon_from_score(score: float, h_min: int = H_MIN, h_max: int = H_MAX, gamma: float = 1.0) -> int:
    """Map a score in [0,1] to a commitment horizon. Monotone non-decreasing in `score`."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {score}")
    if not 1 <= h_min <= h_max:
        raise ValueError(f"need 1 <= h_min <= h_max, got h_min={h_min}, h_max={h_max}")
    return int(np.clip(round(h_max * score**gamma), h_min, h_max))


def fixed(h: int) -> Callable:
    """Constant horizon. h=100 is ACT as published; h=20 is the replan-more-often control."""
    if h < 1:
        raise ValueError(f"fixed horizon must be >= 1, got {h}")

    def horizon_fn(_t: int, _features: dict, _chunk: np.ndarray) -> int:
        return h

    return horizon_fn


def score_gated(
    scorer: Scorer,
    h_min: int = H_MIN,
    h_max: int = H_MAX,
    gamma: float = 1.0,
) -> Callable:
    """Horizon driven by any scalar scorer. The shared body of the three gated conditions.

    `reversibility_gated` and `risk_gated` differ only in which quantity they feed here —
    which is the point of the comparison: same controller, different control variable.
    """

    def horizon_fn(t: int, features: dict, _chunk: np.ndarray) -> int:
        return horizon_from_score(float(scorer(t, features)), h_min, h_max, gamma)

    return horizon_fn


def reversibility_gated(head, h_min: int = H_MIN, h_max: int = H_MAX, gamma: float = 1.0):
    """The claim: commit in proportion to predicted reversibility R̂(s)."""
    return score_gated(lambda _t, f: head.predict_one(f), h_min, h_max, gamma)


def risk_gated(head, h_min: int = H_MIN, h_max: int = H_MAX, gamma: float = 1.0):
    """v1-style comparison: same controller driven by failure risk, so score = 1 - risk.

    If this matches `reversibility_gated`, the claim is not supported and we say so.
    """
    return score_gated(lambda _t, f: 1.0 - head.predict_one(f), h_min, h_max, gamma)


def oracle(labels: dict[int, float], h_min: int = H_MIN, h_max: int = H_MAX, gamma: float = 1.0):
    """Ceiling condition: the same controller fed *labelled* R from branch rollouts.

    The gap between this and `reversibility_gated` is the cost of estimation error, which
    separates "reversibility is the wrong variable" from "our estimate of it is too weak".
    Missing timesteps are interpolated from the nearest labelled ones rather than defaulted
    to a constant, which would quietly turn the oracle into a fixed-horizon policy.
    """
    if not labels:
        raise ValueError("oracle needs at least one labelled timestep")
    ts = np.array(sorted(labels))
    rs = np.array([labels[t] for t in ts], dtype=np.float64)

    def horizon_fn(t: int, _features: dict, _chunk: np.ndarray) -> int:
        r = float(np.interp(t, ts, rs))
        return horizon_from_score(r, h_min, h_max, gamma)

    return horizon_fn
