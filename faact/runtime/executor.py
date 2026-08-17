"""ChunkExecutor — the loop whose commitment horizon is the thing this project controls.

ACT predicts `chunk_size` (=100) actions from one observation. Published ACT executes all
of them before looking again, so for 100 steps the policy is blind: if the cube is nudged
at step 3 of a chunk, nothing can respond until step 100.

The executor makes that horizon explicit. Each replan asks a `horizon_fn` how many steps
of the fresh chunk to actually commit to, executes exactly that many, then replans. Every
horizon policy in `faact.runtime.controller` — fixed, risk-gated, reversibility-gated,
oracle — is a different `horizon_fn` against this same loop, so the ablation compares one
variable and not four code paths.

Optional temporal ensembling averages overlapping chunk predictions with exponential
weights, as in the ACT paper. It is off by default: it changes what "committed horizon"
means, and the ablation needs that quantity to stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class ChunkPredictor(Protocol):
    """Anything that turns an observation into (chunk, features). See ACTWrapper."""

    chunk_size: int

    def predict_chunk(self, obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]: ...


# (timestep, features, chunk) -> number of steps to commit
HorizonFn = Callable[[int, dict[str, np.ndarray], np.ndarray], int]


@dataclass
class Replan:
    """One replanning decision, recorded for the ablation metrics."""

    timestep: int
    horizon: int
    features: dict[str, np.ndarray] = field(repr=False, default_factory=dict)


class ChunkExecutor:
    """Drives chunked execution with a controllable commitment horizon.

    Usage is pull-based so the caller keeps the env loop:

        ex = ChunkExecutor(policy, horizon_fn)
        ex.reset()
        for t in range(T):
            action = ex.act(t, obs)     # replans internally when the chunk runs out
            obs, ... = env.step(action)
    """

    def __init__(
        self,
        policy: ChunkPredictor,
        horizon_fn: HorizonFn,
        temporal_ensemble: bool = False,
        ensemble_coeff: float = 0.01,
    ) -> None:
        self.policy = policy
        self.horizon_fn = horizon_fn
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_coeff = ensemble_coeff
        self.replans: list[Replan] = []
        self._chunk: np.ndarray | None = None
        self._chunk_start = 0  # timestep the current chunk was predicted at
        self._committed = 0  # horizon granted to the current chunk
        # (start_timestep, chunk) for every chunk still overlapping the present.
        self._history: list[tuple[int, np.ndarray]] = []

    def reset(self) -> None:
        """Clear all per-episode state. Must be called after every env.reset()."""
        self.policy.reset()
        self.replans.clear()
        self._chunk = None
        self._chunk_start = 0
        self._committed = 0
        self._history.clear()

    # -- main entry point --------------------------------------------------------------

    def act(self, t: int, obs: dict) -> np.ndarray:
        """Return the action for timestep `t`, replanning first if the commitment expired."""
        if self._needs_replan(t):
            self._replan(t, obs)

        assert self._chunk is not None  # guaranteed by _replan
        offset = t - self._chunk_start
        action = self._chunk[offset]

        if self.temporal_ensemble:
            action = self._ensemble(t)
        return np.asarray(action, dtype=np.float64)

    def _needs_replan(self, t: int) -> bool:
        """Replan on the first step, or once the committed horizon has been consumed."""
        if self._chunk is None:
            return True
        return (t - self._chunk_start) >= self._committed

    def _replan(self, t: int, obs: dict) -> None:
        """Predict a fresh chunk and ask the horizon policy how much of it to commit to."""
        chunk, features = self.policy.predict_chunk(obs)
        chunk = np.asarray(chunk)
        if chunk.ndim != 2:
            raise ValueError(f"expected a (chunk_size, action_dim) chunk, got {chunk.shape}")

        horizon = int(self.horizon_fn(t, features, chunk))
        # A horizon of 0 would replan forever without stepping the env; longer than the
        # chunk would index past the end. Both are controller bugs, so fail loudly.
        if not 1 <= horizon <= chunk.shape[0]:
            raise ValueError(
                f"horizon_fn returned {horizon} at t={t}; must be in [1, {chunk.shape[0]}]"
            )

        self._chunk = chunk
        self._chunk_start = t
        self._committed = horizon
        self._history.append((t, chunk))
        self.replans.append(Replan(timestep=t, horizon=horizon, features=features))

    # -- temporal ensembling -----------------------------------------------------------

    def _ensemble(self, t: int) -> np.ndarray:
        """Exponentially-weighted average of every past chunk's prediction for step `t`.

        Weight exp(-coeff * age) favours the freshest chunk, as in the ACT paper. Chunks
        that no longer reach `t` are dropped so the history cannot grow without bound.
        """
        self._history = [(s, c) for s, c in self._history if t - s < c.shape[0]]
        if not self._history:
            raise RuntimeError(f"no chunk covers timestep {t}")

        preds, weights = [], []
        for start, chunk in self._history:
            age = t - start
            preds.append(chunk[age])
            weights.append(np.exp(-self.ensemble_coeff * age))

        w = np.asarray(weights, dtype=np.float64)
        return np.asarray(preds, dtype=np.float64).T @ (w / w.sum())

    # -- reporting ---------------------------------------------------------------------

    @property
    def n_replans(self) -> int:
        return len(self.replans)

    @property
    def mean_committed_horizon(self) -> float:
        """Mean horizon granted per replan — the headline number for horizon policies."""
        if not self.replans:
            return float("nan")
        return float(np.mean([r.horizon for r in self.replans]))
