"""Thermal governor — keep long CPU-bound jobs from cooking the laptop.

M3 labelling is hours of saturated CPU (MuJoCo runs at 84 steps/s and is the bottleneck by
~13x over the policy). On a fanless-ish laptop that means sustained heat, and a thermally
throttled machine is also a *slow* machine, so backing off is not only kinder to the
hardware — it is often no slower overall than pushing through the throttle.

Signal: `NSProcessInfo.thermalState`, Apple's own thermal-pressure API. No sudo, no
sampling of SMC sensors, no third-party tools. It reports four levels:

    0 nominal    everything is fine
    1 fair       warm, fans up, still running at full speed
    2 serious    the system is actively shedding performance
    3 critical   aggressive throttling; keep running here and you get heat, not results

We work through nominal/fair and pause at serious or above, resuming once it drops back.
`pmset -g therm` is deliberately not used: it reports nothing at all until a thermal event
has already been recorded, so it cannot be polled proactively.

On non-macOS (the RunPod box) there is no such signal, and a datacentre machine does not
need one; the governor degrades to a no-op and says so rather than pretending to protect.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

log = logging.getLogger("faact.thermal")

NOMINAL, FAIR, SERIOUS, CRITICAL = 0, 1, 2, 3
STATE_NAMES = {NOMINAL: "nominal", FAIR: "fair", SERIOUS: "serious", CRITICAL: "critical"}


def thermal_state() -> int | None:
    """Current thermal pressure (0-3), or None where the platform cannot report it."""
    if sys.platform != "darwin":
        return None
    try:
        from Foundation import NSProcessInfo
    except ImportError:
        return None
    return int(NSProcessInfo.processInfo().thermalState())


def thermal_available() -> bool:
    return thermal_state() is not None


@dataclass
class ThermalGovernor:
    """Pauses work while the machine is thermally stressed.

    Call `checkpoint()` between units of work — between episodes, or between label points.
    It returns immediately when cool and blocks while hot, so callers need no thermal logic
    of their own beyond placing the call somewhere interruptible.

    `pause_at=SERIOUS` by default: `fair` is normal for sustained compute and pausing there
    would stall almost permanently, defeating the job rather than protecting the machine.
    """

    pause_at: int = SERIOUS
    resume_at: int = FAIR
    poll_seconds: float = 15.0
    # Give up waiting eventually rather than hanging a job forever on a machine that simply
    # runs hot. Reported loudly; the caller decides what to do about it.
    max_pause_seconds: float = 600.0

    total_paused: float = 0.0
    n_pauses: int = 0
    max_state_seen: int = 0
    _log_each_pause: bool = True
    history: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def sample(self) -> int | None:
        state = thermal_state()
        if state is not None:
            self.max_state_seen = max(self.max_state_seen, state)
            self.history.append((time.time(), state))
        return state

    def checkpoint(self) -> float:
        """Block while the machine is too hot. Returns seconds spent paused."""
        state = self.sample()
        if state is None or state < self.pause_at:
            return 0.0

        self.n_pauses += 1
        start = time.perf_counter()
        if self._log_each_pause:
            log.warning(
                "thermal pressure %s — pausing until it drops to %s",
                STATE_NAMES.get(state, state),
                STATE_NAMES.get(self.resume_at, self.resume_at),
            )

        while True:
            time.sleep(self.poll_seconds)
            waited = time.perf_counter() - start
            state = self.sample()
            if state is None or state <= self.resume_at:
                break
            if waited >= self.max_pause_seconds:
                log.warning(
                    "still %s after %.0fs of cooling; continuing anyway so the job can "
                    "finish — consider fewer workers",
                    STATE_NAMES.get(state, state),
                    waited,
                )
                break

        paused = time.perf_counter() - start
        self.total_paused += paused
        log.info("resumed after %.0fs (state=%s)", paused, STATE_NAMES.get(state, state))
        return paused

    def report(self) -> dict:
        """JSON-safe summary, so a run records how much it was throttled."""
        return {
            "available": thermal_available(),
            "n_pauses": self.n_pauses,
            "total_paused_seconds": round(self.total_paused, 1),
            "max_state_seen": self.max_state_seen,
            "max_state_name": STATE_NAMES.get(self.max_state_seen, "unknown"),
            "pause_at": STATE_NAMES.get(self.pause_at, self.pause_at),
        }


def safe_worker_count(requested: int | None = None, reserve: int = 1) -> int:
    """How many worker processes to run without saturating the machine.

    Counts **performance** cores only. The M5's 6 efficiency cores are much slower at
    MuJoCo, so scheduling workers onto them buys little throughput while adding heat and
    making the machine unpleasant to use. One P-core is reserved for the OS and the caller.
    """
    if requested is not None:
        if requested < 1:
            raise ValueError(f"worker count must be >= 1, got {requested}")
        return requested

    import os
    import subprocess

    perf_cores = None
    if sys.platform == "darwin":
        try:
            perf_cores = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True
                ).strip()
            )
        except (subprocess.SubprocessError, ValueError):
            perf_cores = None

    total = perf_cores or os.cpu_count() or 2
    return max(1, total - reserve)


def limit_torch_threads(n: int = 1) -> None:
    """Pin each worker to one torch thread.

    Without this every worker spawns a thread pool sized to the whole machine, so N workers
    fight over N x cores threads. That is slower than running serially *and* hotter — the
    single most important knob for a multi-process CPU job.
    """
    import torch

    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        # Only settable before any parallel work has started; already-initialised is fine.
        pass
