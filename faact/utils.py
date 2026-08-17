"""Small shared helpers: device resolution, seeding, and wall-clock timing.

Kept deliberately tiny. Anything that grows a second responsibility moves out of here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

log = logging.getLogger("faact")

# All runtime outputs go here. Gitignored — nothing in artifacts/ is ever a program input.
ARTIFACTS = Path(os.environ.get("FAACT_ARTIFACTS", "artifacts"))


def resolve_device(preferred: str | None = None) -> str:
    """Pick a torch device.

    Dev happens on macOS (cpu/mps), runs happen on the RunPod L40S (cuda), so nothing
    downstream is allowed to hardcode a device string.
    """
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    # MPS is fine for shape-level smoke tests but not for anything we report.
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch together.

    Note this does *not* make MuJoCo deterministic on its own — episode reproducibility
    also needs the env seeded at reset. See `faact.envs.state` for the determinism test.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def timed(label: str) -> Iterator[dict[str, float]]:
    """Time a block and log it. Rule 8: every long-running job reports wall clock.

    Yields a dict that is filled in on exit, so callers can persist the duration:
        with timed("branch rollout") as t: ...
        record["seconds"] = t["seconds"]
    """
    out: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start
        log.info("%s took %.1fs (%.2f min)", label, out["seconds"], out["seconds"] / 60)


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON under artifacts/, creating parents. Returns the resolved path."""
    p = Path(path)
    if not p.is_absolute():
        p = ARTIFACTS / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
