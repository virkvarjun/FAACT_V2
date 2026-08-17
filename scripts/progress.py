#!/usr/bin/env python
"""One-line progress report for a running labelling job.

Reads the completed episode shards on disk, so it is safe to run at any time and never
touches the job itself. Prints a single line meant to be tailed or polled:

    14:52  8/40 eps  118 pts  R pre 0.81 post 0.34  thermal fair  paused 0s  eta ~68min

    python scripts/progress.py --total 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.thermal import STATE_NAMES, thermal_state  # noqa: E402
from faact.utils import ARTIFACTS  # noqa: E402


def summarise_shards(shard_dir: Path) -> dict:
    """Aggregate whatever episodes have finished so far."""
    pre, post, secs, paused = [], [], [], 0.0
    n_done = n_excluded = n_points = 0

    for path in sorted(shard_dir.glob("episode_*.json")):
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a shard mid-write; it will be counted on the next poll
        n_done += 1
        secs.append(rec.get("seconds", 0.0))
        paused += rec.get("thermal", {}).get("total_paused_seconds", 0.0)
        if rec.get("excluded"):
            n_excluded += 1
            continue
        onset = rec.get("perturb", {}).get("onset_step", 0)
        for p in rec.get("points", []):
            n_points += 1
            (pre if p["timestep"] < onset else post).append(p["reversibility"])

    return {
        "n_done": n_done,
        "n_excluded": n_excluded,
        "n_points": n_points,
        "mean_seconds": float(np.mean(secs)) if secs else 0.0,
        "R_pre": float(np.mean(pre)) if pre else None,
        "R_post": float(np.mean(post)) if post else None,
        "paused_seconds": paused,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=str(ARTIFACTS / "reversibility_shards"))
    ap.add_argument("--total", type=int, default=40, help="episodes the job was asked for")
    ap.add_argument("--workers", type=int, default=3, help="used only for the ETA estimate")
    args = ap.parse_args()

    shard_dir = Path(args.shards)
    if not shard_dir.exists():
        print(f"{time.strftime('%H:%M')}  no shards yet at {shard_dir}")
        return 0

    s = summarise_shards(shard_dir)
    remaining = max(0, args.total - s["n_done"])
    eta_min = remaining * s["mean_seconds"] / max(1, args.workers) / 60 if s["mean_seconds"] else 0

    state = thermal_state()
    rpre = f"{s['R_pre']:.2f}" if s["R_pre"] is not None else "--"
    rpost = f"{s['R_post']:.2f}" if s["R_post"] is not None else "--"

    print(
        f"{time.strftime('%H:%M')}  {s['n_done']}/{args.total} eps  {s['n_points']} pts  "
        f"R pre {rpre} post {rpost}  "
        f"thermal {STATE_NAMES.get(state, 'n/a')}  "
        f"paused {s['paused_seconds']:.0f}s  "
        f"eta ~{eta_min:.0f}min"
        + (f"  ({s['n_excluded']} excluded)" if s["n_excluded"] else ""),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
