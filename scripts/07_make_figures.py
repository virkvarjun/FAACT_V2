"""M6: build the figures from whatever measured artifacts exist.

Each figure is attempted independently and skipped with a stated reason if its input is
missing, so a partial run produces the figures it can rather than failing wholesale. The
calibration scatter is produced by script 05, since it needs the trained head.

    python scripts/07_make_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faact.viz.figures import (  # noqa: E402
    plot_ablation_bars,
    plot_horizon_trace,
    plot_reversibility_trace,
    point_of_no_return,
)
from faact.utils import ARTIFACTS  # noqa: E402

SHARD_DIR = ARTIFACTS / "reversibility_shards"


def pick_illustrative_episode(shard_dir: Path) -> dict | None:
    """Choose the labelled episode that best shows reversibility doing something.

    Ranked by how far R falls from its pre-onset level to its post-onset minimum. This
    picks the clearest *example*, and the figure caption says so — it is an illustration
    of the mechanism, not evidence about the average case, which is what the ablation
    table and calibration plot are for.
    """
    best, best_drop = None, -np.inf
    for path in sorted(shard_dir.glob("episode_*.json")):
        rec = json.loads(path.read_text())
        if not rec.get("points"):
            continue
        onset = rec["perturb"]["onset_step"]
        pre = [p["reversibility"] for p in rec["points"] if p["timestep"] < onset]
        post = [p["reversibility"] for p in rec["points"] if p["timestep"] >= onset]
        if not pre or not post:
            continue
        drop = float(np.mean(pre) - min(post))
        if drop > best_drop:
            best, best_drop = rec, drop
    return best


def figure_reversibility_trace(out_dir: Path) -> str:
    if not SHARD_DIR.exists():
        return f"skipped: no labels at {SHARD_DIR} (run script 04)"
    rec = pick_illustrative_episode(SHARD_DIR)
    if rec is None:
        return "skipped: no labelled episode has points both before and after its onset"

    t = np.array([p["timestep"] for p in rec["points"]])
    r = np.array([p["reversibility"] for p in rec["points"]])
    ponr = plot_reversibility_trace(
        t, r, onset=rec["perturb"]["onset_step"],
        path=out_dir / "fig1_reversibility_trace.png",
        title=f"Reversibility along a perturbed trajectory ({rec['kind']}, seed {rec['seed']})",
    )
    return (f"fig1_reversibility_trace: seed {rec['seed']} ({rec['kind']}), "
            f"onset {rec['perturb']['onset_step']}, point of no return {ponr}")


def figure_horizon_trace(out_dir: Path, ablation: dict | None) -> str:
    if ablation is None:
        return "skipped: no ablation.json (run script 06)"
    episodes = ablation.get("episodes", {})
    # Find the adaptive condition by BEHAVIOUR rather than by name: whichever condition
    # actually varied its horizon the most. Matching on a name string silently skipped this
    # figure once already, when the condition was renamed from "reversibility_gated" to
    # "gated_oof".
    def spread(name: str) -> int:
        return max((int(np.ptp(e["horizons"])) for e in episodes[name] if e["horizons"]),
                   default=0)

    gated = max(episodes, key=spread) if episodes else None
    if gated is None or spread(gated) == 0:
        return "skipped: no condition in this ablation varied its horizon"

    # Show the episode where the controller varied its horizon the most — a flat trace
    # would illustrate nothing, and picking it silently would overstate the controller.
    best = max(episodes[gated], key=lambda e: np.ptp(e["horizons"]) if e["horizons"] else 0)
    if not best["horizons"] or np.ptp(best["horizons"]) == 0:
        return "skipped: the gated controller never varied its horizon on any episode"

    plot_horizon_trace(
        np.array(best["replan_steps"]),
        np.array(best["horizons"]),
        onset=best["perturb"]["onset_step"],
        path=out_dir / "fig3_horizon_trace.png",
        title=f"Committed horizon under reversibility gating (seed {best['seed']})",
    )
    return (f"fig3_horizon_trace: seed {best['seed']}, horizon range "
            f"{min(best['horizons'])}-{max(best['horizons'])}")


def figure_ablation(out_dir: Path, ablation: dict | None) -> str:
    if ablation is None:
        return "skipped: no ablation.json (run script 06)"
    rows = ablation.get("rows", [])
    if not rows:
        return "skipped: ablation.json has no rows"
    plot_ablation_bars(
        rows, out_dir / "fig4_ablation.png",
        title=f"Ablation on identical perturbed episodes (n={ablation.get('n_episodes_common')})",
    )
    return f"fig4_ablation: {len(rows)} conditions, n={ablation.get('n_episodes_common')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ARTIFACTS / "figures"))
    ap.add_argument("--ablation", default=str(ARTIFACTS / "ablation.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_path = Path(args.ablation)
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None

    results = [
        figure_reversibility_trace(out_dir),
        figure_horizon_trace(out_dir, ablation),
        figure_ablation(out_dir, ablation),
    ]
    calib = ARTIFACTS / "fig_calibration.png"
    results.append(
        f"fig2_calibration: {calib} (from script 05)" if calib.exists()
        else "skipped: no fig_calibration.png (run script 05)"
    )

    print("MILESTONE: M6")
    for line in results:
        print(f"  {line}")
    made = sum(not r.startswith("skipped") for r in results)
    print(f"\nMEASURED: {made}/4 figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
