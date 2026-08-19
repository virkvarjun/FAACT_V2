"""Figures for the talk and the project site.

All plotting lives here so the scripts stay thin and each figure can be unit-tested for the
thing that actually matters — the quantity it derives — rather than for how it looks.

Matplotlib is configured for the Agg backend at import: these run headless on RunPod, and
a figure that needs a display is a figure that fails at 2am.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Consistent language across every figure: one colour means one thing everywhere.
C_REV = "#2a6fdb"      # reversibility
C_ONSET = "#d1495b"    # perturbation onset
C_PONR = "#8b2e3c"     # point of no return
C_HORIZON = "#3f8f5f"  # committed horizon
GREY = "0.55"


def point_of_no_return(
    timesteps: np.ndarray, reversibility: np.ndarray, threshold: float = 0.5
) -> int | None:
    """First timestep after which reversibility never again rises above `threshold`.

    Defined as a *permanent* crossing, not the first dip: a momentary stumble the policy
    recovers from is exactly what reversibility is meant to distinguish from a true point
    of no return. Returns None when the episode never commits to failure — which is the
    common case for successful episodes and must not be rendered as "step 0".
    """
    t = np.asarray(timesteps)
    r = np.asarray(reversibility, dtype=float)
    if t.shape != r.shape:
        raise ValueError(f"timesteps {t.shape} and reversibility {r.shape} must match")
    if r.size == 0:
        return None

    below = r < threshold
    if not below.any() or not below[-1]:
        return None  # ends above threshold, so it never permanently crossed

    # Walk back from the end while the run of sub-threshold values is unbroken.
    i = len(r) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    return int(t[i])


def plot_reversibility_trace(
    timesteps: np.ndarray,
    reversibility: np.ndarray,
    onset: int,
    path: Path,
    threshold: float = 0.5,
    title: str = "",
) -> int | None:
    """Figure 1: R along one trajectory, with onset and point of no return marked."""
    t = np.asarray(timesteps)
    r = np.asarray(reversibility, dtype=float)
    ponr = point_of_no_return(t, r, threshold)

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(t, r, "o-", color=C_REV, lw=2, ms=4, label="measured $R$ (branch rollout)")
    ax.axhline(threshold, color=GREY, ls=":", lw=1)
    ax.axvline(onset, color=C_ONSET, lw=2, label=f"perturbation onset (t={onset})")
    if ponr is not None:
        ax.axvline(ponr, color=C_PONR, ls="--", lw=2, label=f"point of no return (t={ponr})")
        ax.axvspan(ponr, t.max(), color=C_PONR, alpha=0.07)

    ax.set_xlabel("timestep")
    ax.set_ylabel("reversibility $R$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title or "Reversibility along a perturbed trajectory")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _save(fig, path)
    return ponr


def plot_horizon_trace(
    replan_steps: np.ndarray,
    horizons: np.ndarray,
    onset: int,
    path: Path,
    title: str = "",
) -> None:
    """Figure 3: committed horizon over time, overlaid on the known onset.

    Drawn as a step plot because the horizon is piecewise-constant by construction — the
    controller commits h steps and does not revisit the decision until they are consumed.
    A line plot would imply a continuous change that never happens.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.step(replan_steps, horizons, where="post", color=C_HORIZON, lw=2,
            label="committed horizon $h_t$")
    ax.axvline(onset, color=C_ONSET, lw=2, label=f"perturbation onset (t={onset})")
    ax.set_xlabel("timestep")
    ax.set_ylabel("committed horizon (steps)")
    ax.set_title(title or "Commitment horizon reacting to a disturbance")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    _save(fig, path)


def plot_ablation_bars(rows: list[dict], path: Path, title: str = "") -> None:
    """Figure 4: success rate per condition, with n annotated on every bar.

    n is printed on the bars rather than in the caption because these figures get pasted
    into slides on their own, and a success rate without its n is not a result.
    """
    names = [r["condition"] for r in rows]
    rates = [r["success_rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 0.6 * len(rows) + 1.8))
    bars = ax.barh(names, rates, color=C_REV, alpha=0.85)
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row['n_success']}/{row['n_episodes']} = {row['success_rate']:.0%}",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(1.0, max(rates) * 1.35) if rates else 1.0)
    ax.set_xlabel("success rate under perturbation")
    ax.set_title(title or "Ablation: horizon policies on identical perturbed episodes")
    ax.invert_yaxis()
    _save(fig, path)


def plot_horizon_curve(
    horizons: np.ndarray,
    success: np.ndarray,
    path: Path,
    gated_points: list[tuple[float, float, str]] | None = None,
    title: str = "",
) -> None:
    """Figure 5: success against committed horizon, with adaptive controllers overlaid.

    The project's central figure. If the fixed-horizon points trace a rising curve and every
    adaptive controller lands *on* it rather than above it, then the controller's cleverness
    bought nothing — what determined success was simply how long it committed. Plotting the
    adaptive points in the same axes is what makes that visible rather than asserted.
    """
    h = np.asarray(horizons, dtype=float)
    s = np.asarray(success, dtype=float)
    order = np.argsort(h)
    h, s = h[order], s[order]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(h, s, "o-", color=C_REV, lw=2.2, ms=6, label="fixed horizon", zorder=3)

    if gated_points:
        gh = [p[0] for p in gated_points]
        gs = [p[1] for p in gated_points]
        ax.scatter(gh, gs, s=90, marker="D", color=C_ONSET, zorder=4,
                   label="adaptive controllers (mean horizon)")
        for x, y, label in gated_points:
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6),
                        fontsize=7.5, color=C_ONSET)

    ax.set_xlabel("committed horizon (steps)")
    ax.set_ylabel("success rate under perturbation")
    ax.set_ylim(-0.03, max(0.6, float(s.max()) * 1.25))
    ax.set_title(title or "Success is a function of how long the policy commits")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, color=GREY, alpha=.18, lw=.6)
    _save(fig, path)


def _save(fig, path: Path) -> None:
    """Write PNG for slides and SVG for the site, then release the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
