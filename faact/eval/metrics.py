"""Ablation metrics.

Success rate alone cannot distinguish "reversibility control works" from "replanning more
often works". These are the quantities that separate them:

    success_rate            did the task get done
    interventions_per_ep    how often the controller replanned — the cost side
    mean_committed_horizon  how long it dared commit — the behaviour being controlled
    lead_time               how soon after a *known* disturbance onset it reacted
    false_alarm_rate        how often it panicked on episodes that were fine anyway

`lead_time` is only measurable because we choose the onset step ourselves, so it is exact
ground truth rather than an inferred change point. `false_alarm_rate` is the honesty check:
a controller that always commits h=5 would score perfectly on lead time and be useless.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from faact.eval.runner import EpisodeResult

# A replan counts as a "reaction" when it commits noticeably less than the full chunk.
# 50 of a 100-step chunk is the midpoint of the horizon range, so it separates "committed
# long, business as usual" from "shortened up, something looks wrong".
REACTION_HORIZON_THRESHOLD = 50


def success_rate(results: Iterable[EpisodeResult]) -> float:
    results = list(results)
    if not results:
        raise ValueError("success_rate of zero episodes is undefined")
    return sum(r.success for r in results) / len(results)


def interventions_per_episode(results: Iterable[EpisodeResult]) -> float:
    """Mean number of replans per episode — the cost of reacting."""
    results = list(results)
    if not results:
        raise ValueError("undefined for zero episodes")
    return float(np.mean([len(r.replan_steps) for r in results]))


def mean_committed_horizon(results: Iterable[EpisodeResult]) -> float:
    """Mean horizon committed across all replans of all episodes.

    Pooled over replans rather than averaged per episode, so an episode with many short
    commitments is not weighted equal to one with a single long commitment.
    """
    horizons = [h for r in results for h in r.horizons]
    if not horizons:
        raise ValueError("no replans recorded")
    return float(np.mean(horizons))


def lead_time(result: EpisodeResult, threshold: int = REACTION_HORIZON_THRESHOLD) -> float | None:
    """Steps from perturbation onset to the controller's first shortened commitment.

    Returns None when the episode was unperturbed or the controller never reacted, so
    "never reacted" stays distinguishable from "reacted at step 0" — averaging a sentinel
    like -1 into the mean would silently flatter a controller that mostly ignores onsets.

    A negative value means the horizon shortened *before* onset, which is not prescience:
    it means the controller was already cautious. Reported as-is rather than clipped.
    """
    if result.perturb is None:
        return None
    onset = int(result.perturb["onset_step"])
    for t, h in zip(result.replan_steps, result.horizons):
        if h < threshold:
            return float(t - onset)
    return None


def mean_lead_time(results: Iterable[EpisodeResult], threshold: int = REACTION_HORIZON_THRESHOLD):
    """Mean lead time over episodes where the controller reacted at all.

    Returns (mean, n_reacted, n_perturbed) so the coverage is always reported alongside
    the number — a 2-step mean lead time over 3 of 20 episodes is not a good result.
    """
    results = list(results)
    perturbed = [r for r in results if r.perturb is not None]
    leads = [lt for r in perturbed if (lt := lead_time(r, threshold)) is not None]
    mean = float(np.mean(leads)) if leads else float("nan")
    return mean, len(leads), len(perturbed)


def false_alarm_rate(
    results: Iterable[EpisodeResult], threshold: int = REACTION_HORIZON_THRESHOLD
) -> float:
    """Fraction of *successful* episodes in which the controller shortened its horizon.

    On an episode that succeeded, any shortening was unnecessary caution. This is the
    counterweight to lead time: without it, "always commit h=5" looks optimal.
    """
    successes = [r for r in results if r.success]
    if not successes:
        return float("nan")
    alarmed = sum(any(h < threshold for h in r.horizons) for r in successes)
    return alarmed / len(successes)


def summarise(results: Iterable[EpisodeResult], label: str = "") -> dict:
    """All metrics for one ablation condition, as a JSON-safe row."""
    results = list(results)
    mean_lead, n_reacted, n_perturbed = mean_lead_time(results)
    return {
        "condition": label,
        "n_episodes": len(results),
        "success_rate": success_rate(results),
        "n_success": sum(r.success for r in results),
        "interventions_per_episode": interventions_per_episode(results),
        "mean_committed_horizon": mean_committed_horizon(results),
        "mean_lead_time": mean_lead,
        "n_reacted": n_reacted,
        "n_perturbed": n_perturbed,
        "false_alarm_rate": false_alarm_rate(results),
        "mean_max_reward": float(np.mean([r.max_reward for r in results])),
    }


def markdown_table(rows: list[dict]) -> str:
    """Render summarise() rows as the ablation table that goes in the README."""
    header = (
        "| Condition | n | Success | Interventions/ep | Mean horizon | "
        "Lead time (n reacted) | False alarm |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        lead = "—" if np.isnan(r["mean_lead_time"]) else f"{r['mean_lead_time']:+.0f}"
        fa = "—" if np.isnan(r["false_alarm_rate"]) else f"{r['false_alarm_rate']:.0%}"
        lines.append(
            f"| {r['condition']} | {r['n_episodes']} | "
            f"{r['n_success']}/{r['n_episodes']} = {r['success_rate']:.0%} | "
            f"{r['interventions_per_episode']:.1f} | {r['mean_committed_horizon']:.0f} | "
            f"{lead} ({r['n_reacted']}/{r['n_perturbed']}) | {fa} |"
        )
    return "\n".join(lines)
