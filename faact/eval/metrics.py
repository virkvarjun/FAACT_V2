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

# Re-exported rather than redefined: one definition of success for the whole repo.
from faact.eval.runner import EpisodeResult, success_rate  # noqa: F401

# A "reaction" is a commitment materially shorter than what this episode started with.
#
# Defined relative to the episode's own first horizon rather than against an absolute
# threshold, because an absolute one is degenerate for constant-horizon policies: with a
# 50-step cutoff, fixed h=20 scores as "reacting" at every single step of every episode —
# lead time -106 on 30/30 episodes and a 100% false-alarm rate — purely because 20 < 50.
# That says nothing about the controller and pollutes the ablation table.
#
# Relative framing gives the right answer everywhere: a constant-horizon policy never
# shortens relative to itself, so it registers no reactions and no false alarms, while an
# adaptive controller is measured on whether it actually became more cautious.
REACTION_DROP_FRACTION = 0.5


def _reaction_step(result: "EpisodeResult", drop_fraction: float) -> int | None:
    """Timestep of the first materially-shortened commitment, or None if there was none."""
    if len(result.horizons) < 2:
        return None
    baseline = result.horizons[0]
    for t, h in zip(result.replan_steps[1:], result.horizons[1:]):
        if h < drop_fraction * baseline:
            return t
    return None


def is_adaptive(result: "EpisodeResult") -> bool:
    """Did the horizon vary at all? Constant-horizon rows get '—' rather than a fake number."""
    return len(set(result.horizons)) > 1


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


def lead_time(result: EpisodeResult, drop_fraction: float = REACTION_DROP_FRACTION):
    """Steps from perturbation onset to the controller's first materially-shortened commitment.

    Returns None when the episode was unperturbed, the horizon never varied, or the
    controller never reacted — so "never reacted" stays distinguishable from "reacted at
    step 0". Averaging a sentinel like -1 into the mean would silently flatter a controller
    that mostly ignores onsets.

    A negative value means the horizon shortened *before* onset. That is not prescience,
    it is prior caution, and it is reported as-is rather than clipped.
    """
    if result.perturb is None or not is_adaptive(result):
        return None
    t = _reaction_step(result, drop_fraction)
    return None if t is None else float(t - int(result.perturb["onset_step"]))


def mean_lead_time(results: Iterable[EpisodeResult], drop_fraction: float = REACTION_DROP_FRACTION):
    """Mean lead time over episodes where the controller reacted at all.

    Returns (mean, n_reacted, n_perturbed) so coverage is always reported alongside the
    number — a 2-step mean lead time over 3 of 20 episodes is not a good result.
    """
    results = list(results)
    perturbed = [r for r in results if r.perturb is not None]
    leads = [lt for r in perturbed if (lt := lead_time(r, drop_fraction)) is not None]
    mean = float(np.mean(leads)) if leads else float("nan")
    return mean, len(leads), len(perturbed)


def false_alarm_rate(
    results: Iterable[EpisodeResult], drop_fraction: float = REACTION_DROP_FRACTION
) -> float:
    """Fraction of *successful* episodes in which the controller shortened its horizon.

    On an episode that succeeded, any shortening was unnecessary caution. This is the
    counterweight to lead time: without it, a controller that always shortens would look
    optimal. Constant-horizon policies never shorten relative to themselves, so they score
    0% rather than a meaningless 100%.
    """
    successes = [r for r in results if r.success]
    if not successes:
        return float("nan")
    alarmed = sum(
        is_adaptive(r) and _reaction_step(r, drop_fraction) is not None for r in successes
    )
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
