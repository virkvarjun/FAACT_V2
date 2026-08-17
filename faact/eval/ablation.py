"""M5 ablation: five horizon policies, one fixed set of perturbed episodes.

Every condition runs the *same* seeds with the *same* perturbation specs, differing only in
the `horizon_fn` handed to the ChunkExecutor. That is the point: if the conditions differed
in their episode sets too, a success-rate gap would be unattributable.

The five conditions and what each one rules out:

    fixed h=100            ACT as published — the thing being improved on
    fixed h=20             replans 5x more often at no intelligence. **The ablation that
                           matters most**: if this matches the gated conditions, the gain
                           was never about reversibility, only about replanning frequency.
    risk_gated             same controller, driven by failure risk instead of R. Isolates
                           the choice of control variable from the controller itself.
    reversibility_gated    the claim
    oracle                 the same controller fed *labelled* R. Its gap to
                           reversibility_gated is the cost of estimation error, which
                           separates "wrong variable" from "weak estimator".

A negative result is a legitimate outcome and is reported as measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from faact.envs.perturb import PerturbationSpec
from faact.eval.metrics import summarise
from faact.eval.runner import PerturbationNeverFired, run_episode
from faact.runtime.executor import ChunkExecutor


@dataclass
class Condition:
    """One row of the ablation: a name and the horizon policy that defines it."""

    name: str
    horizon_fn_factory: Callable[[int], Callable]
    isolates: str = ""


def run_condition(
    env,
    policy,
    condition: Condition,
    episodes: list[tuple[int, PerturbationSpec]],
    max_steps: int,
    on_episode: Callable[[int, object], None] | None = None,
) -> tuple[list, list[dict]]:
    """Run one condition over a fixed list of (seed, spec) pairs.

    Returns (results, excluded). Episodes that end before their onset are excluded rather
    than counted — they were never perturbed, and including them would inflate the rate
    with unperturbed successes. Because the episode list is shared across conditions, an
    exclusion in one condition does not silently change another's denominator, so the
    caller must reconcile them (see `align_conditions`).
    """
    results, excluded = [], []
    for seed, spec in episodes:
        executor = ChunkExecutor(policy, condition.horizon_fn_factory(seed))
        result = None
        try:
            result = run_episode(env, executor, seed=seed, perturb_spec=spec, max_steps=max_steps)
            results.append(result)
        except PerturbationNeverFired as exc:
            excluded.append({"seed": seed, "steps": exc.steps, "onset": spec.onset_step})
        # `result` stays None for an excluded episode — reporting the *previous* episode's
        # result here would misattribute it to this seed.
        if on_episode is not None:
            on_episode(seed, result)
    return results, excluded


def align_conditions(per_condition: dict[str, list]) -> dict[str, list]:
    """Restrict every condition to the seeds that survived in *all* of them.

    Without this, conditions could be scored on different episode subsets and the
    comparison would be between different experiments. Shortening the horizon changes when
    an episode terminates, so exclusions genuinely can differ between conditions.
    """
    if not per_condition:
        return {}
    common = set.intersection(*({r.seed for r in results} for results in per_condition.values()))
    return {
        name: [r for r in results if r.seed in common] for name, results in per_condition.items()
    }


def build_table(per_condition: dict[str, list]) -> list[dict]:
    """One summarise() row per condition, in the order given."""
    return [summarise(results, label=name) for name, results in per_condition.items()]
