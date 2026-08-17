# CLAUDE.md — working rules for this repo

## Mission

One claim, tested in simulation:

> **Reversibility, not uncertainty, is the right control variable for chunked imitation policies.**

An Action Chunking Transformer (ACT) commits to ~100 actions at a time and is blind to disturbance while it
does. We estimate, at each state, the probability that the task is still recoverable if we replan now —
**reversibility** `R(s)` — and use that single quantity to set how many steps of each chunk we commit to.

Deadline: **Wed 19 Aug 2026**. Single L40S on RunPod. Simulation only. This supports a talk and a project
site, so measured numbers and clean figures matter more than feature completeness.

## Non-negotiable rules

1. **Never report a number you did not measure.** No estimates, no "should be around". If it wasn't run, say so.
2. **Run every command you write.** No un-executed scripts.
3. **Milestone gates are blocking.** On failure: stop, report what was observed, propose a fix.
4. **Commit at every green gate**, with the acceptance output in the commit message.
5. **Pure functions get unit tests before implementation.** Anything touching the simulator gets a smoke test.
6. **No silent fallbacks.** If a feature extractor, checkpoint, or env fails to load, *raise*. v1 drowned in
   placeholder code paths that quietly did nothing.
7. **Boring and working beats clever and unfinished.**
8. **Log wall-clock time for every long-running job** and report it.

## What v1 did, and why it failed

v1 (`virkvarjun/FAACT`) trained ACT, trained an MLP to predict `failure_within_k`, and at runtime sampled
candidate chunks by perturbing observations / MC-dropout and executed the lowest-risk one.

Measured: baseline **0/20**, monitor-only **0/20**, best intervention config **1/10**,
`recovery_after_intervention: 0` in every run.

Root causes this rebuild must avoid:

- **Perturbations were never implemented.** `--perturbation_mode` was an unused placeholder, so "failures"
  were just a weak policy failing — nothing to recover *from*.
- **No headroom.** A 0% baseline cannot be measurably improved.
- **Wrong target.** `failure_within_k` means "near the end of an episode that failed" — proximity to
  termination, not recoverability.
- **Useless candidate actions.** Perturbation-sampled chunks from a deterministic ACT do not span the
  recovery manifold.
- **Unrunnable repo.** The runtime loop lived in an untracked package; the lerobot submodule pin didn't exist.

**Design consequence:** `lerobot` is a normal pip dependency. Do **not** add it as a submodule, do **not**
fork it. ACT internal features come from PyTorch forward hooks.

## Repo conventions

- Package code in `faact/`, entry points in `scripts/NN_name.py`, tests in `tests/`.
- Anything written at runtime goes to `artifacts/` (gitignored). Nothing in `artifacts/` is an input.
- Long jobs run under `tmux`/`nohup` on RunPod — a dropped SSH session already cost v1 a run's metrics.
- Development happens on macOS (CPU, MuJoCo works); training and branch rollouts happen on the RunPod L40S.
  Keep code device-agnostic: resolve devices through `faact.utils.device`, never hardcode `cuda`.

## Reporting format

At each milestone, report exactly:

```
MILESTONE: <id>
GATE: PASS | FAIL
MEASURED: <actual numbers, with n and seed>
WALL CLOCK: <duration of any long job>
FILES: <created or changed>
NEXT: <what happens next>
BLOCKERS: <unresolved>
```
