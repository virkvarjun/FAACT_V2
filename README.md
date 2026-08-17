# FAACT v2 — reversibility-gated execution horizons

Testing one claim in simulation:

> **Reversibility, not uncertainty, is the right control variable for chunked imitation policies.**

An [ACT](https://arxiv.org/abs/2304.13705) policy predicts ~100 actions at a time and executes them open-loop,
so it is blind to disturbance for the whole committed horizon. We estimate **reversibility** `R(s)` — the
probability the task still succeeds if we replan from `s` right now — and use that single scalar to set the
commitment horizon: `h = clip(5, 100, round(100 · R̂^γ))`.

Task: `gym_aloha/AlohaTransferCube-v0`, ALOHA bimanual cube transfer, simulation only.

---

## Status

This table is the source of truth for what has actually been run. Nothing is claimed here that was not
measured; empty cells mean not yet run.

| Milestone | What it delivers | Gate | Status |
|---|---|---|---|
| M-init | Repo, packaging, docs | — | ✅ done |
| M0 | Environment + headless MuJoCo | `00_setup_check.py` prints PASS | ⬜ not started |
| M1 | ACT checkpoint + honest baseline | unperturbed success ≥ 60% over 20 eps | ⬜ not started |
| M2 | Perturbation suite | perturbed success in 25–65% band | ⬜ not started |
| M3 | Reversibility labels via branch rollout | bitwise-deterministic restore; R drops after onset | ⬜ not started |
| M4 | Reversibility head | Spearman(pred, empirical) ≥ 0.5 on held-out episodes | ⬜ not started |
| M5 | Horizon control + ablation | 5-condition table, n stated | ⬜ not started |
| M6 | Figures | — | ⬜ not started |

**No results have been measured yet.** When they are, numbers land in `artifacts/` and are summarised here.

---

## What exists right now

```
faact/                 # (being built — see Status)
scripts/               # numbered entry points, one per milestone
tests/                 # pytest; sim-dependent tests are marked `sim`
docs/
  FAACT_v2_plan.md     # three-day build plan, scope calls, and cuts
  CLAUDE_BUILD_PROMPT.md  # milestone spec this repo is built against
CLAUDE.md              # working rules (measurement honesty, no silent fallbacks)
```

## Setup

Development is on CPU (macOS included); training and branch rollouts need the GPU box.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/00_setup_check.py     # must print PASS for every check
```

Headless boxes (RunPod) need EGL for MuJoCo rendering:

```bash
export MUJOCO_GL=egl        # apt-get install -y libegl1 libgles2 libglvnd0
```

`lerobot` is a **pip dependency, not a git submodule** — v1 pinned a submodule commit that did not exist in
the upstream repo, so a fresh clone could not be set up at all. Exact resolved versions are recorded in
`requirements.lock.txt` once the environment is first built.

## Relation to prior work

- **vs. DEHP** — same lever (execution horizon), different objective: reversibility rather than task success.
  No RL, and labels come free from the simulator.
- **vs. FIPER** — FIPER predicts *whether* an episode will fail; reversibility predicts *whether intervening
  still helps*.
- **vs. FLARE** — FLARE recovers with MLLM-routed reset skills, with no notion of a point of no return.
- **vs. OCR** — OCR is a recovery *mechanism* with no trigger; reversibility is a trigger with no mechanism.
  They compose.

Adjacent and deliberately acknowledged: reversibility-aware RL (Grinsztajn et al.) and the "leave no trace"
safe-RL lineage. Neither applies reversibility to action-chunk horizon control in imitation learning.

## Predecessor

[`virkvarjun/FAACT`](https://github.com/virkvarjun/FAACT) (v1) tried risk-scored candidate-chunk selection and
measured baseline 0/20, best intervention 1/10, and zero recoveries. The diagnosis — perturbations were never
implemented, so there was no headroom and nothing to recover from — is what this repo is built around.
See `CLAUDE.md` for the full post-mortem.
