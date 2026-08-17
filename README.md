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
| M0 | Environment + headless MuJoCo | `00_setup_check.py` prints PASS | ✅ 6/6 PASS (macOS dev box; not yet run on the L40S) |
| M1 | ACT checkpoint + honest baseline | unperturbed success ≥ 60% over 20 eps | ✅ **13/20 = 65%** |
| M2 | Perturbation suite | perturbed success in 25–65% band | ⬜ not started |
| M3 | Reversibility labels via branch rollout | bitwise-deterministic restore; R drops after onset | ⬜ not started |
| M4 | Reversibility head | Spearman(pred, empirical) ≥ 0.5 on held-out episodes | ⬜ not started |
| M5 | Horizon control + ablation | 5-condition table, n stated | ⬜ not started |
| M6 | Figures | — | ⬜ not started |

### M1 baseline, measured

`lerobot/act_aloha_sim_transfer_cube_human` loads directly against lerobot 0.3.2 — **no ACT training was
needed**. Evaluated unperturbed at fixed h=100, seeds 1000–1019:

| n | success | wall clock | device |
|---|---|---|---|
| 20 | **13/20 = 65%** | 86s (4.3 s/ep) | mps (macOS dev box) |

Failures are graded, not degenerate: the 7 failures reached reward 1–2 of 4 (approach and grasp, no
transfer). 65% leaves the headroom that v1's 0/20 did not. Full per-episode records:
`artifacts/baseline_eval.json`. Not yet re-run on the L40S.

---

## What exists right now

```
faact/
  backbone/act_wrapper.py   # ACT chunk prediction + forward-hook feature extraction
  envs/make.py              # env factory, ALOHA constants
  envs/perturb.py           # 4 perturbation kinds + specs
  envs/state.py             # snapshot / restore / re-observe / cube teleport
  runtime/executor.py       # ChunkExecutor — controllable commitment horizon
  runtime/controller.py     # horizon policies: fixed, score-gated, oracle
  eval/runner.py            # run_episode -> EpisodeResult
scripts/
  00_setup_check.py         # M0 gate
  02_eval_baseline.py       # M1 gate
tests/                      # 49 tests; markers `sim` (MuJoCo) and `policy` (checkpoint)
docs/
  FAACT_v2_plan.md          # three-day build plan, scope calls, and cuts
  CLAUDE_BUILD_PROMPT.md    # milestone spec this repo is built against
CLAUDE.md                   # working rules (measurement honesty, no silent fallbacks)
```

Not yet written: `labeling/`, `models/`, `eval/metrics.py`, `eval/ablation.py`, `viz/`, and scripts
`03`–`07`. The Status table above is authoritative.

## Setup

Development is on CPU (macOS included); training and branch rollouts need the GPU box.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/00_setup_check.py     # must print PASS for every check
```

Two system-level dependencies that the setup check exists to catch early:

| Platform | Needed | Why |
|---|---|---|
| RunPod (Linux) | `apt-get install -y libegl1 libgles2 libglvnd0 ffmpeg libav*-dev`, `export MUJOCO_GL=egl` | headless MuJoCo rendering, and torchcodec's FFmpeg bindings |
| macOS (dev) | `brew install ffmpeg`, `export DYLD_LIBRARY_PATH=/opt/homebrew/lib` | torchcodec's `@rpath` resolves against the Python prefix, not brew's libdir, so the install alone is not enough |

A static FFmpeg CLI binary on `PATH` does **not** satisfy this — torchcodec needs the shared libraries.
lerobot's `pyav` backend is not an alternative: it routes through `torchvision.io.VideoReader`, which current
torchvision no longer ships.

`lerobot` is a **pip dependency, not a git submodule** — v1 pinned a submodule commit that did not exist in
the upstream repo, so a fresh clone could not be set up at all. Resolved versions are in
`requirements.lock.txt`; the load-bearing ones are `lerobot 0.3.2`, `gym-aloha 0.1.3`, `mujoco 3.11.0`,
`gymnasium 0.29.1`, `torch 2.13.0`, `numpy 1.26.4`.

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
