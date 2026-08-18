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
| M1 | ACT checkpoint + honest baseline | reproduce LeRobot's official per-seed results | ✅ **48/50 = 96% agreement**, rates match at 76.0% |
| M2 | Perturbation suite | perturbed success in 25–65% band | ⚠️ **in band (50%), but by post-hoc check, not a sweep** |
| M3 | Reversibility labels via branch rollout | bitwise-deterministic restore; R drops after onset | ✅ **0.83 → 0.57**, 493 states |
| M4 | Reversibility head | Spearman(pred, empirical) ≥ 0.5 on held-out episodes | ❌ **0.479** (5-fold pooled) |
| M5 | Horizon control + ablation | table with n stated | ✅ table produced — ❌ **no condition beats fixed h=20; all p ≥ 0.21** |
| M6 | Figures | — | 🟡 fig 1 done, rest pending run 2 |

### M1 baseline, verified against LeRobot's official evaluation

`lerobot/act_aloha_sim_transfer_cube_human` loads directly against lerobot 0.3.2 — **no ACT training was
needed**.

We do not check the policy against a success-rate threshold, because a rate alone cannot tell a correct
inference path from a subtly broken one. The checkpoint repo ships `eval_info.json`: 500 per-episode records
from the official run, each tagged with its seed, starting at seed 1000. Our runner uses the same seeds, so
the comparison is episode by episode (`scripts/01_verify_act_checkpoint.py`).

| n | ours | official, same seeds | per-seed agreement |
|---|---|---|---|
| 20 | 13/20 = 65.0% | 13/20 = 65.0% | **20/20 = 100%** |
| 50 | 38/50 = 76.0% | 38/50 = 76.0% | **48/50 = 96%** |

The two disagreements at n=50 (seeds 1026, 1043) fall one in each direction and cancel. Independently, our
chunk prediction is **bitwise identical** to lerobot's own `select_action` path on the same observation.

**Why 65% is not a shortfall.** The official run reports 83% — over 500 episodes. On its own first 20 seeds
it scores 65.0%, and on its first 50 it scores 76.0%; those seeds are simply a harder-than-average draw:

| seed window | official success |
|---|---|
| 1000–1019 (n=20) | 65.0% |
| 1000–1049 (n=50) | 76.0% |
| 1000–1099 (n=100) | 81.0% |
| 1000–1499 (n=500) | 83.0% |

So a 20-episode run judged against "80–90%" would look like a broken checkpoint when it is exact. **Any
success rate quoted in this repo states its seed window and n.**

Device note: on CPU we reproduce the official per-seed outcomes at n=50 exactly (38/50); MPS differs on a
single episode (39/50) from float noise. Records: `artifacts/act_verification.json`, `artifacts/baseline_eval.json`.
Not yet re-run on the L40S.

### M2 — perturbations, honest status

The magnitude sweep in `scripts/03_calibrate_perturbations.py` was **not run as a full gate**.
The suite was used at its default magnitudes, and the band was verified *post hoc*: under
perturbation, ACT at fixed h=100 scores **15/30 = 50%** (M5 run 1), which is inside the
25–65% target band and leaves headroom in both directions. A proper per-kind sweep would
still be worth running; the duration sweep for `grasp_slip` is verified working
(dur 2/3/6 → 67%, dur 12 → 33% on a 3-episode smoke test).

### M3 — reversibility labels, measured

```
493 labelled states over 40 episodes (0 excluded)
mean R before onset   0.829  (n=140)
mean R after  onset   0.565  (n=353)
fraction of R exactly 0 or 1: 0.83
```

Gate PASS. ~1h45 wall on 3 performance cores of an M5 MacBook, **zero thermal pauses**
(never left macOS "fair"). The run was killed once at 33/40 by a task timeout and resumed
from per-episode shards with no recomputation.

### M4 — reversibility head, measured (gate FAILS)

5-fold **episode-level** cross-validation, pooled out-of-fold predictions over all 493 states:

| feature set | dim | pooled Spearman | per-fold | MAE (baseline 0.415) |
|---|---|---|---|---|
| cheap | 182 | 0.331 | 0.355 ± 0.119 | 0.329 |
| transformer | 1024 | 0.447 | 0.462 ± 0.161 | 0.282 |
| **minimal** | **28** | **0.479** | 0.483 ± 0.138 | **0.281** |
| all | 1206 | 0.390 | 0.423 ± 0.123 | 0.306 |

**Gate FAILS: 0.479 against a 0.50 threshold.** Marginal — 0.50 is inside one per-fold
standard deviation — and MAE beats predict-the-mean by 32%, so there is real signal.

Two findings worth more than the gate:

- **Label noise is not the bottleneck.** Test–retest reliability of the labels is **0.977**
  (treating measured R as the true probability and drawing two independent Binomial(M=8)
  re-measurements), because 83% of R values saturate at 0 or 1. So 0.479 is 49% of the
  achievable ceiling and raising M to 16 or 32 would buy nothing (ceiling 0.992 / 0.999).
  The limit is features and training-set size, not measurement.
- **28 dimensions beat 1024.** Arm state plus the first action predicts reversibility better
  than ACT's internal transformer representation, which overfits at ~300 training states
  per fold. Reversibility looks predictable from where the arm is and what it is about to
  do — no policy internals required.

A single 24/8/8 split was tried first and abandoned as untrustworthy: it selected
`transformer` on validation at 0.509, which then scored 0.265 on test — a 0.24 swing on the
same model, driven by an 8-episode test set.

### M5 — ablation. **Negative result, and the design is underpowered.**

Two runs. Run 1 (n=30) suggested the claim; run 2 (n=60) did not replicate it. Both are reported.

**Run 2 — n=60, seeds 3000–3059, h_min=20** (the run to believe):

| Condition | n | Success | Interventions/ep | Mean horizon | Lead time (n reacted) | False alarm |
|---|---|---|---|---|---|---|
| fixed h=100 | 60 | 28/60 = 47% | 3.6 | 100 | — (0/60) | 0% |
| fixed h=20 | 60 | **35/60 = 58%** | 18.8 | 20 | — (0/60) | 0% |
| reversibility_gated (γ=1) | 60 | 29/60 = 48% | 6.3 | 61 | +123 (25/60) | 14% |
| reversibility_gated (γ=2) | 60 | 32/60 = 53% | 9.3 | 41 | +76 (39/60) | 38% |

**McNemar on paired seeds: every pair p ≥ 0.210.** Nothing is distinguishable from anything,
including h=20 vs h=100. The reversibility-gated controller does **not** beat a fixed short
horizon.

**Run 1 — n=30, h_min=5** (superseded): gated led at 17/30 = 57% against 50% and 53%, with
p=0.688 and p=1.000. That ordering was noise. Doubling n reversed it, which is why run 1 is
not presented as the result.

**Why no conclusion can be drawn either way.** Power analysis (paired McNemar, α=0.05,
power=0.80, discordant rate 0.40 measured from our data), episodes needed *per condition*:

| true gap | episodes needed |
|---|---|
| 5 pp | ~1250 |
| 10 pp | ~308 |
| 15 pp | ~133 |
| 20 pp | ~72 |

At n=60 only gaps of ~20 pp are detectable, and the differences in play are 5–11 pp.
**This is an underpowered design, not evidence of no effect** — and equally, no support for
the claim.

**Also measured:** `fixed h=5` scores **0/30** (run 1). Shortening the horizon is not a free
safety lever: ACT emits absolute joint targets, so replanning every 5 steps without temporal
ensembling makes each new chunk jump to a different target and the motion falls apart.

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
  eval/metrics.py           # success, interventions, horizon, lead time, false alarms
  labeling/branch_rollout.py  # measure R(s) by snapshot-restore-replay
scripts/
  00_setup_check.py         # M0 gate
  01_verify_act_checkpoint.py  # M1 gate — per-seed reproduction of the official eval
  02_eval_baseline.py       # unperturbed success rate at a fixed horizon
  03_calibrate_perturbations.py  # M2 gate — magnitude sweep into the 25-65% band
tests/                      # 68 tests; markers `sim` (MuJoCo) and `policy` (checkpoint)
docs/
  FAACT_v2_plan.md          # three-day build plan, scope calls, and cuts
  CLAUDE_BUILD_PROMPT.md    # milestone spec this repo is built against
CLAUDE.md                   # working rules (measurement honesty, no silent fallbacks)
```

Not yet written: `labeling/dataset.py`, `models/reversibility.py`, `eval/ablation.py`, `viz/`, and scripts
`04`–`07`. The Status table above is authoritative.

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
