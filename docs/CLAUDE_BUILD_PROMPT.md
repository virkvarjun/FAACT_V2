# Build Prompt — FAACT v2

> Paste this as the opening message in a fresh Cursor / Claude Code session at the root of the new empty repo.
> Consider also saving §1–§3 as `CLAUDE.md` in the repo so the rules persist across sessions.

---

## 1. Mission

Build **`faact-v2`**, to be pushed to `https://github.com/virkvarjun/FAACT_V2.git` — a research codebase that
tests one claim in simulation —

> **Reversibility, not uncertainty, is the right control variable for chunked imitation policies.**

An Action Chunking Transformer (ACT) commits to ~100 actions at a time and is blind to disturbance while it does.
We estimate, at each state, the probability that the task is still recoverable if we replan now — call this
**reversibility** `R(s)` — and use that single quantity to control how many steps of each chunk we actually commit to.

Hard deadline: **Wednesday 19 Aug 2026.** Single NVIDIA L40S on RunPod. Simulation only, no hardware.
This supports a technical talk and a project site, so measured numbers and clean figures matter more than
feature completeness.

## 2. Non-negotiable rules

1. **Never report a number you did not measure.** No estimated accuracies, no "should be around". If you did not
   run it, say you did not run it.
2. **Run every command you write.** Do not hand me a script you have not executed at least in smoke-test form.
3. **Milestone gates are blocking.** Each milestone below has an acceptance check. If it fails, stop, report
   what you observed, and propose a fix. Do not proceed to the next milestone with a failing gate.
4. **Commit at every green gate**, with the acceptance output pasted into the commit message.
5. **Pure functions get unit tests before implementation.** Anything touching the simulator gets a smoke test.
6. **No silent fallbacks.** If a feature extractor, checkpoint, or env fails to load, raise. v1 of this project
   drowned in placeholder code paths that quietly did nothing; do not repeat that.
7. **Prefer boring and working over clever and unfinished.** Deadline is three days.
8. **Log wall-clock time for every long-running job** and report it. I need to know where the budget goes.

## 3. Background — what v1 did and why it failed

The predecessor repo (`virkvarjun/FAACT`) tried: train ACT, train an MLP to predict `failure_within_k` steps,
and at runtime sample candidate chunks by perturbing observations / MC-dropout and execute the lowest-risk one.

Measured outcome: baseline success **0/20**, monitor-only **0/20**, best intervention config **1/10**, and
`recovery_after_intervention: 0` in every single run.

Root causes, all of which this rebuild must avoid:

- **Perturbations were never implemented.** `--perturbation_mode` was an unused placeholder. So "failures" were
  just a weak policy failing, not recoverable disturbances. Nothing to recover *from*.
- **No headroom.** A 0% baseline cannot be improved upon measurably.
- **Wrong target.** `failure_within_k` labels mean "near the end of an episode that failed" — proximity to
  termination, not recoverability.
- **Useless candidate actions.** Perturbation-sampled chunks from a deterministic ACT do not span the recovery
  manifold. The v1 author's own conclusion: *"the bottleneck is action usefulness, not just action scoring."*
- **Unrunnable repo.** The runtime loop lived in an untracked package; the lerobot submodule pin did not exist.

**Design consequence:** this repo depends on `lerobot` as a normal pip dependency. Do **not** add it as a git
submodule and do **not** fork it. Extract ACT internal features with PyTorch forward hooks (see M1).

## 4. Target structure

```
faact/
  __init__.py
  backbone/
    act_wrapper.py        # load ACT, predict_chunk(obs) -> (chunk, features), feature hooks
  envs/
    make.py               # env factory, headless config
    perturb.py            # PerturbationSpec + appliers
    state.py              # MuJoCo snapshot / restore
  runtime/
    executor.py           # ChunkExecutor: commitment horizon, temporal ensembling
    controller.py         # horizon policies: fixed, risk-gated, reversibility-gated, oracle
  labeling/
    branch_rollout.py     # reversibility label generation
    dataset.py            # assemble (features, R) training set, episode-level splits
  models/
    reversibility.py      # MLP head regressing R in [0,1]
  eval/
    runner.py             # run_episode -> EpisodeResult
    metrics.py
    ablation.py
  viz/
    figures.py
scripts/
  00_setup_check.py
  01_get_or_train_act.py
  02_eval_baseline.py
  03_collect_perturbed.py
  04_label_reversibility.py
  05_train_reversibility.py
  06_run_ablation.py
  07_make_figures.py
tests/
configs/
CLAUDE.md
README.md
pyproject.toml
```

---

## 5. Milestones

### M-init — Repository

Initialise against the empty remote `https://github.com/virkvarjun/FAACT_V2.git`:

```bash
git init && git branch -M main
git remote add origin https://github.com/virkvarjun/FAACT_V2.git
```

Create `pyproject.toml` (Python 3.10, package name `faact`), a `.gitignore` covering
`artifacts/ outputs/ checkpoints/ *.pt *.npz *.mp4 __pycache__/ .venv/`, and a `README.md` that is **accurate
from the first commit** — describe only what exists. The predecessor repo's README described a directory layout
and four scripts that were never written; that is the single most damaging thing in it. If a section describes
future work, put it under a heading that says so.

**No git submodules.** `lerobot` is a pinned pip dependency. Record the exact resolved version in
`pyproject.toml` and in the README.

Push the initial commit before starting M0, so the remote is live and every milestone gate can be pushed as it
goes green.

### M0 — Environment

Set up Python 3.10, PyTorch with CUDA, `lerobot`, `gym-aloha`, `mujoco`, `gymnasium`.

**Known blockers from the predecessor project — handle these preemptively, do not rediscover them:**

| Blocker | Fix |
|---|---|
| Headless MuJoCo has no display | `MUJOCO_GL=egl` plus EGL system libs (`libegl1`, `libgles2`, `libglvnd0`) |
| `torchcodec` cannot decode dataset videos | Install FFmpeg **shared libraries**, not just the CLI binary |
| `gym-aloha` uses the Gym v0.26 API | `pip install "shimmy[gym-v26]"` as the Gymnasium bridge |
| `lerobot` `import_utils.py` `UnicodeDecodeError` on package metadata | Patch or pin around it; the predecessor repo needed a monkey-patch |
| RunPod SSH drops long jobs | Everything long-running goes under `tmux` or `nohup`. A previous experiment lost its metrics to a dropped session. |

**Acceptance (`scripts/00_setup_check.py`, must print PASS for all):**
- imports `torch` (CUDA available, prints device name), `lerobot`, `gym_aloha`, `mujoco`
- creates `gym_aloha/AlohaTransferCube-v0` with `obs_type="pixels_agent_pos"`, steps it 10 times
- renders one RGB frame headless and writes it to `artifacts/setup_frame.png` (non-black, assert `std > 1.0`)
- loads one episode from `lerobot/aloha_sim_transfer_cube_human` and decodes 5 frames

### M1 — ACT policy and honest baseline

**Do this in the first hour: ACT training is the long pole. Start it before writing anything else.**

`scripts/01_get_or_train_act.py`, in this order:

1. **Try the HuggingFace checkpoint first.** `lerobot/act_aloha_sim_transfer_cube_human` previously failed only
   because it lacks `policy_preprocessor.json` from an older LeRobot version. Attempt to synthesise that config
   against the current schema. If it loads and evaluates sanely, you have saved 6 hours — but **verify by
   evaluation, not by successful loading.**
2. **Otherwise train.** `lerobot-train`, dataset `lerobot/aloha_sim_transfer_cube_human`, policy `act`,
   batch 8, checkpoint every 10K steps, up to 100K. Run under tmux.
   - After 500 steps, report measured `it/s` and the extrapolated total wall clock. If throughput looks
     dataloader-bound, raise `num_workers` before anything else.
   - Evaluate each checkpoint as it lands. **Stop as soon as a checkpoint clears the M1 gate** — do not burn
     hours finishing 100K if 50K suffices.

`faact/backbone/act_wrapper.py`:
- `predict_chunk(obs) -> (chunk[k,14], features: dict[str, np.ndarray])`
- Features via **forward hooks** on the ACT transformer encoder output (pooled) and decoder output. Also expose
  cheap action-space features: `action_first`, `action_prefix_mean_10`, `action_prefix_flat_10`.
- Hook registration must **assert it fired**. If a hook produces nothing, raise — do not return zeros.

**Acceptance gate (blocking):** unperturbed ACT success **≥ 60% over 20 episodes**, fixed horizon h=100,
fixed seed. Report the exact rate and the checkpoint step. If below 60%, stop and report; everything downstream
is meaningless without headroom.

### M2 — Perturbation suite

`faact/envs/perturb.py`. `PerturbationSpec(kind, onset_step, magnitude, duration)`, four kinds:

| Kind | Implementation |
|---|---|
| `object_displace` | At `onset_step`, teleport the cube: write the free-joint qpos through the dm_control physics object, then `physics.forward()`. Magnitude δ ∈ [2,5] cm, random planar direction. |
| `grasp_slip` | Force the gripper channel fully open for `duration` (default 3) steps. |
| `actuation_noise` | Additive Gaussian on actions over a window of 20 steps. |
| `occlusion` | Zero a rectangular image patch in the camera observation over a window. |

Onset is known by construction — persist it in every episode record, it is the ground truth for lead-time.

**Acceptance:** for each kind, 20 episodes, report success rate. Tune magnitudes so perturbed success lands in
the **25–65%** band. Outside that band there is either no effect or no headroom — retune and re-report.
Write `artifacts/perturbation_calibration.json`.

### M3 — Reversibility labelling

`faact/envs/state.py`: `snapshot(env)` / `restore(env, state)` via dm_control
`physics.get_state()` / `physics.set_state()` + `physics.forward()`.

`faact/labeling/branch_rollout.py`:

```
for episode in perturbed_episodes:          # E = 40
    for t in range(0, T, S):                # S = 25
        s = snapshot(env)
        wins = 0
        for m in range(M):                  # M = 8
            restore(env, s)
            wins += rollout_to_completion(policy, fresh_chunk=True, seed=m)
        R[t] = wins / M
```

**Critical for budget:** run branch rollouts **open-loop at h=100**, so this costs ~13K policy forward passes
rather than ~1.3M. MuJoCo becomes the bottleneck, not the GPU. Parallelise across 8 worker processes.
Expected 1–2 hours; if the measured projection exceeds 3 hours, drop to `S=50, M=5` and say so.

**Acceptance:**
- Determinism test: `restore` twice from the same snapshot with the same seed produces **bitwise-identical**
  action sequences. This is a unit test and it must pass — silent state leakage would invalidate every label.
- Sanity: mean `R` is significantly lower after perturbation onset than before. Report both means.
- Writes `artifacts/reversibility_labels.npz` with `features`, `R`, `episode_id`, `timestep`, `perturb_spec`.

### M4 — Reversibility head

`faact/models/reversibility.py`: MLP `input_dim -> 256 -> 128 -> 1`, ReLU, dropout, sigmoid output.
Soft-target BCE against `R ∈ [0,1]`. **Episode-level** train/val/test splits — never split within an episode.

**Acceptance:** Spearman correlation between predicted and empirical `R` on held-out **episodes** ≥ 0.5.
Report Spearman, MAE, and a calibration scatter at `artifacts/fig_calibration.png`. If correlation is below 0.5,
report it honestly and try richer features before changing the target.

### M5 — Horizon control and ablation

`faact/runtime/executor.py` — `ChunkExecutor` executes `h` steps of a chunk then replans, with optional
temporal ensembling (exponential weights over overlapping chunks).

`faact/runtime/controller.py` — horizon policies:
- `fixed(h)`
- `reversibility_gated`: `h = clip(5, 100, round(100 * R_hat**gamma))`, `gamma ∈ {1, 2}`
- `risk_gated`: same map driven by a failure-risk predictor instead of R (the v1-style comparison)
- `oracle`: uses labelled `R` — the ceiling

`scripts/06_run_ablation.py`, all conditions on the **same** perturbed episode set and seed, n=20:

| Condition | Isolates |
|---|---|
| fixed h=100 | ACT as published |
| fixed h=20 | **Is any gain just replanning more often?** The ablation that matters most. |
| risk_gated | Does reversibility beat plain risk? |
| reversibility_gated | The claim |
| oracle | Ceiling / estimation loss |

Metrics: success rate, interventions per episode, mean committed horizon, lead time vs. known onset,
false-alarm rate on successful episodes. Emit `artifacts/ablation.json` and a markdown table.

**Acceptance:** table produced with all five conditions and n stated. **A negative result is an acceptable
outcome** — report it plainly with a diagnosis. Do not tune until the numbers look good and then present the
tuned run as if it were the first.

### M6 — Figures

`scripts/07_make_figures.py`, SVG output:
1. Reversibility over a trajectory with perturbation onset and point-of-no-return marked
2. Predicted vs. empirical R calibration scatter
3. Committed horizon over time overlaid on onset
4. Ablation bar chart
5. Side-by-side videos, same seed and perturbation, fixed-h vs. reversibility-gated

---

## 6. Testing

- `pytest tests/` green at every gate.
- Required unit tests: snapshot/restore determinism; perturbation appliers actually change env state; horizon
  map monotonicity and clipping; episode-level splits leak no episode across sets; metrics functions on
  hand-checked toy inputs.
- Required smoke tests: 2-episode end-to-end run of each script with a tiny config, in CI-able form.

## 7. Reporting

At each milestone, give me exactly this:

```
MILESTONE: <id>
GATE: PASS | FAIL
MEASURED: <the actual numbers, with n and seed>
WALL CLOCK: <duration of any long job>
FILES: <what was created or changed>
NEXT: <what you are about to do>
BLOCKERS: <anything you could not resolve>
```

No prose summaries of code you wrote — I can read the diff. Report measurements and blockers.

## 8. Stretch, only if M0–M6 are green and time remains

**LoRA recovery adapter store.** Because we inject the perturbations, the failure-mode label is free — no MLLM
failure analysis needed, which is what makes this cheap relative to prior work.

1. Recovery-data generator: roll out ACT, inject perturbation at `t`, hand control to ALOHA's **scripted expert**
   (`PickAndTransferPolicy`, privileged state is fine for data generation), keep the continuation iff it succeeds,
   tag it with the perturbation kind.
2. Train one LoRA (rank 8–16, on attention q/k/v + FFN) per perturbation kind on the frozen ACT.
3. Runtime: detector fires → route on perturbation-kind classifier → swap adapter → execute → swap back.
4. **Required ablation or it is not a result:** K separate adapters vs. one adapter trained jointly on all
   recovery data vs. base policy. At K=3 the joint adapter may well match the store; we need to know.

Realistic partial credit: the data generator plus ~50 labelled recovery trajectories, and one trained adapter.

## 9. Start here

1. Kick off M1 ACT acquisition (HF checkpoint attempt, then training under tmux) **before writing other code.**
2. While the GPU works, build M0 setup checks, then `envs/perturb.py`, `envs/state.py`, and
   `labeling/branch_rollout.py` with their unit tests — none of these need a trained policy.
3. Report the M1 throughput measurement as soon as you have 500 steps.
