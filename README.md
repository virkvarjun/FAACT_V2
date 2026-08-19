# FAACT v2 — reversibility-gated execution horizons

The claim under test:

> **Reversibility, not uncertainty, is the right control variable for chunked imitation policies.**

**Measured answer: only within an operating regime, and we found its edge.**

An [ACT](https://arxiv.org/abs/2304.13705) policy predicts ~100 actions at a time and executes them
open-loop, so it is blind to disturbance for the whole committed horizon. We measure **reversibility**
`R(s)` — the probability the task still succeeds if we replan from `s` right now — by branch rollout,
and use it to set the commitment horizon: `h = clip(h_min, 100, round(100 · R̂^γ))`.

Under weak, mistimed disturbances, gating on ground-truth R beats the published fixed horizon
(77% vs 65%, p=0.013). Under correctly-timed calibrated ones it **loses** (39% vs 47%), and success
becomes an almost perfect function of how long the controller commits (ρ=0.94). Shortening the
horizon is a costly intervention for a chunked policy; it only pays where states are at risk but
still recoverable.

Task: `gym_aloha/AlohaTransferCube-v0`, ALOHA bimanual cube transfer, simulation only.

---

## Status

This table is the source of truth for what has actually been run. Every number here was measured;
the gates that failed are marked as failed.

| Milestone | What it delivers | Gate | Status |
|---|---|---|---|
| M-init | Repo, packaging, docs | — | ✅ done |
| M0 | Environment + headless MuJoCo | `00_setup_check.py` prints PASS | ✅ 6/6 PASS — macOS/CPU only, **CUDA path untested** |
| M1 | ACT checkpoint + honest baseline | reproduce LeRobot's official per-seed results | ✅ **48/50 agreement**, rates match at 76.0% |
| M2 | Perturbation suite | in 25–65% band **and** ≥15pt drop | ❌ **2 of 4 kinds effective**; `occlusion` and `grasp_slip` are no-ops here |
| M3 | Reversibility labels via branch rollout | bitwise-deterministic restore; R drops after onset | ✅ **0.78 → 0.24**, 1000 states / 78 eps (plus 990 / 80 in regime A) |
| M4 | Reversibility head | Spearman(pred, empirical) ≥ 0.5 on held-out episodes | ✅ **0.693** (5-fold pooled, 78 eps) |
| M5 | Horizon control + ablation | table with n stated | ❌ **no gain under calibrated disturbances**; all p ≥ 0.061 |
| M6 | Figures + videos | — | ✅ 4 figures, side-by-side videos |

**Headline:** the reversibility estimate was never the bottleneck (0.479 → 0.660 → 0.693 across three
datasets, with control performance unmoved). The binding constraint is the density of
recoverable-but-at-risk states. See [Headline result](#headline-result--the-method-has-an-operating-regime-and-we-measured-its-edge).

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

### M4 — reversibility head. **Gate PASSES at 80 episodes.**

5-fold **episode-level** cross-validation, pooled out-of-fold predictions over every labelled state:

| feature set | dim | n=40 eps (493 states) | **n=80 eps (990 states)** | MAE @80 (baseline 0.414) |
|---|---|---|---|---|
| cheap | 182 | 0.331 | 0.601 | 0.224 |
| **transformer** | 1024 | 0.447 | **0.660** | **0.199** |
| minimal | 28 | 0.479 | 0.629 | 0.209 |
| all | 1206 | 0.390 | 0.642 | 0.195 |

**Pooled Spearman 0.660** (per-fold 0.670 ± 0.076), against a 0.50 gate. MAE 0.199 vs
predict-the-mean 0.414 — a 52% reduction.

The path there is the useful part:

- At 40 episodes the gate **failed at 0.479**, and the 1024-d transformer features were the
  *worst* performer (0.447) while 28-d `minimal` was best. That inverted at 80 episodes.
  It was a sample-size effect: ~300 training states per fold cannot support 1024 dimensions,
  ~600 can.
- **Label noise was correctly ruled out as the cause.** Test–retest reliability of the labels
  is 0.977 (treat measured R as the true probability, draw two independent Binomial(M=8)
  re-measurements), because 83% of R values saturate at 0 or 1. So more branches would have
  bought nothing — the ceiling was already 0.977 — and the bottleneck was training-set size.
  Doubling episodes moved 0.479 → 0.660; raising M would not have.
- A single 24/8/8 split was tried first and abandoned as untrustworthy: it selected
  `transformer` on validation at 0.509, which then scored 0.265 on test — a 0.24 swing on the
  same model from an 8-episode test set.

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

### Headline result — the method has an operating regime, and we measured its edge

Two ablations, on two disturbance regimes, reach opposite conclusions. Both are reported,
because the contrast between them *is* the finding.

**Regime A — 39% of post-onset states already unrecoverable** (window 40–160, 4 kinds, n=79):

| Condition | Success | Mean horizon |
|---|---|---|
| fixed h=100 | 51/79 = 65% | 100 |
| fixed h=20 | 48/79 = 61% | 20 |
| gated_oof (γ=2) | 52/79 = 66% | 49 |
| **oracle_R (γ=2)** | **61/79 = 77%** | 47 |

Oracle beats both baselines (p=0.013 vs h=100, p=0.019 vs h=20). Success is *unrelated* to
mean horizon here: **ρ = −0.03**.

**Regime B — 75% of post-onset states already unrecoverable** (window 150–280, 2 effective
kinds, calibrated magnitudes, n=76):

| Condition | Success | Mean horizon |
|---|---|---|
| **fixed h=100** | **36/76 = 47%** | 100 |
| gated_oof (γ=1) | 35/76 = 46% | 50 |
| oracle_R (γ=1) | 30/76 = 39% | 43 |
| oracle_R (γ=2) | 30/76 = 39% | 41 |
| gated_oof (γ=2) | 30/76 = 39% | 41 |
| fixed h=20 | 25/76 = 33% | 20 |

The oracle now *loses* to the published fixed horizon, and nothing is significant (all
p ≥ 0.061). Success has become an almost perfect function of how long the controller
commits: **ρ = +0.94**, with `fixed h=5` at 0/30 completing the curve.

**The interpretation.** Shortening the commitment horizon is a costly intervention for a
chunked policy — ACT emits absolute joint targets, so frequent replanning degrades the
motion. That cost is only worth paying when there is a population of states that are *at
risk but still recoverable*. In regime A such states are common and gating pays. In regime B
three quarters of disturbed states are already lost, the intervention has nothing to act on,
and only its cost remains.

So the honest claim is neither "reversibility works" nor "it doesn't":

> Reversibility-gated horizon control pays off only where recoverable-but-at-risk states are
> common. Its binding constraint is the density of those states, not the quality of the
> reversibility estimate.

**A methodological warning that follows.** Both regimes sit inside the 25–65% perturbed-success
band that the plan specifies for calibration. They differ enormously in the quantity that
actually matters. **Calibrating disturbances by episode success rate does not guarantee a
useful reversibility distribution** — the band should be set on the fraction of recoverable
states, not on the success rate.

### Corrected disturbances: same accuracy from half the data

The results above use a disturbance window of steps 40–160. Measuring the task's own phase
timings showed first cube contact at median step **154** and transfer completion at **278** —
so those disturbances fired almost entirely *before the cube was grasped*. Two of the four
kinds did essentially nothing (`grasp_slip` opened a gripper holding nothing; `occlusion`
barely matters to a policy running on proprioception). The window is now 150–280, magnitudes
are calibrated, and only the two effective kinds are used.

Re-labelling under that regime (in progress; 42 of 80 episodes at time of writing):

| dataset | episodes | states | pooled Spearman | MAE vs baseline |
|---|---|---|---|---|
| v1 — window 40–160, 4 kinds | 80 | 990 | 0.660 | 0.199 / 0.414 |
| **v2 — window 150–280, 2 kinds** | **42** | **528** | **0.667** | **0.181 / 0.432** |

**The corrected disturbances reach the same predictive quality from half the episodes.**
Notably this is *not* because the labels became less saturated — v1 was 84% of R values at
exactly 0 or 1, v2 is 86%. Reversibility on this task is close to genuinely binary either
way; what changed is that perturbing the manipulation produces states whose recoverability
is actually predictable, where perturbing the approach did not.

Also measured: under this regime the published fixed horizon drops from 65% to **39%**
success, so the disturbances finally leave real headroom for a controller to exploit.

### Generalisation to unseen episodes

Every `gated_oof` episode in the table above comes from the *labelled* set — held out from
its own CV fold, but still inside the labelled distribution. Running the comparison videos
on never-labelled seeds suggested the learned controller might collapse there (1/7 against
5/7 over seven episodes), so it was tested properly.

**n=28 novel seeds (7000–7029), effective perturbation kinds only, corrected onset window:**

| Condition | n | Success | Interventions/ep | Mean horizon |
|---|---|---|---|---|
| fixed h=100 | 28 | 11/28 = 39% | 3.7 | 100 |
| fixed h=20 | 28 | 9/28 = 32% | 19.1 | 20 |
| reversibility_gated (γ=2) | 28 | 8/28 = 29% | 9.7 | 41 |

`fixed h=100` vs gated: 5 vs 2 discordant, **p=0.453**. The seven-episode alarm was noise.
No collapse on unseen episodes, and no benefit either — the same conclusion as the main
table, reached on independent seeds.

Note the absolute rates: with only the two *effective* perturbation kinds at the corrected
onset, the published fixed horizon falls from 65% to **39%**. The disturbances are finally
doing real work, and there is far more headroom for a controller to demonstrate value than
the v1 dataset offered.

### Comparison videos

`scripts/08_make_videos.py` renders matched pairs — same seed, same disturbance, different
horizon rule — selected by outcome divergence, since a pair where both policies succeed
demonstrates nothing. Output is 1280×480 h264 with the disturbance and outcome burned in.
It writes nothing when no pair diverges, rather than producing a persuasive-looking file
that shows no difference.

---

## Reproducing every number

Each script is a milestone gate: it prints `GATE: PASS|FAIL`, writes its evidence to
`artifacts/`, and exits non-zero on failure.

```bash
python scripts/00_setup_check.py --allow-no-cuda   # environment
python scripts/01_verify_act_checkpoint.py         # M1: per-seed vs LeRobot's official eval
python scripts/03_calibrate_perturbations.py       # M2: which disturbances actually work
python scripts/04_label_reversibility.py \
    --episodes 80 --workers 3 \
    --kinds object_displace actuation_noise \
    --shard-dir artifacts/reversibility_shards_v2  # M3: ~3h on 3 performance cores
python scripts/05_train_reversibility.py \
    --shards artifacts/reversibility_shards_v2     # M4: 5-fold CV, seconds
python scripts/06_run_ablation.py --episodes 80 --skip-gated --h-min 20 \
    --gammas 1.0 2.0 --fixed-hs 100 20 \
    --oracle-shards artifacts/reversibility_shards_v2 \
    --oof-heads artifacts/reversibility_heads_oof.pt   # M5: ~45min
python scripts/07_make_figures.py                  # M6
python scripts/08_make_videos.py                   # side-by-side comparisons
python scripts/progress.py --shards <dir> --total 80   # status of a running label job
```

Long jobs go under `nohup`/`tmux` — a background-task timeout killed one 90-minute run
mid-flight. Per-episode shards made that cost nothing: it resumed from disk with no
recomputation.

**Running it on a laptop.** `faact/thermal.py` pauses work when macOS reports thermal
pressure, caps workers at performance-cores-minus-one, and pins each worker to a single torch
thread. Across ~9 hours of labelling the machine never left "fair" and never needed a pause.
The single most important knob is the thread pinning: without it, N workers each spawn a pool
sized to the whole machine and contend, which is slower than running serially and hotter.

---

## What exists right now

```
faact/
  backbone/act_wrapper.py   # ACT chunk prediction + forward-hook features
  envs/make.py              # env factory, ALOHA constants
  envs/perturb.py           # 4 perturbation kinds, calibrated magnitudes
  envs/state.py             # snapshot / restore / re-observe / cube teleport
  runtime/executor.py       # ChunkExecutor — controllable commitment horizon
  runtime/controller.py     # fixed, score-gated, reversibility-gated, oracle
  labeling/branch_rollout.py  # measure R(s) by snapshot-restore-replay
  labeling/dataset.py       # episode-level splits, k-fold, feature groups
  models/reversibility.py   # MLP head, soft-target BCE
  models/calibration.py     # isotonic calibration (measured not to help)
  eval/{runner,metrics,ablation}.py
  viz/figures.py
  thermal.py                # thermal governor for long laptop runs
scripts/                    # 00-08, one per gate, plus progress.py
tests/                      # 146 tests; markers `sim` (MuJoCo), `policy` (checkpoint)
docs/                       # the three-day plan and the milestone spec
```

## Setup

Everything in this repo ran on a MacBook (Apple M5) with no GPU. That is a measured claim, not
a convenience: MuJoCo steps at ~84 steps/s while an ACT forward pass takes ~76 ms, so a
250-step branch rollout is 2.96 s of simulation against 0.23 s of policy. **Branch-rollout
labelling is CPU-bound by roughly 13×, and an accelerator buys almost nothing** — what matters
is performance-core count. The CUDA path exists and is device-agnostic, but is untested.

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
