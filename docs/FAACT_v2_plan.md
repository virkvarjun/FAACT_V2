# FAACT v2 — Three-Day Build Plan

**Deadline:** Wednesday 19 Aug 2026 · **Target:** interview talk + project site · **Compute:** 1× L40S (RunPod)

---

## 0. The honest scope call

Three days, with the runtime loop needing a rebuild, does not fit the full research plan. What follows is cut to
what can actually produce defensible numbers by Wednesday, with explicit fallback positions at each gate.

**In scope**

1. Rebuilt runtime loop (chunk executor + eval runner)
2. Perturbation suite — the thing that makes "recovery" measurable at all
3. Reversibility labelling by branch rollout, and a reversibility head
4. Reversibility-gated dynamic chunking (no-training variant)
5. Honest ablation table + figures + site

**Cut, and why**

| Cut | Reason |
|---|---|
| Full K-adapter LoRA store | Needs recovery demos per mode; realistically day 4–6. Scaffold only (see §6). |
| OCR keypoint inverse policy | Requires training an inverse model. High value, no time. |
| Faithful FIPER re-implementation | Real multi-sample ACE needs a stochastic policy. Use your existing proxies, label them as proxies. |
| Insertion / Panda tasks | One task, done properly, beats four half-done. |
| Multi-seed error bars | Report single-seed with n stated. Do not claim significance. |

**The single biggest schedule risk:** whether you still have the trained ACT checkpoint from the 100K-step run
(the one that produced `outputs/eval_videos/step_100000/`). Checkpoints were gitignored and lived on RunPod.
If that pod is gone and ACT must be retrained from scratch, that is most of Day 1 and everything below slips.
**Check this first, before anything else.**

---

## 1. Why the current results are 0/20, and what fixes it

Diagnosis from the committed artifacts:

- `--perturbation_mode` was never implemented, so nothing was ever *disturbed*. The "failures" were just a weak
  policy failing. There is no recovery signal to detect.
- Baseline success `0/20` means zero headroom. No intervention method can show a lift over a floor of zero.
- The risk label is `failure_within_k`, i.e. "am I near the end of an episode that failed." That is a
  proximity-to-termination detector, not a notion of recoverability.
- `recovery_after_intervention: 0` in every run, and your own note — *"the bottleneck is action usefulness,
  not just action scoring"* — is the correct diagnosis. Candidate chunks sampled by perturbing a deterministic
  ACT do not span the recovery manifold.

The fix is ordered: **working baseline → real perturbations → headroom → then, and only then, a method.**

---

## 2. Day-by-day

### Day 1 (Mon) — baseline + perturbations

**Gate at end of day: unperturbed ACT success ≥ 60% over 20 episodes. If not met, stop and debug; do not proceed.**

1. **Recover or retrain the ACT checkpoint** on `lerobot/aloha_sim_transfer_cube_human`, `AlohaTransferCube-v0`.
   Target the ~80–90% LeRobot reports. Retraining is ~100K steps; start it immediately in the background if needed.
2. **Rebuild the runtime loop.** Minimum viable replacement for the missing `faact/` package:

```
faact/
  backbone/act_wrapper.py     # predict_chunk(obs) -> (chunk[k,14], features dict)
  envs/perturb.py             # PerturbationSpec, apply(env, spec, t)
  runtime/executor.py         # ChunkExecutor: horizon h, temporal ensembling
  runtime/controller.py       # risk + reversibility -> {continue, replan, abort}
  eval/runner.py              # run_episode(...) -> EpisodeResult
  labeling/branch_rollout.py  # snapshot/restore + M continuations
```

Keep the existing `failure_prediction/` package untouched and import into it — do not refactor on a deadline.

3. **Perturbation suite** (`envs/perturb.py`). Four types, each parameterised by onset step `t` and magnitude.
   Onset is known by construction, which gives free ground truth for lead-time.

| Type | Implementation |
|---|---|
| `object_displace` | At step `t`, teleport the cube: set the free-joint qpos via `physics`, then `physics.forward()`. δ ~ U(2, 5) cm, random planar direction. |
| `grasp_slip` | Override the gripper channel to fully open for 3 consecutive steps during transport. |
| `actuation_noise` | Additive Gaussian on actions over a window of 20 steps. |
| `occlusion` | Zero a rectangular patch in one camera image over a window. Cheapest of the four. |

Validate by eyeballing: perturbed success should land somewhere in 30–60%. If perturbation drops success to 0%,
it is too strong and there is again no headroom — tune magnitude until you sit in that band. **This band is the
whole experiment.**

### Day 2 (Tue) — reversibility

4. **Branch-rollout labeller** (`labeling/branch_rollout.py`). The core primitive is dm_control state
   save/restore, which gym-aloha exposes through the underlying physics object:

```python
state = physics.get_state().copy()      # snapshot
...
physics.set_state(state); physics.forward()   # restore
```

Loop:

```
for episode in perturbed_episodes:            # E = 40
    for t in range(0, T, S):                  # S = 25  -> ~16 points
        s = snapshot()
        wins = 0
        for m in range(M):                    # M = 8
            restore(s)
            wins += rollout_to_completion(policy, fresh_chunk=True)
        R[t] = wins / M
```

**Cost:** ≈ 40 × 16 × 8 × 250 ≈ 1.3M env steps. Run branch rollouts **open-loop at h=100** so the policy fires
~13K forward passes rather than 1.3M — MuJoCo, not the GPU, becomes the bottleneck. Parallelise across 8 workers.
Expect roughly 1–2 hours, not a day. If it overruns, drop to `S=50, M=5` and re-check.

5. **Reversibility head.** Reuse `failure_prediction/models/failure_predictor.py` almost verbatim — same
   `input_dim → [256,128] → 1` MLP on the same `feat_*` vectors. Change two things: sigmoid output regressing
   `R ∈ [0,1]`, and soft-target BCE (or MSE) instead of hard-label BCE. Keep episode-level splits from
   `data/splits.py`. Train time: minutes.

   **Produce the calibration scatter — predicted R vs. empirical branch-rollout R.** This is your single best
   figure and the strongest evidence that the quantity is real rather than asserted.

### Day 3 (Wed) — horizon control, ablation, write-up

6. **Reversibility-gated horizon** in `ChunkExecutor`. No training:

   `h_t = clip(h_min, h_max, round(h_max · R(s_t)^γ))`, with `h_min=5, h_max=100, γ∈{1,2}`

7. **Ablation table.** All on the same perturbed episode set, same seed, n=20:

| Condition | What it isolates |
|---|---|
| Fixed h = 100 | ACT as published |
| Fixed h = 20 | Is any gain just "replan more often"? — **the ablation that matters most** |
| Risk-gated horizon (existing MLP) | Does reversibility beat plain risk? |
| Reversibility-gated horizon | The claim |
| Oracle reversibility (labelled R) | Ceiling — how much is lost to estimation error |

Metrics: success under perturbation, interventions/episode, mean committed horizon, lead time vs. known onset,
false-alarm rate on successful episodes.

8. **Figures + site + talk.** See §7.

---

## 3. What the claim is, precisely

> Reversibility, not uncertainty, is the right control variable for chunked policies.

Detection (FIPER, ActProbe), horizon (DEHP, MoH, A³), and recovery (OCR, FLARE) are each solved separately, and
each picks its own trigger. One estimate of *"can I still come back from here"* governs all three: how long to
commit, when to intervene, and when to stop trying. DEHP optimises horizon for task success with online RL;
optimising for reversibility is a different objective, needs no RL, and gets dense labels free from the simulator.

**Positioning to state out loud, unprompted:**

- **vs. DEHP** — same lever (execution horizon), different objective (reversibility, not success), no RL, label-free.
- **vs. FIPER** — FIPER predicts *whether* it will fail; reversibility predicts *whether intervening still helps*.
- **vs. FLARE** — FLARE recovers via MLLM-routed reset skills; no notion of a point of no return.
- **vs. OCR** — OCR is a recovery *mechanism* with no trigger; reversibility is a trigger with no mechanism. They compose.

**Known adjacent literature — do not get blindsided:** reversibility-aware RL (Grinsztajn et al.) and the
"leave no trace" safe-RL lineage. Neither applies it to action-chunk horizon control in imitation learning.
Say this before an interviewer says it to you.

---

## 4. Metrics and what counts as success

By Wednesday, a good outcome is *any* of:

- Reversibility-gated horizon beats **both** fixed-h baselines on perturbed success rate, or
- Predicted R correlates well with empirical R (calibration plot) even if the controller gain is small, or
- A clean negative result with a diagnosis

The third is genuinely acceptable for a talk and better than an overclaimed first. The arc *"v1 tried candidate
selection → measured that it fails → diagnosed why → v2 reframes around reversibility"* is a stronger interview
narrative than a marginal success bump, because it demonstrates you can evaluate your own work.

---

## 5. Repo hygiene — do this regardless, it is 30 minutes

Before anyone clones the repo:

- **Rewrite the README to match reality.** It currently describes `act/`, `monitor/recoverability_head.py`,
  `rollout/perturbations.py`, and four scripts that do not exist, plus metrics that are never computed.
- **Publish or vendor `faact/`.** Four scripts import it; a clean clone `ImportError`s.
- **Fix the submodule.** The pin `7e0b41d` does not exist in `huggingface/lerobot`, so
  `--recurse-submodules` fails. Push your fork with the ACT `return_features` hook and re-point `.gitmodules`.
- **Label the π₀ results as π₀.** The README frames everything as ACT; every committed number is π₀.
- **Correct the citation on your site.** "Detects trajectory failures... and autonomously recovers" is not
  supported by `recovery_after_intervention: 0`. Soften until the new numbers exist.

---

## 6. The LoRA adapter store

**Verdict: the design makes sense, and there is one trick that makes it cheap. It does not fit before Wednesday.
Scaffold it, present it as the designed next step.**

### Why it makes sense here

The reason FLARE needs an MLLM is failure-mode *labelling* — it has to analyse execution videos offline to
discover what went wrong. You do not have that problem: **you inject the perturbation, so you already know the
failure mode.** Perturbation type *is* the mode label, free and noiseless. That removes the expensive component
of FLARE's pipeline and is a legitimate differentiator.

### Design

```
Perturbation type  ──►  recovery demos  ──►  LoRA_k  (rank 8–16 on ACT attn q,k,v + FFN)
   (free label)         (scripted expert)      ~0.5M params vs ACT's ~80M

Runtime:  detector fires
          └─► route on perturbation-type classifier (or argmax predicted ΔR per adapter)
              └─► swap LoRA_k in, execute N steps, swap back to base
```

**Recovery-data generator (the part that makes this feasible at all):** ALOHA sim ships scripted expert policies
(`PickAndTransferPolicy` for transfer-cube) that use privileged state. So:

1. Roll out ACT, inject perturbation at step `t`
2. Hand control to the scripted expert from the perturbed state
3. Keep the continuation iff it succeeds
4. Tag it with the perturbation type

That yields clean, labelled recovery demonstrations with zero human effort and no MLLM. This is the piece worth
presenting even before the adapters are trained.

### The ablation that makes or breaks it

At K = 3–4 modes, **a single LoRA fine-tuned on all recovery data jointly is a strong baseline and may well match
the store.** The store's value — specialisation without interference — only clearly pays off at larger K or when
recovery motions are genuinely distinct. Run `K separate adapters` vs `1 joint adapter` vs `base policy`, or the
store is not a result. Anticipate this question; it is the first one a good interviewer will ask.

### Wednesday-realistic version

- ✅ Implement the scripted-expert recovery-data generator, show ~50 labelled recovery trajectories
- ✅ Slide describing the store, the routing rule, and the joint-adapter ablation
- ⚠️ Train **one** adapter on the dominant perturbation type if Day 3 runs ahead of schedule
- ❌ The full K-adapter store with a learned router

---

## 7. Talk and site

**Narrative spine (5 beats):**

1. Action chunking buys smoothness and costs reactivity — once committed, the robot is blind
2. v1: detect risk, sample candidate chunks, pick the safest → **measured 1/10, zero recoveries**
3. Diagnosis: the bottleneck was action usefulness, not action scoring — and the baseline had no headroom
   because nothing was ever actually perturbed
4. Reframe: reversibility as one control variable governing horizon, intervention, and abort
5. Results, honest limitations, and the adapter store as next step

**Figures, in priority order:**

1. Reversibility heatmap along a trajectory with the point of no return marked
2. Predicted vs. empirical R calibration scatter
3. Committed horizon over time, overlaid on perturbation onset — shows the controller reacting
4. Ablation table
5. Architecture diagram (the reversibility → three-decisions figure)

**Videos:** side-by-side same-seed, same-perturbation, fixed-h vs. reversibility-gated. One clear save is worth
more than a montage. `scripts/record_5_interventions.sh` already exists for this.

**Site:** static page, videos inline and autoplay-muted-loop, figures as SVG, method section with the four
positioning bullets from §3. Match the visual language of your existing project pages.

---

## 8. First three actions

1. Check whether the 100K ACT checkpoint still exists on RunPod — **this gates everything**
2. If not, launch retraining now, in the background, before writing any other code
3. While that runs, write `envs/perturb.py` and `labeling/branch_rollout.py` — neither needs a trained policy to test
