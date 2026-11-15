# VisuoGrasp V4 — Wave 3: conservative residual SAC (2026-11-14) — FAILED, RL training stopped

**Setting: PRIVILEGED residual learning on a scripted controller — not autonomous, not
vision-based.** Nominal controller planned from the simulator object pose; actor observation with
the simulator object pose; contacts = simulator ground truth. Modified V4 environment (not V3.3).

## 1. Implementation changes (Wave 1/2 code paths unchanged)

* `residual.py`: `ResidualConfig` gains `reward_version`, `w_residual` (defaults = Wave 2 behaviour);
  `RES1` (Wave 2) and **`RES2`** (Wave 3); v4r2 term in `ResidualGraspEnv.step`; helpers
  `init_actor_log_std`, `tanh_gaussian_entropy`; logging-only statistic `res_jitter`.
* `agents.py`: `ZeroCenteredWarmupSAC` (warm-up samples the policy, not uniform actions);
  `make_sac(..., algo_cls, **overrides)`; several decision gates per run; actor statistics
  (mean abs(tanh(mu)), mean log std) in `qhealth.csv`.
* New scripts: `train_conservative_residual_sac.py`, `wave3_gate_decision.py`, `wave3_final_eval.py`,
  `wave3_report.py`. Extended: `verify_residual.py --version`, `validate_checkpoints.py
  --residual-version --perturbed-control`, `record_wave2_videos.py --group --residual-version`.

## 2. Reward v4r2 (v4r1 unchanged)

    r_v4r2(t) = r_v4r1(t) - 0.01 * (1/22) * sum_i r_i(t)^2

r_v4r1 is computed by `VisuoGraspV4Env` from the executed 22-D action exactly as before; r ∈ [−1, 1]^22
is the normalised residual. Maximum penalty 0.01 per step. Verified per step: v4r2 − v4r1 =
−0.01·mean(r²) with error ≤ 4.2e-16, dynamics bit-identical (check G).

## 3. Residual bounds (res2) and exploration

| Channel | res1 (Wave 2) | **res2 (Wave 3)** |
|---|---|---|
| finger joint goals | ±0.30 rad | **±0.08 rad** |
| wrist position goal (world axes) | ±2 cm | **±4 mm** |
| wrist orientation goal (hand axes, quaternion product) | ±0.15 rad | **±0.03 rad** |

Exploration: warm-up 5,000 steps sampled from the policy itself (mean exactly 0, std 0.2 before
tanh, state independent; measured mean abs(r) 0.156) instead of 10,000 uniform-random steps;
entropy coefficient auto, initial 0.01 (Wave 2: 0.1); target entropy −5.03 = entropy of
tanh(N(0, 0.2²)) in 22-D (SB3 default −22 ≙ std 0.09). Actor mean layer zero-initialised (initial
deterministic residual exactly 0, checked). Unchanged: SAC, LayerNorm critics, n-step 10, lr 3e-4,
batch 256, γ 0.99, buffer 1M, seed 11.

## 4. Zero-residual equivalence (`wave3/verification/verification_report.json`, all passed)

A: r = 0 vs scripted lock-step, 40 episodes / 12,302 steps bit-identical; B: 100 validation
episodes identical; C: no workspace / tilt / clearance / joint-range violation, quaternion error ≤ 2.2e-16;
D: no reward exploitation; E: training and evaluation action mapping identical (960 steps);
F: observation independent of evaluation-only fields; G: v4r2 formula exact. On the final unseen set
r = 0 equals scripted in all 540 episodes.

## 5. Training (stopped at the first gate)

25,000 of 75,000 steps; 13.6 min wall (8.1 min training at 51.4 steps/s, 5.5 min at the gate);
stop reason `gate_stop`; no numerical instability (q_max 40.7 at 10k → 25.5 at 25k; critic loss
0.10–0.26; n-step check passed), no thermal throttling.

| Step | in-training val (n=50, seeds 8.0M) | zero residual, same episodes |
|---|---|---|
| 12.5k | 52 % (top 76, oblique 28) | 68 % (84, 52) |
| 25k | 60 % (top 80, oblique 40) | 68 % (84, 52) |

Gate 25k — independent validation, seeds 8.5M, 200 paired episodes:

| Policy | All | Top | Oblique | vs r = 0 | fixed / broken | McNemar p |
|---|---|---|---|---|---|---|
| r = 0 (scripted) | 71.5 % | 90 % | 53 % | — | — | — |
| chaos control r ~ U(±1e-3) | 70.5 % | 90 % | 51 % | −1.0 pp | 0 / 2 | 0.50 |
| 12.5k | 57.5 % | 76 % | 39 % | — | — | — |
| **25k (best = last)** | **61.0 %** | 78 % | 44 % | **−10.5 pp [−15.5, −5.5]** | 4 / 25 | 1.0e-4 |

Rule (fixed before training): stop if diff ≤ −5 pp or diff < 0 with p < 0.05 → **stop at 25k**.

Entropy behaviour (`fig_actor_entropy.png`): the deterministic residual left zero within 5k
updates (mean abs(tanh(mu)) 0.45 at 10k, 0.43 at 25k ≈ 40 % of the bounds) and the policy std rose
from 0.2 to ≈ 0.33; alpha rose 0.010 → 0.023, then fell to 0.0086. Once the mean moves into the
tanh non-linearity, the entropy of the squashed policy drops below the target and the automatic
coefficient raises alpha — the exploration pressure the request warned about.

## 6. Final evaluation — new unseen set (seeds 6.1M, 540 paired episodes per policy, run once)

Selection frozen before the run (`frozen_selection.json`): best_model = last_model = 25k.

| Policy | All | Top (n=270) | Oblique (n=270) |
|---|---|---|---|
| Scripted controller = r = 0 | 76.7 % [72.9, 80.0] | 95.6 % [92.4, 97.4] | 57.8 % [51.8, 63.5] |
| Chaos control r ~ U(±1e-3) | 76.9 % [73.1, 80.2] | 95.9 % | 57.8 % |
| **Conservative residual SAC (25k)** | **69.1 % [65.1, 72.8]** | **88.1 % [83.7, 91.5]** | **50.0 % [44.1, 55.9]** |

Paired vs r = 0: **−7.6 pp [−10.2, −5.0]**, McNemar p = 5.8e-9; top −7.4 pp (0 fixed / 20 broken,
p = 1.9e-6); oblique −7.8 pp (6 / 27, p = 3.2e-4). Chaos control: +0.2 pp (2 / 1, p = 1.0).

Outcome categories vs r = 0 (540 episodes):

| | preserved (scripted success kept) | **improved by the correction** | **broken by the correction** | neither |
|---|---|---|---|---|
| Residual SAC | 367 | **6** (all oblique: 3 bottle, 3 L-shape) | **47** (top 20, oblique 27) | 120 |
| Chaos control | 413 | 2 | 1 | 124 |

* Success of the residual policy is almost entirely the scripted controller's: 367 of its 373
  successes are episodes the scripted controller also solves.
* Genuine improvements: 6 episodes, above the chaos floor (2) — the oblique corrections fixed 3
  lifted-then-dropped, 2 pushed and 1 never-lifted episode — but outweighed 7.8 : 1 by breakages.
* Failures caused by the corrections (47): pushed before closing 27, never lifted 12, held without a
  pad pair 4, lifted then dropped 3, partial lift 1.
* By family (r = 0 → residual): bottle 67.5 → 65.0 %, box 90.6 → 83.3 %, cylinder 81.7 → 76.7 %,
  L-shape 70.8 → 64.2 %, ring 35.0 → 5.0 % (ring: 0 fixed / 12 broken, p = 5e-4).
* Corrections: mean abs(r) 0.375 → wrist position 2.9 mm, rotation 0.021 rad, fingers 0.030 rad;
  step-to-step jitter 0.23 (corrections change by a quarter of the bound each step — not a stable
  offset); phase timing preserved (closing at step 47.2 vs 46.6; Wave 2: 151 vs 46); hand 3.4 mm
  from the planned grasp pose at closing (r = 0: 1.4 mm); pushed before closing 33.7 % vs 23.7 %;
  0 simulator failures.

## 7. Best checkpoint

`outputs/rl/wave3/conservative_residual_sac_seed11/best_model.zip` (25k; identical to `last_model.zip`
and `checkpoints/gate_step_25000.zip`). Videos (`wave3/videos/`, all re-runs reproduced):
`top_success_top_box_seed6100000.mp4`, `oblique_success_oblique_bottle_seed6117102.mp4` (a fixed
episode: scripted dropped the bottle), `informative_failure_top_box_seed6104001.mp4` (broken: held
without pad pair), `side_by_side_residual_wins_…seed6117102.mp4`, `side_by_side_scripted_wins_…seed6104001.mp4`.

## 8. Verdict

**Reject conservative residual SAC.** With 5× smaller bounds, a magnitude penalty and zero-centred
exploration the damage shrank from −28 pp (Wave 2) to −7.6 pp, and phase timing is no longer
disrupted, but the learned corrections still break 47 scripted successes for 6 genuine fixes. RL
training is stopped (no Wave 4). Limitations: one seed, 25k steps, one residual design.
