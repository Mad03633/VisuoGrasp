# VisuoGrasp V4 — Wave 2: Residual SAC (2026-11-14)

**Setting: PRIVILEGED residual learning on a scripted controller — not autonomous, not
vision-based.** The nominal controller plans from the simulator object pose and geometry; the
actor observation contains the simulator object pose; the contact signal is the simulator ground
truth (GT oracle). Environment: modified V4 (`V4Config` v4.0-foundation, armature 1e-3, noslip 5),
reward **v4r1 unchanged**, approaches top + oblique (45°). Outcome: **failure** — the trained
residual policy is significantly worse than the scripted controller it corrects.

## 1. Residual control (Step 1)

`visuograsp_v4/residual.py`. The scripted controller is a dead-beat set-point tracker (each step
it commands the full remaining error up to the speed limit), so a residual added to the 22-D
action would be cancelled on the next step (lasting effect ≤ 1 cm / 0.055 rad). The residual
therefore corrects the **set-points** the scripted controller tracks, bounded, r ∈ [−1, 1]^22:

| Channel | Semantics | Bound at abs(r) = 1 |
|---|---|---|
| r[0:16] fingers | joint-space goal offset q_goal + 0.30·r | 0.30 rad per joint |
| r[16:19] wrist position | world-axis goal offset p_goal + 0.02·r | 2 cm per axis |
| r[19:22] wrist orientation | R_goal · exp(0.15·r), hand axes, quaternion product + normalisation | 0.15 rad per axis |

The scripted phase machine (with the corrected set-points in its goal tests), speed limits,
timeouts and all hand-controller projections (workspace box, 110° tilt limit, anti-windup lead
limit, table clearance, joint ranges) are unchanged. Zero offsets skip every arithmetic operation.

Verification (`wave2/verification/verification_report.json`, all required checks passed before training):

| Check | Result |
|---|---|
| A. r = 0 vs scripted, lock-step, 40 DEV episodes | 12,302 steps: executed action, actor observation, reward, flags, qpos/qvel **bit-identical**; outcomes identical (25/40 both) |
| B. r = 0 vs scripted, 100 independent-validation episodes | all 9 compared columns identical; 75 % = 75 % |
| C. stress (zero, random, ±1 constant, bang-bang; 30 episodes) | quaternion norm error ≤ 2.2e-16; 0 workspace / tilt / clearance (at command time) / joint-range violations; 0 non-finite observations; hand–table force 0 |
| D. reward audit (8 residual policies × 10 DEV episodes) | no policy with ≤ zero-residual success has a higher return (no exploitation found); random residuals raise pushing before closing to 60 % (zero: 30 %) |
| E. training vs evaluation action mapping | 849 steps bit-identical |
| F. actor observation (163-D = 130 V4 + phase one-hot 7 + phase progress 1 + nominal goal − target 3 + previous residual 22) | unchanged when evaluation-only tracker fields are mutated |
| Final test | r = 0 equals scripted in all 324 test episodes (0 mismatches in GT, outcome, lift, return, steps) |

## 2. Training (Step 3)

| | |
|---|---|
| Run | `outputs/rl/wave2/residual_sac_seed11` |
| Algorithm | SB3 2.9.2 SAC, LayerNorm critics, n-step 10, lr 3e-4, batch 256, buffer 1M, learning_starts 10k, ent auto (init 0.1), actor mean layer zero-initialised (initial deterministic policy = scripted, checked exactly) |
| Seed / approaches | 11 / top + oblique |
| Steps completed | **75,000** of 150,000 (stopped by the pre-registered 75k gate) |
| Duration | 25.2 min wall (22.1 min training at 56.7 steps/s, 3.2 min at the gate); launched 13:16:31, stopped 13:41:54 |
| Stability | no divergence: q_max peak 59.1, q_p99 peak 48.1 (above the soft limit 50 / bound ≈ 30 early on, falling to q_p99 27.7 at 75k); critic loss 0.30–0.53; alpha 0.1 → 0.003; n-step check passed (max abs return error 1.7e-6); 1 simulator failure in 246 training episodes; no thermal throttling (passive limit 100 %, no pause) |

## 3. Validation

In-training (model selection, seeds 8.0M, 25 per approach; zero residual on the same episodes: top 84 %, oblique 52 %):

| Step | All | Top | Oblique |
|---|---|---|---|
| 25k | 42 % | 56 % | 28 % |
| 50k (best_model) | 50 % | 76 % | 24 % |
| 75k | 36 % | 56 % | 16 % |

Gate at 75k — independent validation (seeds 8.5M, 100 paired episodes):

| Model | All | Top | Oblique | Paired diff vs r = 0 [95 % bootstrap] | only residual / only r = 0 | McNemar p |
|---|---|---|---|---|---|---|
| r = 0 (scripted) | 75 % [65.7, 82.5] | 92 % | 58 % | — | — | — |
| 25k | 43 % | 56 % | 30 % | — | — | — |
| 50k = best | 37 % [28.2, 46.8] | 58 % | 16 % | −38 pp [−48, −28] | 1 / 39 | 7.5e-11 |
| 75k | 39 % [30.0, 48.8] | 56 % | 22 % | −36 pp [−47, −25] | 3 / 39 | 5.6e-9 |

Rule (`wave2_plan_and_decision_rule.json`, fixed before training): stop if both the gate
checkpoint and best_model are ≤ −20 pp (or ≤ −10 pp with p < 0.05). Both were → **stopped at 75k**;
the remaining 75k steps were not spent.

## 4. Final test (Step 4) — seeds 5.1M, 324 identical episodes per policy, used once

| Policy | All | Top (n=162) | Oblique (n=162) |
|---|---|---|---|
| Scripted controller (PRIVILEGED planner) | 77.8 % [72.9, 82.0] | 95.1 % [90.6, 97.5] | 60.5 % [52.8, 67.7] |
| Residual env, r = 0 | 77.8 % (identical, episode by episode) | 95.1 % | 60.5 % |
| **Residual SAC best (50k)** | **49.4 % [44.0, 54.8]** | 67.9 % [60.4, 74.6] | 30.9 % [24.3, 38.4] |
| Residual SAC last (75k) | 45.1 % [39.7, 50.5] | 62.3 % [54.7, 69.4] | 27.8 % [21.5, 35.1] |
| Wave 1 SAC + demos (100k) | 10.5 % [7.6, 14.3] | 13.0 % | 8.0 % |
| Wave 1 SAC scratch (150k) | 0.0 % [0, 1.2] | 0 % | 0 % |

Paired comparisons (same 324 episodes; McNemar exact; 95 % paired-bootstrap CI):

| A vs B | Diff | 95 % CI | only A / only B | p |
|---|---|---|---|---|
| residual best vs scripted (= learning contribution) | **−28.4 pp** | [−34.0, −23.2] | 8 / 100 | 2.4e-21 |
| — top | −27.2 pp | [−34.0, −20.4] | 0 / 44 | 1.1e-13 |
| — oblique | −29.6 pp | [−38.3, −21.0] | 8 / 56 | 5.6e-10 |
| residual last vs residual best | −4.3 pp | [−10.2, +1.5] | 41 / 55 | 0.18 |
| residual best vs Wave 1 demo SAC | +38.9 pp | [+32.7, +45.1] | 142 / 16 | 2.0e-26 |
| Wave 1 demo vs scratch | +10.5 pp | [+7.4, +13.9] | 34 / 0 | 1.2e-10 |

The +38.9 pp of residual SAC over Wave 1 SAC comes from the scripted prior, **not** from learning;
learning itself lowered success by 28.4 pp.

By object family (top + oblique), scripted → residual best: bottle 75.0 → 43.8 %, box 89.8 → 58.3 %,
cylinder 86.1 → 45.8 %, L-shape 72.2 → 59.7 %, ring 20.8 → 0 % (all paired differences negative;
p < 0.05 except ring, p = 0.06). Table: `final_test/summary_by_family*.csv`, `paired.csv`.

## 5. Do the corrections change behaviour? Yes — and mostly for the worse

Residual best on the 324 test episodes (`final_test/behaviour_change.csv`):
mean abs(r) = 0.60 (14 % of components saturated), i.e. wrist position offset 2.2 cm, rotation
offset 0.17 rad, finger offset 0.18 rad per step — large, not small, corrections. Outcome changed
versus r = 0 in 146/324 episodes. Hand position at the start of closing is 2.1 cm from the planned
grasp pose (scripted: 0.14 cm); closing starts at step 151 on average (scripted: 46) because the
moving set-points rarely satisfy the phase goal tests, so phases end by timeout; the object is pushed
before closing in 66 % of episodes (scripted: 22.5 %). Failure modes (top / oblique): pushed before
close 26 / 73, never lifted 6 / 26, lifted then dropped 9 / 4, held without pad pair 9 / 0
(scripted: 3 / 28 pushed, 5 / 28 never lifted).

Likely causes (not tested — would need a new run): 10k steps of uniform-random residuals
(abs(r) ≈ 0.5 ⇒ ≈ 2 cm set-point jitter) fill the buffer with heavily perturbed behaviour; v4r1 has
no cost on residual size, so nothing anchors the actor to r = 0 after it leaves its zero
initialisation; early Q over-estimation (q_max 59 vs ≈ 30 bound) on off-nominal actions; stateless
offsets interact badly with the phase goal tests.

## 6. Files

* Learning curves / critic: `report/fig_training_curves.png`, `fig_validation.png`, `fig_critic.png`;
  final test: `fig_final_test.png`, `fig_final_test_by_family.png`; tables in `report/` and `final_test/`.
* Checkpoints: `residual_sac_seed11/best_model.zip` (50k), `last_model.zip` (75k), `checkpoints/step_{25000,50000,75000}.zip`, `gate_step_75000.zip`.
* Videos (`wave2/videos/`, test seeds, all re-runs reproduced, overlay states the privileged information):
  `top_success_top_box_seed5100002.mp4`, `oblique_success_oblique_box_seed5105105.mp4` (scripted failed here),
  `informative_failure_top_box_seed5100000.mp4` (residual pushes the cube; scripted succeeds),
  `side_by_side_residual_wins_oblique_box_seed5105105.mp4`, `side_by_side_scripted_wins_top_box_seed5100000.mp4`
  (top row scripted, bottom row residual).
* Pipeline checks (not results): `wave2/check_residual_s11`, `wave2/_smoke_final_test`, `wave2/_smoke_videos` (DEV seeds).

## 7. Limitations

One seed; one residual design (res1) and one exploration scheme; privileged state and GT contacts;
the 50k selection is based on 50 episodes; stopped at 75k by rule (no evidence for 75–150k).
