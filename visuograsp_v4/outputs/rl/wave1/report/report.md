# VisuoGrasp V4 — Wave 1 SAC results (2026-11-14)

**Setting: PRIVILEGED state-based teacher, NOT vision-based.** The actor observation contains the
simulator object pose and geometry; the contact signal is the ground-truth oracle (M0-style). The
demo arm additionally uses scripted demonstrations planned from the privileged object pose.
Environment: `VisuoGraspV4Env`, `V4Config` v4.0-foundation (finger armature 1e-3, noslip 5), reward
**v4r1**, approaches top + oblique (45°), 320-step episodes. This is the modified V4 environment,
not the V3.3 environment. Test seeds (5_100_000+) were **not** used in Wave 1.

## 1. What was run

| | SAC scratch | SAC + demos |
|---|---|---|
| Run folder | `outputs/rl/wave1/sac_scratch_s11` | `outputs/rl/wave1/sac_demo_s11` |
| Seed | 11 | 11 |
| Config | SB3 2.9.2 SAC, LayerNorm critics, n-step 10, lr 3e-4, batch 256, UTD 1, γ 0.99, ent_coef auto (init 0.1), learning_starts 10k, buffer 1M | same |
| Demonstrations | — | 20,040 scripted transitions, 65 episodes (seeds 3_110_000+), scripted GT success 73.8 % (top 27/33, oblique 21/32) |
| Steps completed | **150,000** of 300,000 approved | **150,000** of 300,000 approved |
| Stop reason | pre-registered gate rule at 150k (below 20 %, no trend) | same |
| Wall time (launch 11:46:40 → 12:44) | 57.4 min (52.2 min training, 5.2 min waiting at the gate) | 55.5 min (51.5 min training incl. 52 s prefill + 25 s verification, 4.0 min at the gate) |
| Throughput | 47.9 env steps/s (2 runs in parallel, 2 torch threads each) | 48.5 env steps/s |

Both runs ran in parallel on CPU. Total compute ≈ 58 min wall for both runs + ≈ 10 min of independent
validation, videos and checks. An earlier launch at 11:43 was aborted after 5,000 steps because of a
bug in my divergence check (missing SB3 loss values before `learning_starts` were read as NaN — a
false alarm; Q was finite). It is kept in `wave1/_aborted_launch_1143/` with a README; both runs were
relaunched from scratch with the fix.

## 2. Validation protocol

* In-training validation (model selection → `best_model.zip`): 50 episodes every 25k steps
  (5 families × 2 approaches × 5), seeds 8_000_000+, deterministic policy.
* Independent validation (gate decision; not used for selection): 100 episodes per checkpoint
  (5 families × 2 approaches × 10 = 50 top + 50 oblique), seeds 8_500_000+, deterministic policy,
  torch 1 thread. 95 % Wilson intervals.
* Reference on the same 100 seeds: scripted controller with PRIVILEGED planner.
* Gate rule fixed before launch: `wave1/wave1_plan_and_decision_rule.json`; applied by
  `scripts/wave1_gate_decision.py` (output stored in each run's `gate_decision.json`).

## 3. Independent validation, GT success (deterministic policy, PRIVILEGED state)

| Policy | Checkpoint | All (n=100) | Top (n=50) | Oblique (n=50) | Partial lift ≥ 3.5 cm (all) |
|---|---|---|---|---|---|
| SAC scratch | 50k | 1 % [0.2, 5.4] | 2 % [0.4, 10.5] | 0 % [0, 7.1] | 12 % |
| SAC scratch | 100k | 0 % [0, 3.7] | 0 % [0, 7.1] | 0 % [0, 7.1] | 12 % |
| SAC scratch | 150k = best_model = last | 0 % [0, 3.7] | 0 % [0, 7.1] | 0 % [0, 7.1] | 11 % |
| SAC + demos | 50k | 5 % [2.2, 11.2] | 8 % [3.2, 18.8] | 2 % [0.4, 10.5] | 11 % |
| SAC + demos | **100k = best_model** | **8 % [4.1, 15.0]** | **10 % [4.3, 21.4]** | **6 % [2.1, 16.2]** | 29 % |
| SAC + demos | 150k = last | 5 % [2.2, 11.2] | 6 % [2.1, 16.2] | 4 % [1.1, 13.5] | 22 % |
| Scripted (PRIVILEGED planner) | — | 75 % [65.7, 82.5] | 92 % [81.2, 96.8] | 58 % [44.2, 70.6] | 77 % |

Gate (150k): threshold 20 % not met by either arm (best upper Wilson bound 21.4 %, demo top);
trend criterion not met (demo GT 8 % → 5 % and partial lift 29 % → 22 % from 100k to 150k; scratch
flat). Decision: **stop both at 150k**. Sensitivity check (demo, same seeds, torch 6 threads): best
7 %, 150k 5 % — aggregate rates are stable although single episodes are not (see §7).

## 4. Training curves (stochastic policy, training episodes)

Figures: `fig_training_curves.png`, `fig_validation.png`, `fig_critic.png`, `fig_outcomes.png`.
Per 25k-step window: `table_training_windows.csv`.

* Scratch: 1 GT success in 565 training episodes; partial-lift rate 2–24 % per window, no trend.
* Demo: 35 GT successes in 534 training episodes; success rate per window rises from 0–2 %
  (0–25k) to 10–10.5 % (125–150k, top and oblique). This upward trend is in the **stochastic**
  training episodes only; the deterministic policy did not improve from 100k to 150k.
* Episode return: scratch −8.1 → −5.5 (plateau from 40k); demo −8.1 → −5.5 with peaks ≈ −2.5.

## 5. Critic / entropy / stability

* No divergence stop. |Q| exceeded the soft limit 50 (true discounted returns of v4r1 are bounded
  by ≈ 20–30): q_max peak 62.8 (scratch), 73.7 (demo); demo q_p99 at 150k 44.5 and mean Q on demo
  transitions 25.1 → **moderate over-estimation on the demo arm**, not a runaway (hard limit 200).
* Critic loss stable (scratch ≈ 0.01–0.03, demo ≈ 0.5–1). Entropy coefficient decays
  log-linearly from 0.1 to 0.0019 (scratch) / 0.011 (demo): the policy entropy stayed above the
  −22 target the whole time, so the entropy bonus effectively vanished after ≈ 40k steps.
* n-step targets re-computed by hand: max |Δreturn| ≤ 2.4e-6, discounts and done flags exact.
* Numerical failures of the **simulator** (object launched / runaway) under learned policies:
  scratch 18/565 training episodes (3.2 %), demo 9/534 (1.7 %); in independent validation
  1–7 of 100 episodes per checkpoint (scripted baseline: 0/486). They terminate the episode with
  −5 and are counted as failures. Consequence: mean lift is meaningless (one episode reached 1.5e6
  cm); medians and the partial-lift rate are reported instead.
* System: thermal zone 94.85 °C reading throughout (constant value; 61 °C at idle), passive limit
  100 % (no thermal throttling), throttle reasons 0, CPU performance 109–132 % of nominal,
  ≥ 3.5 GB RAM free, no pause. `wave1/system_monitor.csv`.

## 6. Demonstration check (same action semantics and reward)

`sac_demo_s11/demo_verification.json`: all 20,040 stored demo transitions replayed in a fresh
environment with the stored actions → max |Δobs| = 0, |Δnext_obs| = 0, |Δreward| = 0, 0 done /
timeout mismatches; same `V4Config`, observation mode, contact source and approaches as the
training env; reward version v4r1 in both; action space [−1, 1] with SAC `scale_action` = identity
(stored action = executed action, scripted actions clipped to ±0.999).

## 7. Failure analysis

Independent validation outcomes (`fig_outcomes.png`): the dominant failure for both arms is
**object pushed before closing** (44–76 % of episodes per checkpoint/approach), then no contact
(12–44 %). Demo best (100k), top: 8 % success retained, 20 % lifted then dropped, 8 % never lifted,
54 % pushed; oblique: 6 % success, 14 % never lifted, 54 % pushed, 18 % no contact.

Chaotic evaluation: a 2.4e-7 change in the first action (torch 1 vs 6 threads) turned a
`success_retained` episode into `success_lost_later`. All evaluation therefore uses 1 thread and
single-episode outcomes must not be over-interpreted; the first video attempt (default threads,
5 of 15 not reproduced) is kept in `videos/_not_reproduced_default_threads/` and not used.

## 8. Best checkpoints

| Arm | best_model.zip (in-training selection) | Independent validation |
|---|---|---|
| Scratch | step 150k (GT 0/50; chosen by the partial-lift tie-break) | 0 % [0, 3.7] |
| Demo | step 100k (GT 4/50 = 8 %) | 8 % [4.1, 15.0] |

Final checkpoints: `last_model.zip` (150k) in both folders; also `checkpoints/step_{50000,100000,150000}.zip`
and `gate_step_150000.zip`. Limitation: the 4th element of the pre-registered selection key (mean max
lift) is sensitive to launched objects — it made scratch 125k "best" temporarily; the final scratch
choice (150k) was decided by the partial-lift rate, the demo choice by GT success.

## 9. Trajectory videos (overview | side | wrist, seeds from independent validation, reproduced = True)

* Demo best (100k): `sac_demo_s11/videos/gate150k_best_model/` — top success (seed 8500000), oblique
  success (8501102), pushed-before-close (top 8500001, oblique 8500100), lifted-then-dropped (top
  8500002), never-lifted (oblique 8500101).
* Scratch best (150k): `sac_scratch_s11/videos/gate150k_best_model/` — failures only (no success exists).
* Scratch 50k: `sac_scratch_s11/videos/gate150k_checkpoints_step_50000/` — its only GT success
  (top 8504006, success then lost) and failures.

## 10. Scratch vs demo

Demo prefill helped: 8 % vs 0 % at 100k (Wilson intervals [4.1, 15.0] vs [0, 3.7] do not overlap),
35 vs 1 training successes, partial-lift 29 % vs 12 %. But it plateaued far below the scripted
controller it learned from (75 % on the same seeds) and below the 20 % gate. Scratch SAC did not
learn to grasp in 150k steps.

## 11. Limitations

One seed per arm; 100 validation episodes per checkpoint; privileged state; reward v4r1 only; budget
stopped at 150k by rule (no evidence about 150–300k); learned policies trigger simulator failures
that the scripted controller does not.
