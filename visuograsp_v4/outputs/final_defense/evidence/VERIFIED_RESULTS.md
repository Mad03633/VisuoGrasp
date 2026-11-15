# Verified results (source of truth for the defense package)

All numbers below were read from frozen files on 2026-11-15. Paths are relative to
`C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\rl\` unless stated otherwise.
**Setting for every learned policy: PRIVILEGED state-based control** (simulator object pose and
ground-truth pad contacts in the actor observation). No result in this package is vision-based.

## 1. Wave 5 fresh final test (primary result)

* Data: `wave5/final_test/final/episodes.csv`.
  * 7 controllers × 540 identical episodes (seeds 6,700,000+): 27 object cells × 2 approaches
    (top-down, oblique 45°) × 10.
  * 3,780 unique rows; integrity in `analysis/integrity.json`.
  * The set was used once, after freezing (`policies.json`, `frozen_policies/manifest.json`).
  * The reading rule was written before the results (`final_test/final_decision_rule.json`).
* Statistics:
  * `analysis/summary.csv`: Wilson 95 % CI;
  * `analysis/paired_all.csv`: exact McNemar, paired bootstrap 95 % CI (10,000 resamples), Holm
    over 12 pre-listed overall comparisons;
  * `analysis/decision.json`.
* The run was interrupted once by Claude Code memory protection at 3,029 / 3,780 episodes.
  It was resumed with `--resume --workers 1`; the earlier rows are byte-identical to the backup
  (`resume_log.txt`).

| Controller | Kind | GT success (n = 540) [Wilson 95 %] | Top-down (n = 270) | Oblique 45° (n = 270) | Numerical failures |
|---|---|---|---|---|---|
| Scripted expert | analytic planner + phase machine (PRIVILEGED pose) | 80.6 % [77.0, 83.7] (435) | 96.7 % | 64.4 % | 0 |
| BC Wave 4 (`bc4`) | supervised imitation | 77.4 % [73.7, 80.7] (418) | 95.9 % | 58.9 % | 1 |
| Improved BC E5g | supervised imitation | 77.4 % [73.7, 80.7] (418) | 93.7 % | 61.1 % | 1 |
| BC ensemble (`e5_ens`, 6 members) | supervised imitation | 80.7 % [77.2, 83.8] (436) | 95.9 % | 65.6 % | 0 |
| E5c epoch 10 (E6 initialisation) | supervised imitation | 79.4 % [75.8, 82.6] (429) | 96.3 % | 62.6 % | 1 |
| **E6** (E5c + 90k steps AWAC-style BC-anchored RL) | BC + RL fine-tuning | **82.0 % [78.6, 85.0] (443)** | 95.9 % | **68.1 %** | 0 |
| E6b (E5g + 130k steps, same method) | BC + RL fine-tuning | 78.5 % [74.9, 81.8] (424) | 93.3 % | 63.7 % | 1 |

Paired comparisons on the same 540 episodes (`analysis/paired_all.csv`, group = all):

| A − B | Diff (pp) [bootstrap 95 %] | A-only / B-only wins | exact McNemar p | Holm p |
|---|---|---|---|---|
| E6 − E5c (RL contribution) | +2.6 [+0.9, +4.3] | 18 / 4 | 0.0043 | **0.039** |
| E6b − E5g (RL contribution, replication) | +1.1 [−0.9, +3.3] | 20 / 14 | 0.39 | 1.0 |
| BC ensemble − BC Wave 4 | +3.3 [+1.3, +5.4] | 26 / 8 | 0.0029 | **0.029** |
| E5g − BC Wave 4 | 0.0 [−2.6, +2.4] | 23 / 23 | 1.0 | 1.0 |
| E6 − BC Wave 4 | +4.6 [+2.2, +7.0] | 34 / 9 | 0.00017 | **0.0020** |
| E6 − BC ensemble | +1.3 [−0.4, +3.0] | 14 / 7 | 0.19 | 0.95 |
| E6 − scripted expert | +1.5 [0.0, +3.1] | 13 / 5 | 0.096 | 0.58 |
| BC ensemble − scripted expert | +0.2 [−1.3, +1.7] | 8 / 7 | 1.0 | 1.0 |
| BC Wave 4 − scripted expert | −3.1 [−5.4, −0.9] | 9 / 26 | 0.0060 | 0.048 |
| E5g − scripted expert | −3.1 [−4.8, −1.5] | 2 / 19 | 0.0002 | 0.0024 |
| E5c − scripted expert | −1.1 [−2.6, +0.4] | 6 / 12 | 0.24 | 0.95 |
| E6b − scripted expert | −2.0 [−4.1, −0.2] | 8 / 19 | 0.052 | 0.37 |

**E6 versus its initialisation, by approach** (exploratory sub-groups, no multiplicity correction):

| Approach | Change | Wins / losses | p |
|---|---|---|---|
| Oblique | +5.6 pp | 16 / 1 | 0.0003 |
| Top-down | −0.4 pp | 2 / 3 | 1.0 |

In 13 of the 18 E6 wins, the initialisation had pushed the object before closing.

**What may and may not be claimed:**

May be claimed:
* E6 has the highest observed GT success (82.0 %).
* E6 significantly improved its own BC initialisation (Holm p = 0.039).

May not be claimed:
* that E6 is better than the expert (p = 0.096);
* that E6 is better than the BC ensemble (p = 0.19);
* that E6 reliably succeeds ≥ 80 % (the Wilson lower bound is 78.6 %);
* that the RL gain is consistent across runs (E6b: p = 0.39; one training seed per configuration).

**Failure outcomes** (gt_success = False; `evidence/failure_outcomes_by_policy.csv`):

| Controller | Never lifted | Pushed before close | No contact | Dropped | No pad pair | Total failed |
|---|---|---|---|---|---|---|
| Scripted | 57 | 38 | 3 | 7 | 0 | 105 |
| E6 | 46 | 38 | 7 | 3 | 3 | 97 |

**Other findings:**
* **Oblique ring:** 0 / 20 for every controller, including the expert.
* **Numerical failures:** 4 of 3,780 episodes, all `object_runaway` in top-down grasps.
  * All four happened after GT success had already been reached, so the unchanged GT metric counts
    them as successes.
  * Counting them as failures changes no conclusion (`analysis/sensitivity_strict_*.csv`).

## 2. Earlier waves (each on its own fresh test set; absolute rates are not comparable across sets)

| Wave | Method (kind) | Learned GT | Scripted on the same episodes | Test set | Source |
|---|---|---|---|---|---|
| W1 | SAC from scratch (autonomous RL) | 0.0 % | 77.8 % | 5.1M, n = 324 | `wave2/report/report.md` §4 |
| W1 | SAC + scripted demos in replay | 10.5 % | 77.8 % | 5.1M, n = 324 | same |
| W2 | Residual SAC on the scripted controller | 49.4 % | 77.8 % | 5.1M, n = 324 | same |
| W3 | Conservative residual SAC (smaller bounds, zero-centred exploration, v4r2 penalty) | 69.1 % | 76.7 % | 6.1M, n = 540 | `wave3/report/report.md` |
| W4 | Curriculum SAC (E1, no demos) | 18.9 % | 77.8 % | 6.6M, n = 540 | `wave4/report/report.md` §4 |
| W4 | BC + SAC fine-tuning (E2b) | 48.7 % | 77.8 % | 6.6M, n = 540 | same |
| W4 | Behaviour cloning | 74.6 % | 77.8 % | 6.6M, n = 540 | same |
| W4 | BC + anchored SAC, grasp-and-lift sub-task (E4) | 78.9 % | 77.8 % (p = 0.53) | 6.6M sub-task | same |
| W5 | BC ensemble / E6 | 80.7 % / 82.0 % | 80.6 % | 6.7M, n = 540 | section 1 |

* **Wave 1 validation** (8.5M, n = 100): SAC scratch 0 %, SAC + demos 8 % [4.1, 15.0], scripted
  75 % (`wave1/report/report.md`). Both arms were stopped by the pre-registered 150k gate.
* **Same policy on two test sets:** BC Wave 4 scored 74.6 % on 6.6M and 77.4 % on 6.7M. This shows
  test-set variation of about 3 pp, which is why only matched comparisons are valid.

## 3. Training budget

* **Wave 5 RL:** E6 221,000 steps (104.8 min), E6b 170,000 steps (78.1 min); stopped by user
  request after passing all gates.
* **Selected checkpoints:** E6 90k and E6b 130k, chosen on DEV_B (8.9M, 200 episodes;
  `wave5/selection/sel_e6/selection_decision.json`).
* **Wave 5 BC:** 8 runs, 74 min in total.
* **Demonstrations:** 600 clean episodes (seeds 3.6M; success 61.5 %) plus 600 DART episodes
  (σ = 0.05, seeds 3.7M; success 65.0 %).
* **Seeds:** one training seed (11) per RL configuration.

## 4. Scripted baseline including the side approach

Source: `outputs/scripted/scripted_baseline_elliptic_imp10_noslip5_dt2.5ms_sr6_arm0.001/summary_by_approach.csv`
(seeds 5.1M, 162 episodes per approach).

| Approach | GT success |
|---|---|
| Top-down | 95.1 % |
| Oblique | 60.5 % |
| Side 90° | 30.2 % |
| Side 90°, feasible plans only (31 episodes, tall objects) | 61.3 % |

**The side approach exists only in the planner and the scripted baseline. No learned policy was
trained or evaluated on side grasps.**
