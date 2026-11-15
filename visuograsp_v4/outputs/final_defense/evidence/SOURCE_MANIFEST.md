# Source manifest and documented discrepancies

Root: `C:\Users\user\Desktop\AMRL\Project_2_2`.

* No file outside `visuograsp_v4\outputs\final_defense\` was created or changed for this package.
  The only exception is MiKTeX's own configuration: an update check and package auto-install, both
  needed to compile the PDF.
* Frozen results were only read. Their SHA-256 manifest is
  `visuograsp_v4\outputs\rl\wave5\RESULTS_FROZEN_SHA256.txt`; re-checked at the end of the build,
  see `evidence/integrity_check.txt`.

## Claims and sources

| Claim | Source |
|---|---|
| Final Wave 5 numbers, CIs, paired tests, Holm | `visuograsp_v4/outputs/rl/wave5/final_test/final/analysis/{summary,paired_all,decision,integrity}.*` |
| Per-episode outcomes, failure modes | `.../final_test/final/episodes.csv` |
| Reading rule written before results | `.../final_test/final_decision_rule.json` |
| Frozen checkpoints (E6 90k, E6b 130k) | `.../wave5/frozen_policies/manifest.json` (SHA-256) |
| Earlier waves | `visuograsp_v4/outputs/rl/wave{1,2,3,4}/report/report.md` |
| Demonstration statistics, E5/E6 training | `visuograsp_v4/outputs/rl/wave5/report/report.md`; `demos/*/meta.json`; `rl/*/run_config.json`, `finished.json` |
| E6 objective | `visuograsp_v4/wave5.py` (`AWBCSAC.train`) |
| BC loss | `visuograsp_v4/scripts/wave5_bc.py` (sampling ∝ weights, MSE on tanh μ) |
| SAC settings | `visuograsp_v4/config.py` (`SACConfig`), `agents.py` (LayerNorm critics, SB3 n-step buffer) |
| Reward v4r1 | `visuograsp_v4/env.py` (`step`, `_potential`), `config.py` (`RewardConfig`) |
| Observation layout (130-D) | `visuograsp_v4/env.py` (`OBS_LAYOUT`) |
| Wrist / finger action mapping | `visuograsp_v4/controller.py` (`apply_action`), `config.py` (`ControlConfig`) |
| Physics parameters | `visuograsp_v4/config.py`, `README_V4.md` |
| Side-approach scripted results | `visuograsp_v4/outputs/scripted/scripted_baseline_elliptic_imp10_noslip5_dt2.5ms_sr6_arm0.001/summary_by_approach.csv` |
| V3.3 facts and results | `README.md` (root), V3.3 demo videos in `visuograsp_full_outputs/cv_v2/drl_runs_v3_final/videos/v33_final_demos/` |
| Course requirements | `VisuoGrasp_DRL_Grasping_Project_Instructions (1).pdf` (report: 8 pages + appendix; 10-minute talk) |
| Re-rendered demo clips | `final_defense/media/rerendered/rerender_manifest.csv` (all 17 clips reproduce their frozen outcome) |

## Discrepancies found and how they were resolved

1. **`visuograsp_v4/README_V4.md` is out of date.**
   * It says "No RL policy has been trained in V4" and lists RL settings as preparation.
   * That was true on 2026-11-14 before Wave 1. Waves 1–5 trained and evaluated many policies (wave
     reports).
   * The package uses the wave reports and frozen CSVs. The README was not edited (historical file).
2. **The side / lateral approach.**
   * The request lists side grasps among the V4 approach directions.
   * The planner and scripted baseline support side 90° grasps (scripted GT 30.2 %; 61.3 % on
     feasible tall objects).
   * Every learned policy was trained and tested on top-down + oblique only (`approaches=('top','oblique')`).
   * All side-grasp media are labelled "scripted expert".
3. **The camera infrastructure.**
   * V4 contains RGB-D cameras, colour segmentation, PCA object estimation and a D0 geometric
     contact detector (`perception.py`, `obs_mode='vision'`, `contact_source='d0_vision'`).
   * None of the reported V4 policies used them: all use `obs_mode='privileged'` and
     `contact_source='gt_oracle'`.
   * No D2 (CNN–GRU) detector exists for the V4 moving wrist camera.
4. **E6b best checkpoint.** `wave5/HANDOFF.md` first stated "E6b best_model = step 0". A correction
   appended to the same file gives the actual step, 130k, which is the checkpoint used.
5. **E5f epoch-8 value.** `bc/e5f_hist_pw1/STOPPED.txt` quotes DEV_A 0.52, while `dev_evals.csv`
   logs 0.55. The logged value is used.
6. **Numerical failures.**
   * All 4 final-test numerical failures happened after GT success had been reached, so they count
     as successes under the unchanged GT metric.
   * The strict sensitivity analysis (`analysis/sensitivity_strict_*.csv`) is reported alongside.
7. **Wave 2 vs Wave 4 scripted rate.**
   * The scripted controller scored 77.8 % on two different test sets (5.1M, n = 324, and 6.6M,
     n = 540). This is a coincidence of two independent samples, not a copied number.
   * Both values were checked in the reports.
