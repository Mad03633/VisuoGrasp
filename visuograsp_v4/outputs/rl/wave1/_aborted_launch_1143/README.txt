Aborted first Wave 1 launch (2026-11-14 11:43:47). Not a result.
sac_scratch_s11: stopped at env step 5,000 (before learning_starts = 10,000, i.e. before any gradient
update) by a FALSE divergence alarm: the Q-health check read the not-yet-existing SB3 loss values as
NaN. Q-values were finite (q_mean -0.37, q_max 0.54). Bug fixed in agents.py (missing loss -> None).
sac_demo_s11: killed manually during demo prefill (would have hit the same false alarm at 5,000 steps).
Both runs were relaunched under the same names with the fixed code.
