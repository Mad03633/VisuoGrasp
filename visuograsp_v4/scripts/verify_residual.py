"""Wave 2 Step 1-2 verification of residual control (must pass before any residual training).

A  zero-residual equivalence, step by step: VisuoGraspV4Env + ScriptedGrasp versus
   ResidualGraspEnv with r = 0 in lock-step (executed action, actor observation, reward, flags,
   hand / object state compared bit for bit). DEV seeds 9_200_000+.
B  zero-residual equivalence, episode level, on the 100 independent-validation episodes
   (seeds 8_500_000+) against the stored Wave 1 scripted reference CSV.
C  constraint / quaternion stress test with random, constant and bang-bang residuals.
D  reward audit under residual control (exploitation, premature object movement, large residuals).
E  identical action mapping in training (GTEpisodeInfo wrapper) and evaluation env.
F  actor observation: layout and no dependence on evaluation-only tracker fields.
PRIVILEGED setting throughout (simulator object pose, GT-oracle contacts).

    python -m visuograsp_v4.scripts.verify_residual
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pandas as pd

from visuograsp_v4 import rotations as rot
from visuograsp_v4.config import OUTPUT_ROOT, V4Config

OUT = OUTPUT_ROOT / 'rl' / 'wave2' / 'verification'
RCFG = None          # set in main(): residual version under test (res1 = Wave 2, res2 = Wave 3)
SCRIPTED_REF = OUTPUT_ROOT / 'rl' / 'wave1' / 'independent_validation_scripted' / 'scripted_reference' / 'episodes.csv'


def lockstep_zero(cfg, seeds_per_approach=20, base_seed=9_200_000) -> dict:
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.residual import ResidualGraspEnv
    from visuograsp_v4.scripted import ScriptedGrasp
    env_s = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    env_r = ResidualGraspEnv(cfg, RCFG)
    res = {'episodes': 0, 'steps': 0, 'action_mismatch_steps': 0, 'obs_mismatch_steps': 0, 'reward_mismatch_steps': 0,
           'flag_mismatch_steps': 0, 'state_mismatch_steps': 0, 'outcome_mismatches': 0, 'max_abs_action_diff': 0.0,
           'max_abs_obs_diff': 0.0, 'max_abs_reward_diff': 0.0, 'gt_success_scripted': 0, 'gt_success_zero': 0}
    zero = np.zeros(22, np.float32)
    for ai, ap in enumerate(('top', 'oblique')):
        for ep in range(seeds_per_approach):
            seed = base_seed + ai * 100 + ep
            o_s, _ = env_s.reset(seed=seed, options={'approach': ap})
            o_r, _ = env_r.reset(seed=seed, options={'approach': ap})
            agent = ScriptedGrasp(env_s.sim, env_s.approach)
            assert np.array_equal(o_s, o_r[:o_s.shape[0]]), 'reset observation differs'
            while True:
                a_s = agent.act()
                o_s, r_s, t_s, tr_s, i_s = env_s.step(a_s)
                o_r, r_r, t_r, tr_r, i_r = env_r.step(zero)
                a_r = i_r['executed_action']
                d_a = float(np.abs(a_s - a_r).max())
                d_o = float(np.abs(o_s - o_r[:o_s.shape[0]]).max())
                d_r = abs(r_s - r_r)
                res['max_abs_action_diff'] = max(res['max_abs_action_diff'], d_a)
                res['max_abs_obs_diff'] = max(res['max_abs_obs_diff'], d_o)
                res['max_abs_reward_diff'] = max(res['max_abs_reward_diff'], d_r)
                res['action_mismatch_steps'] += int(not np.array_equal(a_s, a_r))
                res['obs_mismatch_steps'] += int(not np.array_equal(o_s, o_r[:o_s.shape[0]]))
                res['reward_mismatch_steps'] += int(r_s != r_r)
                res['flag_mismatch_steps'] += int((t_s, tr_s) != (t_r, tr_r) or i_s['flags'] != i_r['flags'])
                same_state = (np.array_equal(env_s.sim.data.qpos, env_r.sim.data.qpos)
                              and np.array_equal(env_s.sim.data.qvel, env_r.sim.data.qvel))
                res['state_mismatch_steps'] += int(not same_state)
                res['steps'] += 1
                if t_s or tr_s or t_r or tr_r:
                    break
            ts, tz = env_s.sim.tracker, env_r.sim.tracker
            res['outcome_mismatches'] += int((ts.gt_success, ts.max_lift, ts.dropped) != (tz.gt_success, tz.max_lift, tz.dropped))
            res['gt_success_scripted'] += int(ts.gt_success)
            res['gt_success_zero'] += int(tz.gt_success)
            res['episodes'] += 1
    res['passed'] = all(res[k] == 0 for k in ('action_mismatch_steps', 'obs_mismatch_steps', 'reward_mismatch_steps',
                                               'flag_mismatch_steps', 'state_mismatch_steps', 'outcome_mismatches'))
    return res


def episode_level_zero(cfg) -> tuple[dict, pd.DataFrame]:
    from visuograsp_v4.evaluation import run_policy_episode
    from visuograsp_v4.residual import ResidualGraspEnv
    from visuograsp_v4.scripts.validate_checkpoints import ival_grid
    env = ResidualGraspEnv(cfg, RCFG)
    rows = []
    for seed, f, v, m, ap in ival_grid(cfg.objects.families, ('top', 'oblique'), 10):
        rows.append(run_policy_episode(env, lambda o: np.zeros(22, np.float32), seed, f, v, m, ap))
    z = pd.DataFrame(rows)
    ref = pd.read_csv(SCRIPTED_REF, float_precision='round_trip')   # default parser is off by ~1 ulp
    cols = ['gt_success', 'gt_success_any_phalanx', 'outcome', 'max_lift_cm', 'final_lift_cm', 'episode_return',
            'steps', 'hold_frames_gt', 'sim_failure']
    m = z.merge(ref, on=['seed', 'approach'], suffixes=('_zero', '_scripted'))
    mism = {c: int((m[f'{c}_zero'] != m[f'{c}_scripted']).sum()) for c in cols}
    res = {'paired_episodes': len(m), 'mismatches_per_column': mism, 'gt_success_zero': float(z.gt_success.mean()),
           'gt_success_scripted_reference': float(ref.gt_success.mean()), 'passed': len(m) == 100 and sum(mism.values()) == 0}
    return res, m


def stress(cfg, base_seed=9_210_000) -> tuple[dict, pd.DataFrame]:
    from visuograsp_v4.residual import ResidualGraspEnv
    env = ResidualGraspEnv(cfg, RCFG)
    c = cfg.control
    lo, hi = np.asarray(c.workspace_lo), np.asarray(c.workspace_hi)
    rng = np.random.default_rng(0)
    policies = {'zero': lambda k: np.zeros(22), 'random': lambda k: rng.uniform(-1, 1, 22), 'const_plus': lambda k: np.ones(22),
                'const_minus': lambda k: -np.ones(22), 'bang_bang': lambda k: np.ones(22) * (1 if k % 2 else -1)}
    rows = []
    for pi, (name, pol) in enumerate(policies.items()):
        for ep in range(6):
            ap = ('top', 'oblique')[ep % 2]
            env.reset(seed=base_seed + pi * 100 + ep, options={'approach': ap})
            ctrl = env.sim.ctrl
            v = {'quat_norm_err_max': 0.0, 'workspace_violations': 0, 'tilt_violations': 0, 'clearance_violations': 0,
                 'finger_range_violations': 0, 'nonfinite_obs': 0, 'goal_quat_norm_err_max': 0.0,
                 'post_step_low_below_clearance': 0}
            orig_apply = ctrl.apply_action

            def checked_apply(action, orig_apply=orig_apply, ctrl=ctrl, v=v):
                # the controller's invariant: at command time, with the CURRENT finger curl
                kp = ctrl.hand_local_keypoints()
                flags = orig_apply(action)
                low = ctrl.lowest_point_at(ctrl.target_pos, ctrl.target_quat, kp)
                v['clearance_violations'] += int(low < c.table_clearance - 1e-9 and ctrl.target_pos[2] < hi[2] - 1e-9)
                return flags
            ctrl.apply_action = checked_apply
            k = 0
            while True:
                o, r, term, trunc, info = env.step(np.asarray(pol(k), np.float32))
                k += 1
                q = ctrl.target_quat
                v['quat_norm_err_max'] = max(v['quat_norm_err_max'], abs(float(np.linalg.norm(q)) - 1.0))
                _, _, gq = env.agent.goal_pose()
                v['goal_quat_norm_err_max'] = max(v['goal_quat_norm_err_max'], abs(float(np.linalg.norm(gq)) - 1.0))
                v['workspace_violations'] += int(np.any(ctrl.target_pos < lo - 1e-9) or np.any(ctrl.target_pos > hi + 1e-9))
                v['tilt_violations'] += int(rot.finger_tilt_deg(q) > c.max_tilt_deg + 1e-6)
                # informational: after the physics step the fingers have curled further
                v['post_step_low_below_clearance'] += int(ctrl.lowest_point_at(ctrl.target_pos, q) < c.table_clearance - 1e-6)
                fl, fh = env.sim.scene.finger_lo, env.sim.scene.finger_hi
                v['finger_range_violations'] += int(np.any(ctrl.finger_target < fl - 1e-12) or np.any(ctrl.finger_target > fh + 1e-12))
                v['nonfinite_obs'] += int(not np.all(np.isfinite(o)))
                if term or trunc:
                    break
            del ctrl.apply_action                         # restore the class method
            t = env.sim.tracker
            rows.append({'policy': name, 'hand_table_force_max_n': t.hand_table_force_max, 'seed': base_seed + pi * 100 + ep, 'approach': ap, 'steps': k,
                         'gt_success': bool(t.gt_success), 'sim_failure': bool(t.nonfinite or t.runaway),
                         'clearance_raises': t.flag_counts.get('clearance_raise', 0),
                         'lead_clips': t.flag_counts.get('lead_clip_pos', 0), 'tilt_rejects': t.flag_counts.get('tilt_reject', 0),
                         **v})
    df = pd.DataFrame(rows)
    res = {'episodes': len(df), 'max_target_quat_norm_err': float(df.quat_norm_err_max.max()),
           'max_goal_quat_norm_err': float(df.goal_quat_norm_err_max.max()),
           'workspace_violations': int(df.workspace_violations.sum()), 'tilt_violations': int(df.tilt_violations.sum()),
           'clearance_violations': int(df.clearance_violations.sum()),
           'finger_range_violations': int(df.finger_range_violations.sum()), 'nonfinite_obs': int(df.nonfinite_obs.sum()),
           'sim_failures': int(df.sim_failure.sum())}
    res['passed'] = (res['max_target_quat_norm_err'] < 1e-9 and res['max_goal_quat_norm_err'] < 1e-9
                     and res['workspace_violations'] == res['tilt_violations'] == res['clearance_violations']
                     == res['finger_range_violations'] == res['nonfinite_obs'] == 0)
    return res, df


def reward_audit(cfg, base_seed=9_220_000, per_policy=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    from visuograsp_v4.evaluation import run_policy_episode
    from visuograsp_v4.residual import ResidualGraspEnv
    from visuograsp_v4 import objects as objs
    env = ResidualGraspEnv(cfg, RCFG)
    rng = np.random.default_rng(1)

    def const(idx, val):
        def f(o):
            r = np.zeros(22, np.float32)
            r[idx] = val
            return r
        return f
    policies = {'zero': lambda o: np.zeros(22, np.float32), 'random': lambda o: rng.uniform(-1, 1, 22).astype(np.float32),
                'grasp_higher_max': const([18], 1.0), 'grasp_lower_max': const([18], -1.0),
                'shift_x_max': const([16], 1.0), 'squeeze_fingers': const(list(range(16)), 1.0),
                'loosen_fingers': const(list(range(16)), -1.0), 'rotate_max': const([19, 20, 21], 1.0)}
    rows = []
    fams = cfg.objects.families
    for name, pol in policies.items():
        for ep in range(per_policy):
            seed = base_seed + ep
            fam = fams[ep % len(fams)]
            ap = ('top', 'oblique')[ep % 2]
            r0 = np.random.default_rng(seed)
            variant = str(r0.choice(list(objs.VARIANTS[fam])))
            mode = str(r0.choice(list(objs.RESTING_MODES[fam])))
            rewards = []

            def act(o, pol=pol):
                return pol(o)
            # discounted return needs per-step rewards: wrap step
            orig_step = env.step

            def step(a, orig_step=orig_step):
                out = orig_step(a)
                rewards.append(out[1])
                return out
            env.step = step
            row = run_policy_episode(env, act, seed, fam, variant, mode, ap)
            env.step = orig_step
            row['policy'] = name
            row['disc_return'] = float(sum(0.99 ** k * r for k, r in enumerate(rewards)))
            rows.append(row)
    df = pd.DataFrame(rows)
    summ = df.groupby('policy').agg(episodes=('gt_success', 'size'), gt_success=('gt_success', 'mean'),
                                    return_mean=('episode_return', 'mean'), disc_return_mean=('disc_return', 'mean'),
                                    pushed_before_close=('pushed_before_close', 'mean'),
                                    sim_failures=('sim_failure', 'sum'), median_max_lift_cm=('max_lift_cm', 'median'),
                                    res_abs_mean=('res_abs_mean', 'mean')).reset_index()
    return df, summ


def mapping_identity(cfg, base_seed=9_230_000) -> dict:
    from visuograsp_v4.agents import GTEpisodeInfo
    from visuograsp_v4.residual import ResidualGraspEnv
    tr, ev = GTEpisodeInfo(ResidualGraspEnv(cfg, RCFG)), ResidualGraspEnv(cfg, RCFG)
    steps = mism = 0
    for ep in range(3):
        rng = np.random.default_rng(ep)
        o1, _ = tr.reset(seed=base_seed + ep, options={'approach': ('top', 'oblique')[ep % 2]})
        o2, _ = ev.reset(seed=base_seed + ep, options={'approach': ('top', 'oblique')[ep % 2]})
        mism += int(not np.array_equal(o1, o2))
        while True:
            a = rng.uniform(-1, 1, 22).astype(np.float32)
            o1, r1, t1, u1, i1 = tr.step(a)
            o2, r2, t2, u2, i2 = ev.step(a)
            mism += int(not (np.array_equal(o1, o2) and r1 == r2 and (t1, u1) == (t2, u2)
                             and np.array_equal(i1['executed_action'], i2['executed_action'])))
            steps += 1
            if t1 or u1:
                break
    return {'steps': steps, 'mismatched_steps': mism, 'passed': mism == 0}


def reward_version_check(cfg, base_seed=9_250_000) -> dict:
    """G (v4r2 only): same residual sequence in a v4r2 env and a v4r1 env with the same bounds ->
    identical dynamics; per-step reward difference = -w * mean(r^2) exactly as specified."""
    from dataclasses import replace
    from visuograsp_v4.residual import ResidualGraspEnv
    if RCFG.reward_version != 'v4r2':
        return {'skipped': 'reward version is not v4r2', 'passed': True}
    e2 = ResidualGraspEnv(cfg, RCFG)
    e1 = ResidualGraspEnv(cfg, replace(RCFG, reward_version='v4r1', w_residual=0.0))
    steps = dyn_mism = 0
    max_err = max_pen = 0.0
    for ep in range(4):
        rng = np.random.default_rng(100 + ep)
        e1.reset(seed=base_seed + ep, options={'approach': ('top', 'oblique')[ep % 2]})
        e2.reset(seed=base_seed + ep, options={'approach': ('top', 'oblique')[ep % 2]})
        while True:
            a = (rng.uniform(-1, 1, 22) * (0.0 if ep == 0 else 1.0)).astype(np.float32)   # episode 0: r = 0
            o1, r1, t1, u1, i1 = e1.step(a)
            o2, r2, t2, u2, i2 = e2.step(a)
            dyn_mism += int(not (np.array_equal(o1, o2) and (t1, u1) == (t2, u2)
                                 and np.array_equal(e1.sim.data.qpos, e2.sim.data.qpos)))
            expected = -RCFG.w_residual * float(np.mean(np.square(a.astype(np.float64))))
            max_err = max(max_err, abs((r2 - r1) - expected), abs(i2['reward_terms']['residual'] - expected),
                          abs(i2['reward_v4r1'] - r1))
            max_pen = max(max_pen, -expected)
            steps += 1
            if t1 or u1:
                break
    return {'steps': steps, 'dynamics_mismatch_steps': dyn_mism, 'max_abs_reward_formula_error': max_err,
            'max_penalty_per_step': max_pen, 'w_residual': RCFG.w_residual,
            'passed': dyn_mism == 0 and max_err < 1e-12}


def observation_check(cfg) -> dict:
    from visuograsp_v4.env import OBS_LAYOUT
    from visuograsp_v4.residual import EXTRA_LAYOUT, RES_OBS_DIM, ResidualGraspEnv
    env = ResidualGraspEnv(cfg, RCFG)
    o, _ = env.reset(seed=9_240_000, options={'approach': 'top'})
    for _ in range(5):
        o, *_ = env.step(np.zeros(22, np.float32))
    base_obs = env.base._obs()
    x1 = env._obs(base_obs)
    t = env.sim.tracker                       # mutate evaluation-only fields: the observation must not change
    saved = (t.gt_success, t.gt_hold, t.max_lift, t.height_hold)
    t.gt_success, t.gt_hold, t.max_lift, t.height_hold = True, 59, 0.5, 59
    x2 = env._obs(env.base._obs())
    t.gt_success, t.gt_hold, t.max_lift, t.height_hold = saved
    return {'obs_dim': int(RES_OBS_DIM), 'layout': [list(x) for x in OBS_LAYOUT] + [list(x) for x in EXTRA_LAYOUT],
            'privileged_fields_in_actor_obs': ['object_center_est/axes/half_extents/in_hand_frame (simulator pose)',
                                               'grasp_pos_rel/grasp_rot_err (plan from simulator pose)',
                                               'contact_flags/probs (GT oracle)'],
            'evaluation_only_fields_excluded': ['gt_success', 'hold counters', 'lift tracker', 'table force', 'object velocity'],
            'obs_unchanged_when_tracker_mutated': bool(np.array_equal(x1, x2)),
            'passed': bool(np.array_equal(x1, x2)) and o.shape[0] == RES_OBS_DIM}


def main() -> None:
    import argparse
    import torch
    from visuograsp_v4.residual import RESIDUAL_CONFIGS
    global OUT, RCFG
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='res1', choices=sorted(RESIDUAL_CONFIGS))
    ap.add_argument('--group', default=None, help='output group (default: wave2 for res1, wave3 for res2)')
    args = ap.parse_args()
    RCFG = RESIDUAL_CONFIGS[args.version]
    OUT = OUTPUT_ROOT / 'rl' / (args.group or {'res1': 'wave2', 'res2': 'wave3'}[args.version]) / 'verification'
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    cfg = V4Config()
    t0 = time.time()
    report = {'residual_version': RCFG.version, 'residual_config': RCFG.to_dict()}
    report['A_lockstep_zero_residual'] = lockstep_zero(cfg)
    print('[A]', report['A_lockstep_zero_residual'], flush=True)
    report['B_episode_level_zero_residual'], paired = episode_level_zero(cfg)
    paired.to_csv(OUT / 'B_zero_vs_scripted_ival_paired.csv', index=False)
    print('[B]', report['B_episode_level_zero_residual'], flush=True)
    report['C_stress'], sdf = stress(cfg)
    sdf.to_csv(OUT / 'C_stress_episodes.csv', index=False)
    print('[C]', report['C_stress'], flush=True)
    adf, asum = reward_audit(cfg)
    adf.to_csv(OUT / 'D_reward_audit_episodes.csv', index=False)
    asum.to_csv(OUT / 'D_reward_audit_summary.csv', index=False)
    z = asum[asum.policy == 'zero'].iloc[0]
    exploit = asum[(asum.policy != 'zero') & (asum.gt_success <= z.gt_success) & (asum.disc_return_mean > z.disc_return_mean)]
    succ, fail = adf[adf.gt_success.astype(bool)], adf[~adf.gt_success.astype(bool)]
    report['D_reward_audit'] = {
        'summary': asum.round(4).to_dict(orient='records'),
        'policies_with_higher_return_but_not_higher_success_than_zero': exploit.policy.tolist(),
        'mean_disc_return_success_episodes': float(succ.disc_return.mean()) if len(succ) else None,
        'mean_disc_return_failure_episodes': float(fail.disc_return.mean()) if len(fail) else None,
        'max_disc_return_of_a_failure': float(fail.disc_return.max()) if len(fail) else None,
        'min_disc_return_of_a_success': float(succ.disc_return.min()) if len(succ) else None}
    print('[D]\n', asum.round(3).to_string(index=False), flush=True)
    report['E_mapping_identity'] = mapping_identity(cfg)
    print('[E]', report['E_mapping_identity'], flush=True)
    report['F_observation'] = observation_check(cfg)
    print('[F] passed', report['F_observation']['passed'], 'obs_dim', report['F_observation']['obs_dim'], flush=True)
    report['G_reward_version'] = reward_version_check(cfg)
    print('[G]', report['G_reward_version'], flush=True)
    report['all_required_passed'] = bool(report['G_reward_version']['passed'] and report['A_lockstep_zero_residual']['passed'] and report['B_episode_level_zero_residual']['passed']
                                         and report['C_stress']['passed'] and report['E_mapping_identity']['passed']
                                         and report['F_observation']['passed'])
    report['seconds'] = time.time() - t0
    (OUT / 'verification_report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print('ALL REQUIRED CHECKS PASSED' if report['all_required_passed'] else 'CHECKS FAILED', f'({report["seconds"]:.0f} s)')
    print('written to', OUT)


if __name__ == '__main__':
    main()
