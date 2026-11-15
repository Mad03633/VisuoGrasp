"""Phase-6 smoke tests for VisuoGrasp V4 (no pytest dependency; about 2-4 minutes on CPU).

    python -m visuograsp_v4.scripts.run_smoke_tests            # all tests
    python -m visuograsp_v4.scripts.run_smoke_tests --skip-sb3 # without the SB3 interface check

Writes outputs/smoke/<timestamp>/results.json and log.txt. Exit code 1 if any test fails.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime

import mujoco
import numpy as np

from visuograsp_v4 import objects as objs
from visuograsp_v4 import rotations as rot
from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.env import OBS_DIM, PRIVILEGED_DIM, VisuoGraspV4Env, obs_slices
from visuograsp_v4.evaluation import run_scripted_episode
from visuograsp_v4.perception import RGBDPerceptionV4
from visuograsp_v4.sim import VisuoGraspSimV4

CFG = V4Config()
RESULTS: list = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def high_hand_sim(family='box', variant='cube', mode='upright', seed=1):
    """Sim with the hand placed high above the object (free space for motion tests)."""
    sim = VisuoGraspSimV4(CFG)
    rng = np.random.default_rng(seed)
    spec = sim.sample_spec(rng, family, variant, mode, scale=1.05)
    spec.hand_pos = (0.0, 0.0, 0.30)
    spec.hand_quat = tuple(rot.IDENTITY)
    sim.reset(spec, rng)
    return sim


def idle(n):
    return [np.zeros(22, np.float32)] * n


# ---------------------------------------------------------------------------- tests
def test_model_loads():
    sim = VisuoGraspSimV4(CFG)
    shapes = set()
    for fam in CFG.objects.families:
        for var in objs.VARIANTS[fam]:
            for scale in (min(CFG.objects.scale_bins), max(CFG.objects.scale_bins)):
                s = sim.cache.get(fam, var, scale)
                m = s.model
                shapes.add((m.nq, m.nv, m.nu, m.nmocap, m.neq))
                check(m.eq_type[0] == mujoco.mjtEq.mjEQ_WELD, 'equality 0 must be a weld')
    check(shapes == {(30, 28, 16, 1, 1)}, f'unexpected model sizes {shapes}')
    return {'models_compiled': sim.cache.compilations, 'nq_nv_nu_nmocap_neq': sorted(shapes)}


def test_dof_structure():
    sim = high_hand_sim()
    m, s = sim.model, sim.scene
    jt = mujoco.mjtJoint
    hand_joint = m.joint('hand_free')
    check(hand_joint.type[0] == jt.mjJNT_FREE, 'hand must be a free body (6 DoF)')
    hinge = [m.joint(f'f{i}_j{j}') for i in range(4) for j in range(4)]
    check(all(j.type[0] == jt.mjJNT_HINGE for j in hinge), 'finger joints must be hinges')
    check(len(hinge) == 16 and m.nu == 16, 'expected 16 finger actuators')
    env = VisuoGraspV4Env(CFG)
    check(env.action_space.shape == (22,), 'action dim must be 22')
    env.close()
    return {'hand_dof': 6, 'finger_hinges': 16, 'physical_actuators': int(m.nu), 'mocap_targets': int(m.nmocap),
            'action_dim': 22, 'finger_ranges_rad': m.jnt_range[s.finger_joints].tolist()[:4]}


def test_wrist_rotation_changes_orientation():
    out = {}
    for axis_idx, name in enumerate(('roll_x', 'pitch_y', 'yaw_z')):
        sim = high_hand_sim(seed=10 + axis_idx)
        q0 = sim.ctrl.hand_quat()
        a = np.zeros(22)
        a[19 + axis_idx] = 1.0
        for _ in range(20):                          # 20 x 0.06 rad = 1.2 rad commanded
            sim.step(a)
        for z in idle(10):
            sim.step(z)
        q1 = sim.ctrl.hand_quat()
        r = rot.diff_local(q1, q0)
        ang = float(np.linalg.norm(r))
        axis = r / max(ang, 1e-9)
        check(abs(ang - 1.2) < 0.05, f'{name}: rotated {ang:.3f} rad, commanded 1.2')
        check(abs(axis[axis_idx]) > 0.99, f'{name}: rotation axis {axis}')
        out[name] = {'commanded_rad': 1.2, 'measured_rad': round(ang, 4), 'axis_local': np.round(axis, 4).tolist(),
                     'track_err_rot_deg_final': round(math.degrees(rot.angle_between(q1, sim.ctrl.target_quat)), 3)}
    return out


def test_fingertips_follow_rotation():
    sim = high_hand_sim(seed=20)
    s = sim.scene

    def local_tips():
        R = sim.data.xmat[s.hand_body].reshape(3, 3)
        return (sim.fingertips() - sim.data.xpos[s.hand_body]) @ R
    w0, l0 = sim.fingertips(), local_tips()
    a = np.zeros(22)
    a[19] = 1.0
    a[21] = 0.5
    for _ in range(25):
        sim.step(a)
    for z in idle(10):
        sim.step(z)
    w1, l1 = sim.fingertips(), local_tips()
    world_move = float(np.linalg.norm(w1 - w0, axis=1).min())
    local_change = float(np.linalg.norm(l1 - l0, axis=1).max())
    check(world_move > 0.03, f'fingertips barely moved in world: {world_move}')
    check(local_change < 0.002, f'fingertips moved relative to hand: {local_change}')
    return {'min_world_displacement_m': round(world_move, 4), 'max_hand_frame_change_m': round(local_change, 6)}


def test_bounded_control_and_response():
    sim = high_hand_sim(seed=30)
    c, cfg = sim.ctrl, CFG.control
    rng = np.random.default_rng(0)
    max_dp = max_dr = 0.0
    qnorm_err = 0.0
    for _ in range(80):
        p0, q0 = c.target_pos.copy(), c.target_quat.copy()
        sim.step(rng.uniform(-1, 1, 22) * 3.0)       # out-of-range actions must be clipped
        max_dp = max(max_dp, float(np.linalg.norm(c.target_pos - p0)))
        max_dr = max(max_dr, rot.angle_between(c.target_quat, q0))
        qnorm_err = max(qnorm_err, abs(float(np.linalg.norm(c.target_quat)) - 1.0),
                        abs(float(np.linalg.norm(sim.data.mocap_quat[sim.scene.mocap_id])) - 1.0))
        check(np.all(c.finger_target >= sim.scene.finger_lo - 1e-12) and np.all(c.finger_target <= sim.scene.finger_hi + 1e-12),
              'finger targets out of joint range')
    check(max_dp <= cfg.wrist_trans_max * math.sqrt(3) + 1e-9, f'translation step {max_dp}')
    check(max_dr <= cfg.wrist_rot_max * math.sqrt(3) + 1e-9, f'rotation step {max_dr}')
    check(qnorm_err < 1e-9, f'quaternion norm error {qnorm_err}')
    bad = np.full(22, np.nan)
    flags = sim.step(bad)['flags']
    check(flags['nonfinite_action'] and np.all(np.isfinite(sim.data.qpos)), 'NaN action not handled')
    # wrist response: +x translation 8 cm
    sim2 = high_hand_sim(seed=31)
    p_start = sim2.ctrl.hand_pos()
    a = np.zeros(22)
    a[16] = 1.0
    for _ in range(8):
        sim2.step(a)
    for z in idle(8):
        sim2.step(z)
    moved = sim2.ctrl.hand_pos() - p_start
    check(abs(moved[0] - 0.08) < 0.004 and abs(moved[1]) < 0.003 and abs(moved[2]) < 0.003, f'wrist moved {moved}')
    # finger response, one finger at a time (closing all four in free space makes fingers 1-3
    # collide near the hand axis, which is correct physics but not a response test)
    dq = np.zeros(16)
    for f in range(4):
        simf = high_hand_sim(seed=32 + f)
        q_before = simf.data.qpos[simf.scene.finger_qadr].copy()
        a = np.zeros(22)
        a[4 * f:4 * f + 4] = 1.0
        for _ in range(8):                            # target 8 x 0.055 = 0.44 rad per joint
            simf.step(a)
        for z in idle(10):
            simf.step(z)
        dq[4 * f:4 * f + 4] = (simf.data.qpos[simf.scene.finger_qadr] - q_before)[4 * f:4 * f + 4]
    check(np.all(np.abs(dq - 0.44) < 0.04), f'finger joints did not track 0.44 rad: {np.round(dq, 3)}')
    return {'max_target_step_m': round(max_dp, 5), 'max_target_rot_rad': round(max_dr, 5),
            'quat_norm_err': qnorm_err, 'wrist_x_response_m': np.round(moved, 4).tolist(),
            'finger_dq_min_rad': round(float(dq.min()), 3)}


def test_cameras_after_rotation():
    sim = high_hand_sim(seed=40)
    s, d = sim.scene, sim.data
    side_before = d.cam_xpos[s.side_cam].copy()
    a = np.zeros(22)
    a[20] = 1.0                                       # pitch the hand
    a[16] = -0.5
    for _ in range(25):
        sim.step(a)
    mujoco.mj_forward(sim.model, d)
    R = d.xmat[s.hand_body].reshape(3, 3)
    expected = d.xpos[s.hand_body] + R @ np.array([0.0, 0.0, -0.024])
    pos_err = float(np.linalg.norm(d.cam_xpos[s.wrist_cam] - expected))
    rot_err = float(np.abs(d.cam_xmat[s.wrist_cam].reshape(3, 3) - R).max())
    check(pos_err < 1e-6 and rot_err < 1e-6, f'wrist camera not attached: {pos_err}, {rot_err}')
    check(np.allclose(d.cam_xpos[s.side_cam], side_before), 'side camera moved')
    per = RGBDPerceptionV4(s, 128, seed=0)
    stats = {}
    for cam in ('wrist', 'side'):
        rgb, depth = per.frame(cam)
        check(np.isfinite(depth).all() and rgb.std() > 1.0, f'{cam} image invalid')
        stats[cam] = {'rgb_std': round(float(rgb.std()), 2), 'object_pixels': int(per.segment(rgb).sum())}
    check(stats['side']['object_pixels'] > 20, 'object not visible in side camera')
    return {'wrist_cam_pos_err_m': pos_err, 'wrist_cam_rot_err': rot_err,
            'finger_tilt_deg_after': round(rot.finger_tilt_deg(sim.ctrl.hand_quat()), 1), 'images': stats}


def test_object_resting_poses():
    sim = VisuoGraspSimV4(CFG)
    rows, worst_pen = 0, 0.0
    for fam in CFG.objects.families:
        for var in objs.VARIANTS[fam]:
            for mode in objs.RESTING_MODES[fam]:
                for scale in (min(CFG.objects.scale_bins), max(CFG.objects.scale_bins)):
                    rng = np.random.default_rng(rows)
                    spec = sim.sample_spec(rng, fam, var, mode, scale=scale)
                    info = sim.reset(spec, rng)
                    check(info['valid'], f'{fam}/{var}/{mode}/{scale}: settle invalid {info}')
                    check(info['initial_hand_object_contacts'] == 0, 'object starts inside the hand')
                    worst_pen = min(worst_pen, info['settle_min_dist'])
                    rows += 1
    return {'poses_checked': rows, 'worst_object_table_dist_mm': round(1000 * worst_pen, 3),
            'compilations': sim.cache.compilations}


def test_random_rollouts_finite():
    out = {}
    for k, fam in enumerate(CFG.objects.families):
        env = VisuoGraspV4Env(CFG, obs_mode='privileged', contact_source='gt_oracle', families=(fam,))
        obs, _ = env.reset(seed=100 + k)
        rng = np.random.default_rng(k)
        max_obj_speed, min_dist, steps = 0.0, 1.0, 0
        for _ in range(150):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, 22).astype(np.float32))
            steps += 1
            check(np.all(np.isfinite(obs)) and math.isfinite(r), f'{fam}: non-finite obs/reward')
            v = info['privileged']['object_linvel']
            max_obj_speed = max(max_obj_speed, float(np.linalg.norm(v)))
            min_dist = min(min_dist, env.sim.tracker.obj_table_min_dist)
            if term or trunc:
                obs, _ = env.reset(seed=200 + k)
        bad = [w for w in env.sim.tracker.warnings if w in ('mjWARN_BADQACC', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL')]
        check(not bad, f'{fam}: MuJoCo warnings {bad}')
        check(max_obj_speed < 3.0, f'{fam}: object exploded, speed {max_obj_speed}')
        check(min_dist > -0.005, f'{fam}: object penetrated the table by {min_dist}')
        out[fam] = {'steps': steps, 'max_object_speed_mps': round(max_obj_speed, 3),
                    'min_object_table_dist_mm': round(1000 * min_dist, 3)}
        env.close()
    return out


def test_contact_rich_random_rollouts():
    """Scripted approach to the grasp pose, then 120 random 22-D actions next to the object.
    (Free-space random rollouts above rarely touch the object; this one forces contact.)"""
    from visuograsp_v4.scripted import ScriptedGrasp
    sim = VisuoGraspSimV4(CFG)
    out = {}
    for k, (fam, approach) in enumerate((('box', 'top'), ('cylinder', 'side'), ('bottle', 'oblique'),
                                         ('ring', 'top'), ('l_shape', 'oblique'), ('box', 'side'))):
        rng = np.random.default_rng(700 + k)
        spec = sim.sample_spec(rng, fam)
        sim.reset(spec, rng)
        agent = ScriptedGrasp(sim, approach)
        for _ in range(200):
            sim.step(agent.act())
            if agent.phase in ('close', 'lift', 'hold', 'done'):
                break
        max_speed, contact_steps = 0.0, 0
        for _ in range(120):
            o = sim.step(rng.uniform(-1, 1, 22))
            v = sim.data.qvel[sim.scene.obj_vadr:sim.scene.obj_vadr + 6]
            max_speed = max(max_speed, float(np.linalg.norm(v[:3])))
            contact_steps += int(o['contacts']['hand_object'])
            check(o['finite'], f'{fam}: non-finite state')
        t = sim.tracker
        bad = [w for w in t.warnings if w in ('mjWARN_BADQACC', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL')]
        check(not bad, f'{fam}: MuJoCo warnings {bad}')
        check(max_speed < 3.0, f'{fam}: object speed {max_speed} m/s')
        check(t.obj_table_min_dist > -0.005, f'{fam}: table penetration {t.obj_table_min_dist}')
        out[f'{fam}/{approach}'] = {'hand_object_contact_steps': contact_steps, 'max_object_speed_mps': round(max_speed, 3),
                                    'min_object_table_dist_mm': round(1000 * t.obj_table_min_dist, 3),
                                    'max_hand_table_force_n': round(t.hand_table_force_max, 2)}
    check(sum(v['hand_object_contact_steps'] for v in out.values()) > 100, 'rollouts did not produce contact')
    return out


def test_top_and_side_grasp_attempts():
    sim = VisuoGraspSimV4(CFG)
    out = {}
    for label, fam, var, mode, approach, seed in (('top_box', 'box', 'cube', 'upright', 'top', 6_000_001),
                                                    ('oblique_bottle', 'bottle', 'std', 'upright', 'oblique', 6_000_002),
                                                    ('side_bottle', 'bottle', 'std', 'upright', 'side', 6_000_003),
                                                    ('side_box_tall', 'box', 'tall', 'upright', 'side', 6_000_004)):
        rng = np.random.default_rng(seed)
        spec = sim.sample_spec(rng, fam, var, mode, scale=1.05)
        row = run_scripted_episode(sim, spec, approach, seed)
        check(not row['sim_failure'], f'{label}: simulation failure')
        check(row['final_phase'] == 'done', f'{label}: attempt did not complete ({row["final_phase"]})')
        check(row['reached_grasp'], f'{label}: grasp pose not reached')
        check(abs(row['grasp_tilt_deg'] - {'top': 0, 'oblique': 45, 'side': 90}[approach]) < 1.0, 'wrong grasp tilt')
        out[label] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                      if k in ('approach', 'grasp_tilt_deg', 'reached_grasp', 'contact_occurred', 'max_lift_cm',
                               'hold_frames_gt', 'gt_success', 'gt_success_any_phalanx', 'outcome',
                               'hand_table_force_max_n', 'track_err_pos_p95_mm', 'track_err_rot_p95_deg')}
    return out


def test_gym_env_api():
    from stable_baselines3.common.env_checker import check_env
    out = {}
    for mode, contact in (('privileged', 'gt_oracle'), ('vision', 'd0_vision'), ('privileged', 'none')):
        env = VisuoGraspV4Env(CFG, obs_mode=mode, contact_source=contact)
        check_env(env, warn=True, skip_render_check=True)
        obs, info = env.reset(seed=5, options={'family': 'box', 'variant': 'cube', 'resting_mode': 'upright',
                                               'approach': 'top'})
        check(obs.shape == (OBS_DIM,) and obs.dtype == np.float32, 'obs shape/dtype')
        check(env.privileged_state().shape == (PRIVILEGED_DIM,), 'privileged state shape')
        t0 = time.time()
        for _ in range(20):
            obs, r, term, trunc, info = env.step(np.zeros(22, np.float32))
        out[f'{mode}/{contact}'] = {'obs_dim': OBS_DIM, 'privileged_dim': PRIVILEGED_DIM,
                                    'steps_per_s': round(20 / (time.time() - t0), 1),
                                    'object_visible': float(obs[obs_slices()['object_visible']][0])}
        env.close()
    # separation: with contact_source='none' the actor never sees contact even when GT contact exists
    env = VisuoGraspV4Env(CFG, obs_mode='privileged', contact_source='none')
    sim_env = env
    env.reset(seed=6, options={'family': 'box', 'variant': 'cube', 'resting_mode': 'upright', 'approach': 'top'})
    from visuograsp_v4.scripted import ScriptedGrasp
    agent = ScriptedGrasp(sim_env.sim, 'top')
    saw_gt = False
    sl = obs_slices()
    for _ in range(160):
        obs, r, term, trunc, info = env.step(agent.act())
        saw_gt |= bool(info['privileged']['gt_pad_contacts'].sum() > 0)
        check(not np.any(obs[sl['contact_flags']]) and not np.any(obs[sl['contact_probs']]), 'GT contact leaked into obs')
        if term or trunc:
            break
    check(saw_gt, 'test needs GT contact to occur')
    env.close()
    out['contact_none_separation'] = {'gt_contact_occurred': saw_gt, 'actor_contact_channels_all_zero': True}
    return out


def test_vision_estimate_matches_object():
    env = VisuoGraspV4Env(CFG, obs_mode='vision', contact_source='d0_vision')
    errs, med = {}, {}
    for k in range(20):
        fam = CFG.objects.families[k % len(CFG.objects.families)]
        env.reset(seed=300 + k, options={'family': fam, 'approach': 'top'})
        est = env.estimate
        true = env.sim.object_com()
        errs.setdefault(fam, []).append(100 * float(np.linalg.norm(est['center'] - true)))
        med.setdefault(fam, []).append(100 * float(np.linalg.norm(est['median_center'] - true)))
    env.close()
    worst = max(max(v) for v in errs.values())
    check(worst < 3.0, f'vision OBB centre error too large (cm): {errs}')
    return {'resets': 20, 'obb_center_err_cm_mean': {f: round(float(np.mean(v)), 2) for f, v in errs.items()},
            'obb_center_err_cm_max': round(worst, 2),
            'median_center_err_cm_mean': {f: round(float(np.mean(v)), 2) for f, v in med.items()}}


def test_eval_pairing():
    """env.reset(seed, options) must reproduce the scripted-baseline initial state exactly."""
    env = VisuoGraspV4Env(CFG, obs_mode='privileged', contact_source='gt_oracle')
    sim = VisuoGraspSimV4(CFG)
    worst = 0.0
    for k, (fam, var, mode) in enumerate((('box', 'cube', 'upright'), ('bottle', 'std', 'lying'),
                                          ('l_shape', 'long', 'standing'), ('ring', 'wide', 'flat'))):
        seed = 5_100_000 + 37 * k
        rng = np.random.default_rng(seed)
        spec = sim.sample_spec(rng, fam, var, mode)
        sim.reset(spec, rng)
        env.reset(seed=seed, options={'family': fam, 'variant': var, 'resting_mode': mode, 'approach': 'top'})
        d_obj = float(np.linalg.norm(env.sim.data.qpos - sim.data.qpos))
        check(env.sim.spec.mass == spec.mass and env.sim.spec.scale == spec.scale, 'spec differs')
        worst = max(worst, d_obj)
    env.close()
    check(worst < 1e-12, f'initial states differ: {worst}')
    return {'cases': 4, 'max_qpos_difference': worst}


def test_no_recompilation():
    env = VisuoGraspV4Env(CFG, families=('box',))
    for k in range(40):
        env.reset(seed=400 + k)
    n = env.sim.cache.compilations
    possible = len(objs.VARIANTS['box']) * len(CFG.objects.scale_bins)
    env.close()
    check(n <= possible, f'{n} compilations for 40 resets')
    return {'resets': 40, 'compilations': n, 'max_distinct_models': possible}


def test_sb3_sac_interface():
    from stable_baselines3 import SAC
    env = VisuoGraspV4Env(CFG, obs_mode='privileged', contact_source='none')
    model = SAC('MlpPolicy', env, learning_starts=100, buffer_size=2000, batch_size=64, verbose=0, seed=0,
                device='cpu', policy_kwargs={'net_arch': [64, 64]})
    t0 = time.time()
    model.learn(total_timesteps=300)
    env.close()
    return {'timesteps': 300, 'seconds': round(time.time() - t0, 1), 'note': 'interface check only, not training'}


def test_v33_untouched():
    from visuograsp_v4.scripts import integrity_manifest as im
    import json as _json
    baseline = _json.loads(im.BASELINE.read_text(encoding='utf-8'))
    res = im.compare(baseline, im.build_manifest())
    check(res['untouched'] and not res['added_outside_v4'], f'V3.3 tree changed: {res}')
    return {k: (len(v) if isinstance(v, list) else v) for k, v in res.items()}


TESTS = [test_model_loads, test_dof_structure, test_wrist_rotation_changes_orientation,
         test_fingertips_follow_rotation, test_bounded_control_and_response, test_cameras_after_rotation,
         test_object_resting_poses, test_random_rollouts_finite, test_contact_rich_random_rollouts,
         test_top_and_side_grasp_attempts,
         test_gym_env_api, test_vision_estimate_matches_object, test_eval_pairing, test_no_recompilation,
         test_sb3_sac_interface, test_v33_untouched]


def main() -> int:
    # MuJoCo writes MUJOCO_LOG.TXT into the working directory on warnings: keep it inside V4 outputs
    import os
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-sb3', action='store_true')
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    out = OUTPUT_ROOT / 'smoke' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    log = []
    failed = 0
    for test in TESTS:
        if args.skip_sb3 and test is test_sb3_sac_interface:
            continue
        if args.only and test.__name__ not in args.only:
            continue
        t0 = time.time()
        try:
            details = test()
            status = 'PASS'
        except Exception as exc:                      # noqa: BLE001 - report every failure
            details = {'error': repr(exc), 'traceback': traceback.format_exc()}
            status = 'FAIL'
            failed += 1
        dt = time.time() - t0
        RESULTS.append({'test': test.__name__, 'status': status, 'seconds': round(dt, 2), 'details': details})
        line = f'[{status}] {test.__name__} ({dt:.1f} s)'
        print(line, flush=True)
        log.append(line)
        if status == 'FAIL':
            print(details['traceback'], flush=True)
    (out / 'results.json').write_text(json.dumps({'config': CFG.to_dict(), 'mujoco': mujoco.__version__,
                                                  'results': RESULTS}, indent=1, default=str), encoding='utf-8')
    (out / 'log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print(f'{len(RESULTS) - failed}/{len(RESULTS)} passed; results in {out}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
