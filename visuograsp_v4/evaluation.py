"""Run one scripted episode and summarise it with ground-truth metrics (evaluation only)."""
from __future__ import annotations

import math
import time

import numpy as np

from . import rotations as rot
from .scripted import ScriptedGrasp
from .sim import EpisodeSpec, VisuoGraspSimV4


def classify(row: dict) -> str:
    if row['sim_failure']:
        return 'sim_failure'
    if row['gt_success']:
        return 'success_retained' if row['final_lift_cm'] >= 10 else 'success_lost_later'
    if row['object_off_table']:
        return 'object_left_table'
    if row['dropped']:
        return 'lifted_then_dropped'
    if row['height_hold_frames'] >= 60:
        return 'held_without_pad_pair'
    if row['pushed_before_close']:
        return 'object_pushed_before_close'
    if not row['contact_occurred']:
        return 'no_contact'
    return 'partial_lift' if row['max_lift_cm'] >= 3.5 else 'never_lifted'


def run_scripted_episode(sim: VisuoGraspSimV4, spec: EpisodeSpec, approach: str, seed: int,
                         frame_callback=None, scripted_kwargs=None) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    settle = sim.reset(spec, rng)
    agent = ScriptedGrasp(sim, approach, **(scripted_kwargs or {}))
    plan = agent.make_plan()
    start_com = sim.object_com()
    start_quat = sim.object_quat()
    pushed = tipped = False
    reached_grasp_err = float('nan')
    close_contacts = None
    steps = 0
    for steps in range(1, sim.cfg.task.max_steps + 1):
        phase_before = agent.phase
        a = agent.act()
        out = sim.step(a)
        if frame_callback is not None:
            frame_callback(sim, agent, out, steps)
        if phase_before in ('transit', 'descend', 'approach') and agent.phase in ('close',):
            reached_grasp_err = float(np.linalg.norm(sim.ctrl.hand_pos() - plan.grasp_pos))
            disp = float(np.linalg.norm(sim.object_com()[:2] - start_com[:2]))
            pushed = disp > 0.02
            tipped = rot.angle_between(sim.object_quat(), start_quat) > math.radians(20)
        if phase_before == 'close' and agent.phase == 'lift':
            close_contacts = out['contacts']['pads'].copy()
        if not out['finite'] or agent.phase == 'done':
            break
    t = sim.tracker
    sim_failure = bool(t.nonfinite or t.runaway or any(k in t.warnings for k in
                                                       ('mjWARN_BADQACC', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL')))
    row = dict(
        seed=seed, family=spec.family, variant=spec.variant, scale=spec.scale,
        resting_mode=spec.resting_mode, object_yaw_deg=math.degrees(spec.yaw),
        mass=spec.mass, friction=spec.friction, approach=approach,
        azimuth_deg=math.degrees(plan.azimuth), pinch_width_cm=100 * plan.pinch_width,
        hand_start_tilt_deg=rot.finger_tilt_deg(spec.hand_quat),
        hand_start_offset_cm=100 * float(np.linalg.norm(np.asarray(spec.hand_pos[:2]) - start_com[:2])),
        grasp_tilt_deg=rot.finger_tilt_deg(plan.grasp_quat), plan_notes=';'.join(plan.notes),
        plan_feasible=plan.feasible,
        settle_valid=settle['valid'], settle_attempts=settle['attempts'],
        reached_transit=agent.reached.get('transit'), reached_pregrasp=agent.reached.get('descend'),
        reached_grasp=agent.reached.get('approach'), grasp_pose_err_cm=100 * reached_grasp_err,
        pushed_before_close=pushed, tipped_before_close=tipped,
        contact_occurred=t.any_pad_contact, any_hand_object_contact=t.any_hand_object_contact,
        thumb_contact=t.thumb_contact, max_pads_in_contact=t.max_pads_in_contact,
        pads_after_close=''.join(str(int(v)) for v in close_contacts) if close_contacts is not None else '',
        max_lift_cm=100 * t.max_lift, final_lift_cm=100 * t.final_lift,
        hold_frames_gt=t.gt_hold_max, height_hold_frames=t.height_hold_max,
        gt_success=t.gt_success, gt_success_any_phalanx=t.gt_success_any,
        hold_frames_any_phalanx=t.gt_hold_any_max, dropped=t.dropped,
        hand_table_force_max_n=t.hand_table_force_max,
        object_table_min_dist_mm=1000 * t.obj_table_min_dist,
        track_err_pos_p95_mm=1000 * float(np.percentile(t.track_err_pos, 95)) if t.track_err_pos else float('nan'),
        track_err_rot_p95_deg=math.degrees(float(np.percentile(t.track_err_rot, 95))) if t.track_err_rot else float('nan'),
        clearance_raises=t.flag_counts.get('clearance_raise', 0), lead_clips=t.flag_counts.get('lead_clip_pos', 0),
        tilt_rejects=t.flag_counts.get('tilt_reject', 0), warnings=';'.join(f'{k}:{v}' for k, v in t.warnings.items()),
        sim_failure=sim_failure, object_runaway=t.runaway, object_off_table=t.off_table,
        max_object_speed_mps=t.max_object_speed,
        steps=steps, final_phase=agent.phase, wall_s=time.time() - t0,
    )
    row['outcome'] = classify(row)
    return row


def _sim_failure(t) -> bool:
    return bool(t.nonfinite or t.runaway or any(k in t.warnings for k in
                                                ('mjWARN_BADQACC', 'mjWARN_BADQPOS', 'mjWARN_BADQVEL')))


def run_policy_episode(env, act_fn, seed: int, family: str, variant: str, resting_mode: str,
                       approach: str, on_reset=None) -> dict:
    """Run one episode of a policy through the Gymnasium env on the SAME initial state as
    run_scripted_episode for the same seed (paired evaluation; verified by a smoke test).
    ``act_fn(obs) -> action``; ``on_reset(env)`` is called after reset (e.g. to build a scripted
    agent for the new episode). Ground truth is read from the tracker only (evaluation)."""
    t0 = time.time()
    obs, reset_info = env.reset(seed=seed, options={'family': family, 'variant': variant,
                                                    'resting_mode': resting_mode, 'approach': approach})
    if on_reset is not None:
        on_reset(env)
    sim = env.sim
    start_com = sim.object_com()
    pushed = False
    ret = 0.0
    steps = 0
    while True:
        obs, r, term, trunc, info = env.step(act_fn(obs))
        ret += r
        steps += 1
        if sim.object_lift() < 0.01 and float(np.linalg.norm(sim.object_com()[:2] - start_com[:2])) > 0.02:
            pushed = True
        if term or trunc:
            break
    t = sim.tracker
    spec = sim.spec
    row = dict(
        seed=seed, family=family, variant=variant, scale=spec.scale, resting_mode=resting_mode,
        mass=spec.mass, friction=spec.friction, approach=approach, plan_feasible=env.plan.feasible,
        settle_valid=bool(reset_info['settle']['valid']),
        pushed_before_close=pushed, contact_occurred=t.any_pad_contact, thumb_contact=t.thumb_contact,
        max_pads_in_contact=t.max_pads_in_contact, max_lift_cm=100 * t.max_lift, final_lift_cm=100 * t.final_lift,
        hold_frames_gt=t.gt_hold_max, height_hold_frames=t.height_hold_max, gt_success=t.gt_success,
        gt_success_any_phalanx=t.gt_success_any, hold_frames_any_phalanx=t.gt_hold_any_max, dropped=t.dropped,
        hand_table_force_max_n=t.hand_table_force_max, object_table_min_dist_mm=1000 * t.obj_table_min_dist,
        track_err_pos_p95_mm=1000 * float(np.percentile(t.track_err_pos, 95)) if t.track_err_pos else float('nan'),
        clearance_raises=t.flag_counts.get('clearance_raise', 0), tilt_rejects=t.flag_counts.get('tilt_reject', 0),
        warnings=';'.join(f'{k}:{v}' for k, v in t.warnings.items()), sim_failure=_sim_failure(t),
        object_runaway=t.runaway, object_off_table=t.off_table, max_object_speed_mps=t.max_object_speed,
        signal_success=bool(info.get('signal_success', False)), episode_return=ret, steps=steps,
        terminated=bool(term), truncated=bool(trunc), wall_s=time.time() - t0,
    )
    extra = getattr(env, 'episode_stats', None)     # residual env: correction magnitudes, phase timing
    if callable(extra):
        row.update(extra())
    row['outcome'] = classify(row)
    return row
