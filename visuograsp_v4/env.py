"""Gymnasium interface for VisuoGrasp V4 (22-D action, 130-D actor observation).

Actor observation vs privileged state
-------------------------------------
``obs_mode='vision'``     object estimate from the RGB-D point cloud (no simulator object state)
``obs_mode='privileged'`` object estimate replaced by the exact simulator pose/geometry
                          (teacher / oracle setting; must be labelled PRIVILEGED in results)
``contact_source``        'none' | 'd0_vision' (geometric, camera-based) | 'gt_oracle'
                          ('gt_oracle' = simulator contact, the PRIVILEGED M0 baseline)
Simulator-only quantities (true object pose and velocity, true contacts and forces, lift
height, table force) are returned separately by ``privileged_state()`` and in
``info['privileged']``; they are never part of the actor observation.

Reward v4r1 (default, config.RewardConfig; identical for all arms)
------------------------------------------------------------------
Phi(s) = reach + align + contact + lift + push                       (Phi = 0 at a terminal state)
  reach   = 2 (1 - clip(cost / 0.35)),  cost = |v| + 1.5 |v_lat| clip(1 - v_ax / 0.10, 0, 1)
            v = palm - planned grasp pose, both relative to the object estimate; v_ax / v_lat are
            its components along / across the approach axis (draws the hand in along the axis)
  align   = 1 (1 - rot_err / pi)                 rotation to the planned grasp orientation
  contact = 1 (0.5 pads/4 + 0.5 [thumb & other])  contact signal of contact_source (0 for 'none')
  lift    = 3 clip(h / 0.10)                     h: COM height gain (PRIVILEGED, as in V3.3)
  push    = -1 clip(dxy / 5 cm) while h < 2 cm    dxy: object displacement (PRIVILEGED)
r = gamma Phi(s') - Phi(s) + 0.10 [h >= 0.10 & opposition] + 10 once at signal success
  - 3 once on drop - 0.02 min(1, F_table / 10 N) (PRIVILEGED) - 0.0005 |a|^2 - 0.0005 |a - a_prev|^2
  - 5 on a numerical failure
Contact and lift as potentials (not per-step bonuses) remove the "touch but never lift" optimum
of v4r0 (reward audit: discounted value 1.70 vs 0.00 for doing nothing in v4r0, -0.12 in v4r1).
Termination: drop, object off the table, numerical failure; success does NOT terminate in v4r1
(terminating would forfeit the per-step hold reward). Truncation: 320 steps (16 s at 20 Hz).
v4r0 (first V4 reward) is kept selectable for reproducibility.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rotations as rot
from .config import V4Config
from .grasp_planner import GeomSupport, OBBSupport, plan_grasp
from .perception import RGBDPerceptionV4, TemporalConfirmation
from .sim import VisuoGraspSimV4

OBS_LAYOUT = (
    ('finger_qpos_norm', 16), ('finger_qvel', 16), ('finger_target_norm', 16),
    ('wrist_pos', 3), ('wrist_rot6d', 6), ('wrist_linvel', 3), ('wrist_angvel_local', 3),
    ('fingertips_world', 12),
    ('object_center_est', 3), ('object_axes_est', 6), ('object_half_extents_est', 3),
    ('object_in_hand_frame', 3),
    ('approach_cmd', 3), ('grasp_pos_rel', 3), ('grasp_rot_err', 3),
    ('contact_flags', 4), ('contact_probs', 4),
    ('prev_action', 22), ('object_visible', 1),
)
OBS_DIM = sum(n for _, n in OBS_LAYOUT)
PRIVILEGED_LAYOUT = (('object_pos', 3), ('object_quat', 4), ('object_linvel', 3), ('object_angvel', 3),
                     ('gt_pad_contacts', 4), ('gt_pad_force', 4), ('object_lift', 1), ('hand_table_force', 1))
PRIVILEGED_DIM = sum(n for _, n in PRIVILEGED_LAYOUT)
OBS_CLIP = 10.0
GAMMA = 0.99


def obs_slices() -> dict:
    out, i = {}, 0
    for name, n in OBS_LAYOUT:
        out[name] = slice(i, i + n)
        i += n
    return out


class VisuoGraspV4Env(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, cfg: V4Config | None = None, obs_mode: str = 'privileged',
                 contact_source: str = 'none', families=None, approaches=('top', 'oblique', 'side'),
                 terminate_on_success: bool | None = None, render_mode: str | None = None):
        super().__init__()
        assert obs_mode in ('privileged', 'vision')
        assert contact_source in ('none', 'd0_vision', 'gt_oracle')
        self.cfg = cfg or V4Config()
        self.sim = VisuoGraspSimV4(self.cfg)
        self.obs_mode, self.contact_source = obs_mode, contact_source
        self.families = tuple(families or self.cfg.objects.families)
        self.approaches = tuple(approaches)
        self.rw = self.cfg.reward
        self.terminate_on_success = (self.rw.terminate_on_success if terminate_on_success is None
                                     else terminate_on_success)
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32)
        self.slices = obs_slices()
        self.confirm = TemporalConfirmation()
        self._perception = {}
        self.prev_action = np.zeros(22, np.float32)
        self.approach = 'top'
        self.plan = None
        self.estimate = None
        self.phi = 0.0
        self.signal_hold = 0
        self.signal_success = False
        self.bonus_paid = self.drop_paid = False
        self.contact_flags = np.zeros(4, np.float32)
        self.contact_probs = np.zeros(4, np.float32)

    # ------------------------------------------------------------------ estimates
    def _perception_for_scene(self):
        key = id(self.sim.scene)
        if key not in self._perception:
            self._perception[key] = RGBDPerceptionV4(self.sim.scene, self.cfg.task.image_size,
                                                     seed=int(self.np_random.integers(2 ** 31)))
        return self._perception[key]

    def _estimate_object(self) -> dict:
        s = self.sim
        if self.obs_mode == 'privileged':
            R = rot.to_mat(s.object_quat())
            shape = GeomSupport(s.model, s.data, s.scene.object_geoms)
            half = np.array([0.5 * (shape.support(R[:, i]) + shape.support(-R[:, i])) for i in range(3)])
            return {'visible': 1.0, 'center': s.object_com(), 'axes': R, 'half_extents': half,
                    'shape': shape, 'points': np.empty((0, 3))}
        est = self._perception_for_scene().observe()
        if est['visible']:
            est['shape'] = OBBSupport(est['box_center'], est['axes'], est['half_extents'])
        elif self.estimate is not None:          # keep the last valid estimate if the object is lost
            keep = dict(self.estimate)
            keep['visible'] = 0.0
            keep['points'] = np.empty((0, 3))
            return keep
        else:
            est['shape'] = OBBSupport(np.zeros(3), np.eye(3), np.full(3, 0.03))
        return est

    def _contacts(self, est, cs) -> None:
        if self.contact_source == 'none':
            p = np.zeros(4, np.float32)
        elif self.contact_source == 'gt_oracle':          # PRIVILEGED (M0-style)
            p = cs['pads'].astype(np.float32)
        else:
            pts = est.get('points', np.empty((0, 3)))
            p = self._perception_for_scene().d0_probs(pts) if len(pts) else np.zeros(4, np.float32)
        self.contact_probs = p
        self.contact_flags = self.confirm.update(p)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        rng = self.np_random
        fam = opts.get('family') or str(rng.choice(self.families))
        spec = self.sim.sample_spec(rng, fam, opts.get('variant'), opts.get('resting_mode'), opts.get('scale'))
        self.approach = opts.get('approach') or str(rng.choice(self.approaches))
        settle = self.sim.reset(spec, rng)
        self.confirm.reset()
        self.estimate = None
        self.estimate = self._estimate_object()
        keypoints = self.sim.ctrl.hand_local_keypoints()
        est = self.estimate
        self.plan = plan_grasp(est['shape'], rot.from_mat(est['axes']), self.approach,
                               self.sim.ctrl.hand_quat(), self.sim.ctrl.hand_pos(), object_com=est['center'],
                               lowest_fn=lambda p, q: self.sim.ctrl.lowest_point_at(p, q, keypoints))
        self.grasp_rel = self.plan.grasp_pos - est['center']
        self.prev_action = np.zeros(22, np.float32)
        self.signal_hold, self.signal_success = 0, False
        self.bonus_paid = self.drop_paid = False
        cs = self.sim.contact_summary()
        self._contacts(est, cs)
        self.phi = self._potential()
        info = {'spec': spec.__dict__.copy(), 'approach': self.approach, 'settle': settle,
                'plan': self.plan.to_dict(), 'privileged': self._privileged_dict(cs)}
        return self._obs(), info

    def step(self, action):
        a = np.asarray(action, np.float32)
        out = self.sim.step(a)
        cs = out['contacts']
        self.estimate = self._estimate_object()
        self._contacts(self.estimate, cs)
        t, tc = self.sim.tracker, self.cfg.task
        h = out['lift']
        c = self.contact_flags
        opp = bool(c[0] and c[1:].any()) if self.contact_source != 'none' else True
        self.signal_hold = self.signal_hold + 1 if (h >= tc.lift_success and opp) else 0
        self.signal_success |= self.signal_hold >= tc.hold_frames
        obj = self.sim.object_com()
        off_table = bool(np.linalg.norm(obj[:2]) > 0.40 or obj[2] < -0.05)
        failure = not out['finite']          # NaN/Inf or object speed > 5 m/s (numerical failure)
        terminated = bool(failure or t.dropped or off_table or (self.terminate_on_success and self.signal_success))
        truncated = bool(t.steps >= tc.max_steps and not terminated)
        phi_new = 0.0 if terminated else self._potential()
        rw = self.rw
        a_c = np.clip(np.nan_to_num(a), -1, 1)
        terms = {
            'shaping': rw.gamma * phi_new - self.phi,
            'hold_rate': rw.hold_rate if (h >= tc.lift_success and opp) else 0.0,
            'success_bonus': rw.success_bonus if (self.signal_success and not self.bonus_paid) else 0.0,
            'drop': rw.drop if (t.dropped and not self.drop_paid) else 0.0,
            'table_collision': rw.table_collision * min(1.0, cs['hand_table_force'] / 10.0),
            'effort': rw.effort * float(np.sum(a_c ** 2)),
            'smoothness': rw.smoothness * float(np.sum((a_c - self.prev_action) ** 2)),
            'physics_failure': rw.physics_failure if failure else 0.0,
        }
        if rw.version == 'v4r0':      # per-step contact terms (kept for reproducibility)
            terms['contact'] = 0.01 * float(c.sum())
            terms['opposition'] = 0.02 if (self.contact_source != 'none' and opp) else 0.0
        self.bonus_paid |= self.signal_success
        self.drop_paid |= t.dropped
        self.phi = phi_new
        self.prev_action = np.clip(np.nan_to_num(a), -1, 1).astype(np.float32)
        reward = float(sum(terms.values()))
        info = {'reward_terms': terms, 'flags': out['flags'], 'approach': self.approach,
                'signal_success': self.signal_success, 'gt_success_EVAL_ONLY': t.gt_success,
                'gt_success_any_phalanx_EVAL_ONLY': t.gt_success_any, 'off_table': off_table,
                'physics_failure': failure, 'privileged': self._privileged_dict(cs)}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        r = self.sim.scene.renderer(240, 320)
        r.update_scene(self.sim.data, camera='overview')
        return r.render().copy()

    def close(self):
        self.sim.cache.close()

    # ------------------------------------------------------------------ pieces
    def _rel_pose_errors(self):
        s, est = self.sim, self.estimate
        palm = s.ctrl.hand_pos()
        d_rel = float(np.linalg.norm((palm - est['center']) - self.grasp_rel))
        rot_err = rot.diff_local(self.plan.grasp_quat, s.ctrl.hand_quat())
        return d_rel, rot_err

    def _potential(self) -> float:
        rw = self.rw
        d_rel, rot_err = self._rel_pose_errors()
        h = self.sim.object_lift()
        align = rw.w_align * (1.0 - min(1.0, float(np.linalg.norm(rot_err)) / math.pi))
        lift = rw.w_lift * float(np.clip(h / 0.10, 0.0, 1.0))
        if rw.version == 'v4r0':
            return 2.0 * (1.0 - min(1.0, d_rel / 0.25)) + align + lift
        # v4r1: corridor reach -- lateral offset from the approach axis costs extra near the
        # grasp pose, so the hand is drawn in along the approach direction (no side sweep)
        v = (self.sim.ctrl.hand_pos() - self.estimate['center']) - self.grasp_rel
        u = self.plan.u
        axial = float(v @ u)
        lateral = float(np.linalg.norm(v - axial * u))
        cost = float(np.linalg.norm(v)) + rw.lateral_gain * lateral * float(np.clip(1.0 - axial / rw.corridor_len, 0.0, 1.0))
        reach = rw.w_reach * (1.0 - min(1.0, cost / rw.reach_scale))
        contact = 0.0
        if self.contact_source != 'none':
            c = self.contact_flags
            contact = rw.w_contact * (0.5 * float(c.sum()) / 4.0 + 0.5 * float(c[0] and c[1:].any()))
        # PRIVILEGED (simulator object pose): pushing the object before it is lifted costs potential
        disp = float(np.linalg.norm(self.sim.object_com()[:2] - self.sim.rest_com[:2]))
        push = -rw.w_push * float(np.clip(disp / rw.push_scale, 0.0, 1.0)) if h < 0.02 else 0.0
        return reach + align + contact + lift + push

    def _obs(self) -> np.ndarray:
        s, sc, est = self.sim, self.sim.scene, self.estimate
        d = s.data
        span = sc.finger_hi - sc.finger_lo
        R_hand = rot.to_mat(s.ctrl.hand_quat())
        palm = s.ctrl.hand_pos()
        vel = d.qvel[sc.hand_vadr:sc.hand_vadr + 6]
        d_rel_vec = (est['center'] + self.grasp_rel) - palm
        _, rot_err = self._rel_pose_errors()
        parts = {
            'finger_qpos_norm': d.qpos[sc.finger_qadr] / span,
            'finger_qvel': d.qvel[sc.finger_vadr] * 0.05,
            'finger_target_norm': s.ctrl.finger_target / span,
            'wrist_pos': palm * 5.0,
            'wrist_rot6d': rot.rot6d(R_hand),
            'wrist_linvel': vel[:3] * 2.0,
            'wrist_angvel_local': vel[3:] * 0.2,
            'fingertips_world': s.fingertips().reshape(-1) * 5.0,
            'object_center_est': est['center'] * 5.0,
            'object_axes_est': np.concatenate((est['axes'][:, 0], est['axes'][:, 1])),
            'object_half_extents_est': est['half_extents'] * 10.0,
            'object_in_hand_frame': R_hand.T @ (est['center'] - palm) * 5.0,
            'approach_cmd': -self.plan.u,
            'grasp_pos_rel': d_rel_vec * 5.0,
            'grasp_rot_err': rot_err / math.pi,
            'contact_flags': self.contact_flags,
            'contact_probs': self.contact_probs,
            'prev_action': self.prev_action,
            'object_visible': np.array([est['visible']]),
        }
        x = np.concatenate([np.asarray(parts[name], float).reshape(-1) for name, _ in OBS_LAYOUT])
        assert x.shape[0] == OBS_DIM
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _privileged_dict(self, cs) -> dict:
        s, sc = self.sim, self.sim.scene
        v = s.data.qvel[sc.obj_vadr:sc.obj_vadr + 6]
        return {'object_pos': s.object_com(), 'object_quat': s.object_quat(), 'object_linvel': v[:3].copy(),
                'object_angvel': v[3:].copy(), 'gt_pad_contacts': cs['pads'].copy(),
                'gt_pad_force': cs['pad_force'].copy(), 'object_lift': s.object_lift(),
                'hand_table_force': cs['hand_table_force']}

    def privileged_state(self) -> np.ndarray:
        """Critic-only vector (asymmetric actor-critic). Never part of the actor observation."""
        p = self._privileged_dict(self.sim.contact_summary())
        return np.concatenate([np.asarray(p[k], float).reshape(-1) for k, _ in PRIVILEGED_LAYOUT]).astype(np.float32)
