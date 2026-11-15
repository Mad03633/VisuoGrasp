"""Core V4 simulator: object placement and settling, 22-D stepping, ground-truth bookkeeping.

Everything in this class may read simulator state. Separation from what a policy is
allowed to see happens in env.py (actor observation vs privileged state).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import objects as objs
from . import rotations as rot
from .config import V4Config
from .controller import HandController
from .scene import SceneCache


@dataclass
class EpisodeSpec:
    family: str
    variant: str
    scale: float
    resting_mode: str
    yaw: float
    mass: float
    friction: float
    obj_xy: tuple
    hand_pos: tuple
    hand_quat: tuple


@dataclass
class Tracker:
    """Ground-truth episode statistics (evaluation only; never an actor observation)."""
    steps: int = 0
    max_lift: float = 0.0
    final_lift: float = 0.0
    gt_hold: int = 0
    gt_hold_max: int = 0
    height_hold: int = 0
    height_hold_max: int = 0
    gt_success: bool = False            # spec / V3.3 rule: thumb PAD + another PAD, >= 0.2 N
    gt_success_step: int = -1
    gt_hold_any: int = 0
    gt_hold_any_max: int = 0
    gt_success_any: bool = False        # V4 secondary rule: thumb + another finger, any phalanx
    any_pad_contact: bool = False
    thumb_contact: bool = False
    max_pads_in_contact: int = 0
    any_hand_object_contact: bool = False
    had_lift: bool = False
    dropped: bool = False
    hand_table_force_max: float = 0.0
    obj_table_min_dist: float = 1.0
    nonfinite: bool = False
    runaway: bool = False               # object speed > 5 m/s or |z| > 2 m (numerical failure)
    off_table: bool = False             # object COM left the table area (task failure)
    max_object_speed: float = 0.0
    warnings: dict = field(default_factory=dict)
    track_err_pos: list = field(default_factory=list)
    track_err_rot: list = field(default_factory=list)
    flag_counts: dict = field(default_factory=dict)


class VisuoGraspSimV4:
    def __init__(self, cfg: V4Config | None = None, cache: SceneCache | None = None):
        self.cfg = cfg or V4Config()
        self.cache = cache or SceneCache(self.cfg)
        self.scene = None
        self.ctrl: HandController | None = None
        self.spec: EpisodeSpec | None = None
        self.tracker = Tracker()
        self.rest_com = np.zeros(3)
        self.rest_quat = rot.IDENTITY.copy()
        self.settle_attempts = 0

    # ------------------------------------------------------------------ convenience
    @property
    def model(self):
        return self.scene.model

    @property
    def data(self):
        return self.scene.data

    def object_com(self) -> np.ndarray:
        return self.data.xipos[self.scene.object_body].copy()

    def object_quat(self) -> np.ndarray:
        return rot.normalize(self.data.xquat[self.scene.object_body])

    def object_lift(self) -> float:
        return float(self.object_com()[2] - self.rest_com[2])

    def fingertips(self) -> np.ndarray:
        return self.data.site_xpos[self.scene.tip_sites].copy()

    # ------------------------------------------------------------------ sampling
    def sample_spec(self, rng: np.random.Generator, family=None, variant=None, resting_mode=None,
                    scale=None) -> EpisodeSpec:
        oc, tc = self.cfg.objects, self.cfg.task
        family = family or str(rng.choice(oc.families))
        variant = variant or str(rng.choice(list(objs.VARIANTS[family])))
        resting_mode = resting_mode or str(rng.choice(list(objs.RESTING_MODES[family])))
        scale = float(scale if scale is not None else rng.choice(oc.scale_bins))
        yaw = float(rng.uniform(-math.pi, math.pi))
        obj_xy = tuple(rng.uniform(-oc.xy_range, oc.xy_range, 2))
        r = float(rng.uniform(0.0, tc.hand_start_xy_radius))
        phi = float(rng.uniform(-math.pi, math.pi))
        hand_pos = (obj_xy[0] + r * math.cos(phi), obj_xy[1] + r * math.sin(phi),
                    float(rng.uniform(*tc.hand_start_height)))
        tilt = math.radians(float(rng.uniform(0.0, tc.hand_start_max_tilt_deg)))
        tilt_axis_angle = float(rng.uniform(-math.pi, math.pi))
        q_tilt = rot.from_axis_angle([math.cos(tilt_axis_angle), math.sin(tilt_axis_angle), 0.0], tilt)
        q_yaw = rot.from_axis_angle([0, 0, 1], float(rng.uniform(-math.pi, math.pi)))
        hand_quat = tuple(rot.normalize(rot.mul(q_tilt, q_yaw)))
        return EpisodeSpec(family, variant, scale, resting_mode, yaw,
                           float(rng.uniform(*oc.mass_range)), float(rng.uniform(*oc.friction_range)),
                           obj_xy, hand_pos, hand_quat)

    # ------------------------------------------------------------------ reset
    def reset(self, spec: EpisodeSpec, rng: np.random.Generator | None = None) -> dict:
        """Load the cached model, randomise physical parameters, place and settle the object.

        Returns a dict with settle diagnostics. If settling fails, the yaw/xy are resampled
        (up to max_settle_attempts) and the number of attempts is reported.
        """
        rng = rng or np.random.default_rng(0)
        self.scene = self.cache.get(spec.family, spec.variant, spec.scale)
        self.scene.set_mass(spec.mass)
        self.scene.set_friction(spec.friction)
        if self.ctrl is None or self.ctrl.s is not self.scene:
            self.ctrl = HandController(self.scene, self.cfg.control, self.cfg.hand, self.cfg.physics.timestep)
        oc = self.cfg.objects
        info = {}
        for attempt in range(oc.max_settle_attempts):
            info = self._place_and_settle(spec)
            self.settle_attempts = attempt + 1
            if info['valid']:
                break
            spec.yaw = float(rng.uniform(-math.pi, math.pi))
            spec.obj_xy = tuple(rng.uniform(-oc.xy_range, oc.xy_range, 2))
        info['attempts'] = self.settle_attempts
        self.spec = spec
        self.rest_com = self.object_com()
        self.rest_quat = self.object_quat()
        self.tracker = Tracker()
        return info

    def _place_and_settle(self, spec: EpisodeSpec) -> dict:
        m, d, s = self.model, self.data, self.scene
        mujoco.mj_resetData(m, d)
        self.ctrl.reset(spec.hand_pos, spec.hand_quat)
        q_obj = rot.normalize(rot.mul(rot.from_axis_angle([0, 0, 1], spec.yaw),
                                      objs.RESTING_MODES[spec.family][spec.resting_mode]))
        d.qpos[s.obj_qadr:s.obj_qadr + 3] = [spec.obj_xy[0], spec.obj_xy[1], 0.5]
        d.qpos[s.obj_qadr + 3:s.obj_qadr + 7] = q_obj
        mujoco.mj_kinematics(m, d)
        low = objs.lowest_z(m, d, s.object_geoms)
        d.qpos[s.obj_qadr + 2] += 0.0015 - low          # 1.5 mm above the table
        mujoco.mj_forward(m, d)
        initial_contacts = self._hand_object_contact_count()
        n_settle = int(round(self.cfg.objects.settle_seconds / m.opt.timestep))
        self.ctrl._prev_pos, self.ctrl._prev_quat = self.ctrl.target_pos.copy(), self.ctrl.target_quat.copy()
        for _ in range(n_settle):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        v = d.qvel[s.obj_vadr:s.obj_vadr + 6]
        lin, ang = float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))
        pen = self._object_table_min_dist()
        xy = d.xipos[s.object_body, :2]
        tilt_change = rot.angle_between(rot.normalize(d.xquat[s.object_body]), q_obj)
        oc = self.cfg.objects
        valid = (lin < oc.rest_lin_speed and ang < oc.rest_ang_speed and pen > -oc.max_penetration
                 and float(np.linalg.norm(xy)) < oc.max_xy_after_settle and initial_contacts == 0
                 and self._hand_object_contact_count() == 0 and np.all(np.isfinite(d.qpos)))
        # remove residual settling velocity of the object (it is at rest within tolerance)
        d.qvel[s.obj_vadr:s.obj_vadr + 6] = 0.0
        d.qvel[s.hand_vadr:s.hand_vadr + 6] = 0.0
        mujoco.mj_forward(m, d)
        return {'valid': bool(valid), 'settle_lin_speed': lin, 'settle_ang_speed': ang,
                'settle_min_dist': pen, 'settle_rot_change_deg': math.degrees(tilt_change),
                'initial_hand_object_contacts': initial_contacts, 'rest_com_z': float(d.xipos[s.object_body, 2])}

    # ------------------------------------------------------------------ contacts (ground truth)
    def _contact_arrays(self):
        d = self.data
        n = d.ncon
        if n == 0:
            return n, np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        c = d.contact
        return n, np.asarray(c.geom1[:n]), np.asarray(c.geom2[:n]), np.asarray(c.dist[:n])

    def _normal_force(self, k: int) -> float:
        f = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, k, f)
        return float(f[0])

    def _hand_object_contact_count(self) -> int:
        n, g1, g2, _ = self._contact_arrays()
        if n == 0:
            return 0
        s = self.scene
        h1, h2 = np.isin(g1, s.hand_geoms), np.isin(g2, s.hand_geoms)
        o1, o2 = np.isin(g1, s.object_geoms), np.isin(g2, s.object_geoms)
        return int(np.sum((h1 & o2) | (h2 & o1)))

    def _object_table_min_dist(self) -> float:
        n, g1, g2, dist = self._contact_arrays()
        s = self.scene
        mask = ((g1 == s.table_geom) & np.isin(g2, s.object_geoms)) | ((g2 == s.table_geom) & np.isin(g1, s.object_geoms))
        return float(dist[mask].min()) if np.any(mask) else 1.0

    def contact_summary(self) -> dict:
        """GT pad contacts (normal force >= threshold), hand-object and hand-table contacts."""
        s, thr = self.scene, self.cfg.task.contact_threshold_n
        n, g1, g2, dist = self._contact_arrays()
        pads = np.zeros(4, np.float32)
        pad_force = np.zeros(4, np.float32)
        finger_force = np.zeros(4, np.float32)
        hand_obj = False
        table_force = 0.0
        obj_table = 1.0
        for k in range(n):
            a, b = int(g1[k]), int(g2[k])
            a_obj, b_obj = a in s.object_geoms, b in s.object_geoms
            if a_obj or b_obj:
                other = b if a_obj else a
                if other == s.table_geom:
                    obj_table = min(obj_table, float(dist[k]))
                    continue
                if other in s.hand_geoms:
                    hand_obj = True
                    f = self._normal_force(k)
                    link = np.where(s.finger_link_geoms == other)[0]
                    if link.size:
                        finger_force[link[0] // 4] += f
                    idx = np.where(s.pad_geoms == other)[0]
                    if idx.size:
                        pad_force[idx[0]] += f
                        if f >= thr:
                            pads[idx[0]] = 1.0
            elif (a == s.table_geom and b in s.hand_geoms) or (b == s.table_geom and a in s.hand_geoms):
                table_force += self._normal_force(k)
        return {'pads': pads, 'pad_force': pad_force, 'fingers': (finger_force >= thr).astype(np.float32),
                'finger_force': finger_force, 'hand_object': hand_obj,
                'hand_table_force': table_force, 'object_table_min_dist': obj_table}

    # ------------------------------------------------------------------ stepping
    def step(self, action) -> dict:
        flags = self.ctrl.apply_action(action)
        m, d = self.model, self.data
        for k in range(1, self.ctrl.n_sub + 1):
            self.ctrl.substep_target(k)
            mujoco.mj_step(m, d)
        return self._update_tracker(flags)

    def _update_tracker(self, flags: dict) -> dict:
        t, tc, d = self.tracker, self.cfg.task, self.data
        t.steps += 1
        for key, val in flags.items():
            if val:
                t.flag_counts[key] = t.flag_counts.get(key, 0) + 1
        finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))
        t.nonfinite |= not finite
        # Runaway detector: an object can be launched by solver energy injection without any
        # MuJoCo warning (observed with noslip_iterations=5). Treat it as a simulation failure.
        s = self.scene
        obj_v = float(np.linalg.norm(d.qvel[s.obj_vadr:s.obj_vadr + 3])) if finite else float('inf')
        com = d.xipos[s.object_body]
        runaway = (not finite) or obj_v > 5.0 or abs(float(com[2])) > 2.0
        t.runaway |= runaway
        # leaving the 1.2 m x 1.2 m table (e.g. a dropped cylinder rolling away: condim 4 has no
        # rolling friction) is a task failure, not a numerical failure
        t.off_table |= bool(float(np.max(np.abs(com[:2]))) > 0.6 or float(com[2]) < -0.05)
        t.max_object_speed = max(t.max_object_speed, obj_v)
        finite = finite and not runaway
        for i, w in enumerate(d.warning):
            if w.number > 0:
                t.warnings[mujoco.mjtWarning(i).name] = int(w.number)
        cs = self.contact_summary()
        h = self.object_lift()
        t.final_lift = h
        t.max_lift = max(t.max_lift, h)
        pads = cs['pads'] > 0
        t.any_pad_contact |= bool(pads.any())
        t.thumb_contact |= bool(pads[0])
        t.max_pads_in_contact = max(t.max_pads_in_contact, int(pads.sum()))
        t.any_hand_object_contact |= cs['hand_object']
        t.hand_table_force_max = max(t.hand_table_force_max, cs['hand_table_force'])
        t.obj_table_min_dist = min(t.obj_table_min_dist, cs['object_table_min_dist'])
        opp = bool(pads[0] and pads[1:].any())
        t.height_hold = t.height_hold + 1 if h >= tc.lift_success else 0
        t.height_hold_max = max(t.height_hold_max, t.height_hold)
        t.gt_hold = t.gt_hold + 1 if (h >= tc.lift_success and opp) else 0
        t.gt_hold_max = max(t.gt_hold_max, t.gt_hold)
        if not t.gt_success and t.gt_hold >= tc.hold_frames:
            t.gt_success, t.gt_success_step = True, t.steps
        fingers = cs['fingers'] > 0
        opp_any = bool(fingers[0] and fingers[1:].any())
        t.gt_hold_any = t.gt_hold_any + 1 if (h >= tc.lift_success and opp_any) else 0
        t.gt_hold_any_max = max(t.gt_hold_any_max, t.gt_hold_any)
        t.gt_success_any |= t.gt_hold_any >= tc.hold_frames
        if h > tc.lift_drop_arm:
            t.had_lift = True
        if t.had_lift and h < tc.drop_height:
            t.dropped = True
        hp = d.xpos[self.scene.hand_body]
        t.track_err_pos.append(float(np.linalg.norm(hp - self.ctrl.target_pos)))
        t.track_err_rot.append(rot.angle_between(rot.normalize(d.xquat[self.scene.hand_body]), self.ctrl.target_quat))
        return {'contacts': cs, 'lift': h, 'opposition_gt': opp, 'finite': finite, 'flags': flags}
