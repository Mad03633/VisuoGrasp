"""Bounded 22-D control of the 6-DoF wrist (mocap target + weld) and 16 finger servos.

How the wrist is actuated
-------------------------
The hand is a free rigid body. A massless mocap body ``hand_target`` is the commanded
wrist pose; a weld equality constraint pulls the hand towards it with a soft spring-damper
(solref time constant 10 ms, critical damping). The hand therefore moves through the
physics solver: it is pushed back by contacts, carries the object's weight and never
jumps. Only the *target* moves kinematically, and only by bounded increments:

    target_pos  <- target_pos + 0.010 m * a[16:19]          (world axes)
    target_quat <- normalize(target_quat * exp(0.06 rad * a[19:22]))   (hand axes)

followed by four safety projections, in this order:
  1. workspace box on the position;
  2. finger-direction tilt limit (rotation update rejected beyond max_tilt_deg);
  3. anti-windup: the target may lead the real hand by at most 2.5 cm / 0.30 rad,
     which bounds the weld force when the hand is blocked;
  4. table clearance: the hand's lowest point (palm corners and all 16 finger capsules in
     their *current* curl) is predicted at the target pose and the target is raised if it
     would go below the table plane + 4 mm.
Inside a control step the mocap pose is interpolated (lerp + slerp) over the substeps.
"""
from __future__ import annotations

import math

import numpy as np

from . import rotations as rot
from .config import ControlConfig, HandConfig


class HandController:
    def __init__(self, scene, ctrl_cfg: ControlConfig, hand_cfg: HandConfig, physics_dt: float):
        self.s = scene
        self.c = ctrl_cfg
        self.h = hand_cfg
        self.n_sub = int(round(1.0 / ctrl_cfg.policy_hz / physics_dt))
        self.ws_lo = np.asarray(ctrl_cfg.workspace_lo, float)
        self.ws_hi = np.asarray(ctrl_cfg.workspace_hi, float)
        px, py, pz = hand_cfg.palm_half
        self.palm_corners = np.array([[sx * px, sy * py, sz * pz]
                                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        self.target_pos = np.zeros(3)
        self.target_quat = rot.IDENTITY.copy()
        self.finger_target = np.zeros(16)
        self.last_flags: dict = {}

    # ------------------------------------------------------------------ state access
    @property
    def data(self):
        return self.s.data

    def hand_pos(self) -> np.ndarray:
        return self.data.xpos[self.s.hand_body].copy()

    def hand_quat(self) -> np.ndarray:
        return rot.normalize(self.data.xquat[self.s.hand_body])

    # ------------------------------------------------------------------ reset (episode start only)
    def reset(self, pos, quat) -> None:
        """Place hand and target at the same pose. Called only by env.reset, never inside step."""
        d, s = self.data, self.s
        q = rot.normalize(quat)
        d.qpos[s.hand_qadr:s.hand_qadr + 3] = pos
        d.qpos[s.hand_qadr + 3:s.hand_qadr + 7] = q
        d.qvel[s.hand_vadr:s.hand_vadr + 6] = 0.0
        d.qpos[s.finger_qadr] = 0.0
        d.qvel[s.finger_vadr] = 0.0
        d.mocap_pos[s.mocap_id] = pos
        d.mocap_quat[s.mocap_id] = q
        d.ctrl[:] = 0.0
        self.target_pos = np.asarray(pos, float).copy()
        self.target_quat = q.copy()
        self.finger_target = np.zeros(16)

    # ------------------------------------------------------------------ geometry for clearance
    def hand_local_keypoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Palm corners + finger-capsule end points in the hand frame, with radii (current curl)."""
        d, m, s = self.data, self.s.model, self.s
        R = d.xmat[s.hand_body].reshape(3, 3)
        p = d.xpos[s.hand_body]
        g = s.finger_link_geoms
        c = d.geom_xpos[g]
        ax = d.geom_xmat[g].reshape(-1, 3, 3)[:, :, 2]
        half = m.geom_size[g, 1][:, None]
        ends = np.concatenate((c + ax * half, c - ax * half))
        local = (ends - p) @ R
        pts = np.concatenate((self.palm_corners, local))
        radii = np.concatenate((np.zeros(8), np.tile(m.geom_size[g, 0], 2)))
        return pts, radii

    def lowest_point_at(self, pos, quat, keypoints=None) -> float:
        pts, radii = keypoints if keypoints is not None else self.hand_local_keypoints()
        world_z = (pts @ rot.to_mat(quat).T)[:, 2] + pos[2] - radii
        return float(world_z.min())

    # ------------------------------------------------------------------ action -> targets
    def apply_action(self, action) -> dict:
        """Map a 22-D action in [-1, 1] to bounded finger and wrist targets. Returns flags."""
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.shape[0] != 22:
            raise ValueError(f'expected 22 actions, got {a.shape[0]}')
        flags = {'nonfinite_action': bool(not np.all(np.isfinite(a)))}
        a = np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

        # fingers: bounded delta on the position-servo targets
        self.finger_target = np.clip(self.finger_target + self.c.finger_delta_max * a[:16],
                                     self.s.finger_lo, self.s.finger_hi)

        # wrist: bounded delta pose on the mocap target
        old_pos, old_quat = self.target_pos.copy(), self.target_quat.copy()
        new_pos = old_pos + self.c.wrist_trans_max * a[16:19]
        new_quat = rot.normalize(rot.mul(old_quat, rot.from_rotvec(self.c.wrist_rot_max * a[19:22])))

        clipped = np.clip(new_pos, self.ws_lo, self.ws_hi)
        flags['workspace_clip'] = bool(np.any(np.abs(clipped - new_pos) > 1e-12))
        new_pos = clipped

        flags['tilt_reject'] = rot.finger_tilt_deg(new_quat) > self.c.max_tilt_deg
        if flags['tilt_reject']:
            new_quat = old_quat

        hp, hq = self.hand_pos(), self.hand_quat()
        lead = new_pos - hp
        dist = float(np.linalg.norm(lead))
        flags['lead_clip_pos'] = dist > self.c.max_lead_pos
        if flags['lead_clip_pos']:
            new_pos = hp + lead * (self.c.max_lead_pos / dist)
        err = rot.diff_local(new_quat, hq)
        ang = float(np.linalg.norm(err))
        flags['lead_clip_rot'] = ang > self.c.max_lead_rot
        if flags['lead_clip_rot']:
            new_quat = rot.normalize(rot.mul(hq, rot.from_rotvec(err * (self.c.max_lead_rot / ang))))

        low = self.lowest_point_at(new_pos, new_quat)
        flags['clearance_raise'] = low < self.c.table_clearance
        if flags['clearance_raise']:
            new_pos[2] += self.c.table_clearance - low
            new_pos[2] = min(new_pos[2], self.ws_hi[2])

        self._prev_pos, self._prev_quat = old_pos, old_quat
        self.target_pos, self.target_quat = new_pos, rot.normalize(new_quat)
        self.last_flags = flags
        return flags

    def substep_target(self, k: int) -> None:
        """Write the mocap pose for substep k (1..n_sub)."""
        d, s = self.data, self.s
        if self.c.interpolate_substeps:
            t = k / self.n_sub
            d.mocap_pos[s.mocap_id] = (1 - t) * self._prev_pos + t * self.target_pos
            d.mocap_quat[s.mocap_id] = rot.slerp(self._prev_quat, self.target_quat, t)
        else:
            d.mocap_pos[s.mocap_id] = self.target_pos
            d.mocap_quat[s.mocap_id] = self.target_quat
        d.ctrl[:16] = self.finger_target


def pose_to_action(cur_pos, cur_quat, goal_pos, goal_quat, cfg: ControlConfig,
                   trans_speed: float = 1.0, rot_speed: float = 1.0) -> np.ndarray:
    """Wrist part (6 values in [-1, 1]) of the action that moves the target towards a goal
    along a straight line / geodesic, never exceeding the per-step bounds. Used by the
    scripted baseline so that it acts through exactly the same interface as an RL policy."""
    out = np.zeros(6)
    delta = np.asarray(goal_pos, float) - np.asarray(cur_pos, float)
    lim = cfg.wrist_trans_max * trans_speed
    n = float(np.linalg.norm(delta))
    if n > lim:
        delta *= lim / n
    out[:3] = delta / cfg.wrist_trans_max
    r = rot.diff_local(goal_quat, cur_quat)
    lim_r = cfg.wrist_rot_max * rot_speed
    nr = float(np.linalg.norm(r))
    if nr > lim_r:
        r *= lim_r / nr
    out[3:] = r / cfg.wrist_rot_max
    return np.clip(out, -1.0, 1.0)


def finger_to_action(cur_target, goal, cfg: ControlConfig) -> np.ndarray:
    return np.clip((np.asarray(goal, float) - np.asarray(cur_target, float)) / cfg.finger_delta_max, -1, 1)
