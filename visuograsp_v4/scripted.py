"""Scripted V4 grasp baseline (diagnostic, not the RL solution).

Phase machine, all through the same bounded 22-D action interface as an RL policy:
  transit  -> rotate to the grasp orientation while moving above the pre-grasp point
  descend  -> move to the pre-grasp pose (10 cm back along the approach axis)
  approach -> straight-line approach to the grasp pose, slower
  close    -> ramp the 16 finger targets to a fixed closing posture, then settle
  lift     -> raise the wrist 17 cm along world +z, orientation fixed
  hold     -> keep the pose; ground truth needs 60 consecutive frames
PRIVILEGED: the plan uses the simulator's object pose and geometry (diagnostic baseline).
Finger closing is open-loop (no contact feedback).
"""
from __future__ import annotations

import numpy as np

from . import rotations as rot
from .controller import finger_to_action, pose_to_action
from .grasp_planner import GeomSupport, GraspPlan, plan_grasp

CLOSE_POSTURE = np.tile(np.array([0.75, 0.60, 0.45, 0.30]), 4)
PHASES = ('transit', 'descend', 'approach', 'close', 'lift', 'hold', 'done')


class ScriptedGrasp:
    def __init__(self, sim, approach: str, close_posture=CLOSE_POSTURE, close_steps: int = 20,
                 settle_steps: int = 10, hold_steps: int = 75, approach_speed: float = 0.4,
                 lift_speed: float = 0.5, timeouts=None):
        self.sim = sim
        self.approach = approach
        self.close_posture = np.asarray(close_posture, float)
        self.close_steps, self.settle_steps, self.hold_steps = close_steps, settle_steps, hold_steps
        self.approach_speed, self.lift_speed = approach_speed, lift_speed
        self.timeouts = timeouts or {'transit': 80, 'descend': 40, 'approach': 50, 'lift': 60}
        self.phase = 'transit'
        self.phase_step = 0
        self.plan: GraspPlan | None = None
        self.log: list = []
        self.reached = {}

    def make_plan(self) -> GraspPlan:
        s = self.sim
        keypoints = s.ctrl.hand_local_keypoints()          # open hand at episode start
        shape = GeomSupport(s.model, s.data, s.scene.object_geoms)   # PRIVILEGED geometry
        self.plan = plan_grasp(shape, s.object_quat(), self.approach,
                               s.ctrl.hand_quat(), s.ctrl.hand_pos(), object_com=s.object_com(),
                               lowest_fn=lambda p, q: s.ctrl.lowest_point_at(p, q, keypoints))
        return self.plan

    def _goal(self):
        p = self.plan
        return {'transit': (p.transit_pos, 1.0), 'descend': (p.pre_pos, 0.8),
                'approach': (p.grasp_pos, self.approach_speed), 'close': (p.grasp_pos, 0.5),
                'lift': (p.lift_pos, self.lift_speed), 'hold': (p.lift_pos, 0.5)}.get(self.phase, (p.lift_pos, 0.5))

    def _advance(self, nxt: str, reached: bool) -> None:
        self.reached[self.phase] = reached
        self.phase, self.phase_step = nxt, 0

    def act(self) -> np.ndarray:
        if self.plan is None:
            self.make_plan()
        s, c = self.sim, self.sim.ctrl
        cfg = s.cfg.control
        p = self.plan
        goal_pos, speed = self._goal()
        goal_quat = p.grasp_quat
        a = np.zeros(22)
        a[16:22] = pose_to_action(c.target_pos, c.target_quat, goal_pos, goal_quat, cfg, speed, speed)

        if self.phase == 'close':
            frac = min(1.0, (self.phase_step + 1) / self.close_steps)
            a[:16] = finger_to_action(c.finger_target, self.close_posture * frac, cfg)
        else:
            hold_fingers = self.close_posture if self.phase in ('lift', 'hold', 'done') else np.zeros(16)
            a[:16] = finger_to_action(c.finger_target, hold_fingers, cfg)

        # phase transitions use the commanded target AND the real hand pose
        tgt_err = float(np.linalg.norm(c.target_pos - goal_pos))
        rot_err = rot.angle_between(c.target_quat, goal_quat)
        hand_err = float(np.linalg.norm(c.hand_pos() - goal_pos))
        at_goal = tgt_err < 0.004 and rot_err < 0.04 and hand_err < 0.012
        self.phase_step += 1
        if self.phase in ('transit', 'descend', 'approach', 'lift'):
            nxt = {'transit': 'descend', 'descend': 'approach', 'approach': 'close', 'lift': 'hold'}[self.phase]
            if at_goal:
                self._advance(nxt, True)
            elif self.phase_step >= self.timeouts[self.phase]:
                self._advance(nxt, False)
        elif self.phase == 'close' and self.phase_step >= self.close_steps + self.settle_steps:
            self._advance('lift', True)
        elif self.phase == 'hold' and self.phase_step >= self.hold_steps:
            self._advance('done', True)
        self.log.append({'phase': self.phase, 'target_err': tgt_err, 'rot_err': rot_err, 'hand_err': hand_err})
        return a.astype(np.float32)
