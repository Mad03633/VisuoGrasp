"""Residual control on top of the V4 scripted grasp controller (Wave 2).

PRIVILEGED setting: the nominal controller plans from the simulator object pose and geometry,
the actor observation contains the simulator object pose and the contact signal is the ground-
truth oracle. This is residual learning on a privileged scripted controller -- not autonomous and
not vision-based grasping.

Why set-point residuals (not action residuals)
----------------------------------------------
The scripted controller is a dead-beat set-point tracker: every step it commands the full
remaining error of the wrist target and of the finger targets (up to the speed limit). A residual
added to the 22-D *action* would be cancelled on the next step, so its lasting effect would be at
most one step (<= 1 cm, <= 0.055 rad). The residual therefore corrects the SET-POINTS that the
scripted controller tracks:

  fingers  r[0:16]  : finger joint goals   q_goal' = q_goal + 0.30 rad * r[0:16]          (joint space)
  position r[16:19] : wrist position goal  p_goal' = p_goal + 0.02 m * r[16:19]            (world axes)
  rotation r[19:22] : wrist orientation    R_goal' = R_goal * exp(0.15 rad * r[19:22])     (hand axes;
                      unit quaternion via exact quaternion product + normalisation)

with r in [-1, 1]^22 (bounded). The scripted controller then produces its bounded 22-D action
towards the corrected set-points with its own speed limits, phase logic and timeouts, and the hand
controller applies all safety projections (workspace box, finger-tilt limit, anti-windup lead
limit, table clearance) and joint-range clipping. The phase machine uses the corrected set-points
for its "goal reached" tests. r = 0 reproduces the scripted controller bit for bit: zero offsets
skip every arithmetic operation (verified by scripts/verify_residual.py).

Versions (Wave 3 added res2 / v4r2; res1 / v4r1 unchanged):
  res1 (Wave 2): bounds 0.30 rad / 0.02 m / 0.15 rad, reward v4r1.
  res2 (Wave 3): bounds 0.08 rad / 0.004 m / 0.03 rad, reward v4r2.
  v4r2 = v4r1 (every term, computed by VisuoGraspV4Env from the executed action, unchanged)
         - 0.01 * mean_i(r_i^2)          (residual-magnitude penalty, at most -0.01 per step)

Actor observation (163-D) = the 130-D V4 actor observation (privileged object pose, GT-oracle
contacts) + scripted phase one-hot (7) + phase progress (1) + nominal wrist goal minus the current
wrist target (3) + previous residual action (22). No evaluation-only quantity (GT success, hold
counters, lift tracker) is part of it. The reward is the unchanged V4 reward v4r1, computed by
VisuoGraspV4Env from the executed 22-D action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import rotations as rot
from .config import V4Config
from .controller import finger_to_action, pose_to_action
from .env import OBS_CLIP, OBS_DIM, VisuoGraspV4Env
from .scripted import PHASES, ScriptedGrasp

PHASE_LEN = {'transit': 80, 'descend': 40, 'approach': 50, 'close': 30, 'lift': 60, 'hold': 75, 'done': 1}
EXTRA_LAYOUT = (('scripted_phase_onehot', len(PHASES)), ('phase_progress', 1), ('nominal_goal_minus_target', 3),
                ('prev_residual', 22))
RES_OBS_DIM = OBS_DIM + sum(n for _, n in EXTRA_LAYOUT)


@dataclass(frozen=True)
class ResidualConfig:
    version: str = 'res1'
    finger_max: float = 0.30      # rad, finger joint goal offset at |r| = 1
    pos_max: float = 0.02         # m, wrist position goal offset per world axis at |r| = 1
    rot_max: float = 0.15         # rad, wrist orientation goal offset per hand axis at |r| = 1
    reward_version: str = 'v4r1'  # 'v4r1' (Wave 2) or 'v4r2' (v4r1 + residual penalty)
    w_residual: float = 0.0       # v4r2: penalty weight on mean(r^2)

    def to_dict(self) -> dict:
        return asdict(self)


RES1 = ResidualConfig()                                                    # Wave 2
RES2 = ResidualConfig(version='res2', finger_max=0.08, pos_max=0.004, rot_max=0.03,
                      reward_version='v4r2', w_residual=0.01)               # Wave 3
RESIDUAL_CONFIGS = {'res1': RES1, 'res2': RES2}


class ResidualScriptedGrasp(ScriptedGrasp):
    """ScriptedGrasp whose set-points are shifted by a bounded residual (same phase machine)."""

    def __init__(self, sim, approach: str, rcfg: ResidualConfig, **kw):
        super().__init__(sim, approach, **kw)
        self.rcfg = rcfg
        self.residual = np.zeros(22)

    def set_residual(self, r) -> None:
        self.residual = np.clip(np.nan_to_num(np.asarray(r, float).reshape(-1), nan=0.0), -1.0, 1.0)

    def goal_pose(self):
        """Corrected wrist set-point (position, speed, orientation) of the current phase."""
        goal_pos, speed = self._goal()
        goal_quat = self.plan.grasp_quat
        dp, dth = self.residual[16:19], self.residual[19:22]
        if dp.any():
            goal_pos = goal_pos + self.rcfg.pos_max * dp
        if dth.any():
            goal_quat = rot.normalize(rot.mul(goal_quat, rot.from_rotvec(self.rcfg.rot_max * dth)))
        return goal_pos, speed, goal_quat

    def finger_goal(self, nominal: np.ndarray) -> np.ndarray:
        df = self.residual[:16]
        return nominal + self.rcfg.finger_max * df if df.any() else nominal

    def act(self) -> np.ndarray:
        # Same logic as ScriptedGrasp.act, with the corrected set-points (identical for r = 0).
        if self.plan is None:
            self.make_plan()
        s, c = self.sim, self.sim.ctrl
        cfg = s.cfg.control
        goal_pos, speed, goal_quat = self.goal_pose()
        a = np.zeros(22)
        a[16:22] = pose_to_action(c.target_pos, c.target_quat, goal_pos, goal_quat, cfg, speed, speed)
        if self.phase == 'close':
            frac = min(1.0, (self.phase_step + 1) / self.close_steps)
            a[:16] = finger_to_action(c.finger_target, self.finger_goal(self.close_posture * frac), cfg)
        else:
            hold_fingers = self.close_posture if self.phase in ('lift', 'hold', 'done') else np.zeros(16)
            a[:16] = finger_to_action(c.finger_target, self.finger_goal(hold_fingers), cfg)
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


class ResidualGraspEnv(gym.Env):
    """Gymnasium env: action = 22-D residual on the scripted set-points, reward = V4 v4r1."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, cfg: V4Config | None = None, rcfg: ResidualConfig | None = None,
                 approaches=('top', 'oblique'), families=None):
        super().__init__()
        self.base = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle',
                                    families=families, approaches=approaches)
        self.rcfg = rcfg or ResidualConfig()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(RES_OBS_DIM,), dtype=np.float32)
        self.agent: ResidualScriptedGrasp | None = None
        self.prev_res = np.zeros(22, np.float32)
        self._stats: dict = {}

    # convenience pass-through (evaluation helpers read these)
    @property
    def sim(self):
        return self.base.sim

    @property
    def approach(self):
        return self.base.approach

    @property
    def plan(self):
        return self.base.plan

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def rw(self):
        return self.base.rw

    def _extra_obs(self) -> np.ndarray:
        ag, c = self.agent, self.sim.ctrl
        onehot = np.zeros(len(PHASES))
        onehot[PHASES.index(ag.phase)] = 1.0
        progress = min(1.0, ag.phase_step / PHASE_LEN[ag.phase])
        nominal_goal, _ = ag._goal()                   # pure lookup, no side effects
        x = np.concatenate((onehot, [progress], (nominal_goal - c.target_pos) * 10.0, self.prev_res))
        return x

    def _obs(self, base_obs) -> np.ndarray:
        x = np.concatenate((base_obs, self._extra_obs()))
        return np.clip(np.nan_to_num(x, nan=0.0, posinf=OBS_CLIP, neginf=-OBS_CLIP), -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.base.reset(seed=seed, options=options)
        self.agent = ResidualScriptedGrasp(self.base.sim, self.base.approach, self.rcfg)
        self.agent.make_plan()           # same state as the first act() of the scripted controller
        self.prev_res = np.zeros(22, np.float32)
        self._stats = {'n': 0, 'abs_r': 0.0, 'sat': 0.0, 'dp_cm': 0.0, 'dth_rad': 0.0, 'df_rad': 0.0, 'jitter': 0.0,
                       'close_step': -1, 'lift_step': -1, 'grasp_offset_cm': float('nan')}
        info['residual_config'] = self.rcfg.to_dict()
        return self._obs(obs), info

    def step(self, action):
        r = np.clip(np.nan_to_num(np.asarray(action, np.float32).reshape(-1), nan=0.0), -1.0, 1.0)
        self.agent.set_residual(r)
        phase_before = self.agent.phase
        a = self.agent.act()
        obs, rew, term, trunc, info = self.base.step(a)
        if self.rcfg.reward_version == 'v4r2':
            pen = -self.rcfg.w_residual * float(np.mean(np.square(r.astype(np.float64))))
            info['reward_terms'] = dict(info['reward_terms'], residual=pen)
            info['reward_v4r1'] = rew
            rew = rew + pen
        elif self.rcfg.reward_version != 'v4r1':
            raise ValueError(self.rcfg.reward_version)
        st, rc = self._stats, self.rcfg
        st['n'] += 1
        st['abs_r'] += float(np.abs(r).mean())
        st['jitter'] += float(np.abs(r - self.prev_res).mean())      # logging only: step-to-step change
        st['sat'] += float((np.abs(r) > 0.95).mean())
        st['dp_cm'] += 100 * rc.pos_max * float(np.linalg.norm(r[16:19]))
        st['dth_rad'] += rc.rot_max * float(np.linalg.norm(r[19:22]))
        st['df_rad'] += rc.finger_max * float(np.abs(r[:16]).mean())
        if phase_before != 'close' and self.agent.phase == 'close' and st['close_step'] < 0:
            st['close_step'] = st['n']
            st['grasp_offset_cm'] = 100 * float(np.linalg.norm(self.sim.ctrl.hand_pos() - self.agent.plan.grasp_pos))
        if phase_before != 'lift' and self.agent.phase == 'lift' and st['lift_step'] < 0:
            st['lift_step'] = st['n']
        info['scripted_phase'] = self.agent.phase
        info['executed_action'] = a
        self.prev_res = r.astype(np.float32)
        if term or trunc:
            info['residual_episode'] = self.episode_stats()
        return self._obs(obs), rew, term, trunc, info

    def episode_stats(self) -> dict:
        st = self._stats
        n = max(1, st['n'])
        return {'res_abs_mean': st['abs_r'] / n, 'res_jitter': st.get('jitter', 0.0) / n, 'res_saturation': st['sat'] / n, 'res_pos_offset_cm': st['dp_cm'] / n,
                'res_rot_offset_rad': st['dth_rad'] / n, 'res_finger_offset_rad': st['df_rad'] / n,
                'close_step': st['close_step'], 'lift_step': st['lift_step'], 'grasp_offset_cm': st['grasp_offset_cm'],
                'final_phase': self.agent.phase if self.agent else None}

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()


def zero_init_actor_mean(model) -> None:
    """Initial deterministic policy = zero residual = scripted controller (SB3 SAC actor.mu)."""
    import torch
    with torch.no_grad():
        model.actor.mu.weight.zero_()
        model.actor.mu.bias.zero_()


def init_actor_log_std(model, std: float) -> None:
    """State-independent initial exploration std (pre-tanh) for the SB3 SAC actor (Wave 3)."""
    import math
    import torch
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(math.log(std))


def tanh_gaussian_entropy(std: float, dim: int = 22, n: int = 200_000, seed: int = 0) -> float:
    """Monte-Carlo entropy of tanh(N(0, std^2)) per SB3's log-prob (incl. the tanh correction)."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, std, size=n)
    logp = -0.5 * (u / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi) - np.log(1 - np.tanh(u) ** 2 + 1e-6)
    return float(-dim * logp.mean())
