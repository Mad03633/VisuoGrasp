"""Explicit configuration for VisuoGrasp V4. No notebook globals, no monkey-patching.

All numbers that define the physical system live here. A run stores ``asdict(config)``
next to its outputs, so every result can be traced to the exact configuration.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = V4_ROOT / 'outputs'


@dataclass(frozen=True)
class PhysicsConfig:
    """MuJoCo solver settings. Defaults equal the V3.3 contact settings for comparability."""
    timestep: float = 0.0025
    integrator: str = 'implicitfast'
    iterations: int = 65
    cone: str = 'elliptic'
    impratio: float = 10.0
    noslip_iterations: int = 5          # V3.3 value; 0 is the no-noslip sensitivity setting
    gravity: float = -9.81
    contact_solref: tuple = (0.006, 1.0)  # default geom contact (V3.3 value)

    @property
    def label(self) -> str:
        return (f'{self.cone}_imp{self.impratio:g}_noslip{self.noslip_iterations}'
                f'_dt{1000 * self.timestep:g}ms_sr{1000 * self.contact_solref[0]:g}')


@dataclass(frozen=True)
class HandConfig:
    """Four-finger, 16-joint hand. Finger geometry is identical to V3.3; the palm is now a
    free body attached to a mocap target through a weld constraint (6-DoF wrist)."""
    finger_links: tuple = (0.030, 0.027, 0.022, 0.018)
    link_radius: float = 0.0065
    pad_radius: float = 0.009
    finger_base_radius: float = 0.060
    finger_angles: tuple = (math.pi, math.pi / 3, -math.pi / 3, 0.0)   # finger 0 = thumb
    palm_half: tuple = (0.055, 0.055, 0.012)
    palm_mass: float = 0.20
    link_masses: tuple = (0.012, 0.012, 0.012, 0.009)
    joint_range: tuple = (1.45, 1.30, 1.30, 1.30)
    joint_damping: float = 0.025
    joint_armature: float = 0.001       # V3.3 used 5e-5; that value made squeezed grasps explode (see README_V4)
    kp: tuple = (2.5, 1.8, 1.8, 1.8)
    force_limit: float = 3.0
    finger_friction: tuple = (2.0, 0.02, 0.002)
    gravcomp: bool = True               # models an arm that carries the hand's own weight
    weld_solref: tuple = (0.01, 1.0)    # time constant (s), damping ratio
    weld_solimp: tuple = (0.95, 0.99, 0.001)
    grasp_center_offset: float = 0.075  # m along -z_hand: centre of the closed-finger region
    wrist_camera_fovy: float = 95.0


@dataclass(frozen=True)
class ControlConfig:
    """Bounded 22-D action interface: 16 finger deltas + 3 wrist translation + 3 rotation."""
    policy_hz: int = 20
    finger_delta_max: float = 0.055     # rad per step at |a| = 1 (V3.3 value)
    wrist_trans_max: float = 0.010      # m per step per axis, world frame
    wrist_rot_max: float = 0.060        # rad per step per axis, hand frame
    workspace_lo: tuple = (-0.30, -0.30, 0.03)
    workspace_hi: tuple = (0.30, 0.30, 0.45)
    max_tilt_deg: float = 110.0         # finger direction vs world -z; 90 = horizontal fingers
    table_clearance: float = 0.004      # lowest hand point (incl. capsule radius) above the table
    max_lead_pos: float = 0.025         # anti-windup: target may lead the real hand by at most this
    max_lead_rot: float = 0.30          # rad
    interpolate_substeps: bool = True   # move the mocap target smoothly inside one control step

    @property
    def n_actions(self) -> int:
        return 22


@dataclass(frozen=True)
class ObjectConfig:
    families: tuple = ('box', 'cylinder', 'bottle', 'ring', 'l_shape')
    scale_bins: tuple = (0.85, 0.95, 1.05, 1.15)   # cached models, no per-episode recompilation
    nominal_mass: float = 0.15
    mass_range: tuple = (0.05, 0.30)
    friction_range: tuple = (0.5, 1.2)
    xy_range: float = 0.02                         # object position noise on the table (m)
    settle_seconds: float = 0.5
    max_settle_attempts: int = 6
    rest_lin_speed: float = 0.02
    rest_ang_speed: float = 0.30
    max_penetration: float = 0.002
    max_xy_after_settle: float = 0.08


@dataclass(frozen=True)
class TaskConfig:
    max_steps: int = 320
    hold_frames: int = 60               # 3 s at 20 Hz, as in V3.3
    lift_success: float = 0.10          # m of COM height gain
    lift_drop_arm: float = 0.035
    drop_height: float = 0.03
    contact_threshold_n: float = 0.2
    hand_start_height: tuple = (0.26, 0.32)
    hand_start_xy_radius: float = 0.08
    hand_start_max_tilt_deg: float = 25.0
    image_size: int = 128


@dataclass(frozen=True)
class RewardConfig:
    """Reward versions (documented in env.py):
    v4r0 - first V4 reward (per-step contact terms, plain distance-to-grasp-pose potential).
    v4r1 - corridor reach potential, contact as a potential, push potential, no success
           termination (removes the incentive to hover just below the success height).
    """
    version: str = 'v4r1'
    terminate_on_success: bool = False
    gamma: float = 0.99
    w_reach: float = 2.0
    reach_scale: float = 0.35          # m; v4r0 used 0.25 on the plain distance
    lateral_gain: float = 1.5          # extra cost for lateral offset near the grasp pose
    corridor_len: float = 0.10         # m; the pre-grasp distance of the planner
    w_align: float = 1.0
    w_contact: float = 1.0             # v4r1 potential: 0.5 * pads/4 + 0.5 * opposition
    w_lift: float = 3.0
    w_push: float = 1.0                # v4r1 potential: -w * clip(xy displacement / 5 cm) before lift
    push_scale: float = 0.05
    hold_rate: float = 0.10
    success_bonus: float = 10.0
    drop: float = -3.0
    table_collision: float = -0.02
    effort: float = -0.0005
    smoothness: float = -0.0005
    physics_failure: float = -5.0


@dataclass(frozen=True)
class SACConfig:
    """Recommended SAC V4 settings (SB3 2.9.2). V3.3 stabilisers kept: LayerNorm critics, n-step 10."""
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = 'auto_0.1'
    n_steps: int = 10
    net_arch: tuple = (256, 256)
    layernorm_critic: bool = True
    device: str = 'cpu'
    torch_threads: int = 2
    eval_every: int = 25_000
    checkpoint_every: int = 50_000
    q_health_every: int = 5_000


@dataclass(frozen=True)
class V4Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    version: str = 'v4.0-foundation'

    def to_dict(self) -> dict:
        return asdict(self)
