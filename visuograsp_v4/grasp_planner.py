"""Approach-pose generation for top-down, oblique and side grasps.

Conventions (hand frame): fingers extend along -z_hand; the thumb (finger 0) sits at
-x_hand and finger 3 at +x_hand, fingers 1 and 2 at +-60 deg from +x_hand. Closing the
fingers squeezes the object mainly along x_hand. ``u`` is the unit vector from the object
towards the palm; the approach direction (palm -> object) is d = -u.

Approach families:
  top      tilt  0 deg  (u = +z)
  oblique  tilt 45 deg
  side     tilt 90 deg  (u horizontal, fingers horizontal, hand y_axis vertical)

The plan uses the object's pose and geometry. In the scripted baseline these come from
the simulator (PRIVILEGED, labelled as such); the RL environment can feed the same
planner with a camera estimate instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import objects as objs
from . import rotations as rot

APPROACH_TILT_DEG = {'top': 0.0, 'oblique': 45.0, 'side': 90.0}


@dataclass
class GraspPlan:
    approach: str
    azimuth: float
    u: np.ndarray
    grasp_pos: np.ndarray
    grasp_quat: np.ndarray
    pre_pos: np.ndarray
    transit_pos: np.ndarray
    lift_pos: np.ndarray
    pinch_width: float
    target_point: np.ndarray
    notes: list = field(default_factory=list)
    feasible: bool = True      # geometric feasibility for this hand (see notes)

    def to_dict(self) -> dict:
        return {'approach': self.approach, 'azimuth_deg': math.degrees(self.azimuth),
                'pinch_width_cm': 100 * self.pinch_width,
                'grasp_z': float(self.grasp_pos[2]), 'notes': ';'.join(self.notes)}


PALM_GAP = 0.027          # palm half thickness (12 mm) + 15 mm clearance to the object surface
TOP_MIN_PALM_Z = 0.112    # straight fingers: pad bottom 4 mm above the table (V3.3 used 0.118)
PRE_DISTANCE = 0.10       # pre-grasp: back off 10 cm along u
TRANSIT_CLEARANCE = 0.17  # transit height above the object top
LIFT_HEIGHT = 0.17
SIDE_MIN_Z = 0.068        # lowest palm-centre height that keeps the lower finger off the table
PLAN_CLEARANCE = 0.005    # open hand at the grasp pose must clear the table by this much
GRASP_DEPTH = 0.075       # side/oblique: object target point this far from the palm (pad region)
LATERAL_SHIFT = 0.008     # side/oblique: hand offset along +x_hand (clearance for fingers 1/2)


class GeomSupport:
    """Exact support function of the simulated object (PRIVILEGED geometry)."""

    def __init__(self, model, data, geom_ids):
        self.model, self.data, self.geom_ids = model, data, geom_ids

    def support(self, u) -> float:
        return objs.support(self.model, self.data, self.geom_ids, u)


class OBBSupport:
    """Support function of an oriented box (camera estimate: centre, axes columns, half extents)."""

    def __init__(self, center, axes, half_extents):
        self.c = np.asarray(center, float)
        self.A = np.asarray(axes, float)
        self.e = np.asarray(half_extents, float)

    def support(self, u) -> float:
        u = np.asarray(u, float)
        u = u / np.linalg.norm(u)
        return float(u @ self.c + np.sum(np.abs(self.A.T @ u) * self.e))


def _candidate_azimuths(R_obj: np.ndarray) -> list[float]:
    x = R_obj[:, 0]
    if math.hypot(x[0], x[1]) < 0.3:
        x = R_obj[:, 1]
    base = math.atan2(x[1], x[0])
    return [base + k * math.pi / 4 for k in range(8)]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_grasp(shape, object_quat, approach: str, hand_quat,
               hand_pos=None, lift_height: float = LIFT_HEIGHT, azimuth: float | None = None,
               object_com=None, lowest_fn=None) -> GraspPlan:
    """Return pre-grasp, grasp and lift poses for one approach direction.

    ``shape`` answers support queries: GeomSupport (exact simulator geometry, PRIVILEGED)
    or OBBSupport (oriented box estimated from the camera point cloud). ``lowest_fn(pos, quat)``
    predicts the open hand's lowest point; the grasp pose is raised until it clears the table.
    """
    tilt = math.radians(APPROACH_TILT_DEG[approach])
    R_obj = rot.to_mat(object_quat)
    top = shape.support([0.0, 0.0, 1.0])
    cx = 0.5 * (shape.support([1.0, 0, 0]) - shape.support([-1.0, 0, 0]))
    cy = 0.5 * (shape.support([0, 1.0, 0]) - shape.support([0, -1.0, 0]))
    hand_R = rot.to_mat(hand_quat)
    hand_xy = np.asarray(hand_pos if hand_pos is not None else [cx, cy, 0.3], float)
    notes: list[str] = []

    # choose the azimuth / pinch axis that gives the narrowest pinch, prefer small hand motion
    best = None
    for phi in ([azimuth] if azimuth is not None else _candidate_azimuths(R_obj)):
        u = np.array([math.sin(tilt) * math.cos(phi), math.sin(tilt) * math.sin(phi), math.cos(tilt)])
        if approach == 'top':
            x_axis = np.array([math.cos(phi), math.sin(phi), 0.0])          # pinch axis
        else:
            x_axis = np.array([-math.sin(phi), math.cos(phi), 0.0])         # horizontal, perpendicular to u
        width = shape.support(x_axis) + shape.support(-x_axis)
        R_goal = rot.frame_from_z_x(u, x_axis)
        motion = rot.angle_between(rot.from_mat(R_goal), rot.from_mat(hand_R))
        if approach != 'top':
            start_phi = math.atan2(hand_xy[1] - cy, hand_xy[0] - cx)
            motion += 0.5 * abs(_wrap(phi - start_phi))
        score = width + 0.004 * motion
        if best is None or score < best[0]:
            best = (score, phi, u, x_axis, width, R_goal)
    _, phi, u, x_axis, width, R_goal = best

    com_z = float(object_com[2]) if object_com is not None else 0.5 * top
    if approach == 'top':
        target = np.array([cx, cy, top])
        palm = np.array([cx, cy, max(top + 0.030, TOP_MIN_PALM_Z)])
    else:
        if approach == 'side':
            # grasp just above the centre of mass (widest, heaviest part for bottles)
            z_t = max(SIDE_MIN_Z, min(top - 0.020, com_z + 0.012))
            if z_t > top - 0.012:
                notes.append('side_grasp_above_object_top')
        else:
            z_t = max(0.035, top - 0.020)
        target = np.array([cx, cy, z_t])
        near = shape.support(u) - float(u @ target)
        # shift the hand towards +x_hand so fingers 1/2 (at x_hand = +3 cm) pass beside the object
        palm = target + u * max(near + PALM_GAP, GRASP_DEPTH) + x_axis * LATERAL_SHIFT

    grasp_quat = rot.from_mat(R_goal)
    if lowest_fn is not None:
        raised = 0.0
        while lowest_fn(palm, grasp_quat) < PLAN_CLEARANCE and raised < 0.08:
            palm = palm + np.array([0.0, 0.0, 0.002])
            raised += 0.002
        if raised > 0:
            notes.append(f'raised_for_clearance_{100 * raised:.1f}cm')
    pre = palm + u * PRE_DISTANCE
    transit = pre.copy()
    transit[2] = max(pre[2], top + TRANSIT_CLEARANCE)
    lift = palm + np.array([0.0, 0.0, lift_height])
    if width > 2 * 0.060 - 0.010:
        notes.append('object_wider_than_hand')
    feasible = not any(n in notes for n in ('side_grasp_above_object_top', 'object_wider_than_hand'))
    return GraspPlan(approach, phi, u, palm, grasp_quat, pre, transit, lift, width, target, notes, feasible)
