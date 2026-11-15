"""Quaternion helpers. Convention: MuJoCo [w, x, y, z], unit norm, Hamilton product."""
from __future__ import annotations

import math

import mujoco
import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-12:
        return IDENTITY.copy()
    q = q / n
    return -q if q[0] < 0 else q


def mul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def from_rotvec(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return IDENTITY.copy()
    out = np.zeros(4)
    mujoco.mju_axisAngle2Quat(out, v / angle, angle)
    return out


def from_axis_angle(axis, angle_rad) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    return from_rotvec(axis / np.linalg.norm(axis) * angle_rad)


def diff_local(q_target, q_current) -> np.ndarray:
    """Rotation vector r in the frame of q_current with q_current * exp(r) = q_target."""
    out = np.zeros(3)
    mujoco.mju_subQuat(out, np.asarray(q_target, float), np.asarray(q_current, float))
    return out


def angle_between(q_a, q_b) -> float:
    return float(np.linalg.norm(diff_local(q_a, q_b)))


def to_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


def from_mat(R) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(R, float).reshape(-1))
    return normalize(out)


def slerp(q0, q1, t: float) -> np.ndarray:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return normalize(q0 + t * (q1 - q0))
    theta = math.acos(min(1.0, dot))
    s = math.sin(theta)
    return normalize((math.sin((1 - t) * theta) * q0 + math.sin(t * theta) * q1) / s)


def frame_from_z_x(z_axis, x_hint) -> np.ndarray:
    """Right-handed rotation matrix whose z column is z_axis and x column is closest to x_hint."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = alt - np.dot(alt, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def rot6d(R) -> np.ndarray:
    """Continuous 6-D orientation representation: first two columns of R."""
    R = np.asarray(R, float)
    return np.concatenate((R[:, 0], R[:, 1]))


def finger_tilt_deg(q) -> float:
    """Angle between the finger direction (-z of the hand) and world -z. 0 = top-down."""
    d = -to_mat(q)[:, 2]
    return math.degrees(math.acos(float(np.clip(-d[2], -1.0, 1.0))))
