"""V4 object families: geometry, aspect-ratio variants, resting modes and support functions.

Physical modelling changes relative to V3.3:
- Mass is assigned through a uniform geom density, so MuJoCo computes the true centre of
  mass and inertia tensor of each composite shape (V3.3 used an isotropic inertia placed
  at the body origin for every shape).
- Per-episode mass randomisation rescales body mass and inertia together (uniform density
  keeps the centre of mass fixed), so no recompilation is needed.
- Resting poses come from a list of physically stable modes plus a random yaw, are placed
  with an exact support-function computation and are then verified by a settling simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import rotations as rot

OBJECT_RGBA = '.90 .13 .11 1'   # red: compatible with the V3.3 colour segmentation


@dataclass(frozen=True)
class GeomSpec:
    type: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    euler: tuple | None = None
    fromto: tuple | None = None

    def volume(self) -> float:
        if self.type == 'box':
            return 8.0 * self.size[0] * self.size[1] * self.size[2]
        if self.type == 'cylinder':
            return math.pi * self.size[0] ** 2 * 2.0 * self.size[1]
        if self.type == 'capsule':
            r = self.size[0]
            if self.fromto is not None:
                p = np.asarray(self.fromto, float)
                half = 0.5 * float(np.linalg.norm(p[3:] - p[:3]))
            else:
                half = self.size[1]
            return math.pi * r * r * 2.0 * half + 4.0 / 3.0 * math.pi * r ** 3
        if self.type == 'sphere':
            return 4.0 / 3.0 * math.pi * self.size[0] ** 3
        raise ValueError(self.type)


VARIANTS = {
    'box': {
        'cube': dict(hx=0.029, hy=0.029, hz=0.032),        # V3.3 box
        'tall': dict(hx=0.025, hy=0.025, hz=0.055),
        'flat': dict(hx=0.040, hy=0.026, hz=0.020),
    },
    'cylinder': {
        'std': dict(r=0.029, h=0.036),                     # V3.3 cylinder
        'tall': dict(r=0.024, h=0.060),
        'squat': dict(r=0.034, h=0.024),
    },
    'bottle': {
        'std': dict(r=0.029, h=0.036, nr=0.016, nh=0.024),
        'slim': dict(r=0.024, h=0.046, nr=0.012, nh=0.026),
    },
    'ring': {
        'std': dict(R=0.033, r=0.009),                     # V3.3 ring
        'wide': dict(R=0.038, r=0.008),
    },
    'l_shape': {
        'std': dict(a=0.032, t=0.012, hz=0.034),           # close to V3.3 L-shape
        'low': dict(a=0.032, t=0.012, hz=0.018),
        'long': dict(a=0.045, t=0.011, hz=0.026),
    },
}

_RX90 = tuple(rot.from_axis_angle([1, 0, 0], math.pi / 2))
_RY90 = tuple(rot.from_axis_angle([0, 1, 0], math.pi / 2))
RESTING_MODES = {
    'box': {'upright': (1.0, 0.0, 0.0, 0.0), 'side': _RX90, 'end': _RY90},
    'cylinder': {'upright': (1.0, 0.0, 0.0, 0.0), 'lying': _RX90},
    'bottle': {'upright': (1.0, 0.0, 0.0, 0.0), 'lying': _RX90},
    'ring': {'flat': (1.0, 0.0, 0.0, 0.0)},
    'l_shape': {'flat': (1.0, 0.0, 0.0, 0.0), 'standing': _RX90},
}


def geom_specs(family: str, variant: str, scale: float) -> list[GeomSpec]:
    p = {k: v * scale for k, v in VARIANTS[family][variant].items()}
    if family == 'box':
        return [GeomSpec('box', (p['hx'], p['hy'], p['hz']))]
    if family == 'cylinder':
        return [GeomSpec('cylinder', (p['r'], p['h']))]
    if family == 'bottle':
        shoulder_h = 0.008 * scale
        shoulder_r = 0.5 * (p['r'] + p['nr'])
        z_sh = p['h'] + shoulder_h
        z_neck = p['h'] + 2 * shoulder_h + p['nh']
        return [GeomSpec('cylinder', (p['r'], p['h'])),
                GeomSpec('cylinder', (shoulder_r, shoulder_h), pos=(0.0, 0.0, z_sh)),
                GeomSpec('cylinder', (p['nr'], p['nh']), pos=(0.0, 0.0, z_neck))]
    if family == 'ring':
        geoms = []
        for j in range(12):
            a, b = 2 * math.pi * j / 12, 2 * math.pi * (j + 1) / 12
            ft = (p['R'] * math.cos(a), p['R'] * math.sin(a), 0.0,
                  p['R'] * math.cos(b), p['R'] * math.sin(b), 0.0)
            geoms.append(GeomSpec('capsule', (p['r'],), fromto=ft))
        return geoms
    if family == 'l_shape':
        a, t, hz = p['a'], p['t'], p['hz']
        return [GeomSpec('box', (t, a, hz), pos=(-(a - t), 0.0, 0.0)),
                GeomSpec('box', (a, t, hz), pos=(0.0, -(a - t), 0.0))]
    raise ValueError(family)


def object_body_xml(family: str, variant: str, scale: float, nominal_mass: float) -> str:
    specs = geom_specs(family, variant, scale)
    density = nominal_mass / sum(s.volume() for s in specs)
    lines = []
    for k, s in enumerate(specs):
        size = ' '.join(f'{v:.6f}' for v in s.size)
        place = (f'fromto="{" ".join(f"{v:.6f}" for v in s.fromto)}"' if s.fromto is not None
                 else f'pos="{" ".join(f"{v:.6f}" for v in s.pos)}"')
        lines.append(f'<geom name="obj_g{k}" type="{s.type}" size="{size}" {place} density="{density:.4f}" '
                     f'priority="1" rgba="{OBJECT_RGBA}"/>')
    return ('<body name="object" pos="0 0 0.3">\n    <freejoint name="object_free"/>\n    '
            + '\n    '.join(lines) + '\n  </body>')


# ---------------------------------------------------------------------------------------
# Support functions on the compiled model (world frame, current data.geom_x*)
# ---------------------------------------------------------------------------------------
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_CYL = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_CAP = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_SPH = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_ELL = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)


def geom_support(model, data, gid: int, u: np.ndarray) -> float:
    """max_x u.x over geom gid (u unit vector, world frame)."""
    c = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    s = model.geom_size[gid]
    t = int(model.geom_type[gid])
    if t == _BOX:
        return float(u @ c + np.sum(np.abs(R.T @ u) * s[:3]))
    if t == _CYL:
        ua = float(u @ R[:, 2])
        return float(u @ c + abs(ua) * s[1] + s[0] * math.sqrt(max(0.0, 1.0 - ua * ua)))
    if t == _CAP:
        return float(u @ c + abs(float(u @ R[:, 2])) * s[1] + s[0])
    if t == _SPH:
        return float(u @ c + s[0])
    if t == _ELL:
        return float(u @ c + np.linalg.norm(s[:3] * (R.T @ u)))
    raise ValueError(f'unsupported geom type {t}')


def support(model, data, geom_ids, u) -> float:
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    return max(geom_support(model, data, g, u) for g in geom_ids)


def width_along(model, data, geom_ids, u) -> float:
    u = np.asarray(u, float)
    return support(model, data, geom_ids, u) + support(model, data, geom_ids, -u)


def lowest_z(model, data, geom_ids) -> float:
    return -support(model, data, geom_ids, np.array([0.0, 0.0, -1.0]))


def top_z(model, data, geom_ids) -> float:
    return support(model, data, geom_ids, np.array([0.0, 0.0, 1.0]))
