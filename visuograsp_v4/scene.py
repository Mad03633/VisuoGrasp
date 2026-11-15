"""MJCF generation for the V4 scene and a model cache (no per-episode recompilation).

Model structure
---------------
worldbody
  table (plane), two lights, fixed cameras 'side' (V3.3 pose) and 'overview'
  hand_target  (mocap body, visual marker only, no collision)
  hand         (free body, gravcomp) -- palm box, sites palm_center/grasp_center,
               camera 'wrist', 4 fingers x 4 hinge joints (V3.3 finger geometry)
  object       (free body, geoms with uniform density -> true COM and inertia)
equality: weld hand_target -> hand        (the 6-DoF wrist "actuator")
actuator: 16 position servos on finger joints
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from . import objects as objs
from .config import HandConfig, PhysicsConfig, V4Config


def _camera_quat(eye, target) -> str:
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    from .rotations import from_mat
    q = from_mat(np.column_stack((right, up, -f)))
    return ' '.join(f'{v:.9g}' for v in q)


def _finger_xml(h: HandConfig, i: int, angle: float) -> str:
    r = h.finger_base_radius
    x, y = r * math.cos(angle), r * math.sin(angle)
    quat = f'{math.cos(angle / 2):.8f} 0 0 {math.sin(angle / 2):.8f}'
    color = '.12 .74 .91 1' if i == 0 else '.20 .42 .75 1'
    grav = ' gravcomp="1"' if h.gravcomp else ''
    fr = ' '.join(f'{v:g}' for v in h.finger_friction)
    parts = []
    for j, L in enumerate(h.finger_links):
        offset = -h.finger_links[j - 1] if j else 0.0
        radius = h.pad_radius if j == 3 else h.link_radius
        site = f'<site name="tip_{i}" pos="0 0 {-L:.6f}" size=".003"/>' if j == 3 else ''
        parts.append(
            f'<body name="f{i}_b{j}" pos="0 0 {offset:.6f}"{grav}>'
            f'<joint name="f{i}_j{j}" type="hinge" axis="0 1 0" range="0 {h.joint_range[j]}" '
            f'damping="{h.joint_damping}" armature="{h.joint_armature}"/>'
            f'<geom name="f{i}_link{j}" type="capsule" fromto="0 0 0 0 0 {-L:.6f}" size="{radius:.4f}" '
            f'mass="{h.link_masses[j]}" rgba="{color}" friction="{fr}"/>{site}')
    return (f'<body name="finger_{i}" pos="{x:.6f} {y:.6f} -.002" quat="{quat}"{grav}>'
            + ''.join(parts) + '</body>' * 5)


def hand_xml(h: HandConfig) -> str:
    grav = ' gravcomp="1"' if h.gravcomp else ''
    px, py, pz = h.palm_half
    fingers = ''.join(_finger_xml(h, i, a) for i, a in enumerate(h.finger_angles))
    return f"""
  <body name="hand_target" mocap="true" pos="0 0 0.3">
    <geom name="target_marker" type="sphere" size=".006" contype="0" conaffinity="0" group="5" rgba="0 1 0 .5"/>
  </body>
  <body name="hand" pos="0 0 0.3"{grav}>
    <freejoint name="hand_free"/>
    <geom name="palm" type="box" size="{px} {py} {pz}" mass="{h.palm_mass}" rgba=".32 .38 .45 1"/>
    <site name="palm_center" pos="0 0 0" size=".004"/>
    <site name="grasp_center" pos="0 0 {-h.grasp_center_offset}" size=".004" rgba="1 1 0 .6"/>
    <camera name="wrist" pos="0 0 -.024" fovy="{h.wrist_camera_fovy}"/>
    {fingers}
  </body>"""


def actuators_xml(h: HandConfig) -> str:
    return ''.join(
        f'<position name="motor_{i}_{j}" joint="f{i}_j{j}" kp="{h.kp[j]}" forcelimited="true" '
        f'forcerange="-{h.force_limit} {h.force_limit}" ctrlrange="0 {h.joint_range[j]}"/>'
        for i in range(4) for j in range(4))


def scene_xml(cfg: V4Config, family: str, variant: str, scale: float) -> str:
    p: PhysicsConfig = cfg.physics
    h: HandConfig = cfg.hand
    noslip = f' noslip_iterations="{p.noslip_iterations}"' if p.noslip_iterations > 0 else ''
    side_q = _camera_quat((.27, -.24, .23), (0, 0, .047))
    over_q = _camera_quat((.55, -.45, .42), (0, 0, .10))
    wsolref = ' '.join(f'{v:g}' for v in h.weld_solref)
    wsolimp = ' '.join(f'{v:g}' for v in h.weld_solimp)
    csolref = ' '.join(f'{v:g}' for v in p.contact_solref)
    obj = objs.object_body_xml(family, variant, scale, cfg.objects.nominal_mass)
    return f"""<mujoco model="visuograsp_v4_{family}_{variant}_{scale:.2f}">
 <compiler angle="radian" autolimits="true"/>
 <option timestep="{p.timestep}" gravity="0 0 {p.gravity}" integrator="{p.integrator}" iterations="{p.iterations}" cone="{p.cone}" impratio="{p.impratio}"{noslip}/>
 <visual><global offwidth="640" offheight="480"/><map znear="0.004"/></visual>
 <default><geom condim="4" friction="1.2 .01 .002" solref="{csolref}" solimp=".90 .95 .001"/></default>
 <worldbody>
  <light pos="-.3 -.3 .9" diffuse=".9 .9 .9"/>
  <light pos=".3 .35 .7" diffuse=".5 .5 .5"/>
  <geom name="table" type="plane" size=".6 .6 .02" rgba=".67 .73 .79 1" friction="1.1 .01 .001"/>
  <camera name="side" pos=".27 -.24 .23" quat="{side_q}" fovy="51"/>
  <camera name="overview" pos=".55 -.45 .42" quat="{over_q}" fovy="50"/>
  {hand_xml(h)}
  {obj}
 </worldbody>
 <actuator>{actuators_xml(h)}</actuator>
 <equality><weld name="wrist_weld" body1="hand_target" body2="hand" relpose="0 0 0 1 0 0 0" solref="{wsolref}" solimp="{wsolimp}"/></equality>
</mujoco>"""


class CompiledScene:
    """One compiled (family, variant, scale) model with cached ids and nominal inertia."""

    def __init__(self, cfg: V4Config, family: str, variant: str, scale: float):
        self.family, self.variant, self.scale = family, variant, scale
        self.xml = scene_xml(cfg, family, variant, scale)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hand_body = m.body('hand').id
        self.object_body = m.body('object').id
        self.mocap_id = int(m.body_mocapid[m.body('hand_target').id])
        self.hand_qadr = int(m.jnt_qposadr[m.joint('hand_free').id])
        self.hand_vadr = int(m.jnt_dofadr[m.joint('hand_free').id])
        self.obj_qadr = int(m.jnt_qposadr[m.joint('object_free').id])
        self.obj_vadr = int(m.jnt_dofadr[m.joint('object_free').id])
        self.finger_joints = [m.joint(f'f{i}_j{j}').id for i in range(4) for j in range(4)]
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in self.finger_joints])
        self.finger_vadr = np.array([m.jnt_dofadr[j] for j in self.finger_joints])
        self.finger_lo = m.jnt_range[self.finger_joints, 0].copy()
        self.finger_hi = m.jnt_range[self.finger_joints, 1].copy()
        self.pad_geoms = np.array([m.geom(f'f{i}_link3').id for i in range(4)])
        self.tip_sites = np.array([m.site(f'tip_{i}').id for i in range(4)])
        self.palm_geom = m.geom('palm').id
        self.table_geom = m.geom('table').id
        self.grasp_site = m.site('grasp_center').id
        self.hand_geoms = np.array([g for g in range(m.ngeom)
                                    if m.geom_bodyid[g] != 0 and m.geom_bodyid[g] != self.object_body
                                    and m.geom_contype[g] != 0])
        self.finger_link_geoms = np.array([m.geom(f'f{i}_link{j}').id for i in range(4) for j in range(4)])
        self.object_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == self.object_body])
        self.nominal_mass = float(m.body_mass[self.object_body])
        self.nominal_inertia = m.body_inertia[self.object_body].copy()
        self.com_offset = m.body_ipos[self.object_body].copy()
        self.wrist_cam = m.camera('wrist').id
        self.side_cam = m.camera('side').id
        self.renderers: dict = {}

    def set_mass(self, mass: float) -> None:
        f = mass / self.nominal_mass
        self.model.body_mass[self.object_body] = mass
        self.model.body_inertia[self.object_body] = self.nominal_inertia * f

    def set_friction(self, mu: float) -> None:
        self.model.geom_friction[self.object_geoms, 0] = mu

    def renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (height, width)
        if key not in self.renderers:
            self.renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self.renderers[key]

    def close(self) -> None:
        for r in self.renderers.values():
            r.close()
        self.renderers.clear()


class SceneCache:
    """Compile each (family, variant, scale-bin) once per process."""

    def __init__(self, cfg: V4Config):
        self.cfg = cfg
        self._scenes: dict = {}
        self.compilations = 0

    def get(self, family: str, variant: str, scale: float) -> CompiledScene:
        key = (family, variant, round(float(scale), 4))
        if key not in self._scenes:
            self._scenes[key] = CompiledScene(self.cfg, family, variant, scale)
            self.compilations += 1
        return self._scenes[key]

    def close(self) -> None:
        for s in self._scenes.values():
            s.close()
        self._scenes.clear()
