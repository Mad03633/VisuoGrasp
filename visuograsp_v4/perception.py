"""Classical RGB-D perception for V4 (no simulator object state is read here).

Same principles as V3.3 (colour segmentation of the red object, pinhole back-projection,
D0 geometric contact), extended for a camera that moves and rotates with the wrist:
- extrinsics are read from the rendered camera pose (data.cam_xpos / cam_xmat), so the
  wrist camera stays valid under any wrist rotation;
- the object estimate now includes an oriented bounding box from PCA of the fused point
  cloud (centre, axes, half extents) that can drive the grasp planner (OBBSupport).
Limitations: segmentation is a colour threshold (red objects only); there is no learned
segmentation and no D2 contact detector for the new viewpoints yet.
"""
from __future__ import annotations

import cv2
import numpy as np


class TemporalConfirmation:
    """V3.3 rule: on after k frames >= high, off after m frames < low (per finger)."""

    def __init__(self, n=4, k=3, m=2, high=0.55, low=0.35):
        self.n, self.k, self.m, self.high, self.low = n, k, m, high, low
        self.reset()

    def reset(self):
        self.on = np.zeros(self.n, bool)
        self.up = np.zeros(self.n, int)
        self.down = np.zeros(self.n, int)

    def update(self, p) -> np.ndarray:
        p = np.asarray(p)
        self.up = np.where(p >= self.high, self.up + 1, 0)
        self.down = np.where(p < self.low, self.down + 1, 0)
        self.on[(~self.on) & (self.up >= self.k)] = True
        self.on[self.on & (self.down >= self.m)] = False
        return self.on.astype(np.float32)


class RGBDPerceptionV4:
    def __init__(self, scene, size: int = 128, depth_noise: float = 0.001, dropout: float = 0.01,
                 seed: int = 0, cameras=('wrist', 'side')):
        self.s = scene
        self.size = size
        self.depth_noise, self.dropout = depth_noise, dropout
        self.rng = np.random.default_rng(seed)
        self.cameras = tuple(cameras)
        self.cam_ids = {c: scene.model.camera(c).id for c in self.cameras}
        self.renderer = scene.renderer(size, size)

    @staticmethod
    def segment(rgb) -> np.ndarray:
        rgb = rgb.astype(np.int16)
        mask = (rgb[:, :, 0] > 65) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.4) & (rgb[:, :, 0] > rgb[:, :, 2] * 1.35)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)

    def frame(self, camera: str):
        r, d = self.renderer, self.s.data
        r.update_scene(d, camera=camera)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        try:
            r.update_scene(d, camera=camera)
            depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        if self.depth_noise > 0:
            depth = np.maximum(0.001, depth + self.rng.normal(0, self.depth_noise, depth.shape))
            depth[self.rng.random(depth.shape) < self.dropout] = 0.0
        return rgb, depth.astype(np.float32)

    def reconstruct(self, mask, depth, cam_id, max_points=1500) -> np.ndarray:
        vv, uu = np.nonzero(mask & (depth > 0.005) & (depth < 1.5))
        if len(uu) == 0:
            return np.empty((0, 3), np.float32)
        if len(uu) > max_points:
            idx = np.linspace(0, len(uu) - 1, max_points, dtype=int)
            uu, vv = uu[idx], vv[idx]
        z = depth[vv, uu].astype(float)
        m, d = self.s.model, self.s.data
        focal = 0.5 * self.size / np.tan(np.deg2rad(m.cam_fovy[cam_id]) * 0.5)
        c = (self.size - 1) * 0.5
        pts_cam = np.stack(((uu - c) * z / focal, (c - vv) * z / focal, -z), axis=1)
        R = d.cam_xmat[cam_id].reshape(3, 3)
        return (pts_cam @ R.T + d.cam_xpos[cam_id]).astype(np.float32)

    def observe(self, need_frames: bool = False) -> dict:
        pts, frames = [], {}
        for name in self.cameras:
            rgb, depth = self.frame(name)
            mask = self.segment(rgb)
            p = self.reconstruct(mask, depth, self.cam_ids[name])
            if len(p):
                pts.append(p)
            if need_frames:
                frames[name] = {'rgb': rgb, 'depth': depth, 'mask': mask}
        if not pts:
            return {'visible': 0.0, 'center': np.zeros(3), 'axes': np.eye(3), 'half_extents': np.zeros(3),
                    'points': np.empty((0, 3)), 'n_points': 0, 'frames': frames}
        P = np.concatenate(pts).astype(float)
        P = P[P[:, 2] > 0.002]                          # drop table-level noise
        if len(P) < 10:
            return {'visible': 0.0, 'center': np.zeros(3), 'axes': np.eye(3), 'half_extents': np.zeros(3),
                    'points': P, 'n_points': len(P), 'frames': frames}
        center = np.median(P, axis=0)
        X = P - center
        evals, evecs = np.linalg.eigh(np.cov(X.T))
        axes = evecs[:, ::-1]
        for i in range(3):                              # deterministic sign convention
            j = int(np.argmax(np.abs(axes[:, i])))
            if axes[j, i] < 0:
                axes[:, i] *= -1
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        proj = X @ axes
        lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
        box_center = center + axes @ (0.5 * (lo + hi))
        half = 0.5 * (hi - lo)
        # 'center' = OBB centre: about 2x closer to the true COM than the median of the visible
        # surface points (smoke test, 40 resets: mean 1.4-2.1 cm vs 2.7-3.3 cm)
        return {'visible': 1.0, 'center': box_center, 'median_center': center, 'box_center': box_center,
                'axes': axes, 'half_extents': half, 'points': P, 'n_points': len(P), 'frames': frames}

    def d0_probs(self, points) -> np.ndarray:
        """V3.3 D0: gap between object points and each distal pad capsule -> sigmoid."""
        if len(points) == 0:
            return np.zeros(4, np.float32)
        m, d, g = self.s.model, self.s.data, self.s.pad_geoms
        mid = d.geom_xpos[g]
        axis = d.geom_xmat[g].reshape(4, 3, 3)[:, :, 2]
        half, radius = m.geom_size[g, 1], m.geom_size[g, 0]
        start, seg = mid - axis * half[:, None], 2 * axis * half[:, None]
        diff = points[:, None, :] - start[None]
        alpha = np.clip(np.sum(diff * seg[None], axis=2) / np.maximum(np.sum(seg ** 2, axis=1)[None], 1e-12), 0, 1)
        nearest = start[None] + alpha[:, :, None] * seg[None]
        gap = np.linalg.norm(points[:, None, :] - nearest, axis=2).min(axis=0) - radius
        return (1.0 / (1.0 + np.exp(np.clip((gap - 0.006) / 0.004, -40, 40)))).astype(np.float32)
