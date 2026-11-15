"""MP4 recording: overview | side camera | wrist camera, with a ground-truth overlay."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PANEL_W, PANEL_H = 320, 240


def render_panels(sim, cameras=('overview', 'side', 'wrist')) -> np.ndarray:
    r = sim.scene.renderer(PANEL_H, PANEL_W)
    panels = []
    for cam in cameras:
        r.update_scene(sim.data, camera=cam)
        img = r.render().copy()
        cv2.putText(img, cam, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(img)
    return np.concatenate(panels, axis=1)


def overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    y = PANEL_H - 8 - 16 * (len(lines) - 1)
    for line in lines:
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return out


class Mp4Writer:
    def __init__(self, path: Path, fps: int = 10, size=(3 * PANEL_W, PANEL_H)):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f'cv2.VideoWriter could not open {self.path}')
        self.frames = 0

    def add(self, rgb: np.ndarray) -> None:
        self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frames += 1

    def close(self) -> None:
        self.writer.release()
