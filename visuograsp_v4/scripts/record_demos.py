"""Record scripted V4 demonstrations (overview | side | wrist) and verify the outcome.

Each demo replays a fixed seed; the ground-truth outcome is recomputed during recording and
written to demo_manifest.csv. A demo is never labelled from a single frame.

    python -m visuograsp_v4.scripts.record_demos
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from visuograsp_v4 import rotations as rot
from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.evaluation import run_scripted_episode
from visuograsp_v4.sim import VisuoGraspSimV4
from visuograsp_v4.video import Mp4Writer, overlay, render_panels

DEMOS = [
    # label, family, variant, resting_mode, approach, seed
    ('top_box_cube', 'box', 'cube', 'upright', 'top', 7_000_001),
    ('top_cylinder_lying', 'cylinder', 'std', 'lying', 'top', 7_000_002),
    ('top_l_shape_flat', 'l_shape', 'std', 'flat', 'top', 7_000_003),
    ('top_ring_flat', 'ring', 'std', 'flat', 'top', 7_000_004),
    ('oblique_bottle', 'bottle', 'std', 'upright', 'oblique', 7_000_005),
    ('oblique_box_tall', 'box', 'tall', 'upright', 'oblique', 7_000_006),
    ('side_box_tall', 'box', 'tall', 'upright', 'side', 7_000_007),
    ('side_bottle', 'bottle', 'std', 'upright', 'side', 7_000_008),
    ('side_cylinder_tall', 'cylinder', 'tall', 'upright', 'side', 7_000_009),
    ('side_box_cube_marginal_top_edge', 'box', 'cube', 'upright', 'side', 7_000_010),
]


def main() -> None:
    # MuJoCo writes MUJOCO_LOG.TXT into the working directory on warnings: keep it inside V4 outputs
    import os
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()
    cfg = V4Config()
    out = OUTPUT_ROOT / 'demos' / f'scripted_{cfg.physics.label}'
    out.mkdir(parents=True, exist_ok=True)
    sim = VisuoGraspSimV4(cfg)
    rows = []
    for label, family, variant, mode, approach, seed in DEMOS:
        if args.only and label not in args.only:
            continue
        rng = np.random.default_rng(seed)
        spec = sim.sample_spec(rng, family, variant, mode, scale=1.05)
        writer = Mp4Writer(out / f'{label}.mp4', fps=10)

        def cb(s, agent, step_out, step, writer=writer):
            if step % 2:
                return
            t = s.tracker
            pads = ''.join(str(int(v)) for v in step_out['contacts']['pads'])
            img = render_panels(s)
            writer.add(overlay(img, [f'{label}  step {step}  phase {agent.phase}',
                                     f'lift {100 * step_out["lift"]:.1f} cm  GT pads {pads}  hold {t.gt_hold}/60',
                                     f'GT success {int(t.gt_success)}  finger tilt {rot.finger_tilt_deg(s.ctrl.hand_quat()):.0f} deg']))

        row =run_scripted_episode(sim, spec, approach, seed, frame_callback=cb)
        writer.close()
        row.update(label=label, file=f'{label}.mp4', video_frames=writer.frames)
        rows.append(row)
        print(f'[demo] {label}: GT={row["gt_success"]} any={row["gt_success_any_phalanx"]} '
              f'lift={row["max_lift_cm"]:.1f} cm outcome={row["outcome"]} frames={writer.frames}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'demo_manifest.csv', index=False)
    (out / 'README.json').write_text(json.dumps({'note': 'Scripted V4 baseline, PRIVILEGED object pose for '
                                                 'planning. Panels: overview | side | wrist. 10 fps (every 2nd step).'},
                                                indent=1), encoding='utf-8')
    print('written to', out)


if __name__ == '__main__':
    main()
