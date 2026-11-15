"""Final-defense media: re-render selected, ALREADY EVALUATED episodes at 640x480 per camera.

No training, no new evaluation numbers. Policies are the frozen Wave 5 checkpoints; episodes are replayed
deterministically (torch 1 thread) from their seeds, and every clip's GT outcome is compared with the frozen
record (wave5/final_test/final/episodes.csv for policy clips, demos/.../demo_manifest.csv for scripted demos).
PRIVILEGED state-based control: object pose and contacts are simulator ground truth.

    python visuograsp_v4/outputs/final_defense/build/record_defense_clips.py
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]                    # ...\Project_2_2
sys.path.insert(0, str(ROOT))
from visuograsp_v4.config import OUTPUT_ROOT, V4Config  # noqa: E402

FD = OUTPUT_ROOT / 'final_defense'
OUT = FD / 'media' / 'rerendered'
FIN = OUTPUT_ROOT / 'rl' / 'wave5' / 'final_test' / 'final' / 'episodes.csv'
DEMO_MANIFEST = OUTPUT_ROOT / 'demos' / 'scripted_elliptic_imp10_noslip5_dt2.5ms_sr6' / 'demo_manifest.csv'
E6_SPEC = 'sb3:wave5/frozen_policies/e6_best_90k.zip@mask'
W, H, CAP = 640, 480, 96
CAMS = ('overview', 'side', 'wrist')
PRIV = 'PRIVILEGED state: object pose + contacts = simulator ground truth (not vision-based)'

# name, controller label, wave5 policy name (None = scripted demo), spec, seed, family, variant, resting_mode, approach
CLIPS = [
    ('e6_top_box_6700000', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6700000, 'box', 'cube', 'upright', 'top'),
    ('e6_oblique_box_6700100', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6700100, 'box', 'cube', 'upright', 'oblique'),
    ('e6_top_cylinder_6709000', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6709000, 'cylinder', 'std', 'upright', 'top'),
    ('e6_oblique_bottle_6715100', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6715100, 'bottle', 'std', 'upright', 'oblique'),
    ('e6_oblique_lshape_6721100', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6721100, 'l_shape', 'std', 'flat', 'oblique'),
    ('e6_top_ring_6719000', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6719000, 'ring', 'std', 'flat', 'top'),
    ('e6_oblique_box_6706109', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6706109, 'box', 'flat', 'upright', 'oblique'),
    ('scripted_oblique_box_6706109', 'Scripted expert (PRIVILEGED planner)', 'scripted', 'scripted', 6706109, 'box', 'flat', 'upright', 'oblique'),
    ('e6_fail_pushed_oblique_box_6706102', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6706102, 'box', 'flat', 'upright', 'oblique'),
    ('e6_fail_never_lifted_oblique_box_6704100', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6704100, 'box', 'tall', 'side', 'oblique'),
    ('e6_fail_ring_oblique_6719100', 'E6 (BC-anchored RL)', 'e6', E6_SPEC, 6719100, 'ring', 'std', 'flat', 'oblique'),
    ('scripted_demo_side_box_tall_7000007', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000007, 'box', 'tall', 'upright', 'side'),
    ('scripted_demo_side_bottle_7000008', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000008, 'bottle', 'std', 'upright', 'side'),
    ('scripted_demo_top_ring_flat_7000004', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000004, 'ring', 'std', 'flat', 'top'),
    ('scripted_demo_top_lshape_flat_7000003', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000003, 'l_shape', 'std', 'flat', 'top'),
    ('scripted_demo_top_cylinder_lying_7000002', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000002, 'cylinder', 'std', 'lying', 'top'),
    ('scripted_demo_oblique_bottle_7000005', 'Scripted expert (PRIVILEGED planner)', None, 'scripted', 7000005, 'bottle', 'std', 'upright', 'oblique'),
]


def caption(width, lines, h=CAP):
    bar = np.full((h, width, 3), 24, np.uint8)
    y = 26
    for i, ln in enumerate(lines):
        cv2.putText(bar, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62 if i < 2 else 0.5, (255, 255, 255) if i < 2 else (200, 200, 200),
                    1, cv2.LINE_AA)
        y += 28
    return bar


class Recorder:
    """Streams three MP4s per clip (all cameras, side, overview; each with a caption bar) and keyframe PNGs."""

    def __init__(self, name, label, approach, family, seed):
        self.name, self.label, self.approach, self.family, self.seed = name, label, approach, family, seed
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
        four = cv2.VideoWriter_fourcc(*'mp4v')
        self.w_all = cv2.VideoWriter(str(d / f'{name}_all.mp4'), four, 10, (3 * W, H + CAP))
        self.w_side = cv2.VideoWriter(str(d / f'{name}_side.mp4'), four, 10, (W, H + CAP))
        self.w_over = cv2.VideoWriter(str(d / f'{name}_overview.mp4'), four, 10, (W, H + CAP))
        self.buf = collections.deque(maxlen=10)
        self.events, self.frames, self.k = {}, 0, 0

    def frame(self, sim, step, phase=None):
        r = sim.scene.renderer(H, W)
        pan = {}
        for c in CAMS:
            r.update_scene(sim.data, camera=c)
            pan[c] = r.render().copy()
        t = sim.tracker
        pads = [int(v) for v in sim.contact_summary()['pads']]
        lift = 100 * sim.object_lift()
        l1 = f'{self.label} | {self.approach} | {self.family} | seed {self.seed}'
        l2 = f't = {step / 20:4.1f} s   lift {lift:5.1f} cm   GT pads {"".join(map(str, pads))}   hold {t.gt_hold}/60   GT success {int(t.gt_success)}'
        l3 = PRIV + ('' if phase is None else f'   | phase: {phase}')
        bgr = {c: cv2.cvtColor(pan[c], cv2.COLOR_RGB2BGR) for c in CAMS}
        for c in CAMS:
            cv2.putText(bgr[c], c, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        self.w_all.write(np.vstack((np.hstack([bgr[c] for c in CAMS]), caption(3 * W, [l1, l2, l3]))))
        self.w_side.write(np.vstack((bgr['side'], caption(W, [l1, l2, 'PRIVILEGED state (simulator pose + contacts)']))))
        self.w_over.write(np.vstack((bgr['overview'], caption(W, [l1, l2, 'PRIVILEGED state (simulator pose + contacts)']))))
        self.buf.append((self.k, {c: bgr[c] for c in CAMS}))
        ev = []
        if self.k == 0:
            ev.append('start')
        if 'contact' not in self.events and pads[0] and sum(pads) >= 2:
            ev.append('contact')
            if len(self.buf) > 8:
                self._save('approach', *self.buf[-9])
        if 'lift' not in self.events and lift >= 5.0:
            ev.append('lift')
        if 'hold' not in self.events and t.gt_success:
            ev.append('hold')
        for e in ev:
            self._save(e, self.k, {c: bgr[c] for c in CAMS})
        self.last = (self.k, {c: bgr[c] for c in CAMS})
        self.k += 1

    def _save(self, ev, k, pans):
        self.events[ev] = k
        for c, img in pans.items():
            cv2.imwrite(str(self.dir / f'key_{ev}_{c}.png'), img)

    def close(self):
        self._save('end', *self.last)
        for w in (self.w_all, self.w_side, self.w_over):
            w.release()
        return self.k


def run_policy_clip(clip):
    import torch
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.scripts.wave5_eval import make_policy
    from visuograsp_v4.wave4 import evaluate_grid
    name, label, pol, spec, seed, fam, var, mode, ap = clip
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    act, on_reset = make_policy(spec, env)
    rec = Recorder(name, label, ap, fam, seed)
    n = {'i': 0}

    def act_rec(o):
        a = act(o)
        if n['i'] % 2 == 0:
            ph = getattr(getattr(env, '_scr', None), 'phase', None)
            rec.frame(env.sim, n['i'], ph)
        n['i'] += 1
        return a
    with torch.no_grad():
        row = evaluate_grid(env, act_rec, [(seed, fam, var, mode, ap)], on_reset=on_reset).iloc[0]
    frames = rec.close()
    ref = pd.read_csv(FIN)
    ref = ref[(ref.policy == pol) & (ref.seed == seed) & (ref.approach == ap)].iloc[0]
    return row, ref, frames, rec.events


def run_scripted_demo(clip):
    from visuograsp_v4.evaluation import run_scripted_episode
    from visuograsp_v4.sim import VisuoGraspSimV4
    name, label, _, _, seed, fam, var, mode, ap = clip
    sim = VisuoGraspSimV4(V4Config())
    spec = sim.sample_spec(np.random.default_rng(seed), fam, var, mode, scale=1.05)
    rec = Recorder(name, label, ap, fam, seed)

    def cb(s, agent, step_out, step):
        if step % 2 == 0:
            rec.frame(s, step, agent.phase)
    row = run_scripted_episode(sim, spec, ap, seed, frame_callback=cb)
    frames = rec.close()
    ref = pd.read_csv(DEMO_MANIFEST)
    ref = ref[ref.seed == seed].iloc[0]
    return pd.Series(row), ref, frames, rec.events


def main():
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    only = set(sys.argv[1:])
    rows = []
    for clip in CLIPS:
        if only and clip[0] not in only:
            continue
        row, ref, frames, events = (run_scripted_demo if clip[2] is None else run_policy_clip)(clip)
        rep = bool(row.gt_success) == bool(ref.gt_success) and row.outcome == ref.outcome
        rows.append({'clip': clip[0], 'controller': clip[1], 'policy': clip[2] or 'scripted_demo', 'seed': clip[4], 'family': clip[5],
                     'variant': clip[6], 'resting_mode': clip[7], 'approach': clip[8], 'gt_success_rerun': bool(row.gt_success),
                     'outcome_rerun': row.outcome, 'gt_success_frozen': bool(ref.gt_success), 'outcome_frozen': ref.outcome,
                     'reproduced': rep, 'max_lift_cm': round(float(row.max_lift_cm), 2), 'frames': frames,
                     'keyframes': json.dumps(events),
                     'frozen_source': 'wave5/final_test/final/episodes.csv' if clip[2] else 'demos/scripted_elliptic_imp10_noslip5_dt2.5ms_sr6/demo_manifest.csv'})
        print(f'[clip] {clip[0]}: rerun {row.outcome} / frozen {ref.outcome} reproduced={rep} frames={frames} events={events}', flush=True)
    m = OUT / 'rerender_manifest.csv'
    df = pd.DataFrame(rows)
    if m.exists() and only:
        old = pd.read_csv(m)
        df = pd.concat([old[~old.clip.isin(df.clip)], df], ignore_index=True)
    df.to_csv(m, index=False)
    print('written', m)


if __name__ == '__main__':
    main()
