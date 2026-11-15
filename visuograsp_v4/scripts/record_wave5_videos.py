"""Wave 5 videos from the final-test episodes.csv (full task, original V4 initial conditions; PRIVILEGED state).

For one learned policy: first top-down success, first 45-degree success, the most frequent failure, and stacked
clips (top row scripted, bottom row learned) for one episode the learned policy wins and one the scripted
controller wins. Each clip re-runs its episode (torch 1 thread) and checks the GT outcome against the CSV.

    python -m visuograsp_v4.scripts.record_wave5_videos --policy e6 --spec "sb3:wave5/frozen_policies/e6.zip@mask"
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

W5 = OUTPUT_ROOT / 'rl' / 'wave5'
PRIV = 'PRIVILEGED: object pose + contacts = simulator ground truth'


def record(spec, row, label):
    import torch
    from visuograsp_v4 import rotations as rot
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.scripts.wave5_eval import make_policy
    from visuograsp_v4.video import overlay, render_panels
    from visuograsp_v4.wave4 import evaluate_grid
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    act, on_reset = make_policy(spec, env)
    frames, k = [], {'n': 0}

    def act_rec(o):
        a = act(o)
        if k['n'] % 2 == 0:
            s, t = env.sim, env.sim.tracker
            pads = ''.join(str(int(v)) for v in s.contact_summary()['pads'])
            frames.append(overlay(render_panels(s), [
                f'{label} | {row.approach} | {row.family}/{row.variant}/{row.resting_mode} | seed {int(row.seed)}',
                f'step {k["n"]}  lift {100 * s.object_lift():.1f} cm  GT pads {pads}  hold {t.gt_hold}/60  GT success {int(t.gt_success)}',
                f'tilt {rot.finger_tilt_deg(s.ctrl.hand_quat()):.0f} deg | {PRIV}']))
        k['n'] += 1
        return a
    with torch.no_grad():
        df = evaluate_grid(env, act_rec, [(int(row.seed), row.family, row.variant, row.resting_mode, row.approach)], on_reset=on_reset)
    return df.iloc[0], frames


def write(path, frames):
    from visuograsp_v4.video import Mp4Writer
    h, w = frames[0].shape[:2]
    wr = Mp4Writer(path, fps=10, size=(w, h))
    for f in frames:
        wr.add(f)
    wr.close()
    return wr.frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-dir', default='final_test/final')
    ap.add_argument('--policy', required=True)
    ap.add_argument('--spec', required=True)
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    ep = pd.read_csv(W5 / args.eval_dir / 'episodes.csv')
    a = ep[ep.policy == args.policy].set_index(['seed', 'approach'])
    s = ep[ep.policy == 'scripted'].set_index(['seed', 'approach'])
    j = a.join(s[['gt_success', 'outcome']], rsuffix='_scripted').reset_index().sort_values('seed')
    out = W5 / 'videos' / args.policy
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    for apn in ('top', 'oblique'):
        g = j[(j.approach == apn) & (j.outcome == 'success_retained')]
        if len(g):
            picks.append(('single', f'{apn}_success', g.iloc[0]))
    f = j[~j.gt_success.astype(bool)]
    if len(f):
        top = f.outcome.value_counts().index[0]
        picks.append(('single', f'failure_{top}', f[f.outcome == top].iloc[0]))
    win = j[j.gt_success.astype(bool) & ~j.gt_success_scripted.astype(bool)]
    loss = j[~j.gt_success.astype(bool) & j.gt_success_scripted.astype(bool)]
    if len(win):
        picks.append(('pair', 'side_by_side_learned_wins', win.iloc[0]))
    if len(loss):
        picks.append(('pair', 'side_by_side_scripted_wins', loss.iloc[0]))
    label = args.label or f'Wave 5 {args.policy}'
    rows = []
    for mode, tag, r in picks:
        res, fr = record(args.spec, r, label)
        rep = bool(res.gt_success) == bool(r.gt_success) and res.outcome == r.outcome
        name = f'{tag}_{r.approach}_{r.family}_seed{int(r.seed)}.mp4'
        so = r.outcome_scripted
        if mode == 'pair':
            sres, fs = record('scripted', r, 'Scripted controller (PRIVILEGED planner)')
            rep = rep and bool(sres.gt_success) == bool(r.gt_success_scripted)
            n = max(len(fr), len(fs))
            fr += [fr[-1]] * (n - len(fr))
            fs += [fs[-1]] * (n - len(fs))
            nf = write(out / name, [np.concatenate((x, y), axis=0) for x, y in zip(fs, fr)])
            so = sres.outcome
        else:
            nf = write(out / name, fr)
        rows.append({'file': name, 'kind': tag, 'seed': int(r.seed), 'approach': r.approach, 'family': r.family,
                     'learned_outcome': res.outcome, 'learned_outcome_csv': r.outcome, 'scripted_outcome': so,
                     'reproduced': rep, 'frames': nf})
        print(f'[video] {name}: learned {res.outcome} (csv {r.outcome}), scripted {so}, reproduced={rep}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'manifest.csv', index=False)
    print('written to', out)


if __name__ == '__main__':
    main()
