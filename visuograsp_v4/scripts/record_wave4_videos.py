"""Wave 4 videos from a final-test (or selection) episodes.csv: successes and failures of a learned
policy, and stacked side-by-side clips (top row scripted, bottom row learned) for discordant episodes.
Every clip re-runs its episode (torch 1 thread) and checks the GT outcome against the CSV.
Overlay: object pose + contacts = simulator ground truth (PRIVILEGED); protocol (full task / hand-over).

    python -m visuograsp_v4.scripts.record_wave4_videos --eval-dir final_test/full_v1 --policy e1 --source sb3:wave4/curriculum_sac_s11/best_model.zip
    python -m visuograsp_v4.scripts.record_wave4_videos --eval-dir final_test/subtask_v1 --policy e4 --source sb3:wave4/bc_grasp_lift_s11/best_model.zip --handover-stage 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

RL = OUTPUT_ROOT / 'rl'
W4 = RL / 'wave4'
PRIV = 'PRIVILEGED: object pose + contacts = simulator ground truth'


def make_act(source, env):
    import torch
    if source == 'scripted':
        from visuograsp_v4.scripted import ScriptedGrasp
        h = {}
        return (lambda o: h['a'].act()), (lambda e: h.update(a=ScriptedGrasp(e.sim, e.approach)))
    if source.startswith('bc:'):
        from visuograsp_v4.agents import make_sac
        from visuograsp_v4.config import SACConfig
        from visuograsp_v4.wave4 import actor_mean_action
        m = make_sac(env, SACConfig(), 0)
        m.actor.load_state_dict(torch.load(W4 / source[3:] / 'bc_actor_best.pt'))
        return (lambda o: actor_mean_action(m.actor, torch.as_tensor(o[None])).detach().numpy()[0]), None
    from visuograsp_v4.agents import load_sac
    m = load_sac(RL / source[4:])
    return (lambda o: m.predict(o, deterministic=True)[0]), None


def record(source, row, label, handover):
    from visuograsp_v4 import rotations as rot
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.video import overlay, render_panels
    from visuograsp_v4.wave4 import CurriculumGraspEnv, evaluate_grid
    cfg = V4Config()
    stage = None
    if handover is not None and source != 'scripted':
        env, stage = CurriculumGraspEnv(cfg, stage=handover, replay_prev_prob=0.0), handover
    else:
        env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    act, on_reset = make_act(source, env)
    frames, k = [], {'n': 0}
    proto = f'hand-over stage {handover}' if stage else 'full task from reset'

    def act_rec(o):
        a = act(o)
        if k['n'] % 2 == 0:
            s, t = env.sim, env.sim.tracker
            pads = ''.join(str(int(v)) for v in s.contact_summary()['pads'])
            frames.append(overlay(render_panels(s), [
                f'{label} | {proto} | {row.approach} | {row.family}/{row.variant}/{row.resting_mode} | seed {int(row.seed)}',
                f'policy step {k["n"]}  lift {100 * s.object_lift():.1f} cm  GT pads {pads}  hold {t.gt_hold}/60  GT success {int(t.gt_success)}',
                f'tilt {rot.finger_tilt_deg(s.ctrl.hand_quat()):.0f} deg | {PRIV}']))
        k['n'] += 1
        return a
    import torch
    with torch.no_grad():
        df = evaluate_grid(env, act_rec, [(int(row.seed), row.family, row.variant, row.resting_mode, row.approach)],
                           on_reset=on_reset, stage=stage)
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
    ap.add_argument('--eval-dir', required=True, help='relative to outputs/rl/wave4 (contains episodes.csv)')
    ap.add_argument('--policy', required=True, help='policy name in episodes.csv')
    ap.add_argument('--source', required=True)
    ap.add_argument('--handover-stage', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    ep = pd.read_csv(W4 / args.eval_dir / 'episodes.csv')
    a = ep[ep.policy == args.policy].set_index(['seed', 'approach'])
    s = ep[ep.policy == 'scripted'].set_index(['seed', 'approach'])
    j = a.join(s[['gt_success', 'outcome']], rsuffix='_scripted').reset_index().sort_values('seed')
    out = W4 / (args.out or f'videos/{args.policy}_{args.eval_dir.replace("/", "_")}')
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    for apn in ('top', 'oblique'):
        g = j[(j.approach == apn) & j.gt_success.astype(bool)]
        if len(g):
            picks.append(('single', f'{apn}_success', g.iloc[0]))
    f = j[~j.gt_success.astype(bool)]
    if len(f):
        picks.append(('single', f'failure_{f.outcome.value_counts().index[0]}', f[f.outcome == f.outcome.value_counts().index[0]].iloc[0]))
    win = j[j.gt_success.astype(bool) & ~j.gt_success_scripted.astype(bool)]
    loss = j[~j.gt_success.astype(bool) & j.gt_success_scripted.astype(bool)]
    if len(win):
        picks.append(('pair', 'side_by_side_learned_wins', win.iloc[0]))
    if len(loss):
        picks.append(('pair', 'side_by_side_scripted_wins', loss.iloc[0]))
    rows = []
    for mode, tag, r in picks:
        res, fr = record(args.source, r, f'Wave 4 {args.policy}', args.handover_stage)
        rep = bool(res.gt_success) == bool(r.gt_success) and res.outcome == r.outcome
        name = f'{tag}_{r.approach}_{r.family}_seed{int(r.seed)}.mp4'
        so = r.outcome_scripted
        if mode == 'pair':
            sres, fs = record('scripted', r, 'Scripted controller (PRIVILEGED planner)', None)
            rep = rep and bool(sres.gt_success) == bool(r.gt_success_scripted)
            if args.handover_stage is not None:      # align: drop the scripted prefix frames before the hand-over
                pre = int(r.get('handover_prefix_steps', 0) or 0) // 2
                fs = fs[pre:]
            n = max(len(fr), len(fs))
            fr += [fr[-1]] * (n - len(fr))
            fs += [fs[-1]] * (n - len(fs))
            nf = write(out / name, [np.concatenate((x, y), axis=0) for x, y in zip(fs, fr)])
            so = sres.outcome
        else:
            nf = write(out / name, fr)
        rows.append({'file': name, 'kind': tag, 'seed': int(r.seed), 'approach': r.approach, 'family': r.family,
                     'learned_outcome': res.outcome, 'learned_outcome_csv': r.outcome, 'scripted_outcome': so,
                     'reproduced': rep, 'frames': nf, 'protocol': 'hand-over' if args.handover_stage else 'full task'})
        print(f'[video] {name}: learned {res.outcome} (csv {r.outcome}), scripted {so}, reproduced={rep}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'manifest.csv', index=False)
    print('written to', out)


if __name__ == '__main__':
    main()
