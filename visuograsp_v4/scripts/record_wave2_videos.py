"""Wave 2 demonstration videos (overview | side | wrist), picked from the final-test episodes.

  1. successful top-down grasp of residual SAC
  2. successful 45-degree (oblique) grasp of residual SAC
  3. one informative failure of residual SAC (preferably an episode the scripted controller solved)
  4. side by side (stacked): scripted controller vs residual SAC on the same seed, for discordant
     episodes (residual succeeds / scripted fails, and the reverse)
Every episode is re-run (torch 1 thread) and its GT outcome compared with the CSV (reproduced).
Overlay: object pose and contacts are SIMULATOR GROUND TRUTH (privileged).

    python -m visuograsp_v4.scripts.record_wave2_videos --model best_model.zip
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

RL = OUTPUT_ROOT / 'rl'
FINAL = RL / 'wave2' / 'final_test'
PRIV = 'PRIVILEGED: object pose + contacts = simulator ground truth'
RCFG = None          # residual config (res1 = Wave 2 default, res2 = Wave 3)


def run_recorded(kind: str, model, seed, row, label: str):
    """kind: 'scripted' (base env) or 'residual' (residual env + model). Returns (result row, frames)."""
    from visuograsp_v4 import rotations as rot
    from visuograsp_v4.evaluation import run_policy_episode
    from visuograsp_v4.video import overlay, render_panels
    cfg = V4Config()
    holder, frames, k = {}, [], {'n': 0}
    if kind == 'scripted':
        from visuograsp_v4.env import VisuoGraspV4Env
        from visuograsp_v4.scripted import ScriptedGrasp
        env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
        on_reset = lambda e: holder.update(agent=ScriptedGrasp(e.sim, e.approach))
        policy = lambda obs: holder['agent'].act()
    else:
        from visuograsp_v4.residual import ResidualGraspEnv
        env = ResidualGraspEnv(cfg, RCFG, approaches=('top', 'oblique'))
        on_reset = None
        policy = lambda obs: model.predict(obs, deterministic=True)[0]

    def act(obs):
        a = policy(obs)
        if k['n'] % 2 == 0:
            s, t = env.sim, env.sim.tracker
            pads = ''.join(str(int(v)) for v in s.contact_summary()['pads'])
            extra = ''
            if kind == 'residual':
                r = np.asarray(a, float)
                extra = (f'  residual |dp| {100 * RCFG.pos_max * np.linalg.norm(r[16:19]):.2f} cm '
                         f'|df| {RCFG.finger_max * np.abs(r[:16]).mean():.3f} rad')
            frames.append(overlay(render_panels(s), [
                f'{label} | {row.approach} | {row.family}/{row.variant}/{row.resting_mode} | seed {int(seed)}',
                f'step {k["n"]}  lift {100 * s.object_lift():.1f} cm  GT pads {pads}  hold {t.gt_hold}/60  GT success {int(t.gt_success)}{extra}',
                f'tilt {rot.finger_tilt_deg(s.ctrl.hand_quat()):.0f} deg | {PRIV}']))
        k['n'] += 1
        return a
    res = run_policy_episode(env, act, int(seed), row.family, row.variant, row.resting_mode, row.approach, on_reset=on_reset)
    env.close()
    return res, frames


def write(path, frames) -> int:
    from visuograsp_v4.video import Mp4Writer
    h, w = frames[0].shape[:2]
    wr = Mp4Writer(path, fps=10, size=(w, h))
    for f in frames:
        wr.add(f)
    wr.close()
    return wr.frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='best_model.zip')
    ap.add_argument('--policy-label', default='residual_best')
    ap.add_argument('--run', default='residual_sac_seed11')
    ap.add_argument('--final', default='final_test', help='folder under outputs/rl/wave2 with episodes.csv')
    ap.add_argument('--out', default='videos')
    ap.add_argument('--group', default='wave2')
    ap.add_argument('--residual-version', default='res1')
    args = ap.parse_args()
    global RCFG
    from visuograsp_v4.residual import RESIDUAL_CONFIGS
    RCFG = RESIDUAL_CONFIGS[args.residual_version]
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.agents import load_sac
    model = load_sac(RL / args.group / args.run / args.model)
    ep = pd.read_csv(RL / args.group / args.final / 'episodes.csv')
    res = ep[ep.policy == args.policy_label].set_index(['seed', 'approach'])
    scr = ep[ep.policy == 'scripted'].set_index(['seed', 'approach'])
    j = res.join(scr[['gt_success', 'outcome']], rsuffix='_scripted').reset_index().sort_values('seed')
    out = RL / args.group / args.out
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    for ap_name in ('top', 'oblique'):
        g = j[(j.approach == ap_name) & j.gt_success.astype(bool)]
        pref = g[~g.gt_success_scripted.astype(bool)]
        if len(g):
            picks.append(('single', f'{ap_name}_success', (pref if len(pref) else g).iloc[0]))
    f = j[~j.gt_success.astype(bool)]
    pref = f[f.gt_success_scripted.astype(bool)]
    if len(f):
        picks.append(('single', 'informative_failure', (pref if len(pref) else f).iloc[0]))
    win = j[j.gt_success.astype(bool) & ~j.gt_success_scripted.astype(bool)]
    loss = j[~j.gt_success.astype(bool) & j.gt_success_scripted.astype(bool)]
    if len(win):
        picks.append(('pair', 'side_by_side_residual_wins', win.iloc[0]))
    if len(loss):
        picks.append(('pair', 'side_by_side_scripted_wins', loss.iloc[0]))
    if not len(win) and not len(loss):
        picks.append(('pair', 'side_by_side_same_outcome', j.iloc[0]))
    rows = []
    for mode, tag, r in picks:
        seed = int(r.seed)
        rr, fr = run_recorded('residual', model, seed, r, f'Residual SAC ({args.policy_label})')
        rep = bool(rr['gt_success']) == bool(r.gt_success) and rr['outcome'] == r.outcome
        name = f'{tag}_{r.approach}_{r.family}_seed{seed}.mp4'
        if mode == 'pair':
            sr, fs = run_recorded('scripted', None, seed, r, 'Scripted controller (zero residual)')
            rep = rep and bool(sr['gt_success']) == bool(r.gt_success_scripted)
            n = max(len(fr), len(fs))
            fr += [fr[-1]] * (n - len(fr))
            fs += [fs[-1]] * (n - len(fs))
            frames = [np.concatenate((a, b), axis=0) for a, b in zip(fs, fr)]      # top: scripted, bottom: residual
            nf = write(out / name, frames)
            scripted_outcome = sr['outcome']
        else:
            nf = write(out / name, fr)
            scripted_outcome = r.outcome_scripted
        rows.append({'file': name, 'kind': tag, 'seed': seed, 'approach': r.approach, 'family': r.family,
                     'residual_outcome': rr['outcome'], 'residual_outcome_csv': r.outcome,
                     'scripted_outcome': scripted_outcome, 'scripted_outcome_csv': r.outcome_scripted,
                     'reproduced': rep, 'frames': nf, 'layout': 'top row scripted, bottom row residual' if mode == 'pair' else 'single'})
        print(f'[video] {name}: residual {rr["outcome"]} (csv {r.outcome}), scripted {scripted_outcome}, reproduced={rep}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'manifest.csv', index=False)
    print('written to', out)


if __name__ == '__main__':
    main()
