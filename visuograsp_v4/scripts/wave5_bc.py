"""Wave 5 / E5: improved behaviour cloning of the scripted expert (supervised imitation, NOT RL).

Data: Wave 5 demonstrations (wave5_collect_demos.py), GT-successful episodes only; target = the expert's clean
label (for DART episodes the executed action was noisy, the label is not). Loss per sample:
    w_i * || tanh(mu(s_i)) - a_i ||^2 ,  w_i = phase weight (hold/done: --phase-weight, other phases 1)
                                          x approach-balance factor (total weight equal for top and oblique)
Minibatches are drawn with probability proportional to w_i (so the loss is an unweighted MSE on the
resampled batch). Actor = SB3 SAC actor (LayerNorm-SAC policy, --net hidden sizes), Adam, lr linearly 1e-3 -> 1e-4.
Model selection: deterministic GT success on DEV_A (seeds 8_800_000+, 10 per family x approach = 100
episodes) at --eval-epochs; best = highest GT (tie: lower active-phase validation MSE). PRIVILEGED setting.

    python -m visuograsp_v4.scripts.wave5_bc --run e5_clean_dart --data demos_clean demos_dart --epochs 40 --eval-epochs 10 20 30 40
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config

W5 = OUTPUT_ROOT / 'rl' / 'wave5'
DEV_A = 8_800_000
ACTIVE = (0, 1, 2, 3, 4)          # transit, descend, approach, close, lift  (hold = 5, done = 6)


def load(names, approach_filter=None):
    obs, lab, ph, ap, grp = [], [], [], [], []
    stats = []
    for j, n in enumerate(names):
        d = np.load(W5 / 'demos' / n / 'data.npz')
        ep = pd.read_csv(W5 / 'demos' / n / 'episodes.csv')
        ok = ep.set_index('episode').gt_success.to_dict()
        apd = ep.set_index('episode').approach.to_dict()
        m = np.array([ok[int(e)] for e in d['episode']])
        if approach_filter:
            m &= np.array([apd[int(e)] == approach_filter for e in d['episode']])
        obs.append(d['obs'][m]); lab.append(d['label'][m]); ph.append(d['phase'][m])
        ap.append(np.array([apd[int(e)] for e in d['episode'][m]])); grp.append(j * 100_000 + d['episode'][m])
        stats.append({'dataset': n, 'episodes': len(ep), 'successful_episodes': int(ep.gt_success.sum()),
                      'success_by_approach': ep.groupby('approach').gt_success.agg(['sum', 'size']).to_dict(),
                      'used_transitions': int(m.sum())})
    return (np.concatenate(obs), np.concatenate(lab), np.concatenate(ph), np.concatenate(ap), np.concatenate(grp)), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--net', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--phase-weight', type=float, default=0.25)
    ap.add_argument('--no-balance', action='store_true')
    ap.add_argument('--approach-filter', default=None, choices=(None, 'top', 'oblique'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--eval-epochs', type=int, nargs='+', default=[10, 20, 30, 40])
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--dev-per-cell', type=int, default=10)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--mask-prev-action', action='store_true', help='zero the prev_action block (copycat fix)')
    ap.add_argument('--mask-history', action='store_true', help='zero prev_action and all velocity blocks (copycat fix)')
    ap.add_argument('--history-dropout', type=float, default=0.0,
                    help='training-only: zero the history blocks of a sample with this probability (test obs unchanged)')
    ap.add_argument('--ema', type=float, default=0.0, help='EMA decay of the actor weights (evaluated and saved); 0 = off')
    ap.add_argument('--history-noise', type=float, default=0.0,
                    help='training-only: add N(0, (k * feature std)^2) to the history blocks (test obs unchanged)')
    args = ap.parse_args()
    out = W5 / 'bc' / args.run
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    from visuograsp_v4.agents import make_sac
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.wave4 import ALL_FAMILIES, actor_mean_action, dev_grid, evaluate_grid
    (obs, lab, ph, apr, grp), stats = load(args.data, args.approach_filter)
    from visuograsp_v4.wave5 import mask_history, mask_prev_action
    pre = mask_history if args.mask_history else (mask_prev_action if args.mask_prev_action else (lambda o: o))
    obs = pre(obs)
    w = np.where(np.isin(ph, ACTIVE), 1.0, args.phase_weight)
    if not args.no_balance and args.approach_filter is None:
        for a in ('top', 'oblique'):
            m = apr == a
            w[m] *= 0.5 / (w[m].sum() / w.sum())
    ug = np.unique(grp)
    val_g = set(rng.choice(ug, max(1, len(ug) // 10), replace=False).tolist())
    vm = np.array([g in val_g for g in grp])
    xt, yt, wt = torch.as_tensor(obs[~vm]), torch.as_tensor(lab[~vm]), torch.as_tensor(w[~vm] / w[~vm].sum())
    from visuograsp_v4.wave5 import HISTORY
    from visuograsp_v4.wave5 import PREV_ACTION
    blocks = [sl for sl in HISTORY if not (args.mask_prev_action and sl == PREV_ACTION)]   # masked block stays 0
    hist_idx = torch.as_tensor(np.concatenate([np.arange(sl.start, sl.stop) for sl in blocks]))
    hist_std = xt[:, hist_idx].std(dim=0)
    xv, yv = torch.as_tensor(obs[vm]), torch.as_tensor(lab[vm])
    va = torch.as_tensor(np.isin(ph[vm], ACTIVE))
    cfg = V4Config()
    env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    from dataclasses import replace
    model = make_sac(env, replace(SACConfig(), net_arch=tuple(args.net)), args.seed)
    actor = model.actor
    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    import copy
    ema_actor = copy.deepcopy(actor) if args.ema > 0 else None
    pol = ema_actor if ema_actor is not None else actor       # the network that is evaluated and saved
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=args.weight_decay)
    meta = {'run': args.run, 'args': vars(args), 'datasets': stats, 'train_samples': int((~vm).sum()), 'val_samples': int(vm.sum()),
            'approach_share_weighted': {a: float(w[apr == a].sum() / w.sum()) for a in ('top', 'oblique')},
            'active_phase_share_weighted': float(w[np.isin(ph, ACTIVE)].sum() / w.sum()),
            'dev_seed_base': DEV_A, 'note': 'supervised imitation of the PRIVILEGED scripted expert; not RL'}
    (out / 'config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    print('[DATA]', {k: meta[k] for k in ('train_samples', 'val_samples', 'approach_share_weighted', 'active_phase_share_weighted')}, flush=True)
    grid = dev_grid(ALL_FAMILIES, approaches=(args.approach_filter,) if args.approach_filter else ('top', 'oblique'),
                    per_cell=args.dev_per_cell, base=DEV_A)
    n, steps_per_epoch = len(xt), max(1, len(xt) // args.batch)
    total = args.epochs * steps_per_epoch
    curve, evals, it, t0 = [], [], 0, time.time()
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for _ in range(steps_per_epoch):
            for g in opt.param_groups:
                g['lr'] = 1e-3 + (1e-4 - 1e-3) * it / total
            idx = torch.multinomial(wt, args.batch, replacement=True)
            xb = xt[idx]
            if args.history_noise > 0 or args.history_dropout > 0:     # copycat regulariser (training only)
                xb = xb.clone()
                h = xb[:, hist_idx] + args.history_noise * hist_std * torch.randn(len(xb), len(hist_idx))
                keep = (torch.rand(len(xb), 1) >= args.history_dropout).float()
                xb[:, hist_idx] = h * keep
            loss = F.mse_loss(actor_mean_action(actor, xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if ema_actor is not None:
                with torch.no_grad():
                    for pe, pa in zip(ema_actor.parameters(), actor.parameters()):
                        pe.mul_(args.ema).add_(pa, alpha=1 - args.ema)
            tot += float(loss)
            it += 1
        with torch.no_grad():
            pv = actor_mean_action(pol, xv)
            vmse = float(F.mse_loss(pv, yv))
            vmse_act = float(((pv - yv) ** 2).mean(dim=1)[va].mean())
        curve.append({'epoch': ep, 'train_mse_weighted': tot / steps_per_epoch, 'val_mse': vmse, 'val_mse_active': vmse_act,
                      'minutes': (time.time() - t0) / 60})
        pd.DataFrame(curve).to_csv(out / 'curve.csv', index=False)
        if ep in args.eval_epochs:
            torch.save(pol.state_dict(), out / f'actor_epoch{ep}.pt')
            with torch.no_grad():
                dev = evaluate_grid(env, lambda o: actor_mean_action(pol, torch.as_tensor(pre(o)[None])).numpy()[0], grid)
            dev.assign(epoch=ep).to_csv(out / 'dev_episodes.csv', mode='a', header=not (out / 'dev_episodes.csv').exists(), index=False)
            row = {'epoch': ep, 'dev_gt': float(dev.gt_success.mean()), 'dev_episodes': len(dev), 'val_mse_active': vmse_act,
                   'sim_failures': int(dev.sim_failure.sum())}
            for a, g in dev.groupby('approach'):
                row[f'dev_gt_{a}'] = float(g.gt_success.mean())
            evals.append(row)
            pd.DataFrame(evals).to_csv(out / 'dev_evals.csv', index=False)
            print(f'[EVAL] epoch {ep}: dev GT {row["dev_gt"]:.3f} ' + ' '.join(f'{k} {v:.2f}' for k, v in row.items() if k.startswith('dev_gt_'))
                  + f' val_mse_active {vmse_act:.5f}', flush=True)
    best = sorted(evals, key=lambda r: (-r['dev_gt'], r['val_mse_active']))[0]
    import shutil
    shutil.copy2(out / f'actor_epoch{best["epoch"]}.pt', out / 'actor_best.pt')
    (out / 'best.json').write_text(json.dumps({**best, 'selection': 'max DEV_A GT, tie -> lower active-phase val MSE'}, indent=1),
                                   encoding='utf-8')
    print('[DONE] best', best, f'({(time.time() - t0) / 60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
