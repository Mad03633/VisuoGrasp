"""Wave 4 (approach B): behaviour cloning of the scripted expert into the SAC actor, then DAgger.

1. Expert data: scripted controller in VisuoGraspV4Env (PRIVILEGED planner), training seeds
   3_400_000+, top / oblique alternating, all families. Actions clipped to +-0.999 and executed
   exactly as stored (same semantics as the 22-D env action; verified bit-exactly in Wave 1).
2. BC: MSE between tanh(mu(s)) of the SAC actor (LayerNorm-SAC architecture, 256x256) and the
   expert action, on GT-SUCCESSFUL expert episodes only; validation split by episode.
3. DAgger: the BC policy acts (deterministic); a shadow scripted expert advanced on the same states
   labels every visited state; data aggregated; actor re-trained. Seeds 3_500_000 + 1000*iter + k.
4. Every iterate is evaluated (deterministic) on the Wave 4 development set (seeds 8_700_000+, full
   task, original V4 initial conditions) next to the scripted controller on the same episodes; the
   iterate with the best development GT success is kept (bc_actor_best.pt).
The actor observation is the 130-D V4 observation (PRIVILEGED object pose, GT-oracle contacts).

    python -m visuograsp_v4.scripts.wave4_bc_pretrain --expert-episodes 300 --dagger-iters 2 --dagger-episodes 100
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config

OUT = OUTPUT_ROOT / 'rl' / 'wave4' / 'bc_pretrain'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--expert-episodes', type=int, default=300)
    ap.add_argument('--dagger-iters', type=int, default=2)
    ap.add_argument('--dagger-episodes', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--dev-per-cell', type=int, default=4)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    out = OUT if args.out is None else OUTPUT_ROOT / 'rl' / 'wave4' / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    from visuograsp_v4.agents import make_sac
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.scripted import ScriptedGrasp
    from visuograsp_v4.wave4 import (ALL_FAMILIES, actor_mean_action, bc_train, collect_dagger, collect_expert, dev_grid,
                                     evaluate_grid)
    cfg = V4Config()
    env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    t0 = time.time()
    summary = {'privileged': 'actor obs = V4 130-D privileged observation; expert = scripted controller with PRIVILEGED planner',
               'expert_seed_base': 3_400_000, 'dagger_seed_base': 3_500_000, 'dev_seed_base': 8_700_000}
    seeds = np.arange(args.expert_episodes) + 3_400_000
    ex = collect_expert(env, seeds)
    np.savez_compressed(out / 'expert_data.npz', **{k: v for k, v in ex.items() if k != 'episodes'})
    pd.DataFrame(ex['episodes']).to_csv(out / 'expert_episodes.csv', index=False)
    summary['expert'] = {'episodes': len(ex['episodes']), 'transitions': int(len(ex['obs'])),
                         'gt_success': float(np.mean([e['gt_success'] for e in ex['episodes']])),
                         'success_transitions': int(ex['success'].sum()), 'seconds': time.time() - t0}
    print('[EXPERT]', summary['expert'], flush=True)

    model = make_sac(env, SACConfig(), args.seed)            # architecture shared with SAC fine-tuning
    actor = model.actor
    act_fn = lambda o: actor_mean_action(actor, torch.as_tensor(o[None])).detach().numpy()[0]

    grid = dev_grid(ALL_FAMILIES, per_cell=args.dev_per_cell)
    ref_path = OUTPUT_ROOT / 'rl' / 'wave4' / f'dev_scripted_reference_pc{args.dev_per_cell}.csv'
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
    else:
        holder = {}
        ref = evaluate_grid(env, lambda o: holder['a'].act(), grid,
                            on_reset=lambda e: holder.update(a=ScriptedGrasp(e.sim, e.approach)))
        ref.to_csv(ref_path, index=False)
    summary['dev_scripted_gt'] = float(ref.gt_success.mean())

    sel = ex['success']
    data_obs, data_act, data_grp = ex['obs'][sel], ex['act'][sel], ex['episode'][sel]
    iters = []
    best = (-1.0, None)
    for it in range(args.dagger_iters + 1):
        if it > 0:
            dseeds = 3_500_000 + 1000 * it + np.arange(args.dagger_episodes)
            with torch.no_grad():
                dg = collect_dagger(env, act_fn, dseeds)
            pd.DataFrame(dg['episodes']).to_csv(out / f'dagger_iter{it}_rollouts.csv', index=False)
            data_obs = np.concatenate((data_obs, dg['obs']))
            data_act = np.concatenate((data_act, dg['act']))
            data_grp = np.concatenate((data_grp, 100_000 * it + np.arange(len(dg['obs'])) // 320))
            print(f'[DAGGER {it}] learner rollout GT {np.mean([e["gt_success"] for e in dg["episodes"]]):.3f}, '
                  f'+{len(dg["obs"])} labelled states', flush=True)
        res = bc_train(actor, data_obs, data_act, data_grp, epochs=args.epochs if it == 0 else max(10, args.epochs // 2))
        pd.DataFrame(res['curve']).to_csv(out / f'bc_curve_iter{it}.csv', index=False)
        with torch.no_grad():
            dev = evaluate_grid(env, act_fn, grid)
        dev.to_csv(out / f'dev_eval_iter{it}.csv', index=False)
        torch.save(actor.state_dict(), out / f'bc_actor_iter{it}.pt')
        row = {'iter': it, 'train_samples': res['train_samples'], 'final_train_mse': res['curve'][-1]['train_mse'],
               'final_val_mse': res['curve'][-1]['val_mse'], 'dev_gt': float(dev.gt_success.mean()),
               'dev_gt_top': float(dev[dev.approach == 'top'].gt_success.mean()),
               'dev_gt_oblique': float(dev[dev.approach == 'oblique'].gt_success.mean()),
               'dev_partial': float((dev.max_lift_cm >= 3.5).mean()), 'dev_episodes': len(dev),
               'dev_outcomes': dev.outcome.value_counts().to_dict(), 'seconds': time.time() - t0}
        iters.append(row)
        print(f'[BC iter {it}] val mse {row["final_val_mse"]:.4f} dev GT {row["dev_gt"]:.3f} '
              f'(top {row["dev_gt_top"]:.3f}, oblique {row["dev_gt_oblique"]:.3f}) scripted {summary["dev_scripted_gt"]:.3f}', flush=True)
        if row['dev_gt'] > best[0]:
            best = (row['dev_gt'], it)
            torch.save(actor.state_dict(), out / 'bc_actor_best.pt')
            np.savez_compressed(out / 'bc_training_set.npz', obs=data_obs, act=data_act)
    summary['iterations'] = iters
    summary['best_iter'] = best[1]
    summary['best_dev_gt'] = best[0]
    summary['seconds'] = time.time() - t0
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str), encoding='utf-8')
    print('[DONE] best iter', best[1], 'dev GT', best[0], f'({summary["seconds"] / 60:.1f} min)')


if __name__ == '__main__':
    main()
