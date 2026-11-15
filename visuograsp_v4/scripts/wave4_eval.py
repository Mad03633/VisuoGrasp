"""Wave 4 evaluation: cross-run selection (development) and the one-time final test.

  --stage select  seeds 8_750_000 + fam*1000 + approach*100 + ep, 10 per family x approach = 100 episodes
                  (never used in training or in-run selection); candidates vs the scripted controller.
  --stage final   NEW seeds 6_600_000 + cell*1000 + approach*100 + ep, 27 cells x 2 approaches x 10 = 540
                  episodes; run ONCE after the final selection is frozen (frozen_selection.json).
Policy specs (repeatable --policy name=source):
  scripted                        scripted controller (PRIVILEGED planner) in VisuoGraspV4Env
  bc:<dir>                        actor state dict <dir>/bc_actor_best.pt (outputs/rl/wave4/<dir>)
  sb3:<path relative to outputs/rl>  SB3 SAC checkpoint
All policies: base V4 env, original initial conditions, 130-D PRIVILEGED observation, deterministic
(mean) action, torch 1 thread. Outputs: episodes, success with Wilson CIs by approach and family,
outcome counts, paired comparison with the scripted controller (McNemar exact + bootstrap CI) and
paired categories (both succeed / only learned / only scripted / neither).

    python -m visuograsp_v4.scripts.wave4_eval --stage select --tag select_v1 --policy scripted --policy bc=bc:bc_pretrain ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

RL = OUTPUT_ROOT / 'rl'
W4 = RL / 'wave4'


def _run(job) -> pd.DataFrame:
    name, src, stage, episodes = job[:4]
    handover = job[4] if len(job) > 4 else None
    out_dir = job[5] if len(job) > 5 else None     # incremental, resumable per-policy file
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.wave4 import ALL_FAMILIES, dev_grid, evaluate_grid
    cfg = V4Config()
    env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    eval_stage = None
    if handover is not None and src != 'scripted':
        # sub-task evaluation: the learned policy takes over at the scripted hand-over of this stage;
        # the scripted controller on the same seed is identical up to the hand-over (paired)
        from visuograsp_v4.wave4 import CurriculumGraspEnv
        env = CurriculumGraspEnv(cfg, stage=handover, replay_prev_prob=0.0)
        eval_stage = handover
    on_reset = None
    if src == 'scripted':
        from visuograsp_v4.scripted import ScriptedGrasp
        holder = {}
        on_reset = lambda e: holder.update(a=ScriptedGrasp(e.sim, e.approach))
        act = lambda o: holder['a'].act()
    elif src.startswith('bc:'):
        from visuograsp_v4.agents import make_sac
        from visuograsp_v4.config import SACConfig
        from visuograsp_v4.wave4 import actor_mean_action
        model = make_sac(env, SACConfig(), 0)
        model.actor.load_state_dict(torch.load(W4 / src[3:] / 'bc_actor_best.pt'))
        act = lambda o: actor_mean_action(model.actor, torch.as_tensor(o[None])).detach().numpy()[0]
    elif src.startswith('sb3:'):
        from visuograsp_v4.agents import load_sac
        model = load_sac(RL / src[4:])
        act = lambda o: model.predict(o, deterministic=True)[0]
    else:
        raise ValueError(src)
    if stage == 'select':
        grid = dev_grid(ALL_FAMILIES, per_cell=episodes, base=8_750_000)
    else:
        from visuograsp_v4.scripts.wave2_final_test import grid as test_grid
        grid = test_grid(cfg, episodes, 6_600_000)
    part = None if out_dir is None else out_dir / f'partial_{name}.csv'
    done = set()
    if part is not None and part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.seed.astype(int), prev.approach))
    todo = [g for g in grid if (int(g[0]), g[4]) not in done]
    print(f'[{name}] {len(done)} episodes already recorded, {len(todo)} to run', flush=True)
    for i, g in enumerate(todo):
        with torch.no_grad():
            row = evaluate_grid(env, act, [g], on_reset=on_reset, stage=eval_stage)
        row['policy'] = name
        if part is None:
            done_rows = row if i == 0 else pd.concat([done_rows, row], ignore_index=True)
        else:
            row.to_csv(part, mode='a', header=not part.exists(), index=False)   # written after every episode
        if (i + 1) % 100 == 0:
            print(f'[{name}] {len(done) + i + 1}/{len(grid)}', flush=True)
    if part is None:
        return done_rows
    df = pd.read_csv(part).drop_duplicates(['seed', 'approach'], keep='first')
    return df[df.apply(lambda r: (int(r.seed), r.approach) in {(int(g[0]), g[4]) for g in grid}, axis=1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('select', 'final'), required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--policy', action='append', required=True, help='name=source')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--handover-stage', type=int, default=None,
                    help='sub-task protocol: learned policies start at the scripted hand-over of this stage')
    ap.add_argument('--resume', action='store_true', help='continue an interrupted evaluation from its partial_*.csv files')
    args = ap.parse_args()
    out = W4 / ('selection' if args.stage == 'select' else 'final_test') / args.tag
    if (out / 'episodes.csv').exists():
        raise SystemExit(f'{out} already has results (nothing is overwritten)')
    if out.exists() and any(out.glob('partial_*.csv')) and not args.resume:
        raise SystemExit(f'{out} has partial results of an interrupted run; use --resume to continue it')
    if args.stage == 'final' and (out.parent).exists():
        done = [json.loads(f.read_text()).get('handover_stage') for f in out.parent.glob('*/policies.json')
                if (f.parent / 'episodes.csv').exists()]
        if args.handover_stage in done:
            raise SystemExit('this final test protocol has already been run: the fresh test set is used once')
    out.mkdir(parents=True, exist_ok=True)
    pols = dict(p.split('=', 1) for p in args.policy)
    if args.resume and (out / 'policies.json').exists():
        prev = json.loads((out / 'policies.json').read_text(encoding='utf-8'))
        if prev['policies'] != pols or prev['episodes_per_cell'] != args.episodes or prev.get('handover_stage') != args.handover_stage:
            raise SystemExit('resume refused: policies / episodes / protocol differ from the interrupted run')
        with open(out / 'resume_log.txt', 'a', encoding='utf-8') as f:
            f.write('resumed ' + time.strftime('%Y-%m-%dT%H:%M:%S') + '\n')
    else:
        (out / 'policies.json').write_text(json.dumps({'stage': args.stage, 'policies': pols, 'episodes_per_cell': args.episodes,
                                                       'handover_stage': args.handover_stage,
                                                       'frozen': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=1),
                                           encoding='utf-8')
    t0 = time.time()
    jobs = [(n, s, args.stage, args.episodes, args.handover_stage, out) for n, s in pols.items()]
    if args.workers <= 1:
        dfs = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pols))) as ex:
            dfs = list(ex.map(_run, jobs))
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(['policy', 'seed', 'approach'], keep='first')
    df.to_csv(out / 'episodes.csv', index=False)
    from visuograsp_v4.scripts.paired_compare import paired_table
    from visuograsp_v4.scripts.validate_checkpoints import summarise
    pd.concat([summarise(df.assign(approach='all'), ['policy', 'approach']), summarise(df, ['policy', 'approach'])],
              ignore_index=True).to_csv(out / 'summary_by_approach.csv', index=False)
    summarise(df.assign(approach='all'), ['policy', 'family']).to_csv(out / 'summary_by_family.csv', index=False)
    summarise(df, ['policy', 'family', 'approach']).to_csv(out / 'summary_by_family_approach.csv', index=False)
    df.groupby(['policy', 'approach']).outcome.value_counts().unstack(fill_value=0).to_csv(out / 'outcomes.csv')
    if 'scripted' in pols:
        s = df[df.policy == 'scripted']
        pt, cats = [], []
        for n in pols:
            if n == 'scripted':
                continue
            a = df[df.policy == n]
            pt.append(paired_table(a, s, n, 'scripted', by=('approach', 'family')))
            m = a.merge(s[['seed', 'approach', 'gt_success']], on=['seed', 'approach'], suffixes=('', '_scripted'))
            c = np.select([m.gt_success & m.gt_success_scripted, m.gt_success, m.gt_success_scripted],
                          ['both', 'only_learned', 'only_scripted'], 'neither')
            cats.append(pd.Series(c).value_counts().rename(n))
        pd.concat(pt, ignore_index=True).to_csv(out / 'paired_vs_scripted.csv', index=False)
        pd.concat(cats, axis=1).fillna(0).astype(int).T.to_csv(out / 'paired_categories.csv')
    pd.set_option('display.width', 250)
    summ = pd.read_csv(out / 'summary_by_approach.csv')
    print(summ[['policy', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo', 'wilson_hi',
                'partial_lift_rate']].round(3).to_string(index=False))
    if (out / 'paired_categories.csv').exists():
        print(pd.read_csv(out / 'paired_categories.csv', index_col=0).to_string())
    print(f'written to {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
