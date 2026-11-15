"""Wave 5 evaluation: cross-method selection on DEV_B and the one-time fresh final test (PRIVILEGED setting).

  --stage devb   seeds 8_900_000 + fam*1000 + approach*100 + ep, 20 per family x approach = 200 episodes
  --stage final  NEW test grid 6_700_000 + cell*1000 + approach*100 + ep, 10 per cell = 540 episodes (once)
Policy specs (--policy name=spec):
  scripted              scripted controller (PRIVILEGED planner), matched baseline
  bc4                   Wave 4 BC actor (frozen_policies/bc), unmasked observation
  bc5:<run>             Wave 5 BC run (outputs/rl/wave5/bc/<run>/actor_best.pt; net and masking from config.json)
  bc5:<run>#<file>      a specific actor file of that run (e.g. actor_epoch20.pt)
  ens:<spec1>|<spec2>|...  mean action of BC members (bc4 / bc5 specs)
  sb3:<path>[@mask|@hist] SB3 SAC checkpoint relative to outputs/rl; '@mask' prev_action zeroed, '@hist' all history blocks
Per-episode rows are appended to partial_<name>.csv (resumable with --resume, duplicates removed).

    python -m visuograsp_v4.scripts.wave5_eval --stage devb --tag sel_v1 --policy scripted=scripted --policy bc4=bc4 --policy e5=bc5:e5c_mask_pw025
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config

RL = OUTPUT_ROOT / 'rl'
W5 = RL / 'wave5'


def make_policy(spec, env):
    import torch
    from dataclasses import replace
    from visuograsp_v4.wave4 import actor_mean_action
    from visuograsp_v4.wave5 import mask_history, mask_prev_action
    if spec == 'scripted':
        from visuograsp_v4.scripted import ScriptedGrasp
        h = {}
        return (lambda o: h['a'].act()), (lambda e: h.update(a=ScriptedGrasp(e.sim, e.approach)))
    if spec in ('bc4',) or spec.startswith('bc5:'):
        from visuograsp_v4.agents import make_sac
        if spec == 'bc4':
            net, mode, path = (256, 256), None, RL / 'wave4' / 'frozen_policies' / 'bc' / 'bc_actor_best.pt'
        else:
            run_s, _, ckpt = spec[4:].partition('#')          # bc5:<run>[#<actor file>]
            run = W5 / 'bc' / run_s
            c = json.loads((run / 'config.json').read_text())['args']
            mode = 'history' if c.get('mask_history') else ('prev' if c.get('mask_prev_action') else None)
            net, path = tuple(c['net']), run / (ckpt or 'actor_best.pt')
        m = make_sac(env, replace(SACConfig(), net_arch=net), 0)
        m.actor.load_state_dict(torch.load(path))
        pre = {'history': mask_history, 'prev': mask_prev_action}.get(mode, lambda o: o)
        return (lambda o: actor_mean_action(m.actor, torch.as_tensor(pre(o)[None])).detach().numpy()[0]), None
    if spec.startswith('ens:'):
        # action average of several BC members, each with its own input masking (fixed composition)
        members = [make_policy(m, env)[0] for m in spec[4:].split('|')]
        return (lambda o: np.mean([f(o) for f in members], axis=0)), None
    if spec.startswith('sb3:'):
        from visuograsp_v4.agents import load_sac
        path, _, flag = spec[4:].partition('@')
        model = load_sac(RL / path)
        pre = {'mask': mask_prev_action, 'hist': mask_history}.get(flag, lambda o: o)
        return (lambda o: model.predict(pre(o), deterministic=True)[0]), None
    raise ValueError(spec)


def grid_for(stage, cfg):
    from visuograsp_v4.wave4 import ALL_FAMILIES, dev_grid
    if stage == 'devb':
        return dev_grid(ALL_FAMILIES, per_cell=20, base=8_900_000)
    from visuograsp_v4.scripts.wave2_final_test import grid as test_grid
    return test_grid(cfg, 10, 6_700_000)


def _run(job):
    name, spec, stage, out = job
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.wave4 import evaluate_grid
    cfg = V4Config()
    env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    act, on_reset = make_policy(spec, env)
    grid = grid_for(stage, cfg)
    part = out / f'partial_{name}.csv'
    done = set()
    if part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.seed.astype(int), prev.approach))
    todo = [g for g in grid if (int(g[0]), g[4]) not in done]
    print(f'[{name}] {len(done)} recorded, {len(todo)} to run', flush=True)
    for i, g in enumerate(todo):
        with torch.no_grad():
            row = evaluate_grid(env, act, [g], on_reset=on_reset)
        row['policy'] = name
        row.to_csv(part, mode='a', header=not part.exists(), index=False)
        if (i + 1) % 100 == 0:
            print(f'[{name}] {len(done) + i + 1}/{len(grid)}', flush=True)
    keys = {(int(g[0]), g[4]) for g in grid}
    df = pd.read_csv(part).drop_duplicates(['seed', 'approach'], keep='first')
    return df[[(int(s), a) in keys for s, a in zip(df.seed, df.approach)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('devb', 'final'), required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--policy', action='append', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    out = W5 / ('selection' if args.stage == 'devb' else 'final_test') / args.tag
    pols = dict(p.split('=', 1) for p in args.policy)
    if (out / 'episodes.csv').exists():
        raise SystemExit(f'{out} already has results (nothing is overwritten)')
    if args.stage == 'final' and any((W5 / 'final_test').glob('*/episodes.csv')):
        raise SystemExit('the Wave 5 fresh final test has already been run (used once)')
    if out.exists() and any(out.glob('partial_*.csv')) and not args.resume:
        raise SystemExit(f'{out} has partial results; use --resume')
    out.mkdir(parents=True, exist_ok=True)
    if args.resume and (out / 'policies.json').exists():
        if json.loads((out / 'policies.json').read_text())['policies'] != pols:
            raise SystemExit('resume refused: different policies')
        with open(out / 'resume_log.txt', 'a') as f:
            f.write('resumed ' + time.strftime('%Y-%m-%dT%H:%M:%S') + '\n')
    else:
        (out / 'policies.json').write_text(json.dumps({'stage': args.stage, 'policies': pols,
                                                       'frozen': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=1))
    t0 = time.time()
    jobs = [(n, s, args.stage, out) for n, s in pols.items()]
    if args.workers <= 1:
        dfs = [_run(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            dfs = list(ex.map(_run, jobs))
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(['policy', 'seed', 'approach'], keep='first')
    df.to_csv(out / 'episodes.csv', index=False)
    from visuograsp_v4.scripts.paired_compare import paired_table
    from visuograsp_v4.scripts.validate_checkpoints import summarise
    pd.concat([summarise(df.assign(approach='all'), ['policy', 'approach']), summarise(df, ['policy', 'approach'])],
              ignore_index=True).to_csv(out / 'summary_by_approach.csv', index=False)
    summarise(df.assign(approach='all'), ['policy', 'family']).to_csv(out / 'summary_by_family.csv', index=False)
    df.groupby(['policy', 'approach']).outcome.value_counts().unstack(fill_value=0).to_csv(out / 'outcomes.csv')
    g = lambda n: df[df.policy == n]
    pairs = [(a, 'scripted') for a in pols if a != 'scripted'] + [(a, 'bc4') for a in pols if a not in ('scripted', 'bc4') and 'bc4' in pols]
    if 'scripted' in pols:
        pd.concat([paired_table(g(a), g(b), a, b, by=('approach', 'family')) for a, b in pairs], ignore_index=True).to_csv(
            out / 'paired.csv', index=False)
    pd.set_option('display.width', 250)
    s = pd.read_csv(out / 'summary_by_approach.csv')
    print(s[['policy', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo', 'wilson_hi', 'sim_failures']].round(3).to_string(index=False))
    if (out / 'paired.csv').exists():
        p = pd.read_csv(out / 'paired.csv')
        print(p[p.group.isin(['all', 'approach=top', 'approach=oblique'])][['A', 'B', 'group', 'rate_a', 'rate_b', 'diff_a_minus_b',
                                                                            'diff_lo', 'diff_hi', 'only_a', 'only_b', 'mcnemar_exact_p']].round(4).to_string(index=False))
    print(f'written to {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
