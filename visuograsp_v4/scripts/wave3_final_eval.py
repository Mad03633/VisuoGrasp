"""Wave 3 final evaluation on a NEW, previously unseen set (run once, after checkpoint selection is frozen).

Seeds 6_100_000 + cell*1000 + approach*100 + episode (same 27 family/variant/resting-mode cells as
the scripted-baseline grid; approaches top + oblique; 10 episodes per cell = 540 per policy). These
seeds were never used for training, selection, gates or any earlier wave. The exposed Wave 2 test
seeds (5_100_000+) are not used.
Policies (all PRIVILEGED: simulator object pose, GT-oracle contacts; residual env = res2 / v4r2):
  scripted        ScriptedGrasp in VisuoGraspV4Env
  zero_residual   residual env, r = 0 (must equal scripted episode by episode)
  perturbed_1e-3  chaos control: r ~ U(-1e-3, 1e-3) (physically negligible) -> outcome-flip floor
  wave3_best      best_model.zip (in-training selection, seeds 8.0M)
  wave3_last      last_model.zip
Paired outcome categories vs zero residual: preserved (both succeed: scripted success kept),
improved (only the residual policy succeeds), broken (only zero residual succeeds: failure caused by
the corrections), neither.

    python -m visuograsp_v4.scripts.wave3_final_eval
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
from visuograsp_v4.scripts.wave2_final_test import grid

RL = OUTPUT_ROOT / 'rl'
RUN = RL / 'wave3' / 'conservative_residual_sac_seed11'
OUT = RL / 'wave3' / 'final_eval_unseen_6p1M'
SEED_BASE = 6_100_000


def policies(run):
    return {'scripted': ('base', 'SCRIPTED'), 'zero_residual': ('residual', 'ZERO'),
            'perturbed_1e-3': ('residual', 'PERTURBED'), 'wave3_best': ('residual', run / 'best_model.zip'),
            'wave3_last': ('residual', run / 'last_model.zip')}


def _run(job) -> pd.DataFrame:
    name, env_kind, src, episodes, seed_base = job
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.evaluation import run_policy_episode
    cfg = V4Config()
    if env_kind == 'residual':
        from visuograsp_v4.residual import RES2, ResidualGraspEnv
        env = ResidualGraspEnv(cfg, RES2, approaches=('top', 'oblique'))
    else:
        from visuograsp_v4.env import VisuoGraspV4Env
        env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    holder, on_reset = {}, None
    if src == 'SCRIPTED':
        from visuograsp_v4.scripted import ScriptedGrasp
        on_reset = lambda e: holder.update(agent=ScriptedGrasp(e.sim, e.approach))
        act = lambda obs: holder['agent'].act()
    elif src == 'ZERO':
        act = lambda obs: np.zeros(22, np.float32)
    elif src == 'PERTURBED':
        prng = np.random.default_rng(12345)
        act = lambda obs: (1e-3 * prng.uniform(-1, 1, 22)).astype(np.float32)
    else:
        from visuograsp_v4.agents import load_sac
        model = load_sac(src)
        act = lambda obs: model.predict(obs, deterministic=True)[0]
    rows = []
    for seed, f, v, m, ap in grid(cfg, episodes, seed_base):
        row = run_policy_episode(env, act, seed, f, v, m, ap, on_reset=on_reset)
        row['policy'] = name
        rows.append(row)
    return pd.DataFrame(rows)


def categories(df: pd.DataFrame, pol: str, by=None) -> pd.DataFrame:
    a = df[df.policy == pol].set_index(['seed', 'approach'])
    z = df[df.policy == 'zero_residual'].set_index(['seed', 'approach']).loc[a.index]
    cat = np.select([a.gt_success.astype(bool) & z.gt_success.astype(bool), a.gt_success.astype(bool),
                     z.gt_success.astype(bool)], ['preserved', 'improved', 'broken'], 'neither')
    t = pd.DataFrame({'policy': pol, 'category': cat, 'approach': a.index.get_level_values('approach'),
                      'family': a.family.values, 'outcome_policy': a.outcome.values, 'outcome_zero': z.outcome.values})
    return t


def main() -> None:
    global OUT, RUN
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--smoke', action='store_true', help='pipeline check on DEV seeds 9_400_000+ with a check_ run')
    ap.add_argument('--run', default='conservative_residual_sac_seed11')
    args = ap.parse_args()
    RUN = RL / 'wave3' / args.run
    seed_base = SEED_BASE
    if args.smoke:
        seed_base, args.episodes, OUT = 9_400_000, 1, RL / 'wave3' / '_smoke_final_eval'
    if OUT.exists() and any(OUT.glob('*.csv')):
        raise SystemExit(f'{OUT} already has results: the final evaluation is run once (nothing is overwritten)')
    pols = policies(RUN)
    for name, (_, src) in pols.items():
        if not isinstance(src, str) and not src.exists():
            raise SystemExit(f'missing model for {name}: {src}')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {k: str(v[1]) for k, v in pols.items()}
    for k in ('wave3_best', 'wave3_last'):
        meta_f = RUN / ('best_model.json' if k == 'wave3_best' else 'finished.json')
        if meta_f.exists():
            frozen[k + '_meta'] = json.loads(meta_f.read_text(encoding='utf-8'))
    (OUT / 'frozen_selection.json').write_text(json.dumps(frozen, indent=1, default=str), encoding='utf-8')
    t0 = time.time()
    jobs = [(n, k, s, args.episodes, seed_base) for n, (k, s) in pols.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        dfs = list(ex.map(_run, jobs))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT / 'episodes.csv', index=False)

    from visuograsp_v4.scripts.paired_compare import paired_table
    from visuograsp_v4.scripts.validate_checkpoints import summarise
    pd.concat([summarise(df.assign(approach='all'), ['policy', 'approach']), summarise(df, ['policy', 'approach'])],
              ignore_index=True).to_csv(OUT / 'summary_by_approach.csv', index=False)
    summarise(df.assign(approach='all'), ['policy', 'family']).to_csv(OUT / 'summary_by_family.csv', index=False)
    summarise(df, ['policy', 'family', 'approach']).to_csv(OUT / 'summary_by_family_approach.csv', index=False)
    get = lambda n: df[df.policy == n]
    pairs = [('wave3_best', 'zero_residual'), ('wave3_last', 'zero_residual'), ('perturbed_1e-3', 'zero_residual'),
             ('wave3_last', 'wave3_best'), ('scripted', 'zero_residual')]
    paired = pd.concat([paired_table(get(a), get(b), a, b, by=('approach', 'family')) for a, b in pairs], ignore_index=True)
    paired.to_csv(OUT / 'paired.csv', index=False)
    cats = pd.concat([categories(df, p) for p in ('wave3_best', 'wave3_last', 'perturbed_1e-3')], ignore_index=True)
    cats.to_csv(OUT / 'paired_categories_episodes.csv', index=False)
    cat_tab = cats.groupby(['policy', 'approach']).category.value_counts().unstack(fill_value=0)
    cat_all = cats.groupby('policy').category.value_counts().unstack(fill_value=0).assign(approach='all').set_index('approach', append=True)
    cat_tab = pd.concat([cat_all, cat_tab]).sort_index()
    cat_tab.to_csv(OUT / 'paired_categories.csv')
    fail = cats[cats.category.isin(['broken', 'improved'])].groupby(['policy', 'category', 'outcome_policy', 'outcome_zero']).size()
    fail.rename('episodes').reset_index().to_csv(OUT / 'changed_episodes_outcomes.csv', index=False)
    z, s = get('zero_residual').set_index(['seed', 'approach']), get('scripted').set_index(['seed', 'approach'])
    zero_equiv = {c: int((z[c] != s.loc[z.index, c]).sum()) for c in ['gt_success', 'outcome', 'max_lift_cm', 'steps']}
    mags = []
    for n in ('wave3_best', 'wave3_last', 'perturbed_1e-3'):
        r = get(n)
        mags.append({'policy': n, 'mean_abs_residual': float(r.res_abs_mean.mean()), 'res_jitter': float(r.res_jitter.mean()),
                     'saturation': float(r.res_saturation.mean()), 'pos_offset_mm': 10 * float(r.res_pos_offset_cm.mean()),
                     'rot_offset_rad': float(r.res_rot_offset_rad.mean()), 'finger_offset_rad': float(r.res_finger_offset_rad.mean()),
                     'mean_close_step': float(r.close_step.replace(-1, np.nan).mean()),
                     'mean_grasp_offset_cm': float(r.grasp_offset_cm.mean()),
                     'pushed_before_close': float(r.pushed_before_close.mean()), 'sim_failures': int(r.sim_failure.sum())})
    zz = get('zero_residual')
    mags.append({'policy': 'zero_residual', 'mean_abs_residual': 0.0, 'res_jitter': 0.0, 'saturation': 0.0, 'pos_offset_mm': 0.0,
                 'rot_offset_rad': 0.0, 'finger_offset_rad': 0.0, 'mean_close_step': float(zz.close_step.replace(-1, np.nan).mean()),
                 'mean_grasp_offset_cm': float(zz.grasp_offset_cm.mean()), 'pushed_before_close': float(zz.pushed_before_close.mean()),
                 'sim_failures': int(zz.sim_failure.sum())})
    pd.DataFrame(mags).to_csv(OUT / 'residual_magnitude_behaviour.csv', index=False)
    meta = {'seed_base': seed_base, 'episodes_per_policy': int(len(get('scripted'))), 'smoke': args.smoke,
            'zero_vs_scripted_mismatches': zero_equiv, 'torch_threads': 1, 'deterministic': True,
            'privileged': 'simulator object pose in observation and plan; GT-oracle contacts', 'seconds': time.time() - t0}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    summ = pd.read_csv(OUT / 'summary_by_approach.csv')
    print(summ[['policy', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo', 'wilson_hi']].round(3).to_string(index=False))
    pa = paired[paired.group.isin(['all', 'approach=top', 'approach=oblique'])]
    print(pa[['A', 'B', 'group', 'rate_a', 'rate_b', 'diff_a_minus_b', 'diff_lo', 'diff_hi', 'only_a', 'only_b',
              'mcnemar_exact_p']].round(4).to_string(index=False))
    print(cat_tab.to_string())
    print(pd.DataFrame(mags).round(4).to_string(index=False))
    print(json.dumps({'zero_vs_scripted_mismatches': zero_equiv}))
    print(f'written to {OUT} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
