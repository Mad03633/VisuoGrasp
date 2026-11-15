"""Wave 2 final test: all policies on the identical test episodes (run once, after training).

Test grid = the scripted-baseline grid, seeds 5_100_000 + cell*1000 + approach*100 + episode,
approaches top + oblique, 6 episodes per (family, variant, resting mode) cell = 324 episodes per
policy. These seeds were never used for training, model selection or the 75k gate.
Policies (all PRIVILEGED: simulator object pose, GT-oracle contacts):
  scripted            ScriptedGrasp in VisuoGraspV4Env (nominal controller)
  zero_residual       ResidualGraspEnv with r = 0 (must equal scripted episode by episode)
  residual_best       Wave 2 residual SAC, best_model.zip (in-training selection, seeds 8.0M)
  residual_last       Wave 2 residual SAC, last_model.zip
  wave1_demo_best     Wave 1 SAC + demos, best_model.zip (100k)   -- autonomous SAC, base env
  wave1_scratch_best  Wave 1 SAC scratch, best_model.zip (150k)   -- autonomous SAC, base env
Deterministic policies, torch 1 thread (reproducible).

    python -m visuograsp_v4.scripts.wave2_final_test
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
OUT = RL / 'wave2' / 'final_test'
POLICIES = {
    'scripted': ('base', 'SCRIPTED'),
    'zero_residual': ('residual', 'ZERO'),
    'residual_best': ('residual', RL / 'wave2' / 'residual_sac_seed11' / 'best_model.zip'),
    'residual_last': ('residual', RL / 'wave2' / 'residual_sac_seed11' / 'last_model.zip'),
    'wave1_demo_best': ('base', RL / 'wave1' / 'sac_demo_s11' / 'best_model.zip'),
    'wave1_scratch_best': ('base', RL / 'wave1' / 'sac_scratch_s11' / 'best_model.zip'),
}


def grid(cfg, episodes: int, seed_base: int) -> list[tuple]:
    """Same cells and seed formula as evaluate_policy.test_grid, with a selectable seed base
    (5_100_000 = test; 9_300_000 = DEV seeds for the --smoke pipeline check)."""
    from visuograsp_v4 import objects as objs
    from visuograsp_v4.scripts.evaluate_policy import APPROACHES
    cells = [(f, v, m) for f in cfg.objects.families for v in objs.VARIANTS[f] for m in objs.RESTING_MODES[f]]
    out = []
    for gi, (f, v, m) in enumerate(cells):
        for ap in ('top', 'oblique'):
            for ep in range(episodes):
                out.append((seed_base + gi * 1000 + APPROACHES.index(ap) * 100 + ep, f, v, m, ap))
    return out


def _run(job) -> pd.DataFrame:
    name, env_kind, src, episodes, seed_base = job
    import torch
    torch.set_num_threads(1)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.evaluation import run_policy_episode
    cfg = V4Config()
    if env_kind == 'residual':
        from visuograsp_v4.residual import ResidualGraspEnv
        env = ResidualGraspEnv(cfg, approaches=('top', 'oblique'))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--smoke', action='store_true', help='pipeline check: DEV seeds 9_300_000+, check_ run, 1 episode')
    args = ap.parse_args()
    global OUT
    seed_base = 5_100_000
    if args.smoke:
        seed_base, args.episodes, OUT = 9_300_000, 1, RL / 'wave2' / '_smoke_final_test'
        for k in ('residual_best', 'residual_last'):
            POLICIES[k] = ('residual', RL / 'wave2' / 'check_residual_s11' / POLICIES[k][1].name)
    if OUT.exists() and any(OUT.glob('*.csv')):
        raise SystemExit(f'{OUT} already has results: the final test is run once (nothing is overwritten)')
    for name, (_, src) in POLICIES.items():
        if not isinstance(src, str) and not src.exists():
            raise SystemExit(f'missing model for {name}: {src}')
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(n, k, s, args.episodes, seed_base) for n, (k, s) in POLICIES.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        dfs = list(ex.map(_run, jobs))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(OUT / 'episodes.csv', index=False)

    from visuograsp_v4.scripts.paired_compare import paired_table
    from visuograsp_v4.scripts.validate_checkpoints import summarise
    s_all = summarise(df.assign(approach='all'), ['policy', 'approach'])
    s_ap = summarise(df, ['policy', 'approach'])
    pd.concat([s_all, s_ap], ignore_index=True).to_csv(OUT / 'summary_by_approach.csv', index=False)
    summarise(df, ['policy', 'family', 'approach']).to_csv(OUT / 'summary_by_family_approach.csv', index=False)
    summarise(df.assign(approach='all'), ['policy', 'family']).to_csv(OUT / 'summary_by_family.csv', index=False)
    get = lambda n: df[df.policy == n]
    pairs = [('residual_best', 'scripted'), ('residual_best', 'zero_residual'), ('residual_last', 'scripted'),
             ('residual_last', 'residual_best'), ('scripted', 'zero_residual'), ('residual_best', 'wave1_demo_best'),
             ('wave1_demo_best', 'wave1_scratch_best'), ('scripted', 'wave1_demo_best')]
    paired = pd.concat([paired_table(get(a), get(b), a, b, by=('approach', 'family')) for a, b in pairs], ignore_index=True)
    paired.to_csv(OUT / 'paired.csv', index=False)
    # zero residual must reproduce the scripted controller episode by episode
    z, s = get('zero_residual').set_index(['seed', 'approach']), get('scripted').set_index(['seed', 'approach'])
    eq_cols = ['gt_success', 'outcome', 'max_lift_cm', 'episode_return', 'steps']
    zero_equiv = {c: int((z[c] != s.loc[z.index, c]).sum()) for c in eq_cols}
    # behaviour change of the residual policies relative to zero residual (paired)
    beh = []
    for n in ('residual_best', 'residual_last'):
        r = get(n).set_index(['seed', 'approach'])
        zz = z.loc[r.index]
        beh.append({'policy': n, 'episodes': len(r), 'outcome_changed': int((r.outcome != zz.outcome).sum()),
                    'gt_changed': int((r.gt_success != zz.gt_success).sum()),
                    'mean_abs_residual': float(r.res_abs_mean.mean()), 'mean_saturation': float(r.res_saturation.mean()),
                    'mean_pos_offset_cm': float(r.res_pos_offset_cm.mean()), 'mean_rot_offset_rad': float(r.res_rot_offset_rad.mean()),
                    'mean_finger_offset_rad': float(r.res_finger_offset_rad.mean()),
                    'mean_close_step': float(r.close_step.replace(-1, np.nan).mean()),
                    'mean_close_step_zero': float(zz.close_step.replace(-1, np.nan).mean()),
                    'mean_grasp_offset_cm': float(r.grasp_offset_cm.mean()), 'mean_grasp_offset_cm_zero': float(zz.grasp_offset_cm.mean()),
                    'pushed_before_close': float(r.pushed_before_close.mean()), 'pushed_before_close_zero': float(zz.pushed_before_close.mean()),
                    'sim_failures': int(r.sim_failure.sum()), 'sim_failures_zero': int(zz.sim_failure.sum())})
    pd.DataFrame(beh).to_csv(OUT / 'behaviour_change.csv', index=False)
    oc = df.groupby(['policy', 'approach']).outcome.value_counts().unstack(fill_value=0)
    oc.to_csv(OUT / 'outcomes.csv')
    # consistency with the stored scripted baseline (sim path, same seeds)
    from visuograsp_v4.scripts.evaluate_policy import SCRIPTED_CSV
    base = pd.read_csv(SCRIPTED_CSV)[['seed', 'approach', 'gt_success']]
    mm = get('scripted').merge(base, on=['seed', 'approach'], suffixes=('', '_stored'))
    meta = {'episodes_per_policy': int(len(get('scripted'))), 'seed_base': seed_base, 'smoke': args.smoke, 'approaches': ['top', 'oblique'],
            'policies': {k: str(v[1]) for k, v in POLICIES.items()}, 'torch_threads': 1, 'deterministic': True,
            'zero_vs_scripted_mismatches': zero_equiv,
            'scripted_env_vs_stored_baseline_gt_mismatches': int((mm.gt_success != mm.gt_success_stored).sum()),
            'scripted_env_vs_stored_baseline_paired': int(len(mm)),
            'privileged': 'simulator object pose in the observation / plan; GT-oracle contacts',
            'seconds': time.time() - t0}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    summ = pd.read_csv(OUT / 'summary_by_approach.csv')
    print(summ[['policy', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo', 'wilson_hi']].round(3).to_string(index=False))
    pa = paired[paired.group.isin(['all', 'approach=top', 'approach=oblique'])]
    print(pa[['A', 'B', 'group', 'rate_a', 'rate_b', 'diff_a_minus_b', 'diff_lo', 'diff_hi', 'only_a', 'only_b',
              'mcnemar_exact_p']].round(4).to_string(index=False))
    print(pd.DataFrame(beh).round(3).T.to_string())
    print(json.dumps({k: meta[k] for k in ('zero_vs_scripted_mismatches', 'scripted_env_vs_stored_baseline_gt_mismatches')}))
    print(f'written to {OUT} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
