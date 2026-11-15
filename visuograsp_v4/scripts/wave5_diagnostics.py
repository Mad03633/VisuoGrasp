"""Wave 5 diagnostics (no training): BC error by expert phase, BC vs scripted on a fresh development set,
numerical-failure analysis over all stored V4 evaluation CSVs. PRIVILEGED setting throughout.

  dev set DEV_A: seeds 8_800_000 + fam*1000 + approach*100 + ep (10 per family x approach = 100 episodes)

    python -m visuograsp_v4.scripts.wave5_diagnostics
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config

RL = OUTPUT_ROOT / 'rl'
OUT = RL / 'wave5' / 'diagnostics'
DEV_A = 8_800_000


def bc_actor(env):
    from visuograsp_v4.agents import make_sac
    m = make_sac(env, SACConfig(), 0)
    m.actor.load_state_dict(torch.load(RL / 'wave4' / 'frozen_policies' / 'bc' / 'bc_actor_best.pt'))
    return m.actor


def phase_errors(env, actor, n=30) -> pd.DataFrame:
    """Run the scripted expert on DEV_A episodes and record, per step, its phase, its action and the BC
    prediction on the same observation (open-loop imitation error, no compounding)."""
    from visuograsp_v4.scripted import ScriptedGrasp
    from visuograsp_v4.wave4 import ALL_FAMILIES, actor_mean_action, dev_grid
    rows = []
    for seed, f, v, m, ap in dev_grid(ALL_FAMILIES, per_cell=3, base=DEV_A)[:n]:
        obs, _ = env.reset(seed=seed, options={'family': f, 'variant': v, 'resting_mode': m, 'approach': ap})
        ag = ScriptedGrasp(env.sim, env.approach)
        t = 0
        while True:
            phase = ag.phase
            a = np.clip(ag.act(), -0.999, 0.999)
            with torch.no_grad():
                p = actor_mean_action(actor, torch.as_tensor(obs[None])).numpy()[0]
            err = (p - a) ** 2
            rows.append({'seed': seed, 'approach': ap, 'family': f, 't': t, 'phase': phase, 'mse_all': err.mean(),
                         'mse_fingers': err[:16].mean(), 'mse_trans': err[16:19].mean(), 'mse_rot': err[19:].mean(),
                         'abs_a': np.abs(a).mean()})
            obs, r, term, trunc, _ = env.step(a.astype(np.float32))
            t += 1
            if term or trunc:
                break
        rows[-1]['episode_gt'] = bool(env.sim.tracker.gt_success)
    return pd.DataFrame(rows)


def matched(env, actor) -> pd.DataFrame:
    from visuograsp_v4.scripted import ScriptedGrasp
    from visuograsp_v4.wave4 import ALL_FAMILIES, actor_mean_action, dev_grid, evaluate_grid
    grid = dev_grid(ALL_FAMILIES, per_cell=10, base=DEV_A)
    h = {}
    s = evaluate_grid(env, lambda o: h['a'].act(), grid, on_reset=lambda e: h.update(a=ScriptedGrasp(e.sim, e.approach)))
    with torch.no_grad():
        b = evaluate_grid(env, lambda o: actor_mean_action(actor, torch.as_tensor(o[None])).numpy()[0], grid)
    return pd.concat([s.assign(policy='scripted'), b.assign(policy='bc_w4')], ignore_index=True)


def failures() -> pd.DataFrame:
    rows = []
    srcs = {'wave4_final_full': RL / 'wave4/final_test/full_task/episodes.csv',
            'wave4_final_subtask': RL / 'wave4/final_test/subtask_handover5/episodes.csv',
            'wave3_final': RL / 'wave3/final_eval_unseen_6p1M/episodes.csv',
            'wave2_final': RL / 'wave2/final_test/episodes.csv'}
    for name, p in srcs.items():
        if not p.exists():
            continue
        e = pd.read_csv(p)
        for pol, g in e.groupby('policy'):
            w = g.warnings.fillna('').astype(str)
            rows.append({'source': name, 'policy': pol, 'episodes': len(g), 'sim_failures': int(g.sim_failure.sum()),
                         'rate': float(g.sim_failure.mean()), 'runaway': int(g.object_runaway.sum()),
                         'badqacc_warn': int(w.str.contains('BADQACC').sum()), 'badqvel_warn': int(w.str.contains('BADQVEL').sum()),
                         'top_rate': float(g[g.approach == 'top'].sim_failure.mean()),
                         'oblique_rate': float(g[g.approach == 'oblique'].sim_failure.mean()),
                         'worst_family': g.groupby('family').sim_failure.mean().idxmax(),
                         'worst_family_rate': float(g.groupby('family').sim_failure.mean().max()),
                         'fail_outcome_before': '', 'median_max_speed_failed': float(g[g.sim_failure].max_object_speed_mps.median())
                         if g.sim_failure.any() else float('nan')})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    torch.set_num_threads(1)
    from visuograsp_v4.env import VisuoGraspV4Env
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    actor = bc_actor(env)
    pe = phase_errors(env, actor)
    pe.to_csv(OUT / 'bc_phase_errors.csv', index=False)
    ph = pe.groupby('phase').agg(steps=('mse_all', 'size'), mse=('mse_all', 'mean'), mse_fingers=('mse_fingers', 'mean'),
                                 mse_trans=('mse_trans', 'mean'), mse_rot=('mse_rot', 'mean'), abs_a=('abs_a', 'mean'))
    ph['share_of_steps'] = ph.steps / ph.steps.sum()
    ph.to_csv(OUT / 'bc_phase_error_summary.csv')
    print(ph.round(5).to_string())
    mt = matched(env, actor)
    mt.to_csv(OUT / 'dev_a_scripted_vs_bc.csv', index=False)
    s, b = mt[mt.policy == 'scripted'].set_index(['seed', 'approach']), mt[mt.policy == 'bc_w4'].set_index(['seed', 'approach'])
    j = s[['family', 'gt_success', 'outcome']].join(b[['gt_success', 'outcome']], rsuffix='_bc')
    j['cat'] = np.select([j.gt_success & j.gt_success_bc, j.gt_success_bc, j.gt_success], ['both', 'only_bc', 'only_scripted'], 'neither')
    print(mt.groupby(['policy', 'approach']).gt_success.mean().unstack().round(3).to_string())
    print(j.reset_index().groupby(['approach', 'cat']).size().unstack(fill_value=0).to_string())
    print('BC outcomes where scripted succeeded and BC failed:', j[j.cat == 'only_scripted'].outcome_bc.value_counts().to_dict())
    print('scripted outcomes in neither:', j[j.cat == 'neither'].outcome.value_counts().to_dict())
    print(mt.groupby(['policy', 'family']).gt_success.mean().unstack().round(2).to_string())
    fl = failures()
    fl.to_csv(OUT / 'numerical_failures.csv', index=False)
    print(fl.drop(columns=['fail_outcome_before']).round(3).to_string(index=False))
    (OUT / 'summary.json').write_text(json.dumps({
        'dev_a_gt': mt.groupby('policy').gt_success.mean().to_dict(),
        'dev_a_categories': j.cat.value_counts().to_dict()}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
