"""Independent test evaluation of V4 policies, paired with the scripted baseline.

Test grid = the scripted-baseline grid: family x variant x resting mode x approach x N episodes,
seed = 5_100_000 + grid_index*1000 + approach_index*100 + episode (approach index in
('top', 'oblique', 'side') is kept even when only a subset of approaches is evaluated), so every
policy starts from exactly the same initial states as the scripted baseline.

    python -m visuograsp_v4.scripts.evaluate_policy --model outputs/rl/<run>/best_model.zip --label sac_demo_s11
    python -m visuograsp_v4.scripts.evaluate_policy --scripted-env --approaches top --episodes 2   # pairing check
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from visuograsp_v4 import objects as objs
from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.env import VisuoGraspV4Env
from visuograsp_v4.evaluation import run_policy_episode
from visuograsp_v4.scripted import ScriptedGrasp

APPROACHES = ('top', 'oblique', 'side')
TEST_SEED_BASE = 5_100_000
SCRIPTED_CSV = (OUTPUT_ROOT / 'scripted' / 'scripted_baseline_elliptic_imp10_noslip5_dt2.5ms_sr6_arm0.001'
                / 'episodes.csv')


def test_grid(cfg: V4Config, approaches, episodes: int) -> list[tuple]:
    cells = [(f, v, m) for f in cfg.objects.families for v in objs.VARIANTS[f] for m in objs.RESTING_MODES[f]]
    grid = []
    for gi, (f, v, m) in enumerate(cells):
        for ap in approaches:
            ai = APPROACHES.index(ap)
            for ep in range(episodes):
                grid.append((TEST_SEED_BASE + gi * 1000 + ai * 100 + ep, f, v, m, ap))
    return grid


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def paired_vs_scripted(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(SCRIPTED_CSV)[['seed', 'approach', 'gt_success']].rename(columns={'gt_success': 'scripted'})
    m = df.merge(base, on=['seed', 'approach'], how='inner')
    rows = []
    for ap, g in m.groupby('approach'):
        a, b = g.gt_success.astype(bool), g.scripted.astype(bool)
        only_pol, only_scr = int((a & ~b).sum()), int((~a & b).sum())
        p = binomtest(only_pol, only_pol + only_scr).pvalue if only_pol + only_scr else float('nan')
        rows.append({'approach': ap, 'paired_episodes': len(g), 'policy_gt': a.mean(), 'scripted_gt': b.mean(),
                     'both': int((a & b).sum()), 'only_policy': only_pol, 'only_scripted': only_scr,
                     'neither': int((~a & ~b).sum()), 'mcnemar_exact_p': p})
    return pd.DataFrame(rows)


def main() -> None:
    cwd0 = Path.cwd()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--scripted-env', action='store_true', help='run the scripted controller through the env path')
    ap.add_argument('--approaches', nargs='*', default=['top', 'oblique', 'side'])
    ap.add_argument('--episodes', type=int, default=6)
    args = ap.parse_args()
    if args.model and not Path(args.model).is_absolute():
        args.model = str((cwd0 / args.model).resolve())       # resolve before changing directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    cfg = V4Config()
    env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle')
    holder = {}
    if args.scripted_env:
        label = args.label or 'scripted_via_env'
        on_reset = lambda e: holder.update(agent=ScriptedGrasp(e.sim, e.approach))

        def act(obs):
            return holder['agent'].act()
    else:
        from visuograsp_v4.agents import load_sac
        model = load_sac(Path(args.model))
        label = args.label or Path(args.model).parent.name
        on_reset = None

        def act(obs):
            return model.predict(obs, deterministic=True)[0]
    out = OUTPUT_ROOT / 'eval' / label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (seed, f, v, m, ap_name) in enumerate(test_grid(cfg, args.approaches, args.episodes)):
        row = run_policy_episode(env, act, seed, f, v, m, ap_name, on_reset=on_reset)
        row['policy'] = label
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f'{k + 1} episodes', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'episodes.csv', index=False)
    summ = []
    for ap_name, g in df.groupby('approach'):
        k, n = int(g.gt_success.sum()), len(g)
        lo, hi = wilson(k, n)
        summ.append({'approach': ap_name, 'episodes': n, 'gt_success': k / n, 'wilson_lo': lo, 'wilson_hi': hi,
                     'gt_any_phalanx': g.gt_success_any_phalanx.mean(), 'sim_failures': int(g.sim_failure.sum()),
                     'left_table': int(g.object_off_table.sum()), 'mean_max_lift_cm': g.max_lift_cm.mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(out / 'summary.csv', index=False)
    paired = paired_vs_scripted(df)
    paired.to_csv(out / 'paired_vs_scripted.csv', index=False)
    pd.set_option('display.width', 200)
    print(summ.round(3).to_string(index=False))
    print(paired.round(4).to_string(index=False))
    print('written to', out)


if __name__ == '__main__':
    main()
