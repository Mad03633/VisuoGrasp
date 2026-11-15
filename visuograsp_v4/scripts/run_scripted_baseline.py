"""V4 scripted baseline evaluation (diagnostic; uses PRIVILEGED object pose for planning).

Grid: family x variant x resting mode x approach (top / oblique / side), N episodes each.
Seeds: EVAL seeds 5_000_000 + ... (dev tuning used 9xx_xxx seeds only). Physics variants
are written to separate folders and must be reported as separate environment versions.

    python -m visuograsp_v4.scripts.run_scripted_baseline --episodes 4 --noslip 5
    python -m visuograsp_v4.scripts.run_scripted_baseline --episodes 4 --noslip 0
    python -m visuograsp_v4.scripts.run_scripted_baseline --smoke        # 1 episode per approach
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from visuograsp_v4 import objects as objs
from visuograsp_v4.config import OUTPUT_ROOT, PhysicsConfig, V4Config
from visuograsp_v4.evaluation import run_scripted_episode
from visuograsp_v4.sim import VisuoGraspSimV4

APPROACHES = ('top', 'oblique', 'side')
EVAL_SEED_BASE = 5_100_000   # 5_000_000+ was consumed while diagnosing the armature instability


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def summarise(df: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(g.gt_success.sum()), len(g)
        lo, hi = wilson(k, n)
        ka = int(g.gt_success_any_phalanx.sum())
        lo_a, hi_a = wilson(ka, n)
        rows.append({**dict(zip(keys, key)), 'episodes': n,
                     'gt_success': k / n, 'gt_wilson_lo': lo, 'gt_wilson_hi': hi,
                     'gt_any_phalanx': ka / n, 'any_wilson_lo': lo_a, 'any_wilson_hi': hi_a,
                     'contact_rate': g.contact_occurred.mean(),
                     'median_max_lift_cm': g.loc[~g.sim_failure, 'max_lift_cm'].median(),
                     'mean_hold_frames_gt': g.hold_frames_gt.mean(), 'pushed_before_close': g.pushed_before_close.mean(),
                     'reached_grasp_pose': g.reached_grasp.mean(), 'sim_failure': g.sim_failure.mean(),
                     'sim_failures_n': int(g.sim_failure.sum()),
                     'object_left_table_n': int(g.object_off_table.sum()),
                     'hand_table_force_max_n': g.hand_table_force_max_n.max(),
                     'plan_feasible': g.plan_feasible.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    # MuJoCo writes MUJOCO_LOG.TXT into the working directory on warnings: keep it inside V4 outputs
    import os
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=4, help='episodes per (family, variant, mode, approach)')
    ap.add_argument('--noslip', type=int, default=5)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    cfg = V4Config(physics=replace(PhysicsConfig(), noslip_iterations=args.noslip))
    name = args.out or ('scripted_smoke' if args.smoke else f'scripted_baseline_{cfg.physics.label}_arm{cfg.hand.joint_armature:g}')
    out = OUTPUT_ROOT / 'scripted' / name
    out.mkdir(parents=True, exist_ok=True)
    sim = VisuoGraspSimV4(cfg)
    grid = []
    for family in cfg.objects.families:
        for variant in objs.VARIANTS[family]:
            for mode in objs.RESTING_MODES[family]:
                grid.append((family, variant, mode))
    if args.smoke:
        grid = [('box', 'cube', 'upright'), ('bottle', 'std', 'upright')]
    episodes = 1 if args.smoke else args.episodes
    rows = []
    started = time.time()
    for gi, (family, variant, mode) in enumerate(grid):
        for ai, approach in enumerate(APPROACHES):
            for ep in range(episodes):
                seed = EVAL_SEED_BASE + gi * 1000 + ai * 100 + ep
                rng = np.random.default_rng(seed)
                spec = sim.sample_spec(rng, family, variant, mode)
                row = run_scripted_episode(sim, spec, approach, seed)
                row['physics'] = cfg.physics.label
                rows.append(row)
        print(f'[{gi + 1}/{len(grid)}] {family}/{variant}/{mode} done, {time.time() - started:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'episodes.csv', index=False)
    by_approach = summarise(df, ['approach'])
    by_family = summarise(df, ['family', 'approach'])
    by_mode = summarise(df, ['family', 'resting_mode', 'approach'])
    feasible = summarise(df[df.plan_feasible], ['approach'])
    by_approach.to_csv(out / 'summary_by_approach.csv', index=False)
    by_family.to_csv(out / 'summary_by_family_approach.csv', index=False)
    by_mode.to_csv(out / 'summary_by_family_mode_approach.csv', index=False)
    feasible.to_csv(out / 'summary_by_approach_feasible_only.csv', index=False)
    df.groupby(['approach', 'outcome']).size().unstack(fill_value=0).to_csv(out / 'outcomes.csv')
    meta = {'config': cfg.to_dict(), 'episodes_per_cell': episodes, 'grid_cells': len(grid),
            'seed_rule': 'EVAL_SEED_BASE + grid_index*1000 + approach_index*100 + episode',
            'eval_seed_base': EVAL_SEED_BASE, 'privileged': 'planner uses simulator object pose/geometry',
            'wall_seconds': time.time() - started, 'compilations': sim.cache.compilations,
            'mujoco': mujoco.__version__, 'python': platform.python_version(), 'platform': platform.platform()}
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    pd.set_option('display.width', 220)
    print(by_approach.round(3).to_string(index=False))
    print(feasible[['approach', 'episodes', 'gt_success', 'gt_wilson_lo', 'gt_wilson_hi']].round(3).to_string(index=False))
    print('written to', out)


if __name__ == '__main__':
    main()
