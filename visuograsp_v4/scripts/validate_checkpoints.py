"""Independent validation of saved SAC checkpoints (Wave 1 gate and final validation).

Seeds 8_500_000 + family*1000 + approach*100 + episode (approach index in ('top', 'oblique',
'side')). This set is disjoint from the in-training model-selection set (8_000_000+, used to pick
best_model.zip), from demo seeds (3_000_000+) and from the test grid (5_100_000+, NOT used here).
Variant and resting mode are drawn from the seed. Deterministic policy (mean action).
PRIVILEGED setting: the actor observation contains the simulator object pose, contacts = GT oracle.

    python -m visuograsp_v4.scripts.validate_checkpoints --run wave1/sac_demo_s11 --tag gate150k \
        --models checkpoints/step_50000.zip checkpoints/step_100000.zip best_model.zip --per-cell 10
    python -m visuograsp_v4.scripts.validate_checkpoints --scripted --tag scripted_reference --per-cell 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

IVAL_SEED_BASE = 8_500_000
ALL_APPROACHES = ('top', 'oblique', 'side')


def ival_grid(families, approaches, per_cell: int) -> list[tuple]:
    from visuograsp_v4 import objects as objs
    grid = []
    for fi, fam in enumerate(families):
        for ap in approaches:
            ai = ALL_APPROACHES.index(ap)
            for ep in range(per_cell):
                seed = IVAL_SEED_BASE + fi * 1000 + ai * 100 + ep
                rng = np.random.default_rng(seed)
                variant = str(rng.choice(list(objs.VARIANTS[fam])))
                mode = str(rng.choice(list(objs.RESTING_MODES[fam])))
                grid.append((seed, fam, variant, mode, ap))
    return grid


def model_step(path: Path) -> int | None:
    m = re.search(r'step_(\d+)', path.name)
    if m:
        return int(m.group(1))
    meta = path.with_suffix('.json')
    if meta.exists():
        return int(json.loads(meta.read_text(encoding='utf-8'))['step'])
    if path.name == 'last_model.zip' and (path.parent / 'finished.json').exists():
        return int(json.loads((path.parent / 'finished.json').read_text(encoding='utf-8'))['steps'])
    return None


def _run(job) -> pd.DataFrame:
    """One worker: evaluate one model (or the scripted controller) on the whole grid."""
    label, model_path, approaches, per_cell, threads = job[:5]
    env_kind = job[5] if len(job) > 5 else 'base'        # 'base' (V4 env) or 'residual' (Wave 2/3)
    res_version = job[6] if len(job) > 6 else 'res1'
    import torch
    torch.set_num_threads(threads)               # 1 = reproducible reference setting
    os.chdir(OUTPUT_ROOT)                        # MUJOCO_LOG.TXT stays inside V4 outputs
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.evaluation import run_policy_episode
    cfg = V4Config()
    if env_kind == 'residual':
        from visuograsp_v4.residual import RESIDUAL_CONFIGS, ResidualGraspEnv
        env = ResidualGraspEnv(cfg, RESIDUAL_CONFIGS[res_version], approaches=tuple(approaches))
    else:
        env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=tuple(approaches))
    holder = {}
    if model_path == 'ZERO_RESIDUAL':
        assert env_kind == 'residual'
        on_reset = None
        act = lambda obs: np.zeros(22, np.float32)
    elif model_path == 'PERTURBED_1e-3':
        # chaos control: physically negligible random residual (1e-3 of the bounds), fixed RNG
        assert env_kind == 'residual'
        on_reset = None
        prng = np.random.default_rng(12345)
        act = lambda obs: (1e-3 * prng.uniform(-1, 1, 22)).astype(np.float32)
    elif model_path is None:
        from visuograsp_v4.scripted import ScriptedGrasp
        on_reset = lambda e: holder.update(agent=ScriptedGrasp(e.sim, e.approach))
        act = lambda obs: holder['agent'].act()
    else:
        from visuograsp_v4.agents import load_sac
        model = load_sac(model_path)
        on_reset = None
        act = lambda obs: model.predict(obs, deterministic=True)[0]
    rows = []
    for seed, f, v, m, ap in ival_grid(cfg.objects.families, approaches, per_cell):
        row = run_policy_episode(env, act, seed, f, v, m, ap, on_reset=on_reset)
        row['model'] = label
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by) -> pd.DataFrame:
    from visuograsp_v4.scripts.evaluate_policy import wilson as _wilson
    wilson = lambda k, n: tuple(min(1.0, max(0.0, x)) for x in _wilson(k, n))
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(g.gt_success.sum()), len(g)
        lo, hi = wilson(k, n)
        ka = int(g.gt_success_any_phalanx.sum())
        alo, ahi = wilson(ka, n)
        part = int((g.max_lift_cm >= 3.5).sum())
        plo, phi = wilson(part, n)
        r = dict(zip(by, key))
        r.update(episodes=n, gt_success_n=k, gt_success=k / n, wilson_lo=lo, wilson_hi=hi,
                 gt_any_phalanx=ka / n, any_wilson_lo=alo, any_wilson_hi=ahi,
                 partial_lift_rate=part / n, partial_wilson_lo=plo, partial_wilson_hi=phi,
                 contact_rate=float(g.contact_occurred.mean()), mean_max_lift_cm=float(g.max_lift_cm.mean()),
                 mean_return=float(g.episode_return.mean()), sim_failures=int(g.sim_failure.sum()),
                 left_table=int(g.object_off_table.sum()))
        for oc, c in g.outcome.value_counts().items():
            r[f'outcome_{oc}'] = int(c)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    cwd0 = Path.cwd()
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='run folder relative to outputs/rl (e.g. wave1/sac_demo_s11)')
    ap.add_argument('--models', nargs='*', default=[], help='model files relative to the run folder')
    ap.add_argument('--scripted', action='store_true', help='PRIVILEGED scripted controller on the same seeds')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--approaches', nargs='*', default=['top', 'oblique'])
    ap.add_argument('--per-cell', type=int, default=10)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--out', default=None, help='output folder for --scripted (relative to outputs/rl)')
    ap.add_argument('--env', choices=('base', 'residual'), default='base')
    ap.add_argument('--zero-residual', action='store_true', help='add the r = 0 policy (residual env only)')
    ap.add_argument('--perturbed-control', action='store_true',
                    help='add the chaos control r ~ U(-1e-3, 1e-3) (residual env only)')
    ap.add_argument('--residual-version', default='res1')
    ap.add_argument('--torch-threads', type=int, default=1,
                    help='1 (default) is the reproducible setting; >1 only as a sensitivity check')
    args = ap.parse_args()
    rl = OUTPUT_ROOT / 'rl'
    jobs = []
    if args.scripted:
        out = rl / (args.out or 'wave1/independent_validation_scripted') / args.tag
        jobs.append(('scripted_PRIVILEGED', None, args.approaches, args.per_cell, args.torch_threads))
        steps = {'scripted_PRIVILEGED': None}
    else:
        run = rl / args.run
        out = run / 'independent_validation' / args.tag
        steps = {}
        for mpath in args.models:
            p = (run / mpath) if not Path(mpath).is_absolute() else Path(mpath)
            if not p.exists():
                raise SystemExit(f'missing model {p}')
            label = p.stem if p.parent == run else f'{p.parent.name}/{p.stem}'
            steps[label] = model_step(p)
            jobs.append((label, p, args.approaches, args.per_cell, args.torch_threads, args.env, args.residual_version))
        if args.zero_residual:
            steps['zero_residual'] = 0
            jobs.append(('zero_residual', 'ZERO_RESIDUAL', args.approaches, args.per_cell, args.torch_threads, args.env,
                         args.residual_version))
        if args.perturbed_control:
            steps['perturbed_1e-3'] = 0
            jobs.append(('perturbed_1e-3', 'PERTURBED_1e-3', args.approaches, args.per_cell, args.torch_threads, args.env,
                         args.residual_version))
    if out.exists() and any(out.glob('*.csv')):
        raise SystemExit(f'{out} already has results; use a new --tag (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    workers = args.workers or min(len(jobs), 4)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            dfs = list(ex.map(_run, jobs))
    else:
        dfs = [_run(j) for j in jobs]
    df = pd.concat(dfs, ignore_index=True)
    df['step'] = df.model.map(steps)
    df.to_csv(out / 'episodes.csv', index=False)
    s_all = summarise(df.assign(approach='all'), ['model', 'approach'])
    s_ap = summarise(df, ['model', 'approach'])
    summ = pd.concat([s_all, s_ap], ignore_index=True)
    summ['step'] = summ.model.map(steps)
    summ = summ.sort_values(['step', 'model', 'approach'], na_position='last')
    summ.to_csv(out / 'summary.csv', index=False)
    summarise(df, ['model', 'family', 'approach']).to_csv(out / 'summary_by_family.csv', index=False)
    meta = {'seed_base': IVAL_SEED_BASE, 'per_cell': args.per_cell, 'approaches': args.approaches,
            'episodes_per_model': len(dfs[0]), 'models': {k: (str(v) if v is not None else None) for k, v in steps.items()},
            'deterministic': True, 'torch_threads': args.torch_threads, 'env': args.env,
            'residual_version': args.residual_version if args.env == 'residual' else None, 'privileged': 'object pose from simulator in the observation; GT-oracle contacts',
            'seconds': time.time() - t0, 'note': 'independent validation; test seeds 5_100_000+ not used'}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    cols = ['model', 'step', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo', 'wilson_hi',
            'gt_any_phalanx', 'partial_lift_rate', 'contact_rate', 'mean_max_lift_cm', 'mean_return']
    print(summ[cols].round(3).to_string(index=False))
    print(f'written to {out} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
