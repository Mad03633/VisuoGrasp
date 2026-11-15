"""Wave 5 demonstration collection (scripted expert, PRIVILEGED planner) with phase labels and optional DART noise.

Per step: observation, LABEL = the expert's clean action (clipped to +-0.999), EXECUTED action (= label, or
label + N(0, sigma^2) clipped, for DART episodes), reward, next observation, done / timeout, expert phase.
DART (Laskey et al. 2017): the expert itself stays in control (its phase machine sees the states it reaches),
so labels remain consistent with the observation, unlike DAgger with a learner in control (Wave 4 failure).
Training seeds only (3_600_000+ / 3_700_000+); never validation or test seeds.

    python -m visuograsp_v4.scripts.wave5_collect_demos --seed-base 3600000 --episodes 600 --oblique-frac 0.67 --out demos_clean
    python -m visuograsp_v4.scripts.wave5_collect_demos --seed-base 3700000 --episodes 600 --oblique-frac 0.67 --dart-sigma 0.05 --out demos_dart
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config

OUT = OUTPUT_ROOT / 'rl' / 'wave5' / 'demos'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-base', type=int, required=True)
    ap.add_argument('--episodes', type=int, required=True)
    ap.add_argument('--oblique-frac', type=float, default=0.5)
    ap.add_argument('--dart-sigma', type=float, default=0.0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    assert 3_600_000 <= args.seed_base < 3_800_000, 'Wave 5 training seeds only'
    out = OUT / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.scripted import PHASES, ScriptedGrasp
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    rng = np.random.default_rng(args.seed_base)
    cols = {k: [] for k in ('obs', 'label', 'act', 'rew', 'next_obs', 'done', 'timeout', 'episode', 'phase')}
    eps = []
    t0 = time.time()
    n_obl = int(round(args.episodes * args.oblique_frac))
    approaches = ['oblique'] * n_obl + ['top'] * (args.episodes - n_obl)
    rng.shuffle(approaches)
    for k in range(args.episodes):
        seed, apn = args.seed_base + k, approaches[k]
        obs, _ = env.reset(seed=seed, options={'approach': apn})
        ag = ScriptedGrasp(env.sim, env.approach)
        n0 = len(cols['obs'])
        while True:
            phase = PHASES.index(ag.phase)
            label = np.clip(ag.act(), -0.999, 0.999).astype(np.float32)
            a = label if args.dart_sigma <= 0 else np.clip(label + rng.normal(0, args.dart_sigma, 22), -0.999, 0.999).astype(np.float32)
            nxt, r, term, trunc, _ = env.step(a)
            for key, val in (('obs', obs), ('label', label), ('act', a), ('rew', r), ('next_obs', nxt), ('done', term or trunc),
                             ('timeout', trunc and not term), ('episode', k), ('phase', phase)):
                cols[key].append(val)
            obs = nxt
            if term or trunc:
                break
        t = env.sim.tracker
        eps.append({'episode': k, 'seed': seed, 'approach': apn, 'family': env.sim.spec.family, 'steps': len(cols['obs']) - n0,
                    'gt_success': bool(t.gt_success), 'sim_failure': bool(t.nonfinite or t.runaway), 'max_lift_cm': 100 * t.max_lift})
        if (k + 1) % 100 == 0:
            print(f'[{args.out}] {k + 1}/{args.episodes} episodes, success so far {np.mean([e["gt_success"] for e in eps]):.3f}', flush=True)
    ed = pd.DataFrame(eps)
    ed.to_csv(out / 'episodes.csv', index=False)
    np.savez_compressed(out / 'data.npz', **{k: np.asarray(v, np.float32 if k in ('obs', 'label', 'act', 'rew', 'next_obs') else
                                                         (bool if k in ('done', 'timeout') else np.int32)) for k, v in cols.items()})
    meta = {'seed_base': args.seed_base, 'episodes': args.episodes, 'oblique_frac': args.oblique_frac, 'dart_sigma': args.dart_sigma,
            'transitions': len(cols['obs']), 'gt_success': float(ed.gt_success.mean()),
            'gt_success_by_approach': ed.groupby('approach').gt_success.mean().to_dict(),
            'sim_failures': int(ed.sim_failure.sum()), 'phases': list(PHASES), 'seconds': time.time() - t0,
            'privileged': 'scripted expert plans from the simulator object pose'}
    (out / 'meta.json').write_text(json.dumps(meta, indent=1, default=float), encoding='utf-8')
    print('[DONE]', json.dumps(meta, default=float), flush=True)


if __name__ == '__main__':
    main()
