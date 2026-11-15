"""Throughput benchmark for SAC V4 training (short; no training result is produced).

Measures: env-only steps/s; SAC steps/s with gradient updates (UTD 1, batch 256, LayerNorm
critics, n-step 10) for several torch thread counts and devices; deterministic evaluation episode
time; and N parallel training processes (contention on this laptop).

    python -m visuograsp_v4.scripts.benchmark_throughput                 # full benchmark (~5 min)
    python -m visuograsp_v4.scripts.benchmark_throughput --single 2 cpu  # one SAC measurement (JSON)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np

from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config

UPDATE_STEPS = 3000


def make_env():
    from visuograsp_v4.env import VisuoGraspV4Env
    return VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))


def env_only(steps=2000) -> dict:
    env = make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t0, resets = time.time(), 0
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 22).astype(np.float32))
        if term or trunc:
            env.reset()
            resets += 1
    dt = time.time() - t0
    return {'env_steps_per_s': steps / dt, 'resets': resets}


def sac_single(threads: int, device: str) -> dict:
    import torch
    from visuograsp_v4.agents import make_sac
    cfg = replace(SACConfig(), torch_threads=threads, device=device, learning_starts=1000)
    env = make_env()
    model = make_sac(env, cfg, seed=0)
    t0 = time.time()
    model.learn(1000, progress_bar=False)                      # collection only
    collect = 1000 / (time.time() - t0)
    t0 = time.time()
    model.learn(UPDATE_STEPS, reset_num_timesteps=False, progress_bar=False)
    train = UPDATE_STEPS / (time.time() - t0)
    # deterministic evaluation episode time
    from visuograsp_v4.evaluation import run_policy_episode
    ev = make_env()
    t0 = time.time()
    for k in range(4):
        run_policy_episode(ev, lambda o: model.predict(o, deterministic=True)[0], 5_100_000 + k, 'box', 'cube',
                           'upright', 'top')
    eval_s = (time.time() - t0) / 4
    return {'threads': threads, 'device': device, 'collect_steps_per_s': collect, 'train_steps_per_s': train,
            'eval_seconds_per_episode': eval_s, 'torch': torch.__version__}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', nargs=2, metavar=('THREADS', 'DEVICE'))
    ap.add_argument('--parallel', type=int, nargs='*', default=[3, 4])
    args = ap.parse_args()
    if args.single:
        print(json.dumps(sac_single(int(args.single[0]), args.single[1])))
        return
    out = OUTPUT_ROOT / 'benchmark'
    out.mkdir(parents=True, exist_ok=True)
    res = {'env_only': env_only()}
    print('[env]', res['env_only'], flush=True)
    res['sac_single'] = []
    for threads, device in ((1, 'cpu'), (2, 'cpu'), (4, 'cpu'), (2, 'cuda')):
        r = sac_single(threads, device)
        res['sac_single'].append(r)
        print('[sac]', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res['parallel'] = []
    for n in args.parallel:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, '-m', 'visuograsp_v4.scripts.benchmark_throughput', '--single', '2', 'cpu'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  cwd=str(OUTPUT_ROOT.parents[1]), env={**os.environ, 'PYTHONPATH': str(OUTPUT_ROOT.parents[1])})
                 for _ in range(n)]
        outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        per = [o['train_steps_per_s'] for o in outs]
        r = {'processes': n, 'threads_each': 2, 'per_process_train_steps_per_s': per,
             'aggregate_train_steps_per_s': float(sum(per)), 'wall_s': time.time() - t0}
        res['parallel'].append(r)
        print('[parallel]', {k: (np.round(v, 1).tolist() if isinstance(v, list) else round(v, 1)) for k, v in r.items()}, flush=True)
    (out / 'throughput.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print('written to', out / 'throughput.json')


if __name__ == '__main__':
    main()
