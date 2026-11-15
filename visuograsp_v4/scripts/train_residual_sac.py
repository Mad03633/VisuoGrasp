"""Train Residual SAC on top of the V4 scripted controller (Wave 2). Writes only to outputs/rl/wave2/<run>/.

PRIVILEGED setting: nominal controller planned from the simulator object pose, actor observation
with the simulator object pose, GT-oracle contacts. Residual = bounded set-point corrections
(see residual.py). The actor's mean layer is zero-initialised, so the initial deterministic policy
is exactly the scripted controller. Training refuses to start unless
outputs/rl/wave2/verification/verification_report.json reports all required checks passed.

    python -m visuograsp_v4.scripts.train_residual_sac --seed 11 --total-steps 150000 --gate-at 75000
    python -m visuograsp_v4.scripts.train_residual_sac --seed 11 --total-steps 3000 --gate-at 2000 --pipeline-check
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime

import mujoco
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.logger import configure

from visuograsp_v4.agents import GTEpisodeInfo, V4TrainCallback, make_sac, validation_grid
from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config
from visuograsp_v4.residual import ResidualConfig, ResidualGraspEnv, zero_init_actor_mean
from visuograsp_v4.scripts.train_sac import code_fingerprint, keep_awake

VERIFICATION = OUTPUT_ROOT / 'rl' / 'wave2' / 'verification' / 'verification_report.json'


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--total-steps', type=int, required=True)
    ap.add_argument('--gate-at', type=int, default=None)
    ap.add_argument('--approaches', nargs='*', default=['top', 'oblique'])
    ap.add_argument('--val-per-cell', type=int, default=5)
    ap.add_argument('--checkpoint-every', type=int, default=25_000)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--pipeline-check', action='store_true')
    args = ap.parse_args()

    ver = json.loads(VERIFICATION.read_text(encoding='utf-8')) if VERIFICATION.exists() else {}
    if not ver.get('all_required_passed'):
        raise SystemExit(f'residual verification not passed ({VERIFICATION}); run scripts/verify_residual.py first')

    sac_cfg = replace(SACConfig(), checkpoint_every=args.checkpoint_every)
    if args.threads:
        sac_cfg = replace(sac_cfg, torch_threads=args.threads)
    if args.pipeline_check:
        sac_cfg = replace(sac_cfg, learning_starts=500, eval_every=1000, checkpoint_every=1000, q_health_every=1000)
    cfg, rcfg = V4Config(), ResidualConfig()
    name = args.run_name or f'{"check_" if args.pipeline_check else ""}residual_sac_seed{args.seed}'
    out = OUTPUT_ROOT / 'rl' / 'wave2' / name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty; choose a new --run-name (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)

    def make_env():
        return ResidualGraspEnv(cfg, rcfg, approaches=tuple(args.approaches))
    env, eval_env = GTEpisodeInfo(make_env()), make_env()
    model = make_sac(env, sac_cfg, args.seed)
    zero_init_actor_mean(model)
    probe = np.random.default_rng(0).normal(size=(256, env.observation_space.shape[0])).astype(np.float32)
    init_zero = bool(np.all(model.predict(probe, deterministic=True)[0] == 0.0))
    if not init_zero:
        raise SystemExit('initial deterministic residual is not exactly zero')
    model.set_logger(configure(str(out / 'sb3'), ['csv']))
    meta = {'started': datetime.now().isoformat(timespec='seconds'), 'mode': 'residual_sac', 'seed': args.seed,
            'total_steps': args.total_steps, 'gate_at': args.gate_at, 'approaches': args.approaches,
            'pipeline_check': args.pipeline_check, 'val_per_cell': args.val_per_cell,
            'privileged': 'nominal scripted controller planned from the simulator object pose; actor observation '
                          'contains the simulator object pose; contacts = GT oracle. Not autonomous, not vision-based.',
            'residual_config': rcfg.to_dict(), 'initial_deterministic_residual_exactly_zero': init_zero,
            'verification_report': str(VERIFICATION), 'v4_config': cfg.to_dict(), 'sac_config': asdict(sac_cfg),
            'obs_dim': int(env.observation_space.shape[0]), 'code_sha256_16': code_fingerprint(),
            'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__,
                         'sb3': stable_baselines3.__version__, 'torch': torch.__version__},
            'seeds': {'sb3_seed': args.seed, 'validation_seed_base': 8_000_000, 'independent_validation_seed_base': 8_500_000,
                      'test_seed_base': 5_100_000}, 'pid': os.getpid()}
    (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    grid = validation_grid(cfg.objects.families, args.approaches, 1 if args.pipeline_check else args.val_per_cell)
    cb = V4TrainCallback(out, sac_cfg, eval_env, grid, gate_at=args.gate_at)
    keep_awake(True)
    t0 = time.time()
    error = None
    try:
        model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        (out / 'crash_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        try:
            model.save(str(out / 'crash_model.zip'))
        except Exception:
            pass
    finally:
        keep_awake(False)
        cb._on_training_end()
    secs = time.time() - t0
    if error is None:
        model.save(str(out / 'last_model.zip'))
    (out / 'finished.json').write_text(json.dumps({
        'steps': int(model.num_timesteps), 'requested_steps': args.total_steps, 'seconds_wall': secs,
        'seconds_training_excl_waits': cb.training_seconds(), 'seconds_waiting_gate_or_pause': cb.wait_seconds,
        'steps_per_s_training': model.num_timesteps / max(1e-6, cb.training_seconds()),
        'stop_reason': cb.stop_reason or ('error' if error else 'completed'), 'error': error,
        'best_validation_gt': cb.best, 'best_step': cb.best_step, 'best_key': cb.best_key,
        'finished': datetime.now().isoformat(timespec='seconds')}, indent=1, default=str), encoding='utf-8')
    print(f'[DONE] {name}: {model.num_timesteps:,} steps in {secs / 60:.1f} min, stop={cb.stop_reason or error or "completed"}, '
          f'best val GT {cb.best:.3f} @ {cb.best_step}', flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
