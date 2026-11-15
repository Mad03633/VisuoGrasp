"""Wave 3: conservative residual SAC on the V4 scripted controller. Writes only to outputs/rl/wave3/<run>/.

Differences from Wave 2 (train_residual_sac.py, unchanged):
  * residual res2: bounds 0.08 rad (fingers) / 0.004 m (wrist position) / 0.03 rad (wrist orientation)
  * reward v4r2 = v4r1 - 0.01 * mean(r^2)  (v4r1 itself unchanged)
  * zero-centred exploration: the warm-up (learning_starts = 5,000) samples the policy itself
    (mean exactly 0, state-independent std 0.2 before tanh) instead of uniform random actions
  * entropy: ent_coef auto with initial value 0.01 (Wave 2: 0.1) and target entropy -5.0 = the
    entropy of tanh(N(0, 0.2^2)) in 22 dims (SB3 default -22 corresponds to std 0.09); the automatic
    coefficient therefore holds exploration near std 0.2 instead of pushing it up or letting it collapse
  * decision gates at 25k and 50k (run waits for gate_decision.json), maximum 75k
Unchanged: SB3 2.9.2 SAC, LayerNorm critics, n-step 10, lr 3e-4, batch 256, gamma 0.99, tau 0.005,
buffer 1M, UTD 1, actor mean layer zero-initialised (initial deterministic residual exactly 0).
PRIVILEGED: nominal controller planned from the simulator object pose; actor observation contains the
simulator object pose; contacts = GT oracle. Not autonomous, not vision-based.

    python -m visuograsp_v4.scripts.train_conservative_residual_sac --seed 11 --total-steps 75000 --gates 25000 50000
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

from visuograsp_v4.agents import GTEpisodeInfo, V4TrainCallback, ZeroCenteredWarmupSAC, make_sac, validation_grid
from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config
from visuograsp_v4.residual import (RESIDUAL_CONFIGS, ResidualGraspEnv, init_actor_log_std, tanh_gaussian_entropy,
                                    zero_init_actor_mean)
from visuograsp_v4.scripts.train_sac import code_fingerprint, keep_awake

VERIFICATION = OUTPUT_ROOT / 'rl' / 'wave3' / 'verification' / 'verification_report.json'


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--total-steps', type=int, required=True)
    ap.add_argument('--gates', type=int, nargs='*', default=[25_000, 50_000])
    ap.add_argument('--residual-version', default='res2')
    ap.add_argument('--init-std', type=float, default=0.2)
    ap.add_argument('--ent-init', type=float, default=0.01)
    ap.add_argument('--learning-starts', type=int, default=5_000)
    ap.add_argument('--eval-every', type=int, default=12_500)
    ap.add_argument('--val-per-cell', type=int, default=5)
    ap.add_argument('--checkpoint-every', type=int, default=12_500)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--pipeline-check', action='store_true')
    args = ap.parse_args()

    ver = json.loads(VERIFICATION.read_text(encoding='utf-8')) if VERIFICATION.exists() else {}
    if not ver.get('all_required_passed') or ver.get('residual_version') != args.residual_version:
        raise SystemExit(f'residual verification for {args.residual_version} not passed ({VERIFICATION})')
    rcfg = RESIDUAL_CONFIGS[args.residual_version]
    target_entropy = round(tanh_gaussian_entropy(args.init_std), 2)
    sac_cfg = replace(SACConfig(), learning_starts=args.learning_starts, ent_coef=f'auto_{args.ent_init}',
                      eval_every=args.eval_every, checkpoint_every=args.checkpoint_every, torch_threads=args.threads)
    if args.pipeline_check:
        sac_cfg = replace(sac_cfg, learning_starts=500, eval_every=1000, checkpoint_every=1000, q_health_every=1000)
    cfg = V4Config()
    name = args.run_name or f'{"check_" if args.pipeline_check else ""}conservative_residual_sac_seed{args.seed}'
    out = OUTPUT_ROOT / 'rl' / 'wave3' / name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty; choose a new --run-name (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)

    def make_env():
        return ResidualGraspEnv(cfg, rcfg, approaches=('top', 'oblique'))
    env, eval_env = GTEpisodeInfo(make_env()), make_env()
    model = make_sac(env, sac_cfg, args.seed, algo_cls=ZeroCenteredWarmupSAC, target_entropy=target_entropy)
    zero_init_actor_mean(model)
    init_actor_log_std(model, args.init_std)
    probe = np.random.default_rng(0).normal(size=(256, env.observation_space.shape[0])).astype(np.float32)
    init_zero = bool(np.all(model.predict(probe, deterministic=True)[0] == 0.0))
    samples = model.predict(probe, deterministic=False)[0]
    if not init_zero:
        raise SystemExit('initial deterministic residual is not exactly zero')
    model.set_logger(configure(str(out / 'sb3'), ['csv']))
    meta = {'started': datetime.now().isoformat(timespec='seconds'), 'mode': 'conservative_residual_sac (Wave 3)',
            'seed': args.seed, 'total_steps': args.total_steps, 'gates': args.gates, 'approaches': ['top', 'oblique'],
            'pipeline_check': args.pipeline_check, 'val_per_cell': args.val_per_cell,
            'privileged': 'nominal scripted controller planned from the simulator object pose; actor observation '
                          'contains the simulator object pose; contacts = GT oracle. Not autonomous, not vision-based.',
            'residual_config': rcfg.to_dict(), 'reward': 'v4r2 = v4r1 - 0.01 * mean(r^2)',
            'exploration': {'warmup': 'policy samples (zero mean) for learning_starts steps, no uniform actions',
                            'init_std_pre_tanh': args.init_std, 'ent_coef_init': args.ent_init,
                            'target_entropy': target_entropy,
                            'initial_stochastic_abs_residual_mean': float(np.abs(samples).mean())},
            'initial_deterministic_residual_exactly_zero': init_zero, 'verification_report': str(VERIFICATION),
            'v4_config': cfg.to_dict(), 'sac_config': asdict(sac_cfg), 'obs_dim': int(env.observation_space.shape[0]),
            'code_sha256_16': code_fingerprint(),
            'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__,
                         'sb3': stable_baselines3.__version__, 'torch': torch.__version__},
            'seeds': {'sb3_seed': args.seed, 'model_selection': 8_000_000, 'gate_validation': 8_500_000,
                      'final_evaluation_unseen': 6_100_000, 'exposed_test_not_used': 5_100_000},
            'pid': os.getpid()}
    (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    print('[INIT]', meta['exploration'], flush=True)
    grid = validation_grid(cfg.objects.families, ['top', 'oblique'], 1 if args.pipeline_check else args.val_per_cell)
    cb = V4TrainCallback(out, sac_cfg, eval_env, grid, gate_at=args.gates or None)
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
