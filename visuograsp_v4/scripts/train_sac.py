"""Train SAC on VisuoGrasp V4 (state-based teacher setting). Writes only to outputs/rl/<run>/.

Arms (both PRIVILEGED: simulator object pose in the observation, GT-oracle contact = M0-style):
  scratch  SAC from scratch
  demo     SAC + replay buffer prefilled with scripted-controller transitions (PRIVILEGED planner)

    python -m visuograsp_v4.scripts.train_sac --arm scratch --seed 11 --total-steps 300000
    python -m visuograsp_v4.scripts.train_sac --arm demo --seed 11 --total-steps 300000
    python -m visuograsp_v4.scripts.train_sac --arm demo --seed 11 --total-steps 4000 --pipeline-check

``--gate-at N`` makes the run wait at step N (checkpoint saved, ``gate_waiting.json`` written)
until ``gate_decision.json`` = {"decision": "continue"|"stop", "reason": ...} appears in the run
folder (the replay buffer stays in memory, so continuing is exact). ``stop_request.json`` /
``pause_request.json`` in the run folder stop (after saving) or pause the run.

``--pipeline-check`` shrinks learning_starts / evaluation / checkpoint intervals so that every
logging path runs within a few thousand steps. It is not a training result.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime

import mujoco
import stable_baselines3
import torch
from stable_baselines3.common.logger import configure

from visuograsp_v4.agents import (GTEpisodeInfo, V4TrainCallback, make_sac, prefill_with_scripted,
                                  validation_grid, verify_demo_transitions, verify_nstep)
from visuograsp_v4.config import OUTPUT_ROOT, V4_ROOT, SACConfig, V4Config
from visuograsp_v4.env import VisuoGraspV4Env

TRAIN_DEMO_SEED_BASE = 3_000_000      # demo episodes; disjoint from validation 8M and test 5.1M


def code_fingerprint() -> dict:
    files = sorted(V4_ROOT.glob('*.py')) + sorted((V4_ROOT / 'scripts').glob('*.py'))
    return {f.relative_to(V4_ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files}


def keep_awake(on: bool) -> None:
    """Windows: keep the system out of idle sleep while this process trains (reset on exit)."""
    if sys.platform == 'win32':
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('scratch', 'demo'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--total-steps', type=int, required=True)
    ap.add_argument('--approaches', nargs='*', default=['top', 'oblique'])
    ap.add_argument('--demo-transitions', type=int, default=20_000)
    ap.add_argument('--val-per-cell', type=int, default=3)
    ap.add_argument('--gate-at', type=int, default=None)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--device', default=None)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--pipeline-check', action='store_true')
    ap.add_argument('--learning-starts', type=int, default=None, help='pipeline checks only')
    args = ap.parse_args()

    sac_cfg = SACConfig()
    if args.threads:
        sac_cfg = replace(sac_cfg, torch_threads=args.threads)
    if args.device:
        sac_cfg = replace(sac_cfg, device=args.device)
    if args.pipeline_check:
        sac_cfg = replace(sac_cfg, learning_starts=args.learning_starts or 500, eval_every=2000,
                          checkpoint_every=2000, q_health_every=1000)
    elif args.learning_starts is not None:
        raise SystemExit('--learning-starts is only allowed with --pipeline-check')
    cfg = V4Config()
    name = args.run_name or (f'{"check_" if args.pipeline_check else ""}sac_{args.arm}_'
                             f'{"-".join(args.approaches)}_seed{args.seed}')
    out = OUTPUT_ROOT / 'rl' / name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty; choose a new --run-name (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)

    def make_env():
        return VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle',
                               approaches=tuple(args.approaches))
    env, eval_env = GTEpisodeInfo(make_env()), make_env()
    model = make_sac(env, sac_cfg, args.seed)
    model.set_logger(configure(str(out / 'sb3'), ['csv']))
    meta = {'started': datetime.now().isoformat(timespec='seconds'), 'arm': args.arm, 'seed': args.seed,
            'total_steps': args.total_steps, 'gate_at': args.gate_at, 'approaches': args.approaches,
            'pipeline_check': args.pipeline_check, 'val_per_cell': args.val_per_cell,
            'privileged': 'observation uses simulator object pose; contact = GT oracle (M0-style)'
                          + ('; replay prefilled with scripted (privileged) demonstrations' if args.arm == 'demo' else ''),
            'v4_config': cfg.to_dict(), 'sac_config': asdict(sac_cfg), 'code_sha256_16': code_fingerprint(),
            'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__,
                         'sb3': stable_baselines3.__version__, 'torch': torch.__version__},
            'seeds': {'sb3_seed': args.seed, 'demo_seed_base': TRAIN_DEMO_SEED_BASE + 10_000 * args.seed,
                      'validation_seed_base': 8_000_000, 'test_seed_base': 5_100_000},
            'pid': os.getpid()}
    n_demo = 0
    if args.arm == 'demo':
        t0 = time.time()
        pre = prefill_with_scripted(model, make_env(), args.demo_transitions,
                                    TRAIN_DEMO_SEED_BASE + 10_000 * args.seed, args.approaches)
        records = pre.pop('episode_records')
        pre['seconds'] = time.time() - t0
        n_demo = pre['transitions']
        import pandas as pd
        pd.DataFrame(records).to_csv(out / 'demo_episodes.csv', index=False)
        t1 = time.time()
        ver = verify_demo_transitions(model, make_env, env, records)
        ver['nstep'] = verify_nstep(model.replay_buffer)
        ver['seconds'] = time.time() - t1
        (out / 'demo_verification.json').write_text(json.dumps(ver, indent=1, default=str), encoding='utf-8')
        meta['demo_prefill'] = pre
        meta['demo_verification_passed'] = bool(ver['passed'] and ver['nstep']['passed'])
        print('[DEMO]', pre, flush=True)
        print('[DEMO-VERIFY] passed' if meta['demo_verification_passed'] else '[DEMO-VERIFY] FAILED',
              {k: ver[k] for k in ('replayed_transitions', 'max_abs_obs_diff', 'max_abs_reward_diff',
                                   'done_mismatches')}, flush=True)
        if not meta['demo_verification_passed']:
            (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
            raise SystemExit('demonstration verification failed; see demo_verification.json')
    (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    grid = validation_grid(cfg.objects.families, args.approaches, 1 if args.pipeline_check else args.val_per_cell)
    cb = V4TrainCallback(out, sac_cfg, eval_env, grid, gate_at=args.gate_at, n_demo=n_demo)
    keep_awake(True)
    t0 = time.time()
    error = None
    try:
        model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    except BaseException as exc:          # keep evidence: save the model and the traceback
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
