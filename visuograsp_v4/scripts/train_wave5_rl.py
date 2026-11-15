"""Wave 5 / E6: AWAC-style, BC-anchored RL fine-tuning of a BC actor on the full V4 task (PRIVILEGED setting).

See visuograsp_v4/wave5.py for the objective. Evaluation every --eval-every steps: deterministic full-task GT
on DEV_A (seeds 8_800_000+, --dev-per-cell per family x approach) next to the frozen BC initialisation on the
same episodes (computed once at step 0), plus the drift ||tanh(mu) - tanh(mu_BC)||^2 on fixed demo states.
Automatic stop: two consecutive evaluations after the actor is unfrozen with GT < BC GT - --stop-drop.
best_model.zip = best DEV_A GT (ties -> earlier). Writes outputs/rl/wave5/rl/<run>/.

    python -m visuograsp_v4.scripts.train_wave5_rl --run e6_awac_s11 --init-actor wave5/bc/<run>/actor_best.pt --total-steps 150000 --gates 50000
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime

import mujoco
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure

from visuograsp_v4.agents import GTEpisodeInfo, V4TrainCallback, make_sac
from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config
from visuograsp_v4.env import VisuoGraspV4Env
from visuograsp_v4.scripts.train_sac import code_fingerprint, keep_awake
from visuograsp_v4.scripts.train_wave4 import Wave4EvalCallback, build_demo_buffer
from visuograsp_v4.wave4 import ALL_FAMILIES, actor_mean_action, dev_grid
from visuograsp_v4.wave5 import AWBCSAC, MaskPrevAction, mask_history, mask_prev_action

RL = OUTPUT_ROOT / 'rl'
W5 = RL / 'wave5'
DEV_A = 8_800_000


class E6Eval(Wave4EvalCallback):
    """Adds the BC reference, drift logging and the automatic degradation stop."""

    def __init__(self, *a, bc_ref_gt=None, drift_obs=None, ref_actor=None, stop_drop=0.08, actor_on_step=0, **kw):
        super().__init__(*a, **kw)
        self.bc_ref_gt, self.drift_obs, self.ref_actor = bc_ref_gt, drift_obs, ref_actor
        self.stop_drop, self.actor_on_step, self.bad, self.stop_reason = stop_drop, actor_on_step, 0, None

    def _evaluate(self, step):
        super()._evaluate(step)
        ev = pd.read_csv(self.out / 'wave4_eval.csv')
        gt = float(ev.full_gt.iloc[-1])
        with torch.no_grad():
            drift = float(((actor_mean_action(self.model.actor, self.drift_obs) - actor_mean_action(self.ref_actor, self.drift_obs)) ** 2).mean())
        self._append('e6_monitor.csv', pd.DataFrame([{'step': step, 'dev_gt': gt, 'bc_ref_gt': self.bc_ref_gt, 'drift_mse_to_bc': drift}]))
        if step > self.actor_on_step and self.bc_ref_gt is not None:
            self.bad = self.bad + 1 if gt < self.bc_ref_gt - self.stop_drop else 0
            if self.bad >= 2:
                self.stop_reason = f'degradation: dev GT {gt:.3f} < BC {self.bc_ref_gt:.3f} - {self.stop_drop} twice'
                print('[STOP]', self.stop_reason, flush=True)

    def _on_step(self):
        ok = super()._on_step()
        return ok and self.stop_reason is None


def load_demos(names):
    parts = [dict(np.load(W5 / 'demos' / n / 'data.npz')) for n in names]
    eps = [pd.read_csv(W5 / 'demos' / n / 'episodes.csv') for n in names]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ('obs', 'next_obs', 'act', 'rew', 'done', 'timeout', 'label', 'episode')}
    succ = np.concatenate([np.array([e.set_index('episode').gt_success[int(i)] for i in p['episode']]) for p, e in zip(parts, eps)])
    return data, succ


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--init-actor', required=True, help='actor state dict relative to outputs/rl')
    ap.add_argument('--net', type=int, nargs='+', default=[256, 256])
    ap.add_argument('--demos', nargs='+', default=['demos_clean', 'demos_dart'])
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--total-steps', type=int, required=True)
    ap.add_argument('--gates', type=int, nargs='*', default=[])
    ap.add_argument('--sigma', type=float, default=0.05)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--lambda-bc', type=float, default=0.5)
    ap.add_argument('--demo-frac', type=float, default=0.25)
    ap.add_argument('--actor-delay', type=int, default=20_000)
    ap.add_argument('--actor-lr', type=float, default=1e-4)
    ap.add_argument('--q-clip', type=float, nargs=2, default=[-25.0, 35.0])
    ap.add_argument('--learning-starts', type=int, default=5_000)
    ap.add_argument('--eval-every', type=int, default=10_000)
    ap.add_argument('--dev-per-cell', type=int, default=6)
    ap.add_argument('--stop-drop', type=float, default=0.08)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--pipeline-check', action='store_true')
    ap.add_argument('--mask-prev-action', action='store_true', help='actor sees prev_action zeroed (must match the BC init)')
    ap.add_argument('--mask-history', action='store_true', help='actor sees prev_action and velocities zeroed (must match the BC init)')
    args = ap.parse_args()
    out = W5 / 'rl' / args.run
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    cfg = V4Config()
    sac_cfg = replace(SACConfig(), net_arch=tuple(args.net), ent_coef=0.0, learning_starts=args.learning_starts,
                      torch_threads=args.threads, eval_every=10 ** 12, checkpoint_every=25_000)
    if args.pipeline_check:
        sac_cfg = replace(sac_cfg, learning_starts=300, checkpoint_every=1000, q_health_every=1000)
    mode = 'history' if args.mask_history else ('prev' if args.mask_prev_action else None)
    wrap = (lambda e: MaskPrevAction(e, mode)) if mode else (lambda e: e)
    env = GTEpisodeInfo(wrap(VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))))
    full_env = wrap(VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique')))
    model = make_sac(env, sac_cfg, args.seed, algo_cls=AWBCSAC)
    model.actor.load_state_dict(torch.load(RL / args.init_actor))
    ref_actor = copy.deepcopy(model.actor)
    for p in ref_actor.parameters():
        p.requires_grad_(False)
    data, succ = load_demos(args.demos)
    if mode:
        fn = mask_history if mode == 'history' else mask_prev_action
        data['obs'], data['next_obs'] = fn(data['obs']), fn(data['next_obs'])
    demo = build_demo_buffer(model, data)
    model.setup_awr(tau=args.tau, lambda_bc=args.lambda_bc, fixed_log_std=float(np.log(args.sigma)),
                    bc_obs=data['obs'][succ], bc_act=data['label'][succ], demo_buffer=demo, demo_frac=args.demo_frac,
                    actor_delay=300 if args.pipeline_check else args.actor_delay, actor_lr=args.actor_lr, q_clip=tuple(args.q_clip),
                    beta=None, lambda0=0.0, lambda_min=0.0)
    grid = dev_grid(ALL_FAMILIES, per_cell=1 if args.pipeline_check else args.dev_per_cell, base=DEV_A)
    from visuograsp_v4.wave4 import evaluate_grid
    with torch.no_grad():
        bc_dev = evaluate_grid(full_env, lambda o: actor_mean_action(ref_actor, torch.as_tensor(o[None])).numpy()[0], grid)
    bc_dev.to_csv(out / 'bc_reference_dev.csv', index=False)
    bc_gt = float(bc_dev.gt_success.mean())
    drift_obs = torch.as_tensor(data['obs'][succ][np.random.default_rng(0).integers(0, int(succ.sum()), 4096)])
    meta = {'started': datetime.now().isoformat(timespec='seconds'), 'method': 'E6 AWAC-style BC-anchored fine-tuning (wave5.AWBCSAC)',
            'args': vars(args), 'bc_reference_dev_gt': bc_gt, 'dev_seed_base': DEV_A, 'dev_episodes': len(grid),
            'demo_transitions': int(len(data['obs'])), 'bc_pairs': int(succ.sum()), 'sac_config': asdict(sac_cfg),
            'privileged': 'actor observation = V4 130-D observation with simulator object pose and GT-oracle contacts',
            'reward': 'v4r1 (unchanged)', 'v4_config': cfg.to_dict(), 'code_sha256_16': code_fingerprint(),
            'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__, 'sb3': stable_baselines3.__version__,
                         'torch': torch.__version__}, 'pid': os.getpid()}
    model.set_logger(configure(str(out / 'sb3'), ['csv']))
    (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    print(f'[INIT] BC reference dev GT {bc_gt:.3f} on {len(grid)} episodes', flush=True)
    v4cb = V4TrainCallback(out, sac_cfg, full_env, [], gate_at=args.gates or None)
    actor_on = args.learning_starts + (300 if args.pipeline_check else args.actor_delay)
    e6 = E6Eval(out, 'bcsac', 1000 if args.pipeline_check else args.eval_every, full_env, grid,
                bc_ref_gt=bc_gt, drift_obs=drift_obs, ref_actor=ref_actor, stop_drop=args.stop_drop, actor_on_step=actor_on)
    keep_awake(True)
    t0, error = time.time(), None
    try:
        model.learn(total_timesteps=args.total_steps, callback=CallbackList([e6, v4cb]), progress_bar=False)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        (out / 'crash_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        keep_awake(False)
        v4cb._on_training_end()
    if error is None:
        model.save(str(out / 'last_model.zip'))
    (out / 'finished.json').write_text(json.dumps({
        'steps': int(model.num_timesteps), 'seconds_wall': time.time() - t0, 'stop_reason': e6.stop_reason or v4cb.stop_reason
        or ('error' if error else 'completed'), 'error': error, 'bc_reference_dev_gt': bc_gt, 'best_key': e6.best_key,
        'finished': datetime.now().isoformat(timespec='seconds')}, indent=1, default=str), encoding='utf-8')
    print(f'[DONE] {args.run}: {model.num_timesteps:,} steps, stop={e6.stop_reason or v4cb.stop_reason or error or "completed"}', flush=True)


if __name__ == '__main__':
    main()
