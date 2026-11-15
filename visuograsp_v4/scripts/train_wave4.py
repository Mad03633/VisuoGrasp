"""Wave 4 training: (A+C) curriculum SAC or (B+E) BC-pretrained, BC-regularised SAC. Writes to outputs/rl/wave4/<run>/.

  --method curriculum  SAC from scratch on CurriculumGraspEnv: stages 1-4 (wave4.STAGES), advanced
                       when the deterministic GT success on the stage's development episodes reaches
                       --advance-threshold; earlier stages are rehearsed with probability 0.25.
  --method bcsac       actor initialised from the BC/DAgger actor (wave4_bc_pretrain.py); BCSAC on the
                       full task with demo-buffer mixing, decaying BC regulariser, Q-normalised actor
                       objective, critic warm-up (actor frozen), lower actor learning rate.
Both: SB3 2.9.2 SAC with LayerNorm critics, n-step 10, gamma 0.99, batch 256, buffer 1M; reward v4r1;
actor observation = V4 130-D PRIVILEGED observation. Every --eval-every steps: deterministic GT success
on the full task (original V4 initial conditions; development seeds 8_700_000+, 40 episodes) and, for
the curriculum, on the current stage. best_model.zip = best full-task development GT. Decision gates
(--gates) make the run wait for gate_decision.json (pilot -> longer run without restarting).

    python -m visuograsp_v4.scripts.train_wave4 --method curriculum --seed 11 --total-steps 300000 --gates 50000 150000
    python -m visuograsp_v4.scripts.train_wave4 --method bcsac --seed 11 --total-steps 300000 --gates 50000 150000
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
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

from visuograsp_v4.agents import GTEpisodeInfo, V4TrainCallback, make_sac
from visuograsp_v4.config import OUTPUT_ROOT, SACConfig, V4Config
from visuograsp_v4.env import VisuoGraspV4Env
from visuograsp_v4.scripts.train_sac import code_fingerprint, keep_awake
from visuograsp_v4.wave4 import (ALL_FAMILIES, STAGES, BCSAC, CurriculumGraspEnv, dev_grid, evaluate_grid,
                                 set_actor_log_std)

W4 = OUTPUT_ROOT / 'rl' / 'wave4'


class Wave4EvalCallback(BaseCallback):
    def __init__(self, out, method, eval_every, full_env, full_grid, stage_env=None, stage_grids=None,
                 advance_threshold=0.5, stage=4, stage_setter=None, fixed_stage=False, full_eval=True, select_on='full',
                 verbose=1):
        super().__init__(verbose)
        self.out, self.method, self.eval_every = out, method, eval_every
        self.full_env, self.full_grid = full_env, full_grid
        self.stage_env, self.stage_grids = stage_env, stage_grids or {}
        self.advance_threshold, self.stage = advance_threshold, stage
        self.stage_setter = stage_setter          # direct reference to the training CurriculumGraspEnv.set_stage
        self.fixed_stage, self.full_eval, self.select_on = fixed_stage, full_eval, select_on
        self.next_eval = 0
        self.best_key = None
        self.t0 = time.time()

    def _append(self, name, df):
        path = self.out / name
        df.to_csv(path, mode='a', header=not path.exists(), index=False)

    def _evaluate(self, step):
        act = lambda o: self.model.predict(o, deterministic=True)[0]
        row = {'step': step, 'stage': self.stage, 'hours': (time.time() - self.t0) / 3600}
        advanced = False
        if self.method == 'curriculum' and (self.stage < 4 or self.fixed_stage):
            sd = evaluate_grid(self.stage_env, act, self.stage_grids[self.stage], stage=self.stage)
            self._append('stage_eval_episodes.csv', sd.assign(step=step, eval_stage=self.stage))
            row.update(stage_gt=float(sd.gt_success.mean()), stage_partial=float((sd.max_lift_cm >= 3.5).mean()),
                       stage_episodes=len(sd))
            if not self.fixed_stage and row['stage_gt'] >= self.advance_threshold:
                self.stage += 1
                self.stage_setter(self.stage)
                advanced = True
        if self.full_eval:
            fd = evaluate_grid(self.full_env, act, self.full_grid)
            self._append('full_eval_episodes.csv', fd.assign(step=step))
            row.update(full_gt=float(fd.gt_success.mean()), full_gt_top=float(fd[fd.approach == 'top'].gt_success.mean()),
                       full_gt_oblique=float(fd[fd.approach == 'oblique'].gt_success.mean()),
                       full_partial=float((fd.max_lift_cm >= 3.5).mean()), full_contact=float(fd.contact_occurred.mean()),
                       full_pushed=float(fd.pushed_before_close.mean()), full_episodes=len(fd))
            key = (row['full_gt'], row['full_partial'], row['stage'], row.get('stage_gt', 0.0))
            if self.select_on == 'stage':
                key = (row['stage_gt'], row['stage_partial'])
        else:
            row.update(full_gt=float('nan'), full_gt_top=float('nan'), full_gt_oblique=float('nan'), full_partial=float('nan'))
            key = (row['stage_gt'], row['stage_partial'])
        row.update(advanced=int(advanced), new_stage=self.stage)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.model.save(str(self.out / 'best_model.zip'))
            order = (['stage_dev_gt', 'stage_partial_lift'] if (self.select_on == 'stage' or not self.full_eval)
                     else ['full_task_dev_gt', 'full_partial_lift', 'stage', 'stage_gt'])
            (self.out / 'best_model.json').write_text(json.dumps({'step': step, 'key': key, 'key_order': order},
                                                                 indent=1), encoding='utf-8')
        row['best_step'] = json.loads((self.out / 'best_model.json').read_text())['step']
        self._append('wave4_eval.csv', pd.DataFrame([row]))
        if self.verbose:
            st = f' stage{row["stage"]} GT {row.get("stage_gt", float("nan")):.3f}' if self.method == 'curriculum' else ''
            print(f'[EVAL] step {step:,}:{st} | full task GT {row["full_gt"]:.3f} (top {row["full_gt_top"]:.3f}, '
                  f'oblique {row["full_gt_oblique"]:.3f}) partial {row["full_partial"]:.2f}'
                  + (f' -> ADVANCE to stage {self.stage}' if advanced else ''), flush=True)

    def _on_training_start(self):
        self._evaluate(0)
        self.next_eval = self.eval_every

    def _on_step(self):
        step = int(self.num_timesteps)
        if step >= self.next_eval:
            self.next_eval += self.eval_every
            self._evaluate(step)
        return True


def build_demo_buffer(model, data) -> NStepReplayBuffer:
    n = len(data['obs'])
    buf = NStepReplayBuffer(n + 10, model.observation_space, model.action_space, device=model.device, n_envs=1,
                            handle_timeout_termination=True, n_steps=SACConfig().n_steps, gamma=model.gamma)
    for i in range(n):
        buf.add(data['obs'][i][None], data['next_obs'][i][None], data['act'][i][None], np.array([data['rew'][i]], np.float32),
                np.array([bool(data['done'][i])]), [{'TimeLimit.truncated': bool(data['timeout'][i])}])
    return buf


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=('curriculum', 'bcsac'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--total-steps', type=int, required=True)
    ap.add_argument('--gates', type=int, nargs='*', default=[])
    ap.add_argument('--eval-every', type=int, default=10_000)
    ap.add_argument('--dev-per-cell', type=int, default=4)
    ap.add_argument('--stage-dev-per-cell', type=int, default=5)
    ap.add_argument('--advance-threshold', type=float, default=0.5)
    ap.add_argument('--start-stage', type=int, default=1)
    ap.add_argument('--fixed-stage', type=int, default=None, help='E3: train and select on one stage, no advancement')
    ap.add_argument('--bcsac-stage', type=int, default=None, help='E4: BC-SAC trained on a fixed curriculum stage')
    ap.add_argument('--learning-starts', type=int, default=5_000)
    ap.add_argument('--bc-dir', default='bc_pretrain')
    ap.add_argument('--init-std', type=float, default=0.1)
    ap.add_argument('--demo-frac', type=float, default=0.25)
    ap.add_argument('--lambda0', type=float, default=1.0)
    ap.add_argument('--lambda-min', type=float, default=0.1)
    ap.add_argument('--decay-updates', type=int, default=100_000)
    ap.add_argument('--actor-delay', type=int, default=10_000)
    ap.add_argument('--actor-lr', type=float, default=1e-4)
    ap.add_argument('--beta', type=float, default=2.5)
    ap.add_argument('--ent-init', type=float, default=0.001)
    ap.add_argument('--prior0', type=float, default=0.0, help='E2b: weight of the anchor to the frozen BC actor')
    ap.add_argument('--prior-min', type=float, default=0.0)
    ap.add_argument('--prior-decay-updates', type=int, default=200_000)
    ap.add_argument('--q-clip', type=float, nargs=2, default=None, help='E2b: clamp of the target Q (lo hi)')
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--pipeline-check', action='store_true')
    args = ap.parse_args()
    out = W4 / args.run_name
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} exists and is not empty (nothing is overwritten)')
    out.mkdir(parents=True, exist_ok=True)
    cfg = V4Config()
    if args.fixed_stage is not None:
        args.start_stage = args.fixed_stage
    sac_cfg = replace(SACConfig(), learning_starts=args.learning_starts, torch_threads=args.threads,
                      eval_every=10 ** 12, checkpoint_every=25_000)
    if args.method == 'bcsac':
        sac_cfg = replace(sac_cfg, ent_coef=f'auto_{args.ent_init}')
    if args.pipeline_check:
        sac_cfg = replace(sac_cfg, learning_starts=300, checkpoint_every=1000, q_health_every=1000)
    meta = {'started': datetime.now().isoformat(timespec='seconds'), 'method': args.method, 'args': vars(args),
            'privileged': 'actor observation = V4 130-D observation with simulator object pose and GT-oracle contacts',
            'v4_config': cfg.to_dict(), 'reward': 'v4r1 (unchanged)', 'code_sha256_16': code_fingerprint(),
            'versions': {'python': platform.python_version(), 'mujoco': mujoco.__version__,
                         'sb3': stable_baselines3.__version__, 'torch': torch.__version__},
            'seeds': {'sb3_seed': args.seed, 'dev_selection': 8_700_000, 'bc_expert': 3_400_000, 'dagger': 3_500_000,
                      'final_test_reserved': 6_600_000, 'not_used': ['5_100_000+', '6_100_000+']}, 'pid': os.getpid()}
    full_env = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    full_grid = dev_grid(ALL_FAMILIES, per_cell=1 if args.pipeline_check else args.dev_per_cell)
    stage_env = stage_grids = cur_env = None
    if args.method == 'curriculum':
        cur_env = CurriculumGraspEnv(cfg, stage=args.start_stage,
                                     replay_prev_prob=0.0 if args.fixed_stage is not None else 0.25)
        train_env = GTEpisodeInfo(cur_env)
        stage_env = CurriculumGraspEnv(cfg, stage=args.start_stage, replay_prev_prob=0.0)
        stage_grids = {k: dev_grid(STAGES[k]['families'], per_cell=1 if args.pipeline_check else args.stage_dev_per_cell)
                       for k in (1, 2, 3, 5)}
        model = make_sac(train_env, sac_cfg, args.seed)
        meta['curriculum'] = {'stages': {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                                         for k, v in STAGES.items()},
                              'advance_threshold': args.advance_threshold, 'rehearsal_prob': 0.25, 'start_stage': args.start_stage}
        meta['sac_config'] = asdict(sac_cfg)
    else:
        bc = W4 / args.bc_dir
        if args.bcsac_stage is not None:          # E4: fine-tune on hand-over states of one stage
            cur_env = CurriculumGraspEnv(cfg, stage=args.bcsac_stage, replay_prev_prob=0.0)
            train_env = GTEpisodeInfo(cur_env)
            stage_env = CurriculumGraspEnv(cfg, stage=args.bcsac_stage, replay_prev_prob=0.0)
            stage_grids = {args.bcsac_stage: dev_grid(STAGES[args.bcsac_stage]['families'],
                                                      per_cell=1 if args.pipeline_check else args.stage_dev_per_cell)}
        else:
            train_env = GTEpisodeInfo(VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique')))
        model = make_sac(train_env, sac_cfg, args.seed, algo_cls=BCSAC)
        model.actor.load_state_dict(torch.load(bc / 'bc_actor_best.pt'))
        set_actor_log_std(model.actor, args.init_std)
        ex = dict(np.load(bc / 'expert_data.npz'))
        bcs = dict(np.load(bc / 'bc_training_set.npz'))
        demo = build_demo_buffer(model, ex)
        with torch.no_grad():
            idx = np.random.default_rng(0).integers(0, len(bcs['obs']), 4096)
            _, lp = model.actor.action_log_prob(torch.as_tensor(bcs['obs'][idx]))
            target_entropy = float(-lp.mean())
        model.target_entropy = target_entropy
        prior = None
        if args.prior0 > 0:
            import copy
            prior = copy.deepcopy(model.actor)
        model.setup_bc(bcs['obs'], bcs['act'], demo_buffer=demo, demo_frac=args.demo_frac, lambda0=args.lambda0,
                       lambda_min=args.lambda_min, decay_updates=args.decay_updates,
                       actor_delay=300 if args.pipeline_check else args.actor_delay, beta=args.beta, actor_lr=args.actor_lr,
                       prior_actor=prior, prior0=args.prior0, prior_min=args.prior_min,
                       prior_decay_updates=args.prior_decay_updates, q_clip=tuple(args.q_clip) if args.q_clip else None)
        meta['bcsac'] = {'bc_dir': str(bc), 'bc_summary': json.loads((bc / 'summary.json').read_text()) if (bc / 'summary.json').exists() else None,
                         'init_std': args.init_std, 'target_entropy_measured': target_entropy, 'ent_coef_init': args.ent_init,
                         'demo_transitions': int(len(ex['obs'])), 'bc_pairs': int(len(bcs['obs'])), 'demo_frac': args.demo_frac,
                         'lambda0': args.lambda0, 'lambda_min': args.lambda_min, 'decay_updates': args.decay_updates,
                         'actor_delay': args.actor_delay, 'beta_qnorm': args.beta, 'actor_lr': args.actor_lr,
                         'warmup': 'policy sampling for learning_starts steps', 'prior0': args.prior0,
                         'prior_min': args.prior_min, 'prior_decay_updates': args.prior_decay_updates, 'q_clip': args.q_clip}
        meta['sac_config'] = asdict(sac_cfg)
    model.set_logger(configure(str(out / 'sb3'), ['csv']))
    (out / 'run_config.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
    v4cb = V4TrainCallback(out, sac_cfg, full_env, [], gate_at=args.gates or None)
    e4 = args.method == 'bcsac' and args.bcsac_stage is not None
    w4cb = Wave4EvalCallback(out, 'curriculum' if e4 else args.method, 1000 if args.pipeline_check else args.eval_every,
                             full_env, full_grid, stage_env, stage_grids, args.advance_threshold,
                             stage=args.bcsac_stage if e4 else (args.start_stage if args.method == 'curriculum' else 4),
                             stage_setter=cur_env.set_stage if cur_env is not None else None,
                             fixed_stage=args.fixed_stage is not None or e4, full_eval=args.fixed_stage is None,
                             select_on='stage' if (args.fixed_stage is not None or e4) else 'full')
    keep_awake(True)
    t0 = time.time()
    error = None
    try:
        model.learn(total_timesteps=args.total_steps, callback=CallbackList([w4cb, v4cb]), progress_bar=False)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        (out / 'crash_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        try:
            model.save(str(out / 'crash_model.zip'))
        except Exception:
            pass
    finally:
        keep_awake(False)
        v4cb._on_training_end()
    secs = time.time() - t0
    if error is None:
        model.save(str(out / 'last_model.zip'))
    prefix = cur_env.prefix_steps_total if cur_env is not None else 0
    (out / 'finished.json').write_text(json.dumps({
        'steps': int(model.num_timesteps), 'requested_steps': args.total_steps, 'scripted_prefix_steps_not_learned': int(prefix),
        'seconds_wall': secs, 'seconds_training_excl_waits': v4cb.training_seconds(), 'seconds_waiting': v4cb.wait_seconds,
        'stop_reason': v4cb.stop_reason or ('error' if error else 'completed'), 'error': error,
        'final_stage': w4cb.stage, 'best_key': w4cb.best_key, 'finished': datetime.now().isoformat(timespec='seconds')},
        indent=1, default=str), encoding='utf-8')
    print(f'[DONE] {args.run_name}: {model.num_timesteps:,} steps in {secs / 60:.1f} min, '
          f'stop={v4cb.stop_reason or error or "completed"}, final stage {w4cb.stage}', flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
