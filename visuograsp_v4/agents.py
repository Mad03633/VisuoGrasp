"""SAC for VisuoGrasp V4 (Stable-Baselines3 2.9.2), without notebook monkey-patching.

* LayerNormSACPolicy: LayerNorm after every hidden Linear layer of both critics (and their
  targets) -- the V3.3 stabiliser, here as an importable subclass so checkpoints load cleanly.
* n-step returns (n = 10) through SB3's NStepReplayBuffer (``n_steps`` argument).
* Optional scripted-demonstration prefill of the replay buffer (learning from demonstrations).
  The scripted controller uses PRIVILEGED object pose; any arm that uses it must say so.
  ``verify_demo_transitions`` replays every stored demo episode in a fresh environment and checks
  observation, action, reward and done flags against the buffer (same action semantics and the
  same reward definition as the training environment); ``verify_nstep`` recomputes n-step returns.
* GTEpisodeInfo: logging-only wrapper that adds the ground-truth episode outcome to ``info``.
* V4TrainCallback: fixed validation episodes (model selection on GT success, train seeds
  disjoint from validation and test seeds), periodic checkpoints, soft-Q health log with a
  divergence stop, episode log with failure reasons, an optional decision gate (the run waits at
  a given step until ``gate_decision.json`` says continue/stop) and file-based pause/stop requests
  (used by the system monitor on thermal throttling).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.sac.policies import SACPolicy

from .config import SACConfig
from .evaluation import _sim_failure, classify, run_policy_episode
from .scripted import ScriptedGrasp

VALIDATION_SEED_BASE = 8_000_000    # model selection only; test grid uses 5_100_000+
# Discounted returns of reward v4r1 are bounded by about 30 (shaping telescopes to <= 8, hold
# reward <= 0.1 / (1 - 0.99) = 10, success bonus 10); |Q| far above that means divergence.
Q_SOFT_LIMIT = 50.0
Q_HARD_LIMIT = 200.0


class LayerNormSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        critic = super().make_critic(features_extractor)
        for idx, q_net in enumerate(list(critic.q_networks)):
            layers = []
            for module in q_net:
                layers.append(module)
                if isinstance(module, nn.Linear) and module.out_features != 1:
                    layers.append(nn.LayerNorm(module.out_features))
            new_net = nn.Sequential(*layers)
            critic.add_module(f'qf{idx}', new_net)
            critic.q_networks[idx] = new_net
        return critic


class ZeroCenteredWarmupSAC(SAC):
    """Wave 3: during the warm-up (num_timesteps < learning_starts) actions are sampled from the
    current (zero-mean, small-std) policy instead of uniformly from the whole action space.
    After the warm-up it is identical to SB3 SAC."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
            scaled_action = self.policy.scale_action(unscaled_action)
            return self.policy.unscale_action(scaled_action), scaled_action
        return super()._sample_action(learning_starts, action_noise, n_envs)


def make_sac(env, cfg: SACConfig, seed: int, algo_cls=SAC, **overrides) -> SAC:
    torch.set_num_threads(cfg.torch_threads)
    policy = LayerNormSACPolicy if cfg.layernorm_critic else 'MlpPolicy'
    return algo_cls(policy, env, learning_rate=cfg.learning_rate, buffer_size=cfg.buffer_size,
                    batch_size=cfg.batch_size, learning_starts=cfg.learning_starts, train_freq=cfg.train_freq,
                    gradient_steps=cfg.gradient_steps, gamma=cfg.gamma, tau=cfg.tau, ent_coef=cfg.ent_coef,
                    n_steps=cfg.n_steps, policy_kwargs={'net_arch': list(cfg.net_arch)}, device=cfg.device,
                    seed=seed, verbose=0, **overrides)


def load_sac(path, device='cpu') -> SAC:
    model = SAC.load(str(path), device=device, custom_objects={'policy_class': LayerNormSACPolicy})
    model.replay_buffer = None
    return model


# ---------------------------------------------------------------------- episode summaries
def episode_summary(sim, pushed: bool) -> dict:
    """Ground-truth outcome of the episode that just ended (evaluation / logging only)."""
    t = sim.tracker
    row = dict(sim_failure=_sim_failure(t), gt_success=bool(t.gt_success), gt_any=bool(t.gt_success_any),
               max_lift_cm=100 * t.max_lift, final_lift_cm=100 * t.final_lift, object_off_table=bool(t.off_table),
               dropped=bool(t.dropped), height_hold_frames=int(t.height_hold_max), hold_frames_gt=int(t.gt_hold_max),
               pushed_before_close=bool(pushed), contact_occurred=bool(t.any_pad_contact),
               thumb_contact=bool(t.thumb_contact), max_pads=int(t.max_pads_in_contact))
    row['outcome'] = classify(row)
    return row


class GTEpisodeInfo(gym.Wrapper):
    """Logging only: adds ``info['gt_episode']`` (ground-truth outcome) at the last step of an
    episode. Observation, action, reward and termination are passed through unchanged."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._start = self.env.unwrapped.sim.object_com().copy()
        self._pushed = False
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        sim = self.env.unwrapped.sim
        if sim.object_lift() < 0.01 and float(np.linalg.norm(sim.object_com()[:2] - self._start[:2])) > 0.02:
            self._pushed = True
        if term or trunc:
            info['gt_episode'] = episode_summary(sim, self._pushed)
        return obs, r, term, trunc, info


# ---------------------------------------------------------------------- demonstrations
def prefill_with_scripted(model: SAC, env, n_transitions: int, seed_base: int, approaches) -> dict:
    """Add scripted-controller transitions to the replay buffer (PRIVILEGED planner).

    Uses a separate env instance with the training configuration; seeds seed_base+k are
    training seeds, disjoint from validation (8_000_000+) and test (5_100_000+) seeds.
    Returns per-episode buffer index ranges so that every transition can be verified.
    """
    added = episodes = successes = 0
    k = 0
    records = []
    buf = model.replay_buffer
    while added < n_transitions:
        approach = approaches[k % len(approaches)]
        seed = seed_base + k
        obs, _ = env.reset(seed=seed, options={'approach': approach})
        agent = ScriptedGrasp(env.sim, env.approach)
        start, ret, n = buf.pos, 0.0, 0
        while True:
            a = np.clip(agent.act(), -0.999, 0.999).astype(np.float32)
            nxt, r, term, trunc, info = env.step(a)
            buf.add(obs[None], nxt[None], a[None], np.array([r], np.float32),
                    np.array([term or trunc]), [{'TimeLimit.truncated': bool(trunc and not term)}])
            obs = nxt
            added += 1
            n += 1
            ret += r
            if term or trunc:
                break
        episodes += 1
        successes += int(env.sim.tracker.gt_success)
        records.append({'seed': seed, 'approach': approach, 'start': int(start), 'end': int(buf.pos), 'length': n,
                        'return': ret, 'gt_success': bool(env.sim.tracker.gt_success),
                        'terminated': bool(term), 'truncated': bool(trunc),
                        'outcome': episode_summary(env.sim, False)['outcome']})
        k += 1
    return {'transitions': added, 'episodes': episodes, 'scripted_gt_success': successes / max(1, episodes),
            'episode_records': records}


def verify_demo_transitions(model: SAC, make_env, train_env, records) -> dict:
    """Replay every stored demonstration episode in a fresh env and compare with the buffer.

    Checks: identical env configuration / reward version / action space as the training env;
    SAC's action scaling is the identity on [-1, 1] (stored action = executed action);
    observations, rewards, done and timeout flags of every stored transition are reproduced by
    stepping the current V4 environment with the stored actions.
    """
    buf = model.replay_buffer
    env = make_env()
    te = train_env.unwrapped
    acts = buf.actions[:records[-1]['end'], 0]
    res = {
        'reward_version_demo_env': env.rw.version, 'reward_version_train_env': te.rw.version,
        'env_config_equal': env.cfg == te.cfg,
        'obs_mode_equal': env.obs_mode == te.obs_mode, 'contact_source_equal': env.contact_source == te.contact_source,
        'approaches_equal': env.approaches == te.approaches,
        'action_space_equal': bool(np.array_equal(env.action_space.low, te.action_space.low)
                                   and np.array_equal(env.action_space.high, te.action_space.high)),
        'action_space_low_high': [float(te.action_space.low.min()), float(te.action_space.high.max())],
        'sac_scale_action_identity': bool(np.allclose(model.policy.scale_action(acts), acts, atol=1e-7)),
        'demo_action_min': float(acts.min()), 'demo_action_max': float(acts.max()),
        'buffer_gamma': float(buf.gamma), 'reward_gamma': float(te.rw.gamma), 'buffer_n_steps': int(buf.n_steps),
    }
    max_obs = max_next = max_rew = 0.0
    done_mismatch = timeout_mismatch = 0
    episodes = transitions = 0
    for rec in records:
        obs, _ = env.reset(seed=rec['seed'], options={'approach': rec['approach']})
        for i in range(rec['start'], rec['end']):
            max_obs = max(max_obs, float(np.abs(obs - buf.observations[i, 0]).max()))
            nxt, r, term, trunc, info = env.step(buf.actions[i, 0].copy())
            max_next = max(max_next, float(np.abs(nxt - buf.next_observations[i, 0]).max()))
            max_rew = max(max_rew, abs(float(np.float32(r)) - float(buf.rewards[i, 0])))
            done_mismatch += int(bool(term or trunc) != bool(buf.dones[i, 0]))
            timeout_mismatch += int(bool(trunc and not term) != bool(buf.timeouts[i, 0]))
            if i == rec['start']:
                res.setdefault('reward_terms_keys', sorted(info['reward_terms']))
            obs = nxt
            transitions += 1
        episodes += 1
    res.update(replayed_episodes=episodes, replayed_transitions=transitions, max_abs_obs_diff=max_obs,
               max_abs_next_obs_diff=max_next, max_abs_reward_diff=max_rew, done_mismatches=done_mismatch,
               timeout_mismatches=timeout_mismatch)
    env.close()
    res['passed'] = bool(res['env_config_equal'] and res['obs_mode_equal'] and res['contact_source_equal']
                         and res['approaches_equal'] and res['action_space_equal'] and res['sac_scale_action_identity']
                         and res['reward_version_demo_env'] == res['reward_version_train_env'] == 'v4r1'
                         and res['demo_action_min'] >= -1.0 and res['demo_action_max'] <= 1.0
                         and res['buffer_gamma'] == res['reward_gamma']
                         and max_obs <= 1e-5 and max_next <= 1e-5 and max_rew <= 1e-5
                         and done_mismatch == 0 and timeout_mismatch == 0 and transitions == records[-1]['end'])
    return res


def verify_nstep(buf, n_samples: int = 2000, seed: int = 0) -> dict:
    """Recompute n-step targets by hand for random buffer indices and compare with SB3."""
    size = buf.buffer_size if buf.full else buf.pos
    rng = np.random.default_rng(seed)
    inds = rng.integers(0, size, n_samples)
    smp = buf._get_samples(inds)
    g, n = buf.gamma, buf.n_steps
    max_r = max_disc = max_obs = 0.0
    done_mismatch = 0
    for b, i in enumerate(inds):
        ret, k = 0.0, 0
        for k in range(n):
            j = (i + k) % buf.buffer_size
            ret += g ** k * float(buf.rewards[j, 0])
            if buf.dones[j, 0] or buf.timeouts[j, 0] or j == buf.pos - 1:
                break
        final_done = float(buf.dones[j, 0] and not buf.timeouts[j, 0])
        max_r = max(max_r, abs(ret - float(smp.rewards[b, 0])))
        max_disc = max(max_disc, abs(g ** (k + 1) - float(smp.discounts[b, 0])))
        max_obs = max(max_obs, float(np.abs(buf.next_observations[j, 0] - smp.next_observations[b].cpu().numpy()).max()))
        done_mismatch += int(final_done != float(smp.dones[b, 0]))
    return {'samples': int(n_samples), 'buffer_size_used': int(size), 'n_steps': int(n), 'gamma': float(g),
            'max_abs_return_diff': max_r, 'max_abs_discount_diff': max_disc, 'max_abs_next_obs_diff': max_obs,
            'done_mismatches': done_mismatch,
            'passed': bool(max_r < 1e-4 and max_disc < 1e-6 and max_obs < 1e-6 and done_mismatch == 0)}


# ---------------------------------------------------------------------- validation
def validation_grid(families, approaches, per_cell: int, seed_base: int = VALIDATION_SEED_BASE) -> list[tuple]:
    grid = []
    for fi, fam in enumerate(families):
        for ai, ap in enumerate(approaches):
            for ep in range(per_cell):
                grid.append((seed_base + fi * 1000 + ai * 100 + ep, fam, ap))
    return grid


def evaluate_on_grid(model: SAC, env, grid) -> pd.DataFrame:
    rows = []
    for seed, fam, ap in grid:
        rng = np.random.default_rng(seed)              # variant / resting mode drawn from the seed
        from . import objects as objs
        variant = str(rng.choice(list(objs.VARIANTS[fam])))
        mode = str(rng.choice(list(objs.RESTING_MODES[fam])))
        act = lambda o: model.predict(o, deterministic=True)[0]
        rows.append(run_policy_episode(env, act, seed, fam, variant, mode, ap))
    return pd.DataFrame(rows)


def selection_key(df: pd.DataFrame) -> tuple:
    """Model-selection score (fixed before training): GT success first; ties (e.g. all zero)
    are broken by GT success with any phalanx, then the partial-lift rate (>= 3.5 cm), then the
    mean maximum lift."""
    return (round(float(df.gt_success.mean()), 6), round(float(df.gt_success_any_phalanx.mean()), 6),
            round(float((df.max_lift_cm >= 3.5).mean()), 6), round(float(df.max_lift_cm.mean()), 4))


# ---------------------------------------------------------------------- callback
class V4TrainCallback(BaseCallback):
    def __init__(self, out_dir: Path, cfg: SACConfig, eval_env, grid, gate_at=None,
                 n_demo: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.out = Path(out_dir)
        self.cfg = cfg
        self.eval_env = eval_env
        self.grid = grid
        self.best = -1.0
        self.best_key = None
        self.best_step = None
        self.next_eval = cfg.eval_every
        self.next_ckpt = cfg.checkpoint_every
        self.next_q = cfg.q_health_every
        self.next_control = 500
        gates = [] if gate_at is None else ([gate_at] if isinstance(gate_at, int) else sorted(gate_at))
        self.gates = list(gates)                 # one or more decision gates (Wave 3: 25k and 50k)
        self.gate_at = self.gates[0] if self.gates else None
        self.gate_done = not self.gates
        self.n_demo = int(n_demo)
        self.episodes: list = []
        self.started = time.time()
        self.wait_seconds = 0.0          # time spent at the gate or paused (not training)
        self.last_q = (time.time(), 0, 0.0)
        self.bad_q = 0
        self.nstep_checked = False
        self.stop_reason: str | None = None
        (self.out / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- helpers
    def _append(self, name: str, rows) -> None:
        path = self.out / name
        pd.DataFrame(rows).to_csv(path, mode='a', header=not path.exists(), index=False)

    def _event(self, kind: str, **kw) -> None:
        self._append('control_log.csv', [{'time': datetime.now().isoformat(timespec='seconds'),
                                          'step': int(self.num_timesteps), 'event': kind,
                                          'detail': json.dumps(kw, default=str)}])
        if self.verbose:
            print(f'[CTRL] step {self.num_timesteps:,}: {kind} {kw}', flush=True)

    def _save(self, name: str) -> str:
        path = self.out / 'checkpoints' / name
        self.model.save(str(path))
        return str(path)

    def _stop(self, reason: str, **kw) -> None:
        self.stop_reason = reason
        (self.out / 'stop_reason.json').write_text(json.dumps({'step': int(self.num_timesteps), 'reason': reason,
                                                               **kw}, indent=1, default=str), encoding='utf-8')
        self._event('stop', reason=reason, **kw)

    def training_seconds(self) -> float:
        return time.time() - self.started - self.wait_seconds

    # ---------------------------------------------------------------- Q health / divergence
    def _q_health(self, step: int) -> None:
        buf = self.model.replay_buffer
        if buf is None or buf.size() < 1000:
            return
        if not self.nstep_checked and buf.size() >= 2000:
            self.nstep_checked = True
            chk = verify_nstep(buf)
            (self.out / 'nstep_check.json').write_text(json.dumps(chk, indent=1), encoding='utf-8')
            if not chk['passed']:
                self._stop('nstep_check_failed', check=chk)
                return
        with torch.no_grad():
            batch = buf.sample(1024)
            q = torch.cat(self.model.critic(batch.observations, batch.actions), dim=1).min(dim=1).values
            mu, log_std, _ = self.model.actor.get_action_dist_params(batch.observations)
            actor_mean_abs = float(torch.tanh(mu).abs().mean())        # deterministic action magnitude
            actor_log_std = float(log_std.mean())
            q_demo = float('nan')
            if self.n_demo > 0:
                idx = np.random.randint(0, self.n_demo, 512)
                obs_d = torch.as_tensor(buf.observations[idx, 0], device=self.model.device)
                act_d = torch.as_tensor(buf.actions[idx, 0], device=self.model.device)
                q_demo = float(torch.cat(self.model.critic(obs_d, act_d), dim=1).min(dim=1).values.mean())
        alpha = (float(torch.exp(self.model.log_ent_coef.detach()).item()) if getattr(self.model, 'log_ent_coef', None) is not None
                 else float(self.model.ent_coef_tensor))          # fixed alpha (e.g. Wave 5 E6: alpha = 0)
        now = time.time()
        t_prev, s_prev, w_prev = self.last_q
        sps = (step - s_prev) / max(1e-6, (now - t_prev) - (self.wait_seconds - w_prev))
        self.last_q = (now, step, self.wait_seconds)
        lg = self.model.logger.name_to_value
        row = {'step': step, 'q_mean': float(q.mean()), 'q_std': float(q.std()), 'q_min': float(q.min()),
               'q_p01': float(torch.quantile(q, .01)), 'q_p99': float(torch.quantile(q, .99)),
               'q_max': float(q.max()), 'q_demo_mean': q_demo, 'ent_coef': alpha,
               'actor_mean_abs': actor_mean_abs, 'actor_log_std_mean': actor_log_std,
               'critic_loss_last': lg.get('train/critic_loss'),       # None before learning_starts
               'actor_loss_last': lg.get('train/actor_loss'),
               'buffer_size': int(buf.size()), 'steps_per_s': sps,
               'train_hours': self.training_seconds() / 3600}
        try:
            import psutil
            row['rss_gb'] = psutil.Process().memory_info().rss / 2 ** 30
        except Exception:
            pass
        self._append('qhealth.csv', [row])
        losses = [v for v in (row['critic_loss_last'], row['actor_loss_last']) if v is not None]
        finite = bool(torch.isfinite(q).all()) and bool(np.isfinite(alpha)) and all(np.isfinite(v) for v in losses)
        if not finite:
            self._stop('non_finite_q_or_loss', checkpoint=self._save(f'diverged_step_{step}.zip'), row=row)
            return
        qabs = max(abs(row['q_max']), abs(row['q_min']))
        if qabs > Q_HARD_LIMIT:
            self.bad_q += 1
            self._event('q_above_hard_limit', q_abs_max=qabs, consecutive=self.bad_q)
            if self.bad_q >= 2:
                self._stop('q_divergence', checkpoint=self._save(f'diverged_step_{step}.zip'), row=row)
        else:
            self.bad_q = 0
            if qabs > Q_SOFT_LIMIT:
                self._event('q_above_soft_limit', q_abs_max=qabs)

    # ---------------------------------------------------------------- control files
    def _wait_until(self, done_fn, poll: float = 10.0) -> None:
        t0 = time.time()
        while not done_fn():
            time.sleep(poll)
        self.wait_seconds += time.time() - t0

    def _control(self) -> None:
        stop_f, pause_f = self.out / 'stop_request.json', self.out / 'pause_request.json'
        if stop_f.exists():
            self._stop('stop_request', request=stop_f.read_text(encoding='utf-8')[:500],
                       checkpoint=self._save(f'stopped_step_{self.num_timesteps}.zip'))
            return
        if pause_f.exists():
            ck = self._save(f'pause_step_{self.num_timesteps}.zip')
            self._event('pause', request=pause_f.read_text(encoding='utf-8')[:500], checkpoint=ck)
            t0 = time.time()
            self._wait_until(lambda: not pause_f.exists() or stop_f.exists())
            self._event('resume', paused_s=round(time.time() - t0, 1))
            if stop_f.exists():
                self._control()

    def _gate(self, step: int) -> None:
        self.gate_done = True
        ck = self._save(f'gate_step_{step}.zip')
        dec_f = self.out / 'gate_decision.json'
        info = {'step': step, 'since': datetime.now().isoformat(timespec='seconds'), 'checkpoint': ck,
                'best_in_training_validation': self.best, 'best_step': self.best_step,
                'training_hours': self.training_seconds() / 3600,
                'how': 'write gate_decision.json with {"decision": "continue"|"stop", "reason": ...}'}
        (self.out / 'gate_waiting.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
        self._event('gate_waiting', checkpoint=ck)
        t0 = time.time()
        self._wait_until(lambda: dec_f.exists() or (self.out / 'stop_request.json').exists())
        if dec_f.exists():
            dec = json.loads(dec_f.read_text(encoding='utf-8'))
        else:
            dec = {'decision': 'stop', 'reason': 'stop_request during gate'}
        self._event('gate_decision', waited_s=round(time.time() - t0, 1), **dec)
        (self.out / 'gate_waiting.json').unlink(missing_ok=True)
        if len(self.gates) > 1 and dec_f.exists():     # keep the decision, free the name for the next gate
            dec_f.replace(self.out / f'gate_decision_step_{step}.json')
        if dec.get('decision') != 'continue':
            self._stop('gate_stop', decision=dec)

    # ---------------------------------------------------------------- main hook
    def _on_step(self) -> bool:
        step = int(self.num_timesteps)
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                gt = info.get('gt_episode', {})
                self.episodes.append({'step': step, 'return': float(info['episode']['r']),
                                      'length': int(info['episode']['l']),
                                      'gt_success': int(info.get('gt_success_EVAL_ONLY', False)),
                                      'signal_success': int(info.get('signal_success', False)),
                                      'approach': info.get('approach'),
                                      **{k: gt[k] for k in ('gt_any', 'max_lift_cm', 'final_lift_cm', 'dropped',
                                                            'object_off_table', 'sim_failure', 'contact_occurred',
                                                            'thumb_contact', 'max_pads', 'hold_frames_gt',
                                                            'pushed_before_close', 'outcome') if k in gt},
                                      **info.get('residual_episode', {}), **info.get('extra_episode', {})})
        if len(self.episodes) >= 20:
            self._append('train_episodes.csv', self.episodes)
            self.episodes = []
        if step >= self.next_q:
            self.next_q += self.cfg.q_health_every
            self._q_health(step)
            if self.stop_reason:
                return False
        if step >= self.next_ckpt:
            self.next_ckpt += self.cfg.checkpoint_every
            self._save(f'step_{step}.zip')
        if step >= self.next_eval:
            self.next_eval += self.cfg.eval_every
            t_eval = time.time()
            df = evaluate_on_grid(self.model, self.eval_env, self.grid)
            rate = float(df.gt_success.mean())
            key = selection_key(df)
            improved = self.best_key is None or key > self.best_key
            if improved:
                self.best, self.best_key, self.best_step = rate, key, step
                self.model.save(str(self.out / 'best_model.zip'))
                (self.out / 'best_model.json').write_text(json.dumps(
                    {'step': step, 'selection_key': key, 'key_order': ['gt_success', 'gt_any_phalanx',
                                                                       'partial_lift_rate', 'mean_max_lift_cm'],
                     'validation_episodes': len(df), 'seed_base': VALIDATION_SEED_BASE}, indent=1), encoding='utf-8')
            row = {'step': step, 'val_gt_success': rate, 'val_gt_any_phalanx': float(df.gt_success_any_phalanx.mean()),
                   'val_partial_lift_rate': float((df.max_lift_cm >= 3.5).mean()),
                   'val_max_lift_cm': float(df.max_lift_cm.mean()), 'val_contact_rate': float(df.contact_occurred.mean()),
                   'val_return': float(df.episode_return.mean()), 'val_episodes': len(df), 'best': self.best,
                   'best_step': self.best_step, 'improved': int(improved), 'eval_s': time.time() - t_eval,
                   'hours': self.training_seconds() / 3600}
            for ap, g in df.groupby('approach'):
                row[f'val_gt_{ap}'] = float(g.gt_success.mean())
                row[f'val_partial_{ap}'] = float((g.max_lift_cm >= 3.5).mean())
            self._append('validation.csv', [row])
            df.assign(step=step).to_csv(self.out / 'validation_episodes.csv', mode='a',
                                        header=not (self.out / 'validation_episodes.csv').exists(), index=False)
            if self.verbose:
                print(f'[VAL] step {step:,}: GT {rate:.3f} (top {row.get("val_gt_top", float("nan")):.3f}, '
                      f'oblique {row.get("val_gt_oblique", float("nan")):.3f}) best {self.best:.3f}@{self.best_step} '
                      f'lift {row["val_max_lift_cm"]:.1f} cm, {row["hours"]:.2f} h', flush=True)
        if self.gates and step >= self.gates[0]:
            self._gate(step)
            self.gates.pop(0)
            self.gate_done = not self.gates
            if self.stop_reason:
                return False
        if step >= self.next_control:
            self.next_control += 500
            self._control()
            if self.stop_reason:
                return False
        return True

    def _on_training_end(self) -> None:
        if self.episodes:
            self._append('train_episodes.csv', self.episodes)
            self.episodes = []
