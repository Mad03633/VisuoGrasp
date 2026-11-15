"""Wave 4: curriculum start states, action masking, behaviour cloning / DAgger and BC-regularised SAC.

PRIVILEGED setting (as in Waves 1-3): the actor observation is the 130-D V4 observation with the
simulator object pose and GT-oracle contacts. The scripted controller is used only (a) to produce
curriculum start states and (b) as the demonstrator / DAgger labeller; it is never part of the
learned policy at test time on the full task (stage 4).

Curriculum (CurriculumGraspEnv)
-------------------------------
A start state of stage k is produced by resetting the ORIGINAL V4 episode and letting the scripted
controller drive the simulation (same 22-D action interface, physics stepped normally, no state
teleporting) until its phase machine reaches the hand-over phase; then the learner takes over.
  stage 1  grasp & lift   hand-over at the start of 'close' (hand at the planned grasp pose,
                          fingers open); families box + cylinder; wrist rotation LOCKED (19 DoF)
  stage 2  short approach hand-over at the start of 'approach' (pre-grasp pose, 10 cm back along
                          the approach axis); + bottle, L-shape; all 22 DoF
  stage 3  long approach  hand-over at the start of 'descend' (transit pose above the object,
                          orientation aligned); all five families; all 22 DoF
  stage 4  full task      no hand-over: original V4 initial conditions (autonomous from reset)
Locked action dimensions are set to 0 (the wrist target keeps its orientation). The prefix steps
are not learner transitions; the episode time limit (320 steps) counts them, as in V4.

BC-regularised SAC (BCSAC)
--------------------------
Critic: standard SAC target on a batch that mixes the agent buffer and a demonstration buffer
(fraction demo_frac from the demo buffer, n-step returns in both).
Actor: L = beta * mean(alpha * log pi(a|s) - Q(s, a)) / kappa + lambda(t) * mean ||tanh(mu(s_d)) - a_d||^2
  kappa = mean |Q(s, a)| (detached, >= 1; TD3+BC normalisation), (s_d, a_d) = successful-demo pairs,
  lambda(t) = max(lambda_min, lambda_0 * (1 - t / T_decay)), t = actor updates.
The actor (and alpha) are frozen for the first actor_delay gradient steps (critic warm-up).
E2b additions (prior-regularised variant, used after E2 collapsed):
  + lambda_p(t) * mean_batch ||tanh(mu(s)) - tanh(mu_prior(s))||^2 on ALL batch states (agent + demo),
    mu_prior = frozen copy of the BC actor (KL-to-prior style anchoring), lambda_p decays linearly;
  target Q clamped to [q_lo, q_hi] = the range of discounted v4r1 returns (|returns| <= ~30).
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update

from .config import V4Config
from .env import OBS_CLIP, OBS_DIM, VisuoGraspV4Env
from .scripted import ScriptedGrasp

ALL_FAMILIES = ('box', 'cylinder', 'bottle', 'ring', 'l_shape')
STAGES = {
    1: {'name': 'grasp_lift', 'handover': 'close', 'families': ('box', 'cylinder'), 'locked': (19, 20, 21)},
    2: {'name': 'short_approach', 'handover': 'approach', 'families': ('box', 'cylinder', 'bottle', 'l_shape'), 'locked': ()},
    3: {'name': 'long_approach', 'handover': 'descend', 'families': ALL_FAMILIES, 'locked': ()},
    4: {'name': 'full_task', 'handover': None, 'families': ALL_FAMILIES, 'locked': ()},
    5: {'name': 'grasp_lift_all_families', 'handover': 'close', 'families': ALL_FAMILIES, 'locked': (19, 20, 21)},  # E3 (fixed)
}


class CurriculumGraspEnv(gym.Env):
    """V4 env whose episodes start at a scripted hand-over state of the current curriculum stage."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, cfg: V4Config | None = None, stage: int = 4, approaches=('top', 'oblique'),
                 replay_prev_prob: float = 0.25):
        super().__init__()
        self.base = VisuoGraspV4Env(cfg, obs_mode='privileged', contact_source='gt_oracle', approaches=approaches)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(OBS_DIM,), dtype=np.float32)
        self.stage = stage
        self.replay_prev_prob = replay_prev_prob
        self.ep_stage = stage
        self.handover_info: dict = {}
        self.prefix_steps_total = 0

    # pass-through for evaluation helpers
    @property
    def sim(self):
        return self.base.sim

    @property
    def plan(self):
        return self.base.plan

    @property
    def approach(self):
        return self.base.approach

    def set_stage(self, stage: int) -> None:
        self.stage = int(stage)

    def _mask(self, stage: int) -> np.ndarray:
        m = np.ones(22, np.float32)
        m[list(STAGES[stage]['locked'])] = 0.0
        return m

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = dict(options or {})
        stage = int(opts.pop('stage', self.stage))
        if 'stage' not in (options or {}) and stage > 1 and self.np_random.random() < self.replay_prev_prob:
            stage = int(self.np_random.integers(1, stage))           # rehearse an earlier stage
        self.ep_stage = stage
        st = STAGES[stage]
        if 'family' not in opts:
            opts['family'] = str(self.np_random.choice(st['families']))
        obs, info = self.base.reset(seed=seed, options=opts)
        start_com = self.sim.object_com().copy()
        prefix = 0
        if st['handover'] is not None:
            agent = ScriptedGrasp(self.sim, self.base.approach)
            agent.make_plan()
            while agent.phase != st['handover']:
                obs, r, term, trunc, step_info = self.base.step(agent.act())
                prefix += 1
                if term or trunc or prefix > 250:
                    break
        self.prefix_steps_total += prefix
        t = self.sim.tracker
        self.handover_info = {'stage': stage, 'prefix_steps': prefix,
                              'object_displacement_cm': 100 * float(np.linalg.norm(self.sim.object_com()[:2] - start_com[:2])),
                              'object_lift_cm': 100 * self.sim.object_lift(), 'hand_table_force_n': float(t.hand_table_force_max),
                              'any_contact': bool(t.any_hand_object_contact), 'finite': bool(np.all(np.isfinite(obs))),
                              'valid_handover': bool(prefix <= 250 and not (t.nonfinite or t.runaway or t.dropped))}
        self.mask = self._mask(stage)
        info = dict(info, handover=self.handover_info)
        return obs, info

    def step(self, action):
        a = np.asarray(action, np.float32) * self.mask
        obs, r, term, trunc, info = self.base.step(a)
        info['stage'] = self.ep_stage
        if term or trunc:
            info['extra_episode'] = {'stage': self.ep_stage, 'prefix_steps': self.handover_info.get('prefix_steps', 0)}
        return obs, r, term, trunc, info

    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()


# ---------------------------------------------------------------------- expert data / BC / DAgger
def collect_expert(env: VisuoGraspV4Env, seeds, approaches=('top', 'oblique')) -> dict:
    """Scripted episodes in the base env. Actions clipped to +-0.999 and executed exactly as stored."""
    obs_l, act_l, rew_l, nxt_l, done_l, tout_l, ep_l, succ_l = [], [], [], [], [], [], [], []
    episodes = []
    for k, seed in enumerate(seeds):
        ap = approaches[k % len(approaches)]
        obs, _ = env.reset(seed=int(seed), options={'approach': ap})
        agent = ScriptedGrasp(env.sim, env.approach)
        n0 = len(obs_l)
        while True:
            a = np.clip(agent.act(), -0.999, 0.999).astype(np.float32)
            nxt, r, term, trunc, _ = env.step(a)
            obs_l.append(obs); act_l.append(a); rew_l.append(r); nxt_l.append(nxt)
            done_l.append(term or trunc); tout_l.append(trunc and not term); ep_l.append(k)
            obs = nxt
            if term or trunc:
                break
        ok = bool(env.sim.tracker.gt_success)
        succ_l.extend([ok] * (len(obs_l) - n0))
        episodes.append({'seed': int(seed), 'approach': ap, 'family': env.sim.spec.family, 'steps': len(obs_l) - n0,
                         'gt_success': ok})
    return {'obs': np.array(obs_l, np.float32), 'act': np.array(act_l, np.float32), 'rew': np.array(rew_l, np.float32),
            'next_obs': np.array(nxt_l, np.float32), 'done': np.array(done_l), 'timeout': np.array(tout_l),
            'episode': np.array(ep_l), 'success': np.array(succ_l), 'episodes': episodes}


def collect_dagger(env: VisuoGraspV4Env, actor_fn, seeds, approaches=('top', 'oblique')) -> dict:
    """Learner acts (deterministic); a shadow scripted expert, advanced on the SAME states, labels
    every visited state with the action it would command (DAgger labels)."""
    obs_l, lab_l, episodes = [], [], []
    for k, seed in enumerate(seeds):
        ap = approaches[k % len(approaches)]
        obs, _ = env.reset(seed=int(seed), options={'approach': ap})
        expert = ScriptedGrasp(env.sim, env.approach)
        while True:
            label = np.clip(expert.act(), -0.999, 0.999).astype(np.float32)
            obs_l.append(obs); lab_l.append(label)
            obs, r, term, trunc, _ = env.step(actor_fn(obs))
            if term or trunc:
                break
        episodes.append({'seed': int(seed), 'approach': ap, 'gt_success': bool(env.sim.tracker.gt_success)})
    return {'obs': np.array(obs_l, np.float32), 'act': np.array(lab_l, np.float32), 'episodes': episodes}


def actor_mean_action(actor, obs: torch.Tensor) -> torch.Tensor:
    mu, _, _ = actor.get_action_dist_params(obs)
    return torch.tanh(mu)


def bc_train(actor, obs: np.ndarray, act: np.ndarray, groups: np.ndarray, epochs: int = 30, lr: float = 1e-3,
             batch: int = 512, val_frac: float = 0.1, seed: int = 0) -> dict:
    """Supervised pre-training of the SAC actor's mean: MSE between tanh(mu(s)) and the expert action.
    Validation split by episode group. Returns the learning curve."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    val_g = set(rng.choice(ug, max(1, int(len(ug) * val_frac)), replace=False).tolist())
    vm = np.array([g in val_g for g in groups])
    xt, yt = torch.as_tensor(obs[~vm]), torch.as_tensor(act[~vm])
    xv, yv = torch.as_tensor(obs[vm]), torch.as_tensor(act[vm])
    params = list(actor.latent_pi.parameters()) + list(actor.mu.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    curve = []
    n = len(xt)
    for ep in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = F.mse_loss(actor_mean_action(actor, xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        with torch.no_grad():
            vl = float(F.mse_loss(actor_mean_action(actor, xv), yv)) if len(xv) else float('nan')
        curve.append({'epoch': ep + 1, 'train_mse': tot / n, 'val_mse': vl})
    return {'curve': curve, 'train_samples': int(n), 'val_samples': int(len(xv))}


def set_actor_log_std(actor, std: float) -> None:
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(math.log(std))


# ---------------------------------------------------------------------- BC-regularised SAC
class BCSAC(SAC):
    """SAC with demo-buffer mixing, a decaying BC regulariser on successful demo pairs, Q-normalised
    actor objective and an actor (and alpha) warm-up delay. With bc_obs=None, demo_buffer=None,
    actor_delay=0 and beta=None it reduces to SB3's SAC.train()."""

    def setup_bc(self, bc_obs=None, bc_act=None, demo_buffer=None, demo_frac=0.25, lambda0=1.0, lambda_min=0.1,
                 decay_updates=100_000, actor_delay=10_000, beta=2.5, actor_lr=None, warmup_policy_sampling=True,
                 prior_actor=None, prior0=0.0, prior_min=0.0, prior_decay_updates=200_000, q_clip=None):
        self.bc_obs = None if bc_obs is None else torch.as_tensor(bc_obs, device=self.device)
        self.bc_act = None if bc_act is None else torch.as_tensor(bc_act, device=self.device)
        self.demo_buffer, self.demo_frac = demo_buffer, demo_frac
        self.lambda0, self.lambda_min, self.decay_updates = lambda0, lambda_min, decay_updates
        self.actor_delay, self.beta, self.actor_lr = actor_delay, beta, actor_lr
        self.warmup_policy_sampling = warmup_policy_sampling
        self.prior_actor, self.prior0, self.prior_min = prior_actor, prior0, prior_min
        self.prior_decay_updates, self.q_clip = prior_decay_updates, q_clip
        if prior_actor is not None:
            for prm in prior_actor.parameters():
                prm.requires_grad_(False)
        self.bc_log = {}

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if getattr(self, 'warmup_policy_sampling', False) and self.num_timesteps < learning_starts:
            unscaled, _ = self.predict(self._last_obs, deterministic=False)
            scaled = self.policy.scale_action(unscaled)
            return self.policy.unscale_action(scaled), scaled
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def _batch(self, batch_size):
        if self.demo_buffer is None or self.demo_frac <= 0:
            return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        nd = int(round(batch_size * self.demo_frac))
        a = self.replay_buffer.sample(batch_size - nd, env=self._vec_normalize_env)
        d = self.demo_buffer.sample(nd, env=self._vec_normalize_env)
        cat = lambda x, y: torch.cat((x, y), dim=0)
        return type(a)(cat(a.observations, d.observations), cat(a.actions, d.actions), cat(a.next_observations, d.next_observations),
                       cat(a.dones, d.dones), cat(a.rewards, d.rewards), cat(a.discounts, d.discounts))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        if self.actor_lr is not None:
            for g in self.actor.optimizer.param_groups:
                g['lr'] = self.actor_lr
        logs = {'actor': [], 'critic': [], 'bc': [], 'alpha': [], 'kappa': [], 'lambda': []}
        for _ in range(gradient_steps):
            data = self._batch(batch_size)
            discounts = data.discounts if data.discounts is not None else self.gamma
            actor_on = self._n_updates >= self.actor_delay
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                if actor_on:
                    ent_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_loss.backward()
                    self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            logs['alpha'].append(float(ent_coef))
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * discounts * next_q
                if self.q_clip is not None:
                    target_q = target_q.clamp(self.q_clip[0], self.q_clip[1])
            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            logs['critic'].append(float(critic_loss))
            if actor_on:
                q_pi = torch.min(torch.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True)[0]
                sac_term = (ent_coef * log_prob - q_pi)
                if self.beta is not None:
                    kappa = q_pi.abs().mean().detach().clamp(min=1.0)
                    actor_loss = self.beta * sac_term.mean() / kappa
                    logs['kappa'].append(float(kappa))
                else:
                    actor_loss = sac_term.mean()
                if self.bc_obs is not None:
                    t = self._n_updates - self.actor_delay
                    lam = max(self.lambda_min, self.lambda0 * (1.0 - t / max(1, self.decay_updates)))
                    idx = torch.randint(0, len(self.bc_obs), (batch_size,), device=self.device)
                    bc = F.mse_loss(actor_mean_action(self.actor, self.bc_obs[idx]), self.bc_act[idx])
                    actor_loss = actor_loss + lam * bc
                    logs['bc'].append(float(bc))
                    logs['lambda'].append(lam)
                if self.prior_actor is not None and self.prior0 > 0:
                    t = self._n_updates - self.actor_delay
                    lp = max(self.prior_min, self.prior0 * (1.0 - t / max(1, self.prior_decay_updates)))
                    with torch.no_grad():
                        a_ref = actor_mean_action(self.prior_actor, data.observations)
                    prior = F.mse_loss(actor_mean_action(self.actor, data.observations), a_ref)
                    actor_loss = actor_loss + lp * prior
                    logs.setdefault('prior', []).append(float(prior))
                    logs.setdefault('prior_lambda', []).append(lp)
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                logs['actor'].append(float(actor_loss))
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/ent_coef', float(np.mean(logs['alpha'])))
        self.logger.record('train/critic_loss', float(np.mean(logs['critic'])))
        if logs['actor']:
            self.logger.record('train/actor_loss', float(np.mean(logs['actor'])))
        if logs['bc']:
            self.logger.record('train/bc_mse', float(np.mean(logs['bc'])))
            self.logger.record('train/bc_lambda', float(np.mean(logs['lambda'])))
        if logs['kappa']:
            self.logger.record('train/q_norm_kappa', float(np.mean(logs['kappa'])))
        if logs.get('prior'):
            self.logger.record('train/prior_mse', float(np.mean(logs['prior'])))
            self.logger.record('train/prior_lambda', float(np.mean(logs['prior_lambda'])))
        self.bc_log = {k: (float(np.mean(v)) if v else None) for k, v in logs.items()}


# ---------------------------------------------------------------------- development evaluation
DEV_SEED_BASE = 8_700_000          # Wave 4 development / selection seeds (never used before)
APPROACH_INDEX = ('top', 'oblique', 'side')


def dev_grid(families, approaches=('top', 'oblique'), per_cell: int = 4, base: int = DEV_SEED_BASE) -> list[tuple]:
    """(seed, family, variant, resting_mode, approach); seed = base + family_idx*1000 + approach_idx*100 + ep,
    family_idx in ALL_FAMILIES order (identical seeds for a family in every stage's dev set)."""
    from . import objects as objs
    out = []
    for fam in families:
        fi = ALL_FAMILIES.index(fam)
        for ap in approaches:
            for ep in range(per_cell):
                seed = base + fi * 1000 + APPROACH_INDEX.index(ap) * 100 + ep
                rng = np.random.default_rng(seed)
                out.append((seed, fam, str(rng.choice(list(objs.VARIANTS[fam]))), str(rng.choice(list(objs.RESTING_MODES[fam]))), ap))
    return out


def evaluate_grid(env, act_fn, grid, on_reset=None, stage: int | None = None) -> 'pd.DataFrame':
    """Deterministic evaluation on fixed episodes. For CurriculumGraspEnv pass the stage explicitly."""
    import pandas as pd
    from .evaluation import run_policy_episode
    rows = []
    for seed, f, v, m, ap in grid:
        if stage is not None:
            orig = env.reset

            def reset(*, seed=None, options=None, orig=orig):
                return orig(seed=seed, options=dict(options or {}, stage=stage))
            env.reset = reset
        row = run_policy_episode(env, act_fn, seed, f, v, m, ap, on_reset=on_reset)
        if stage is not None:
            del env.reset
            row.update({f'handover_{k}': v for k, v in env.handover_info.items()})
        rows.append(row)
    return pd.DataFrame(rows)
