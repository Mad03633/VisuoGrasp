"""Wave 5 / E6: advantage-weighted, BC-anchored fine-tuning of a BC actor (AWAC-style), PRIVILEGED setting.

Why not E2/E2b again: both updated the actor along dQ/da of a critic trained on near-BC data; even a small
drift in that direction (E2b anchor MSE ~0.009) increased pushing and lowered GT success. Here the actor never
follows dQ/da. It regresses its mean towards actions that were actually executed and evaluated:

  critic:  SB3 soft-Q target with fixed alpha = 0 (no entropy bonus), next action sampled from the policy with a
           FIXED small std, target clamped to [q_lo, q_hi]; batches mix agent and demonstration transitions
  actor:   L = mean_B[ w(s,a) * mean_dims (tanh(mu(s)) - a)^2 ] + lambda_bc * mean_D[ mean_dims (tanh(mu(s_d)) - a_d)^2 ]
           A(s,a) = min_i Q_i(s,a) - min_i Q_i(s, tanh(mu(s)))           (detached)
           w = exp(clip(A / (tau * std_B(A)), max = log w_max)), normalised to mean 1 over the batch
           B = mixed replay batch (agent + demo, executed actions), D = successful-demo (state, expert label) pairs
  log std: frozen at log(sigma) (state independent); exploration a = tanh(mu + sigma * eps)
  actor frozen for the first actor_delay updates (critic warm-up)
Advantage weights only re-weight actions present in the data, so the policy stays inside the data support.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from .wave4 import BCSAC, actor_mean_action


class AWBCSAC(BCSAC):
    def setup_awr(self, tau=1.0, w_max=20.0, lambda_bc=0.5, fixed_log_std=None, **bc_kwargs):
        self.setup_bc(**bc_kwargs)
        self.awr_tau, self.w_max, self.lambda_bc_const = tau, w_max, lambda_bc
        if fixed_log_std is not None:
            with torch.no_grad():
                self.actor.log_std.weight.zero_()
                self.actor.log_std.bias.fill_(fixed_log_std)
            for p in self.actor.log_std.parameters():
                p.requires_grad_(False)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        if self.actor_lr is not None:
            for g in self.actor.optimizer.param_groups:
                g['lr'] = self.actor_lr
        logs = {'critic': [], 'actor': [], 'awr': [], 'bc': [], 'w_max': [], 'w_ess': [], 'adv_std': []}
        for _ in range(gradient_steps):
            data = self._batch(batch_size)
            discounts = data.discounts if data.discounts is not None else self.gamma
            with torch.no_grad():
                next_actions, _ = self.actor.action_log_prob(data.next_observations)
                next_q = torch.min(torch.cat(self.critic_target(data.next_observations, next_actions), dim=1), dim=1, keepdim=True)[0]
                target_q = data.rewards + (1 - data.dones) * discounts * next_q          # alpha = 0
                if self.q_clip is not None:
                    target_q = target_q.clamp(self.q_clip[0], self.q_clip[1])
            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            logs['critic'].append(float(critic_loss))
            if self._n_updates >= self.actor_delay:
                mu_a = actor_mean_action(self.actor, data.observations)
                with torch.no_grad():
                    q_sa = torch.min(torch.cat(self.critic(data.observations, data.actions), dim=1), dim=1)[0]
                    q_pi = torch.min(torch.cat(self.critic(data.observations, mu_a.detach()), dim=1), dim=1)[0]
                    adv = q_sa - q_pi
                    z = adv / (self.awr_tau * adv.std().clamp(min=1e-6))
                    w = torch.exp(z.clamp(max=float(np.log(self.w_max))))
                    w = w / w.mean()
                awr = (w * ((mu_a - data.actions) ** 2).mean(dim=1)).mean()
                loss = awr
                if self.bc_obs is not None and self.lambda_bc_const > 0:
                    idx = torch.randint(0, len(self.bc_obs), (batch_size,), device=self.device)
                    bc = F.mse_loss(actor_mean_action(self.actor, self.bc_obs[idx]), self.bc_act[idx])
                    loss = loss + self.lambda_bc_const * bc
                    logs['bc'].append(float(bc))
                self.actor.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()
                logs['actor'].append(float(loss)); logs['awr'].append(float(awr))
                logs['w_max'].append(float(w.max())); logs['w_ess'].append(float(w.sum() ** 2 / (w ** 2).sum() / len(w)))
                logs['adv_std'].append(float(adv.std()))
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            self._n_updates += 1
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/critic_loss', float(np.mean(logs['critic'])))
        for k in ('actor', 'awr', 'bc', 'w_max', 'w_ess', 'adv_std'):
            if logs[k]:
                self.logger.record(f'train/{k}', float(np.mean(logs[k])))
        self.bc_log = {k: (float(np.mean(v)) if v else None) for k, v in logs.items()}


# ---------------------------------------------------------------------- copycat fix (Wave 5 diagnosis)
import gymnasium as gym

from .env import obs_slices

PREV_ACTION = obs_slices()['prev_action']          # slice(107, 129): previous executed action in the V4 observation


HISTORY = [obs_slices()[k] for k in ('prev_action', 'finger_qvel', 'wrist_linvel', 'wrist_angvel_local')]


def mask_history(obs):
    """Zero every block that leaks the previous command: prev_action and the joint / wrist velocities (the hand
    follows its target, so its velocity is ~ the previous wrist command). Copycat policies anchored on these
    blocks lift slowly and accelerate only once moving (Wave 5 diagnosis, E5a/E5c)."""
    o = obs.clone() if hasattr(obs, 'clone') else np.array(obs, copy=True)
    for sl in HISTORY:
        o[..., sl] = 0
    return o


def mask_prev_action(obs):
    """Zero the previous-action block of a V4 observation (array or tensor, single or batch). BC policies that
    see prev_action learn to copy it (copycat problem) and then cannot switch abruptly, e.g. from the settle
    phase (0) to lifting (+0.5); masked policies must infer the phase from the state."""
    o = obs.clone() if hasattr(obs, 'clone') else np.array(obs, copy=True)
    o[..., PREV_ACTION] = 0
    return o


class MaskPrevAction(gym.ObservationWrapper):
    """Same environment and action semantics; only the actor's view of prev_action (mode 'prev') or of all
    history blocks (mode 'history') is zeroed."""

    def __init__(self, env, mode='prev'):
        super().__init__(env)
        self.mask_fn = mask_history if mode == 'history' else mask_prev_action

    def observation(self, observation):
        return self.mask_fn(observation).astype(np.float32)

    # pass-through used by the evaluation helpers (gymnasium 1.x wrappers do not forward attributes)
    @property
    def sim(self):
        return self.env.unwrapped.sim

    @property
    def plan(self):
        return self.env.unwrapped.plan

    @property
    def approach(self):
        return self.env.unwrapped.approach
