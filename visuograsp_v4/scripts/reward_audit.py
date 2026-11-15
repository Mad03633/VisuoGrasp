"""Reward audit: does the reward rank reference behaviours in the intended order?

Behaviours (all through the 22-D env interface, privileged observation, GT-oracle contact):
  scripted          full scripted grasp (approach, align, close, lift, hold)
  scripted_no_lift  approach + close, then stay at the grasp pose (contact without lifting)
  scripted_no_close approach to the grasp pose with open fingers
  zero              zero action (hand stays at its start pose)
  random            uniform random actions
Ranked by the DISCOUNTED return (gamma 0.99, + gamma^T Phi(s_T) at a time limit), which is what SAC
optimises; undiscounted sums contain a -(1-gamma)*sum(Phi) shaping artefact.
Intended: success > every non-lifting behaviour; no non-lifting behaviour should beat doing nothing
by a large margin (that would be a local optimum).
Seeds: DEV seeds 9_100_000+ (not used for any evaluation).

    python -m visuograsp_v4.scripts.reward_audit --episodes 2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, RewardConfig, V4Config
from visuograsp_v4.controller import finger_to_action
from visuograsp_v4.env import VisuoGraspV4Env
from visuograsp_v4.scripted import ScriptedGrasp

SEED_BASE = 9_100_000


def make_policy(name, env, rng):
    if name == 'zero':
        return lambda: np.zeros(22, np.float32)
    if name == 'random':
        return lambda: rng.uniform(-1, 1, 22).astype(np.float32)
    agent = ScriptedGrasp(env.sim, env.approach)

    def act():
        a = agent.act()
        if name == 'scripted_no_close':
            a[:16] = finger_to_action(env.sim.ctrl.finger_target, np.zeros(16), env.cfg.control)
        if name == 'scripted_no_lift' and agent.phase in ('lift', 'hold', 'done'):
            a[16:22] = 0.0
        return a
    return act


def main() -> None:
    import os
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUTPUT_ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--versions', nargs='*', default=['v4r0', 'v4r1'])
    args = ap.parse_args()
    policies = ('scripted', 'scripted_no_lift', 'scripted_no_close', 'zero', 'random')
    rows = []
    t0 = time.time()
    for version in args.versions:
        rw = replace(RewardConfig(), version=version, terminate_on_success=(version == 'v4r0'))
        env = VisuoGraspV4Env(V4Config(reward=rw), obs_mode='privileged', contact_source='gt_oracle')
        for fi, fam in enumerate(env.cfg.objects.families):
            for ai, approach in enumerate(('top', 'oblique', 'side')):
                for ep in range(args.episodes):
                    seed = SEED_BASE + fi * 1000 + ai * 100 + ep
                    for pol in policies:
                        env.reset(seed=seed, options={'family': fam, 'approach': approach})
                        act = make_policy(pol, env, np.random.default_rng(seed))
                        ret, terms, steps = 0.0, {}, 0
                        disc, disc_unshaped, g = 0.0, 0.0, 1.0
                        while True:
                            _, r, term, trunc, info = env.step(act())
                            ret += r
                            disc += g * r
                            disc_unshaped += g * (r - info['reward_terms']['shaping'])
                            g *= rw.gamma
                            steps += 1
                            for k, v in info['reward_terms'].items():
                                terms[k] = terms.get(k, 0.0) + v
                            if term or trunc:
                                break
                        if trunc and not term:      # SB3 bootstraps at a time limit: add gamma^T * Phi(s_T)
                            disc += g * env.phi
                        rows.append(dict(version=version, policy=pol, family=fam, approach=approach, seed=seed,
                                         ret=ret, disc_return=disc, disc_unshaped=disc_unshaped, steps=steps, gt_success=env.sim.tracker.gt_success,
                                         max_lift_cm=100 * env.sim.tracker.max_lift,
                                         **{f'sum_{k}': v for k, v in terms.items()}))
        env.close()
        print(f'{version} done, {time.time() - t0:.0f} s', flush=True)
    df = pd.DataFrame(rows)
    out = OUTPUT_ROOT / 'reward_audit'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'episodes.csv', index=False)
    summary = df.groupby(['version', 'policy']).agg(n=('ret', 'size'), undiscounted=('ret', 'mean'),
                                                    disc_return=('disc_return', 'mean'), disc_unshaped=('disc_unshaped', 'mean'),
                                                    gt_success=('gt_success', 'mean'),
                                                    mean_steps=('steps', 'mean')).reset_index()
    by_outcome = df[df.policy == 'scripted'].groupby(['version', 'gt_success']).disc_return.agg(['size', 'mean', 'min', 'max']).reset_index()
    # rank check: within each (version, seed, family, approach), how often is the full script's return the highest?
    piv = df.pivot_table(index=['version', 'seed', 'family', 'approach'], columns='policy', values='disc_return').reset_index()
    succ = df[df.policy == 'scripted'].set_index(['version', 'seed', 'family', 'approach']).gt_success
    piv['scripted_success'] = piv.set_index(['version', 'seed', 'family', 'approach']).index.map(succ)
    piv['scripted_best'] = piv[list(policies)].idxmax(axis=1) == 'scripted'
    rank = piv.groupby(['version', 'scripted_success']).scripted_best.agg(['size', 'mean']).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    by_outcome.to_csv(out / 'scripted_return_by_outcome.csv', index=False)
    rank.to_csv(out / 'rank_check.csv', index=False)
    (out / 'meta.json').write_text(json.dumps({'seed_base': SEED_BASE, 'episodes_per_cell': args.episodes,
                                               'note': 'privileged obs, gt_oracle contacts, DEV seeds'}, indent=1))
    pd.set_option('display.width', 200)
    print(summary.round(2).to_string(index=False))
    print(by_outcome.round(2).to_string(index=False))
    print(rank.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
