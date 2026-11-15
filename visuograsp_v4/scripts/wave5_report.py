"""Wave 5 report figures (reads outputs only, except the small copycat trace which re-runs 2 dev episodes).
PRIVILEGED setting; supervised imitation (E5) is not RL; E6 = AWAC-style RL fine-tuning of a BC actor.

    python -m visuograsp_v4.scripts.wave5_report
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts.evaluate_policy import wilson
from visuograsp_v4.scripts.wave1_report import GRID, INK, INK2, REF, style

W5 = OUTPUT_ROOT / 'rl' / 'wave5'
OUT = W5 / 'report'
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # validated categorical order
LAB = {'e5a_256': 'E5a no fix', 'e5b_512x3': 'E5b no fix, 512x3', 'e5c_mask_pw025': 'E5c prev_action masked',
       'e5d_mask_pw1': 'E5d prev masked, phase w 1', 'e5e_hist_pw025': 'E5e history masked',
       'e5f_hist_pw1': 'E5f history masked, phase w 1', 'e5g_histdrop': 'E5g history dropout+noise',
       'e5h_histdrop_ema': 'E5h E5g + weight EMA'}


def wl(k, n):
    lo, hi = wilson(k, n)
    return max(0, lo), min(1, hi)


def fig_bc_epochs():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)
    groups = [['e5a_256', 'e5c_mask_pw025', 'e5g_histdrop'], ['e5d_mask_pw1', 'e5e_hist_pw025', 'e5h_histdrop_ema']]
    for ax, grp in zip(axes, groups):
        for i, r in enumerate(grp):
            d = pd.read_csv(W5 / 'bc' / r / 'dev_evals.csv')
            ax.plot(d.epoch, d.dev_gt, color=PAL[i], lw=2, marker='o', ms=5, label=LAB[r])
        ax.axhline(0.69, color=PAL[3], ls='--', lw=1.2, label='BC Wave 4 (same 100 episodes)')
        ax.axhline(0.75, color=REF, ls=':', lw=1.5, label='scripted (same 100 episodes)')
        style(ax, 'Closed-loop GT success on DEV_A (100 episodes) vs training epoch', 'GT success', xlabel='epoch')
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.02, 1.0)
        ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('E5 behaviour cloning: lower supervised error does not mean better grasping (copycat inputs)', fontsize=11, color=INK)
    fig.savefig(OUT / 'fig_w5_bc_epochs.png', dpi=130)
    plt.close(fig)


def fig_copycat_trace():
    import torch
    os.chdir(OUTPUT_ROOT)
    torch.set_num_threads(1)
    from visuograsp_v4.agents import make_sac
    from visuograsp_v4.config import SACConfig, V4Config
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.wave4 import actor_mean_action, dev_grid
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    for i, (name, path) in enumerate((('BC Wave 4 (69 % DEV_A)', OUTPUT_ROOT / 'rl/wave4/frozen_policies/bc/bc_actor_best.pt'),
                                      ('E5a epoch 10, copycat (29 % DEV_A)', W5 / 'bc/e5a_256/actor_epoch10.pt'))):
        m = make_sac(env, SACConfig(), 0)
        m.actor.load_state_dict(torch.load(path))
        seed, f, v, mo, ap = dev_grid(('box',), approaches=('top',), per_cell=1, base=8_800_000)[0]
        obs, _ = env.reset(seed=seed, options={'family': f, 'variant': v, 'resting_mode': mo, 'approach': ap})
        lift = []
        for t in range(320):
            with torch.no_grad():
                a = actor_mean_action(m.actor, torch.as_tensor(obs[None])).numpy()[0]
            obs, r, term, trunc, _ = env.step(a)
            lift.append(100 * env.sim.object_lift())
            if term or trunc:
                break
        ax.plot(range(1, len(lift) + 1), lift, color=PAL[i], lw=2, label=name)
    ax.axhline(10, color=REF, ls='--', lw=1, label='10 cm success height')
    style(ax, 'Object lift, DEV_A episode 8800000 (box, top)', 'lift (cm)', xlabel='control step')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / 'fig_w5_copycat_trace.png', dpi=130)
    plt.close(fig)


def fig_e6():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for i, (run, lab) in enumerate((('e6_awac_s11', 'E6 (init E5c ep10, prev masked)'), ('e6b_awac_e5g_s11', 'E6b (init E5g)'))):
        d = W5 / 'rl' / run
        if not (d / 'e6_monitor.csv').exists():
            continue
        m = pd.read_csv(d / 'e6_monitor.csv')
        n = json.loads((d / 'run_config.json').read_text())['dev_episodes']
        lo, hi = zip(*[wl(int(round(g * n)), n) for g in m.dev_gt])
        axes[0].fill_between(m.step, lo, hi, color=PAL[i], alpha=0.12, lw=0)
        axes[0].plot(m.step, m.dev_gt, color=PAL[i], lw=2, marker='o', ms=4, label=f'{lab} (n={n})')
        axes[0].axhline(float(m.bc_ref_gt.iloc[0]), color=PAL[i], ls='--', lw=1, label=f'{lab}: its BC initialisation')
        axes[1].plot(m.step, m.drift_mse_to_bc, color=PAL[i], lw=2, label=lab)
        q = pd.read_csv(d / 'qhealth.csv')
        axes[2].plot(q.step, q.q_mean, color=PAL[i], lw=2, label=f'{lab}: mean Q')
        axes[2].plot(q.step, q.q_max, color=PAL[i], lw=1, ls='--', label=f'{lab}: max Q')
    style(axes[0], 'Full-task GT on DEV_A (deterministic)', 'GT success')
    style(axes[1], 'Drift from the BC actor: mean ||tanh(mu) - tanh(mu_BC)||^2', 'MSE')
    axes[2].axhline(30, color=REF, ls=':', lw=1, label='return bound (~30)')
    style(axes[2], 'Critic on replay samples', 'Q')
    for a in axes:
        a.legend(fontsize=7, frameon=False)
    fig.suptitle('E6: AWAC-style BC-anchored RL fine-tuning (actor frozen for the first 20k updates)', fontsize=11, color=INK)
    fig.savefig(OUT / 'fig_w5_e6.png', dpi=130)
    plt.close(fig)


def fig_eval(folder, title, fname):
    sp = folder / 'summary_by_approach.csv'
    if not sp.exists():
        return
    s = pd.read_csv(sp)
    order = list(json.loads((folder / 'policies.json').read_text())['policies'])
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.55 * len(order) + 1.9), constrained_layout=True, sharey=True)
    for j, ap in enumerate(('all', 'top', 'oblique')):
        ax = axes[j]
        for i, p in enumerate(order):
            r = s[(s.policy == p) & (s.approach == ap)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = REF if p == 'scripted' else PAL[i % len(PAL)]
            ax.errorbar(r.gt_success, i, xerr=[[r.gt_success - r.wilson_lo], [r.wilson_hi - r.gt_success]], fmt='o', ms=8, color=c,
                        ecolor=c, capsize=3)
            ax.text(r.wilson_hi + 0.02, i, f'{r.gt_success:.1%} ({int(r.gt_success_n)}/{int(r.episodes)})', va='center', fontsize=8,
                    color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.invert_yaxis()
        ax.set_title(f'{ap}: GT success (95% Wilson)', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for spn in ('top', 'right'):
            ax.spines[spn].set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)


def fig_final_paired():
    """Paired differences on the 540 fresh final-test episodes (analysis/paired_all.csv), bootstrap 95 % CI."""
    p = W5 / 'final_test' / 'final' / 'analysis' / 'paired_all.csv'
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True, sharey=True)
    for ax, grp in zip(axes, ('all', 'approach=top', 'approach=oblique')):
        t = d[d.group == grp].reset_index(drop=True)
        for i, r in t.iterrows():
            ax.errorbar(100 * r.diff_a_minus_b, i, xerr=[[100 * (r.diff_a_minus_b - r.diff_lo)], [100 * (r.diff_hi - r.diff_a_minus_b)]],
                        fmt='o', ms=8, color=INK2, ecolor=INK2, capsize=3)
            ax.text(100 * max(r.diff_hi, 0) + 0.8, i, f'{100 * r.diff_a_minus_b:+.1f} pp  ({int(r.only_a)}/{int(r.only_b)}, p={r.mcnemar_exact_p:.2g})',
                    va='center', fontsize=7.5, color=INK)
        ax.axvline(0, color=REF, lw=1, ls='--')
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([f'{a} - {b}' for a, b in zip(t.A, t.B)], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(100 * min(d.diff_lo[d.group.isin(['all', 'approach=top', 'approach=oblique'])].min(), 0) - 2,
                    100 * d.diff_hi[d.group.isin(['all', 'approach=top', 'approach=oblique'])].max() + 14)
        ax.set_xlabel('difference in GT success A - B (percentage points)', fontsize=8, color=INK2)
        ax.set_title(f'{grp.replace("approach=", "")}: paired difference, 95% bootstrap CI', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for spn in ('top', 'right'):
            ax.spines[spn].set_visible(False)
    fig.suptitle('Wave 5 FINAL TEST, identical 540 fresh episodes (6.7M); labels: discordant only A / only B, exact McNemar p '
                 '(raw; Holm-adjusted in paired_all.csv). PRIVILEGED', fontsize=10, color=INK)
    fig.savefig(OUT / 'fig_w5_final_paired.png', dpi=130)
    plt.close(fig)


def fig_final_family():
    p = W5 / 'final_test' / 'final' / 'analysis' / 'by_family_long.csv'
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.approach == 'all']
    order = list(json.loads((W5 / 'final_test' / 'final' / 'policies.json').read_text())['policies'])
    fams = sorted(d.family.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(3.3 * len(fams), 0.45 * len(order) + 1.9), constrained_layout=True, sharey=True)
    for ax, fam in zip(axes, fams):
        for i, pol in enumerate(order):
            r = d[(d.family == fam) & (d.policy == pol)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = REF if pol == 'scripted' else PAL[i % len(PAL)]
            ax.errorbar(r.gt_success, i, xerr=[[r.gt_success - r.wilson_lo], [r.wilson_hi - r.gt_success]], fmt='o', ms=7, color=c, ecolor=c, capsize=2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_title(f'{fam} (n={int(d[d.family == fam].episodes.iloc[0])} per policy)', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for spn in ('top', 'right'):
            ax.spines[spn].set_visible(False)
    fig.suptitle('Wave 5 FINAL TEST: GT success by object family (95% Wilson), fresh seeds 6.7M; PRIVILEGED', fontsize=10, color=INK)
    fig.savefig(OUT / 'fig_w5_final_family.png', dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_bc_epochs()
    fig_copycat_trace()
    fig_e6()
    for sub, t, f in (('selection/sel_bc', 'DEV_B (200 episodes): scripted, BC Wave 4, E5g', 'fig_w5_devb_bc.png'),
                      ('selection/sel_e6', 'DEV_B (200 episodes): RL fine-tuned candidates', 'fig_w5_devb_e6.png'),
                      ('final_test/final', 'Wave 5 FINAL TEST (fresh seeds 6.7M, 540 episodes; PRIVILEGED)', 'fig_w5_final.png')):
        fig_eval(W5 / sub, t, f)
    fig_final_paired()
    fig_final_family()
    print('written to', OUT)


if __name__ == '__main__':
    main()
