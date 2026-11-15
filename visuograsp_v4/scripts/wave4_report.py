"""Wave 4 report figures and tables (reads run outputs only; writes outputs/rl/wave4/report/).
All results PRIVILEGED (simulator object pose, GT-oracle contacts); reward v4r1.

    python -m visuograsp_v4.scripts.wave4_report
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts.evaluate_policy import wilson
from visuograsp_v4.scripts.wave1_report import GRID, INK, INK2, REF, style

W4 = OUTPUT_ROOT / 'rl' / 'wave4'
OUT = W4 / 'report'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'     # validated categorical slots 1-3
RUNS = {'E1 curriculum SAC': 'curriculum_sac_s11', 'E2 BC+SAC': 'bcsac_s11', 'E2b BC+SAC prior': 'bcsac_prior_s11',
        'E3 grasp-lift SAC': 'grasp_lift_sac_s11', 'E4 BC grasp-lift SAC': 'bc_grasp_lift_s11'}


def read(run, name):
    p = W4 / run / name
    return pd.read_csv(p) if p.exists() else None


def wl(p, n):
    lo, hi = wilson(int(round(p * n)), n)
    return max(0.0, lo), min(1.0, hi)


def band(ax, x, p, n, color):
    lo, hi = zip(*[wl(v, n) for v in p])
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)


LABELS = {'scripted': 'Scripted controller (PRIVILEGED planner)', 'bc': 'BC (imitation of scripted)',
          'e1_curriculum_sac': 'E1 curriculum SAC (no demos)', 'e2b_bc_sac_prior': 'E2b BC + SAC (anchored)',
          'wave1_demo_sac': 'Wave 1 SAC + demo prefill', 'e3_grasp_lift_sac': 'E3 SAC from scratch (hand-over)',
          'e4_bc_grasp_lift_sac': 'E4 BC + SAC (hand-over)', 'bc_handover': 'BC (hand-over)'}


def fig_eval(folder, title, fname):
    sp = folder / 'summary_by_approach.csv'
    if not sp.exists():
        return None
    summ = pd.read_csv(sp)
    order = [p for p in json.loads((folder / 'policies.json').read_text())['policies']]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.55 * len(order) + 1.9), constrained_layout=True, sharey=True)
    for j, ap in enumerate(('all', 'top', 'oblique')):
        ax = axes[j]
        for i, pol in enumerate(order):
            r = summ[(summ.policy == pol) & (summ.approach == ap)]
            if r.empty:
                continue
            r = r.iloc[0]
            c = REF if pol == 'scripted' else (C3 if pol.startswith('bc') else (C2 if 'wave1' in pol else C1))
            ax.errorbar(r.gt_success, i, xerr=[[r.gt_success - r.wilson_lo], [r.wilson_hi - r.gt_success]], fmt='o', ms=8,
                        color=c, ecolor=c, capsize=3)
            ax.text(r.wilson_hi + 0.02, i, f'{r.gt_success:.1%} ({int(r.gt_success_n)}/{int(r.episodes)})', va='center',
                    fontsize=8, color=INK)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS.get(p, p) for p in order], fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.invert_yaxis()
        ax.set_title(f'{ap}: GT success (95% Wilson)', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for spn in ('top', 'right'):
            ax.spines[spn].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    return summ


def final_figures():
    for sub, title, fname in (('final_test/full_task', 'Wave 4 FINAL TEST, full task from the original V4 initial conditions '
                               '(fresh seeds 6.6M, 540 episodes per policy; PRIVILEGED state)', 'fig_w4_final_full_task.png'),
                              ('final_test/subtask_handover5', 'Wave 4 FINAL TEST, grasp-and-lift sub-task: learned policies take over '
                               'at the scripted hand-over (fresh seeds 6.6M, 540 episodes; PRIVILEGED)', 'fig_w4_final_subtask.png'),
                              ('selection/full_v1', 'Wave 4 selection set, full task (seeds 8.75M, 100 episodes)', 'fig_w4_selection_full.png'),
                              ('selection/subtask_v1', 'Wave 4 selection set, sub-task hand-over (seeds 8.75M, 100 episodes)',
                               'fig_w4_selection_subtask.png')):
        fig_eval(W4 / sub, title, fname)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_full = pd.read_csv(W4 / 'dev_scripted_reference_pc4.csv').gt_success.mean()
    ref_s5_path = W4 / 'dev_scripted_reference_stage5_pc5.csv'
    ref_s5 = pd.read_csv(ref_s5_path).gt_success.mean() if ref_s5_path.exists() else None
    bc = json.loads((W4 / 'bc_pretrain' / 'summary.json').read_text())
    bc_dev = bc['iterations'][0]['dev_gt']

    # ---- BC
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for it, c in ((0, C1), (1, C2)):
        cv = pd.read_csv(W4 / 'bc_pretrain' / f'bc_curve_iter{it}.csv')
        axes[0].plot(cv.epoch, cv.val_mse, color=c, lw=2, label=f'iteration {it} validation MSE')
    axes[0].set_yscale('log')
    style(axes[0], 'Behaviour cloning: action MSE (validation episodes)', 'MSE', xlabel='epoch')
    axes[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    labels = ['scripted expert', 'BC iter 0', 'BC iter 1 (DAgger)']
    vals = [ref_full, bc['iterations'][0]['dev_gt'], bc['iterations'][1]['dev_gt']]
    for i, (lab, v) in enumerate(zip(labels, vals)):
        lo, hi = wl(v, 40)
        axes[1].errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='o', ms=8, color=[REF, C1, C2][i], capsize=3)
        axes[1].text(hi + 0.02, i, f'{v:.1%}', va='center', fontsize=8, color=INK)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlim(0, 1.1)
    axes[1].invert_yaxis()
    axes[1].set_title('Full-task dev GT success (40 episodes, 95% Wilson)', fontsize=10, loc='left', color=INK)
    axes[1].grid(True, axis='x', color=GRID)
    for a in axes:
        a.legend(fontsize=8, frameon=False) if a is axes[0] else None
    fig.suptitle('Wave 4 behaviour cloning of the scripted (PRIVILEGED) expert', fontsize=11, color=INK)
    fig.savefig(OUT / 'fig_w4_bc.png', dpi=130)
    plt.close(fig)

    # ---- E1 curriculum
    ev = read('curriculum_sac_s11', 'wave4_eval.csv')
    if ev is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        colors = {1: C1, 2: C2, 3: C3}
        for st, g in ev.dropna(subset=['stage_gt']).groupby('stage'):
            n = int(g.stage_episodes.iloc[0])
            band(axes[0], g.step, g.stage_gt, n, colors.get(st, INK2))
            axes[0].plot(g.step, g.stage_gt, color=colors.get(st, INK2), lw=2, marker='o', ms=4, label=f'stage {st} dev GT (n={n})')
        for s in ev[ev.advanced == 1].step:
            axes[0].axvline(s, color=INK2, ls=':', lw=1)
        axes[0].axhline(0.5, color=REF, ls='--', lw=1, label='advance threshold 0.5')
        style(axes[0], 'E1 curriculum: GT success on the current stage', 'GT success')
        axes[0].set_ylim(-0.02, 1.02)
        band(axes[1], ev.step, ev.full_gt, 40, C1)
        axes[1].plot(ev.step, ev.full_gt, color=C1, lw=2, marker='o', ms=4, label='E1 full task dev GT (n=40)')
        axes[1].plot(ev.step, ev.full_partial, color=C2, lw=1.5, label='E1 full task partial lift')
        axes[1].axhline(ref_full, color=REF, ls='--', lw=1.5, label=f'scripted, same episodes ({ref_full:.0%})')
        style(axes[1], 'E1: transfer to the full task (original initial conditions)', 'rate')
        axes[1].set_ylim(-0.02, 1.02)
        ep = read('curriculum_sac_s11', 'train_episodes.csv')
        if ep is not None and 'stage' in ep:
            for st, g in ep.groupby('stage'):
                axes[2].plot(g.step, g.gt_success.rolling(50, min_periods=10).mean(), color=colors.get(st, INK2), lw=2,
                             label=f'stage {int(st)} training episodes')
        style(axes[2], 'E1 training GT success by stage (rolling 50, stochastic)', 'GT success')
        axes[2].set_ylim(-0.02, 1.02)
        for a in axes:
            a.legend(fontsize=8, frameon=False)
        fig.suptitle('E1 curriculum SAC from scratch (no demonstrations in the learner); dotted lines = stage advances',
                     fontsize=11, color=INK)
        fig.savefig(OUT / 'fig_w4_curriculum.png', dpi=130)
        plt.close(fig)

    # ---- E2 / E2b fine-tuning
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for lab, run, c in (('E2 BC+SAC', 'bcsac_s11', C2), ('E2b BC+SAC prior', 'bcsac_prior_s11', C1)):
        ev, q = read(run, 'wave4_eval.csv'), read(run, 'qhealth.csv')
        if ev is None:
            continue
        band(axes[0], ev.step, ev.full_gt, 40, c)
        axes[0].plot(ev.step, ev.full_gt, color=c, lw=2, marker='o', ms=4, label=lab)
        if q is not None:
            axes[1].plot(q.step, q.q_mean, color=c, lw=2, label=f'{lab}: mean Q')
            axes[1].plot(q.step, q.q_max, color=c, lw=1, ls='--', label=f'{lab}: max Q')
            axes[2].plot(q.step, q.actor_mean_abs, color=c, lw=2, label=lab)
    axes[0].axhline(bc_dev, color=C3, ls='--', lw=1.5, label=f'BC initialisation ({bc_dev:.0%})')
    axes[0].axhline(ref_full, color=REF, ls=':', lw=1.5, label=f'scripted ({ref_full:.0%})')
    style(axes[0], 'Full-task dev GT success (n=40 per point)', 'GT success')
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].axhline(30, color=REF, ls=':', lw=1, label='bound of discounted v4r1 returns (~30)')
    style(axes[1], 'Critic on replay samples', 'Q')
    style(axes[2], 'Mean |deterministic action| on replay states', 'abs(a)')
    for a in axes:
        a.legend(fontsize=7, frameon=False)
    fig.suptitle('RL fine-tuning of the BC policy on the full task (actor frozen during critic warm-up)', fontsize=11, color=INK)
    fig.savefig(OUT / 'fig_w4_finetune.png', dpi=130)
    plt.close(fig)

    # ---- E3 / E4 grasp-lift sub-task
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for lab, run, c in (('E3 SAC from scratch', 'grasp_lift_sac_s11', C1), ('E4 BC init + SAC (anchored)', 'bc_grasp_lift_s11', C2)):
        ev = read(run, 'wave4_eval.csv')
        if ev is None:
            continue
        g = ev.dropna(subset=['stage_gt'])
        band(ax, g.step, g.stage_gt, 50, c)
        ax.plot(g.step, g.stage_gt, color=c, lw=2, marker='o', ms=4, label=lab)
    if ref_s5 is not None:
        ax.axhline(ref_s5, color=REF, ls='--', lw=1.5, label=f'scripted closing/lifting, same states ({ref_s5:.0%})')
    style(ax, 'Grasp-and-lift sub-task from the scripted hand-over (stage 5, all families, n=50)', 'GT success')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / 'fig_w4_grasp_lift.png', dpi=130)
    plt.close(fig)

    # ---- tables
    rows = []
    for lab, run in RUNS.items():
        ev, fin = read(run, 'wave4_eval.csv'), (W4 / run / 'finished.json')
        if ev is None:
            continue
        f = json.loads(fin.read_text()) if fin.exists() else {}
        r = {'experiment': lab, 'run': run, 'steps': f.get('steps', int(ev.step.max())), 'stop_reason': f.get('stop_reason', 'running'),
             'wall_min': f.get('seconds_wall', float('nan')) / 60, 'final_stage': f.get('final_stage')}
        if ev.full_gt.notna().any():
            r.update(full_dev_gt_first=float(ev.full_gt.iloc[0]), full_dev_gt_best=float(ev.full_gt.max()),
                     full_dev_gt_last=float(ev.full_gt.iloc[-1]))
        if 'stage_gt' in ev and ev.stage_gt.notna().any():
            r.update(stage_dev_gt_best=float(ev.stage_gt.max()), stage_dev_gt_last=float(ev.stage_gt.dropna().iloc[-1]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / 'table_runs.csv', index=False)
    final_figures()
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('written to', OUT)


if __name__ == '__main__':
    main()
