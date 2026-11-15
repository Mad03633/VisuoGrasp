"""Wave 3 report figures: conservative residual SAC (res2 / v4r2). Reads run outputs only and writes
to outputs/rl/wave3/report/. The zero-residual reference on the model-selection set is the cached
Wave 2 file (r = 0 equals the scripted controller for every residual version; verified A/B).
PRIVILEGED setting (simulator object pose, GT-oracle contacts).

    python -m visuograsp_v4.scripts.wave3_report
"""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts import wave2_report as w2
from visuograsp_v4.scripts.wave1_report import GRID, INK, INK2, REF, style

W3 = OUTPUT_ROOT / 'rl' / 'wave3'
RUN = W3 / 'conservative_residual_sac_seed11'
OUT = W3 / 'report'
FINAL = W3 / 'final_eval_unseen_6p1M'
LABEL = {'scripted': 'Scripted controller', 'zero_residual': 'Residual env, r = 0', 'perturbed_1e-3': 'Chaos control, r ~ U(+-1e-3)',
         'wave3_best': 'Conservative residual SAC (25k, best = last)', 'wave3_last': 'Conservative residual SAC last'}


def fig_actor(q: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    axes[0].plot(q.step, q.actor_mean_abs, color=w2.C_RES, lw=2, label='mean |tanh(mu)| on replay states')
    style(axes[0], 'Deterministic residual magnitude (normalised; 1 = bound)', 'abs(r)')
    axes[0].set_ylim(0, 1)
    axes[1].plot(q.step, np.exp(q.actor_log_std_mean), color=w2.C_RES, lw=2, label='exp(mean log std), pre-tanh')
    axes[1].axhline(0.2, color=REF, ls='--', lw=1.5, label='initial / target std 0.2')
    style(axes[1], 'Exploration std', 'std')
    axes[2].plot(q.step, q.ent_coef, color=w2.C_RES, lw=2, label='alpha (auto, init 0.01, target H -5.0)')
    axes[2].set_yscale('log')
    style(axes[2], 'Entropy coefficient', 'alpha')
    for a in axes:
        a.legend(fontsize=8, frameon=False)
    fig.suptitle('Wave 3 actor / entropy diagnostics (logged every 5k steps)', fontsize=11, color=INK)
    fig.savefig(out / 'fig_actor_entropy.png', dpi=130)
    plt.close(fig)


def fig_final(summ: pd.DataFrame, cats: pd.DataFrame, out):
    pols = [p for p in ('scripted', 'zero_residual', 'perturbed_1e-3', 'wave3_best') if p in set(summ.policy)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.6 * len(pols) + 1.8), constrained_layout=True, sharey=True)
    for j, ap in enumerate(('all', 'top', 'oblique')):
        ax = axes[j]
        for i, p in enumerate(pols):
            r = summ[(summ.policy == p) & (summ.approach == ap)].iloc[0]
            c = w2.C_RES if p.startswith('wave3') else REF
            ax.errorbar(r.gt_success, i, xerr=[[r.gt_success - r.wilson_lo], [r.wilson_hi - r.gt_success]], fmt='o', ms=8,
                        color=c, ecolor=c, capsize=3)
            ax.text(r.wilson_hi + 0.02, i, f'{r.gt_success:.1%} ({int(r.gt_success_n)}/{int(r.episodes)})', va='center',
                    fontsize=8, color=INK)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels([LABEL[p] for p in pols], fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.set_title(f'Unseen final set GT success, {ap} (95% Wilson)', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle('Seeds 6.1M, never used before; all policies PRIVILEGED', fontsize=10, color=INK)
    fig.savefig(out / 'fig_final_eval.png', dpi=130)
    plt.close(fig)
    # paired categories vs zero residual: residual policy and chaos control
    order = ['improved', 'broken']
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    rows = []
    for p in ('wave3_best', 'perturbed_1e-3'):
        c = cats[cats.policy == p].category.value_counts()
        rows.append((p, int(c.get('improved', 0)), int(c.get('broken', 0))))
    for i, (p, imp, brk) in enumerate(rows):
        ax.barh(i - 0.18, imp, height=0.34, color=w2.C_RES, label='improved (only residual succeeds)' if i == 0 else None)
        ax.barh(i + 0.18, brk, height=0.34, color=w2.C_ALT, label='broken (only r = 0 succeeds)' if i == 0 else None)
        ax.text(imp + 0.5, i - 0.18, str(imp), va='center', fontsize=8, color=INK)
        ax.text(brk + 0.5, i + 0.18, str(brk), va='center', fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[p] for p, _, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('episodes (of 540 paired)', fontsize=9, color=INK2)
    ax.set_title('Outcome changes vs zero residual on identical episodes', fontsize=10, loc='left', color=INK)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.grid(True, axis='x', color=GRID)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / 'fig_paired_changes.png', dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(RUN / 'train_episodes.csv')
    val = pd.read_csv(RUN / 'validation.csv')
    q = pd.read_csv(RUN / 'qhealth.csv')
    sb = pd.read_csv(RUN / 'sb3' / 'progress.csv')
    iv = pd.concat([pd.read_csv(f).assign(tag=f.parent.name)
                    for f in sorted((RUN / 'independent_validation').glob('*/summary.csv'))], ignore_index=True)
    iv = iv[iv.model != 'perturbed_1e-3']
    ref_iv = {ap: float(iv[(iv.model == 'zero_residual') & (iv.approach == ap)].gt_success.iloc[0]) for ap in w2.APPROACHES}
    zero_sel = pd.read_csv(OUTPUT_ROOT / 'rl' / 'wave2' / 'reference_zero_residual_selection_set.csv')
    w2.fig_training(ep, ref_iv, OUT)
    w2.fig_validation(val, zero_sel, iv, OUT)
    w2.fig_critic(q, sb, OUT)
    fig_actor(q, OUT)
    if (FINAL / 'summary_by_approach.csv').exists():
        fig_final(pd.read_csv(FINAL / 'summary_by_approach.csv'), pd.read_csv(FINAL / 'paired_categories_episodes.csv'), OUT)
    print(q[['step', 'q_mean', 'q_p99', 'q_max', 'ent_coef', 'actor_mean_abs', 'actor_log_std_mean', 'critic_loss_last']].round(4).to_string(index=False))
    print('written to', OUT)


if __name__ == '__main__':
    main()
