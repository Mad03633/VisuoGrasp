"""Wave 2 report: residual SAC learning curves, critic diagnostics, validation against the
zero-residual (= scripted) baseline, final-test comparison. Reads run outputs; writes to
outputs/rl/wave2/report/. The only computation it may run is the zero-residual policy on the
in-training model-selection set (50 episodes, seeds 8.0M) as a reference line (cached CSV).
PRIVILEGED setting throughout (simulator object pose, GT-oracle contacts).

    python -m visuograsp_v4.scripts.wave2_report
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.scripts.evaluate_policy import wilson
from visuograsp_v4.scripts.wave1_report import GRID, INK, INK2, REF, style

W2 = OUTPUT_ROOT / 'rl' / 'wave2'
RUN = W2 / 'residual_sac_seed11'
OUT = W2 / 'report'
C_RES = '#2a78d6'           # validated categorical slot 1
C_ALT = '#eb6834'           # slot 2 (second series in a panel)
APPROACHES = ('top', 'oblique')
POLICY_ORDER = ['scripted', 'zero_residual', 'residual_best', 'residual_last', 'wave1_demo_best', 'wave1_scratch_best']
POLICY_LABEL = {'scripted': 'Scripted controller', 'zero_residual': 'Residual env, r = 0',
                'residual_best': 'Residual SAC (best)', 'residual_last': 'Residual SAC (last)',
                'wave1_demo_best': 'Wave 1 SAC + demos (100k)', 'wave1_scratch_best': 'Wave 1 SAC scratch (150k)'}


def wl(k, n):
    lo, hi = wilson(k, n)
    return max(0.0, lo), min(1.0, hi)


def zero_on_selection_set() -> pd.DataFrame:
    path = W2 / 'reference_zero_residual_selection_set.csv'
    if path.exists():
        return pd.read_csv(path)
    import torch
    torch.set_num_threads(1)
    cwd = os.getcwd()
    os.chdir(OUTPUT_ROOT)
    from visuograsp_v4 import objects as objs
    from visuograsp_v4.agents import validation_grid
    from visuograsp_v4.evaluation import run_policy_episode
    from visuograsp_v4.residual import ResidualGraspEnv
    cfg = V4Config()
    env = ResidualGraspEnv(cfg, approaches=APPROACHES)
    rows = []
    for seed, fam, ap in validation_grid(cfg.objects.families, list(APPROACHES), 5):
        rng = np.random.default_rng(seed)
        v = str(rng.choice(list(objs.VARIANTS[fam])))
        m = str(rng.choice(list(objs.RESTING_MODES[fam])))
        rows.append(run_policy_episode(env, lambda o: np.zeros(22, np.float32), seed, fam, v, m, ap))
    os.chdir(cwd)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def fig_training(ep: pd.DataFrame, ref_iv: dict, out):
    ep = ep.sort_values('step')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    axes[0, 0].plot(ep.step, ep['return'].rolling(50, min_periods=10).mean(), color=C_RES, lw=2, label='residual SAC')
    style(axes[0, 0], 'Training episode return (rolling 50)', 'return')
    for j, ap in enumerate(APPROACHES):
        g = ep[ep.approach == ap]
        ax = axes[0, j + 1]
        ax.plot(g.step, g.gt_success.rolling(50, min_periods=10).mean(), color=C_RES, lw=2, label='residual SAC (stochastic)')
        if ap in ref_iv:
            ax.axhline(ref_iv[ap], color=REF, ls='--', lw=1.5, label=f'zero residual = scripted ({ref_iv[ap]:.0%}, indep. val.)')
        style(ax, f'Training GT success, {ap} (rolling 50 episodes)', 'success rate')
        ax.set_ylim(-0.02, 1.02)
    ax = axes[1, 0]
    ax.plot(ep.step, ep.res_pos_offset_cm.rolling(50, min_periods=10).mean(), color=C_RES, lw=2, label='wrist position offset |dp| (cm)')
    style(ax, 'Mean residual set-point offset per step (rolling 50)', 'cm')
    ax = axes[1, 1]
    ax.plot(ep.step, ep.res_finger_offset_rad.rolling(50, min_periods=10).mean(), color=C_RES, lw=2, label='finger offset |df| (rad)')
    ax.plot(ep.step, ep.res_rot_offset_rad.rolling(50, min_periods=10).mean(), color=C_ALT, lw=2, label='wrist rotation offset |dth| (rad)')
    style(ax, 'Mean residual angle offsets per step (rolling 50)', 'rad')
    ax = axes[1, 2]
    ax.plot(ep.step, ep.pushed_before_close.astype(float).rolling(50, min_periods=10).mean(), color=C_RES, lw=2,
            label='object pushed before close')
    ax.plot(ep.step, ep.sim_failure.astype(float).rolling(50, min_periods=10).mean(), color=C_ALT, lw=2, label='simulator failure')
    style(ax, 'Training failure indicators (rolling 50)', 'rate')
    ax.set_ylim(-0.02, 1.02)
    for a in axes.flat:
        a.legend(fontsize=8, frameon=False)
    fig.suptitle('Wave 2 residual SAC training (training episodes, stochastic policy; PRIVILEGED state)', fontsize=11, color=INK)
    fig.savefig(out / 'fig_training_curves.png', dpi=130)
    plt.close(fig)


def fig_validation(val: pd.DataFrame, zero_sel: pd.DataFrame, iv: pd.DataFrame | None, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    n = int(round(val.val_episodes.iloc[0] / 2))
    for j, ap in enumerate(APPROACHES):
        ax = axes[0, j]
        col = f'val_gt_{ap}'
        lo, hi = zip(*[wl(int(round(p * n)), n) for p in val[col]])
        ax.fill_between(val.step, lo, hi, color=C_RES, alpha=0.12, lw=0)
        ax.plot(val.step, val[col], color=C_RES, lw=2, marker='o', ms=4, label=f'residual SAC (n={n}/eval)')
        z = zero_sel[zero_sel.approach == ap].gt_success.mean()
        ax.axhline(z, color=REF, ls='--', lw=1.5, label=f'zero residual = scripted, same episodes ({z:.0%})')
        style(ax, f'In-training validation GT success, {ap} (selection set, seeds 8.0M)', 'GT success rate')
        if iv is not None:
            ax = axes[1, j]
            g = iv[(iv.approach == ap) & (iv.model != 'zero_residual')].dropna(subset=['step']).sort_values('step')
            g = g.drop_duplicates(['step'])
            ax.errorbar(g.step, g.gt_success, yerr=[g.gt_success - g.wilson_lo, g.wilson_hi - g.gt_success], fmt='o', ms=8,
                        color=C_RES, ecolor=C_RES, capsize=3, label='residual SAC checkpoints (n=50 each)')
            z = iv[(iv.approach == ap) & (iv.model == 'zero_residual')]
            if len(z):
                ax.axhline(float(z.gt_success.iloc[0]), color=REF, ls='--', lw=1.5,
                           label=f'zero residual = scripted, same episodes ({float(z.gt_success.iloc[0]):.0%})')
            style(ax, f'Independent validation GT success, {ap} (seeds 8.5M, 95% Wilson)', 'GT success rate')
    for a in axes.flat:
        a.set_ylim(-0.02, 1.02)
        a.legend(fontsize=8, frameon=False, loc='lower left')
    fig.suptitle('Wave 2 validation vs the zero-residual baseline (deterministic; PRIVILEGED state)', fontsize=11, color=INK)
    fig.savefig(out / 'fig_validation.png', dpi=130)
    plt.close(fig)


def fig_critic(q: pd.DataFrame, sb: pd.DataFrame, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    axes[0, 0].fill_between(q.step, q.q_p01, q.q_p99, color=C_RES, alpha=0.15, lw=0)
    axes[0, 0].plot(q.step, q.q_mean, color=C_RES, lw=2, label='mean (band p1-p99)')
    axes[0, 0].plot(q.step, q.q_max, color=C_ALT, lw=1.5, label='max of 1024 samples')
    style(axes[0, 0], 'Soft-Q (min of twin critics) on replay samples', 'Q')
    axes[0, 1].plot(q.step, q.ent_coef, color=C_RES, lw=2, label='alpha')
    axes[0, 1].set_yscale('log')
    style(axes[0, 1], 'Entropy coefficient', 'alpha')
    s = sb.dropna(subset=['train/critic_loss'])
    axes[1, 0].plot(s['time/total_timesteps'], s['train/critic_loss'].rolling(20, min_periods=1).mean(), color=C_RES, lw=2,
                    label='critic loss (rolling 20 log rows)')
    axes[1, 0].set_yscale('log')
    style(axes[1, 0], 'Critic loss', 'loss')
    axes[1, 1].plot(s['time/total_timesteps'], s['train/actor_loss'].rolling(20, min_periods=1).mean(), color=C_RES, lw=2,
                    label='actor loss (rolling 20 log rows)')
    style(axes[1, 1], 'Actor loss', 'loss')
    for a in axes.flat:
        a.legend(fontsize=8, frameon=False)
    fig.suptitle('Wave 2 critic / actor diagnostics', fontsize=11, color=INK)
    fig.savefig(out / 'fig_critic.png', dpi=130)
    plt.close(fig)


def fig_final(summ: pd.DataFrame, fam: pd.DataFrame, out):
    pols = [p for p in POLICY_ORDER if p in set(summ.policy)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.55 * len(pols) + 1.8), constrained_layout=True, sharey=True)
    for j, ap in enumerate(('all', 'top', 'oblique')):
        ax = axes[j]
        for i, p in enumerate(pols):
            r = summ[(summ.policy == p) & (summ.approach == ap)].iloc[0]
            c = C_RES if p.startswith('residual') else (REF if p in ('scripted', 'zero_residual') else C_ALT)
            ax.errorbar(r.gt_success, i, xerr=[[r.gt_success - r.wilson_lo], [r.wilson_hi - r.gt_success]], fmt='o', ms=8,
                        color=c, ecolor=c, capsize=3)
            ax.text(r.wilson_hi + 0.02, i, f'{r.gt_success:.0%} ({int(r.gt_success_n)}/{int(r.episodes)})', va='center',
                    fontsize=8, color=INK)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels([POLICY_LABEL[p] for p in pols], fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.set_title(f'Final test GT success, {ap} (95% Wilson)', fontsize=10, loc='left', color=INK)
        ax.grid(True, axis='x', color=GRID)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    fig.suptitle('Test seeds 5.1M (never used for selection); all policies PRIVILEGED', fontsize=10, color=INK)
    fig.savefig(out / 'fig_final_test.png', dpi=130)
    plt.close(fig)
    fams = sorted(fam.family.unique())
    sel = [p for p in ('scripted', 'residual_best', 'wave1_demo_best') if p in set(fam.policy)]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    w = 0.8 / len(sel)
    colors = {'scripted': REF, 'residual_best': C_RES, 'wave1_demo_best': C_ALT}
    for k, p in enumerate(sel):
        g = fam[fam.policy == p].set_index('family').loc[fams]
        x = np.arange(len(fams)) + (k - (len(sel) - 1) / 2) * w
        ax.bar(x, g.gt_success, width=w * 0.92, color=colors[p], label=POLICY_LABEL[p])
        ax.errorbar(x, g.gt_success, yerr=[g.gt_success - g.wilson_lo, g.wilson_hi - g.gt_success], fmt='none', ecolor=INK, capsize=2, lw=1)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Final test GT success by object family (top + oblique; 95% Wilson)', fontsize=10, loc='left', color=INK)
    ax.grid(True, axis='y', color=GRID)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(out / 'fig_final_test_by_family.png', dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ep = pd.read_csv(RUN / 'train_episodes.csv')
    val = pd.read_csv(RUN / 'validation.csv')
    q = pd.read_csv(RUN / 'qhealth.csv')
    sb = pd.read_csv(RUN / 'sb3' / 'progress.csv')
    ivs = []
    for f in sorted((RUN / 'independent_validation').glob('*/summary.csv')):
        ivs.append(pd.read_csv(f).assign(tag=f.parent.name))
    iv = pd.concat(ivs, ignore_index=True) if ivs else None
    ref_iv = {}
    if iv is not None:
        for ap in APPROACHES:
            z = iv[(iv.model == 'zero_residual') & (iv.approach == ap)]
            if len(z):
                ref_iv[ap] = float(z.gt_success.iloc[0])
    zero_sel = zero_on_selection_set()
    fig_training(ep, ref_iv, OUT)
    fig_validation(val, zero_sel, iv, OUT)
    fig_critic(q, sb, OUT)
    if iv is not None:
        iv.to_csv(OUT / 'table_independent_validation.csv', index=False)
    final = W2 / 'final_test'
    if (final / 'summary_by_approach.csv').exists():
        summ = pd.read_csv(final / 'summary_by_approach.csv')
        fam = pd.read_csv(final / 'summary_by_family.csv')
        fig_final(summ, fam, OUT)
    ep['w'] = (ep.step - 1) // 25_000
    rows = []
    for (w, ap), g in ep.groupby(['w', 'approach']):
        k, n = int(g.gt_success.sum()), len(g)
        lo, hi = wl(k, n)
        rows.append({'steps': f'{25 * w}k-{25 * (w + 1)}k', 'approach': ap, 'episodes': n, 'gt_success': k / n, 'wilson_lo': lo,
                     'wilson_hi': hi, 'pushed_before_close': float(g.pushed_before_close.mean()),
                     'sim_failures': int(g.sim_failure.sum()), 'res_abs_mean': float(g.res_abs_mean.mean()),
                     'res_pos_offset_cm': float(g.res_pos_offset_cm.mean()), 'res_finger_offset_rad': float(g.res_finger_offset_rad.mean())})
    pd.DataFrame(rows).to_csv(OUT / 'table_training_windows.csv', index=False)
    zs = {ap: float(zero_sel[zero_sel.approach == ap].gt_success.mean()) for ap in APPROACHES}
    zs['all'] = float(zero_sel.gt_success.mean())
    (OUT / 'reference_zero_selection_set.json').write_text(json.dumps(zs, indent=1), encoding='utf-8')
    print('zero residual on selection set:', zs)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('written to', OUT)


if __name__ == '__main__':
    main()
