"""Wave 1 report: learning curves, validation tables with Wilson CIs, scratch vs demo comparison.

Reads only run outputs (train_episodes.csv, validation*.csv, qhealth.csv, sb3/progress.csv,
independent_validation/*/summary.csv, finished.json, control_log.csv) and writes into
outputs/rl/wave1/report/. Nothing is re-run here.
PRIVILEGED setting for every number: simulator object pose in the actor observation, GT-oracle
contacts; the demo arm also uses scripted demonstrations planned from the privileged pose.

    python -m visuograsp_v4.scripts.wave1_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts.evaluate_policy import wilson

WAVE = OUTPUT_ROOT / 'rl' / 'wave1'
RUNS = {'scratch': WAVE / 'sac_scratch_s11', 'demo': WAVE / 'sac_demo_s11'}
LABEL = {'scratch': 'SAC scratch', 'demo': 'SAC + demos'}
COLOR = {'scratch': '#2a78d6', 'demo': '#eb6834'}       # validated categorical slots 1-2 (light surface)
REF = '#6b6a64'                                         # neutral gray: scripted reference (not a series)
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
APPROACHES = ('top', 'oblique')
OUTCOMES = ('success_retained', 'success_lost_later', 'held_without_pad_pair', 'lifted_then_dropped',
            'partial_lift', 'never_lifted', 'object_pushed_before_close', 'no_contact', 'object_left_table',
            'sim_failure')


def style(ax, title, ylabel=None, xlabel='environment steps'):
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:g}k'))


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load() -> dict:
    data = {}
    for arm, run in RUNS.items():
        d = {'run': run}
        d['ep'] = read(run / 'train_episodes.csv')
        d['val'] = read(run / 'validation.csv')
        d['q'] = read(run / 'qhealth.csv')
        d['sb3'] = read(run / 'sb3' / 'progress.csv')
        d['ctrl'] = read(run / 'control_log.csv')
        d['finished'] = json.loads((run / 'finished.json').read_text()) if (run / 'finished.json').exists() else None
        d['cfg'] = json.loads((run / 'run_config.json').read_text()) if (run / 'run_config.json').exists() else None
        ivs = []
        for f in sorted((run / 'independent_validation').glob('*/summary.csv')):
            if f.parent.name.startswith('sensitivity'):
                continue
            ivs.append(pd.read_csv(f).assign(tag=f.parent.name, arm=arm))
        d['iv'] = pd.concat(ivs, ignore_index=True) if ivs else None
        ive = []
        for f in sorted((run / 'independent_validation').glob('*/episodes.csv')):
            if f.parent.name.startswith('sensitivity'):
                continue
            ive.append(pd.read_csv(f).assign(tag=f.parent.name, arm=arm))
        d['iv_ep'] = pd.concat(ive, ignore_index=True) if ive else None
        data[arm] = d
    ref = WAVE / 'independent_validation_scripted' / 'scripted_reference' / 'summary.csv'
    data['scripted_ref'] = read(ref)
    return data


def fig_training(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for arm in RUNS:
        ep = data[arm]['ep']
        if ep is None or ep.empty:
            continue
        ep = ep.sort_values('step')
        ep['partial'] = (ep.max_lift_cm >= 3.5).astype(float)
        c = COLOR[arm]
        axes[0, 0].plot(ep.step, ep['return'].rolling(50, min_periods=10).mean(), color=c, lw=2, label=LABEL[arm])
        axes[1, 0].plot(ep.step, ep.max_lift_cm.rolling(50, min_periods=10).median(), color=c, lw=2, label=LABEL[arm])
        for j, ap in enumerate(APPROACHES):
            g = ep[ep.approach == ap]
            axes[0, j + 1].plot(g.step, g.gt_success.rolling(50, min_periods=10).mean(), color=c, lw=2, label=LABEL[arm])
            axes[1, j + 1].plot(g.step, g.partial.rolling(50, min_periods=10).mean(), color=c, lw=2, label=LABEL[arm])
    style(axes[0, 0], 'Training episode return (rolling 50 episodes)', 'return')
    style(axes[1, 0], 'Training max object lift, rolling MEDIAN of 50\n(mean is dominated by objects launched in numerical failures)', 'cm')
    for j, ap in enumerate(APPROACHES):
        style(axes[0, j + 1], f'Training GT success, {ap} (rolling 50 episodes)', 'success rate')
        style(axes[1, j + 1], f'Training partial lift >= 3.5 cm, {ap} (rolling 50)', 'rate')
        for i in range(2):
            axes[i, j + 1].set_ylim(-0.02, 1.02)
    for ax in axes.flat:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Wave 1 training curves (stochastic policy, training episodes; PRIVILEGED state)', fontsize=11, color=INK)
    fig.savefig(out / 'fig_training_curves.png', dpi=130)
    plt.close(fig)


def fig_validation(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharey=True)
    ref = data['scripted_ref']
    for arm in RUNS:
        c = COLOR[arm]
        val = data[arm]['val']
        if val is not None:
            for j, ap in enumerate(APPROACHES):
                col = f'val_gt_{ap}'
                if col in val:
                    n = int(round(val.val_episodes.iloc[0] / 2)) if 'val_episodes' in val else 25
                    lo, hi = zip(*[wilson(int(round(p * n)), n) for p in val[col]])
                    axes[0, j].fill_between(val.step, lo, hi, color=c, alpha=0.12, lw=0)
                    axes[0, j].plot(val.step, val[col], color=c, lw=2, marker='o', ms=4, label=f'{LABEL[arm]} (n={n}/eval)')
        iv = data[arm]['iv']
        if iv is not None:
            iv = iv.dropna(subset=['step']).drop_duplicates(['model', 'step', 'approach']).sort_values('step')
            for j, ap in enumerate(APPROACHES):
                g = iv[iv.approach == ap]
                if g.empty:
                    continue
                off = -1500 if arm == 'scratch' else 1500
                axes[1, j].errorbar(g.step + off, g.gt_success, yerr=[g.gt_success - g.wilson_lo, g.wilson_hi - g.gt_success],
                                    fmt='o', ms=8, color=c, ecolor=c, elinewidth=1.5, capsize=3,
                                    label=f'{LABEL[arm]} (n={int(g.episodes.iloc[0])}/checkpoint)')
    for j, ap in enumerate(APPROACHES):
        style(axes[0, j], f'In-training validation GT success, {ap} (model selection set 8.0M)', 'GT success rate')
        style(axes[1, j], f'Independent validation GT success, {ap} (seeds 8.5M, 95% Wilson CI)', 'GT success rate')
        if ref is not None:
            r = ref[ref.approach == ap]
            if not r.empty:
                axes[1, j].axhline(float(r.gt_success.iloc[0]), color=REF, ls='--', lw=1.5,
                                   label=f'scripted, PRIVILEGED planner ({float(r.gt_success.iloc[0]):.0%})')
        axes[1, j].axhline(0.20, color=INK2, ls=':', lw=1, label='20% gate threshold')
        for i in range(2):
            axes[i, j].set_ylim(-0.02, 1.02)
            axes[i, j].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle('Wave 1 validation (deterministic policy; PRIVILEGED state)', fontsize=11, color=INK)
    fig.savefig(out / 'fig_validation.png', dpi=130)
    plt.close(fig)


def fig_critic(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    for arm in RUNS:
        c = COLOR[arm]
        q, sb = data[arm]['q'], data[arm]['sb3']
        if q is not None:
            axes[0, 0].fill_between(q.step, q.q_p01, q.q_p99, color=c, alpha=0.12, lw=0)
            axes[0, 0].plot(q.step, q.q_mean, color=c, lw=2, label=f'{LABEL[arm]} mean (band p1-p99)')
            axes[0, 1].plot(q.step, q.ent_coef, color=c, lw=2, label=LABEL[arm])
        if sb is not None and 'train/critic_loss' in sb:
            s = sb.dropna(subset=['train/critic_loss'])
            axes[1, 0].plot(s['time/total_timesteps'], s['train/critic_loss'].rolling(20, min_periods=1).mean(),
                            color=c, lw=2, label=LABEL[arm])
            axes[1, 1].plot(s['time/total_timesteps'], s['train/actor_loss'].rolling(20, min_periods=1).mean(),
                            color=c, lw=2, label=LABEL[arm])
    style(axes[0, 0], 'Soft-Q (min of twin critics) on replay samples', 'Q')
    style(axes[0, 1], 'Entropy coefficient alpha', 'alpha')
    axes[0, 1].set_yscale('log')
    style(axes[1, 0], 'Critic loss (last update per log row, rolling 20 rows)', 'loss')
    axes[1, 0].set_yscale('log')
    style(axes[1, 1], 'Actor loss (rolling 20 rows)', 'loss')
    for ax in axes.flat:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Wave 1 critic / actor diagnostics', fontsize=11, color=INK)
    fig.savefig(out / 'fig_critic.png', dpi=130)
    plt.close(fig)


def fig_outcomes(data, out, rows):
    """rows: list of (arm, tag, model) whose independent-validation outcomes are shown."""
    if not rows:
        return
    cmap = plt.get_cmap('tab20')
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * len(rows) + 2), constrained_layout=True, sharey=True)
    for j, ap in enumerate(APPROACHES):
        ax = axes[j]
        labels = []
        for i, (arm, tag, model) in enumerate(rows):
            e = data[arm]['iv_ep']
            g = e[(e.tag == tag) & (e.model == model) & (e.approach == ap)]
            n = len(g)
            left = 0.0
            for k, oc in enumerate(OUTCOMES):
                v = float((g.outcome == oc).sum()) / max(1, n)
                if v > 0:
                    ax.barh(i, v, left=left, color=cmap(k), edgecolor='white', linewidth=1.5,
                            label=oc if i == 0 or oc not in ax.get_legend_handles_labels()[1] else None)
                    if v >= 0.08:
                        ax.text(left + v / 2, i, f'{v:.0%}', ha='center', va='center', fontsize=7, color=INK)
                    left += v
            labels.append(f'{LABEL[arm]} | {model} (n={n})')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(f'Independent validation outcomes, {ap}', fontsize=10, loc='left', color=INK)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8)
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.savefig(out / 'fig_outcomes.png', dpi=130, bbox_inches='tight')
    plt.close(fig)


def training_windows(data) -> pd.DataFrame:
    rows = []
    for arm in RUNS:
        ep = data[arm]['ep']
        if ep is None:
            continue
        ep = ep.assign(window=(ep.step - 1) // 25_000)
        for (w, ap), g in ep.groupby(['window', 'approach']):
            k, n = int(g.gt_success.sum()), len(g)
            lo, hi = wilson(k, n)
            r = {'arm': arm, 'steps': f'{25 * w}k-{25 * (w + 1)}k', 'approach': ap, 'episodes': n,
                 'gt_success': k / n, 'wilson_lo': max(0.0, lo), 'wilson_hi': min(1.0, hi),
                 'partial_lift_rate': float((g.max_lift_cm >= 3.5).mean()), 'median_max_lift_cm': float(g.max_lift_cm.median()),
                 'mean_return': float(g['return'].mean()),
                 'sim_failures': int(g.sim_failure.sum()) if 'sim_failure' in g else None}
            if 'outcome' in g:
                for oc, c in g.outcome.value_counts().items():
                    r[f'outcome_{oc}'] = int(c)
            rows.append(r)
    return pd.DataFrame(rows)


def run_table(data) -> pd.DataFrame:
    rows = []
    for arm in RUNS:
        f, cfg, ctrl = data[arm]['finished'], data[arm]['cfg'], data[arm]['ctrl']
        q = data[arm]['q']
        r = {'arm': arm, 'privileged': 'object pose + GT contacts' + (' + scripted demos' if arm == 'demo' else '')}
        if cfg:
            r['started'] = cfg['started']
            if 'demo_prefill' in cfg:
                r['demo_transitions'] = cfg['demo_prefill']['transitions']
                r['demo_episodes'] = cfg['demo_prefill']['episodes']
                r['demo_scripted_gt'] = cfg['demo_prefill']['scripted_gt_success']
                r['demo_verification_passed'] = cfg.get('demo_verification_passed')
        if f:
            r.update(steps_completed=f['steps'], stop_reason=f['stop_reason'], finished=f.get('finished'),
                     wall_hours=f['seconds_wall'] / 3600, training_hours_excl_waits=f['seconds_training_excl_waits'] / 3600,
                     waiting_hours=f['seconds_waiting_gate_or_pause'] / 3600, steps_per_s=f['steps_per_s_training'],
                     best_step_in_training_val=f['best_step'], best_in_training_val_gt=f['best_validation_gt'])
        elif q is not None:
            r.update(steps_completed=int(q.step.max()), stop_reason='running')
        if q is not None:
            r.update(q_mean_last=float(q.q_mean.iloc[-1]), q_max_overall=float(q.q_max.max()),
                     q_min_overall=float(q.q_min.min()), ent_coef_last=float(q.ent_coef.iloc[-1]))
        if ctrl is not None:
            r['control_events'] = ';'.join(ctrl.event.astype(str))
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    out = WAVE / 'report'
    out.mkdir(parents=True, exist_ok=True)
    data = load()
    fig_training(data, out)
    fig_validation(data, out)
    fig_critic(data, out)
    runs = run_table(data)
    runs.to_csv(out / 'table_runs.csv', index=False)
    tw = training_windows(data)
    tw.to_csv(out / 'table_training_windows.csv', index=False)
    ivs = [data[a]['iv'] for a in RUNS if data[a]['iv'] is not None]
    iv = pd.concat(ivs, ignore_index=True) if ivs else pd.DataFrame()
    if data['scripted_ref'] is not None:
        iv = pd.concat([iv, data['scripted_ref'].assign(arm='scripted_PRIVILEGED', tag='scripted_reference')],
                       ignore_index=True)
    iv.to_csv(out / 'table_independent_validation.csv', index=False)
    outcome_rows = []
    for arm in RUNS:
        e = data[arm]['iv_ep']
        if e is None:
            continue
        last_tag = sorted(e.tag.unique())[-1]
        for model in e[e.tag == last_tag].model.unique():
            outcome_rows.append((arm, last_tag, model))
    fig_outcomes(data, out, outcome_rows)
    pd.set_option('display.width', 250)
    print(runs.T.to_string())
    cols = ['arm', 'tag', 'model', 'step', 'approach', 'episodes', 'gt_success_n', 'gt_success', 'wilson_lo',
            'wilson_hi', 'partial_lift_rate', 'mean_max_lift_cm']
    if not iv.empty:
        print(iv[[c for c in cols if c in iv]].round(3).to_string(index=False))
    print('written to', out)


if __name__ == '__main__':
    main()
