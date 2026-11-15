"""Final-defense figures. Every number is read from frozen result files (nothing is typed in by hand except the
earlier-wave summary numbers, which are read from each wave's report tables listed in WAVES_SOURCE).
Outputs: report/figures/*.pdf (vector, for LaTeX) and presentation/assets/*.png (300 dpi).

    python visuograsp_v4/outputs/final_defense/build/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from visuograsp_v4.config import OUTPUT_ROOT  # noqa: E402

FD = OUTPUT_ROOT / 'final_defense'
FIG = FD / 'report' / 'figures'
ASSET = FD / 'presentation' / 'assets'
FIN = OUTPUT_ROOT / 'rl' / 'wave5' / 'final_test' / 'final'
for d in (FIG, ASSET):
    d.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3df'
CLS = {'expert': '#6b6a64', 'auto': '#eb6834', 'residual': '#eda100', 'imit': '#2a78d6', 'bcrl': '#1baf7a'}
NAMES = {'scripted': 'Scripted expert', 'bc4': 'BC Wave 4', 'e5g': 'Improved BC (E5g)', 'e5_ens': 'BC ensemble',
         'e5c_ep10': 'E5c (E6 init.)', 'e6': 'E6 BC-anchored RL', 'e6b': 'E6b BC-anchored RL'}
KIND = {'scripted': 'expert', 'bc4': 'imit', 'e5g': 'imit', 'e5_ens': 'imit', 'e5c_ep10': 'imit', 'e6': 'bcrl', 'e6b': 'bcrl'}
ORDER = ['scripted', 'bc4', 'e5g', 'e5_ens', 'e5c_ep10', 'e6', 'e6b']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 8.5, 'axes.titlecolor': INK,
                     'pdf.fonttype': 42})


def clean(ax, grid_axis='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(ASSET / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def final_results(s):
    """Overall GT success with Wilson CI (left) and top-down vs oblique (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={'width_ratios': [1.15, 1]}, sharey=True)
    for i, p in enumerate(ORDER):
        r = s[(s.policy == p) & (s.approach == 'all')].iloc[0]
        c = CLS[KIND[p]]
        axes[0].errorbar(100 * r.gt_success, i, xerr=[[100 * (r.gt_success - r.wilson_lo)], [100 * (r.wilson_hi - r.gt_success)]],
                         fmt='o', ms=6.5 if p == 'e6' else 5, color=c, ecolor=c, capsize=2.5, lw=1.4,
                         mec='white' if p != 'e6' else INK, mew=0.8)
        axes[0].text(100 * r.wilson_hi + 0.6, i, f'{100 * r.gt_success:.1f}%  [{100 * r.wilson_lo:.1f}, {100 * r.wilson_hi:.1f}]',
                     va='center', fontsize=7, color=INK, fontweight='bold' if p == 'e6' else 'normal')
        for ap, mk in (('top', 's'), ('oblique', '^')):
            q = s[(s.policy == p) & (s.approach == ap)].iloc[0]
            axes[1].plot(100 * q.gt_success, i, mk, color=c, ms=5.5, mec='white', mew=0.6)
        t = s[(s.policy == p) & (s.approach == 'top')].iloc[0].gt_success
        o = s[(s.policy == p) & (s.approach == 'oblique')].iloc[0].gt_success
        axes[1].plot([100 * o, 100 * t], [i, i], color=c, lw=1, alpha=0.6, zorder=0)
        axes[1].text(100 * o - 1.0, i, f'{100 * o:.1f}', ha='right', va='center', fontsize=6.5, color=INK2)
        axes[1].text(100 * t + 1.0, i, f'{100 * t:.1f}', ha='left', va='center', fontsize=6.5, color=INK2)
    sc = s[(s.policy == 'scripted') & (s.approach == 'all')].iloc[0].gt_success
    axes[0].axvline(100 * sc, color=CLS['expert'], ls=':', lw=1)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([NAMES[p] for p in ORDER])
    axes[0].invert_yaxis()
    axes[0].set_xlim(70, 93)
    axes[0].set_xlabel('GT grasp success, all 540 episodes (%)')
    axes[0].set_title('Overall success, 95% Wilson CI', loc='left')
    axes[1].set_xlim(48, 104)
    axes[1].set_xlabel('GT success (%)')
    axes[1].set_title('Oblique 45° (triangle) vs top-down (square), n = 270 each', loc='left')
    for a in axes:
        clean(a)
    save(fig, 'fig_final_results')


def paired(pt):
    rows = [('e6', 'e5c_ep10', 'E6 − its BC init. (RL)'), ('e6b', 'e5g', 'E6b − its BC init. (RL)'),
            ('e5_ens', 'bc4', 'BC ensemble − BC Wave 4'), ('e5g', 'bc4', 'Improved BC − BC Wave 4'),
            ('e6', 'bc4', 'E6 − BC Wave 4'), ('e6', 'e5_ens', 'E6 − BC ensemble'), ('e6', 'scripted', 'E6 − scripted expert'),
            ('e5_ens', 'scripted', 'BC ensemble − scripted expert')]
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    for i, (a, b, lab) in enumerate(rows):
        r = pt[(pt.A == a) & (pt.B == b) & (pt.group == 'all')].iloc[0]
        sig = r.holm_p < 0.05
        c = CLS['bcrl'] if a in ('e6', 'e6b') else CLS['imit']
        ax.errorbar(100 * r.diff_a_minus_b, i, xerr=[[100 * (r.diff_a_minus_b - r.diff_lo)], [100 * (r.diff_hi - r.diff_a_minus_b)]],
                    fmt='o', ms=6, color=c, ecolor=c, capsize=2.5, lw=1.4, mfc=c if sig else 'white', mew=1.2)
        ax.text(100 * max(r.diff_hi, 0) + 0.4, i, f'{100 * r.diff_a_minus_b:+.1f} pp   wins/losses {int(r.only_a)}/{int(r.only_b)}   '
                f'McNemar p = {r.mcnemar_exact_p:.2g}, Holm p = {r.holm_p:.2g}', va='center', fontsize=7,
                color=INK, fontweight='bold' if sig else 'normal')
    ax.axvline(0, color=INK2, lw=0.9, ls='--')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-4, 19)
    ax.set_xlabel('Paired difference in GT success on the same 540 episodes (percentage points), 95% bootstrap CI')
    ax.set_title('Paired comparisons (filled marker = significant after Holm correction over 12 pre-listed tests)', loc='left')
    clean(ax)
    save(fig, 'fig_paired')


WAVES_SOURCE = {
    'wave2/report/report.md §4': 'W1 SAC scratch 0.0 %, W1 SAC + demo replay 10.5 %, W2 residual SAC 49.4 %, scripted 77.8 % (5.1M, n = 324)',
    'wave3/report/report.md': 'W3 conservative residual SAC 69.1 %, scripted 76.7 % (6.1M, n = 540)',
    'wave4/report/report.md §4': 'W4 curriculum SAC 18.9 %, BC + SAC (E2b) 48.7 %, BC 74.6 %, scripted 77.8 % (6.6M, n = 540)',
    'wave5/final_test/final/analysis/summary.csv': 'W5 BC ensemble 80.7 %, E6 82.0 %, scripted 80.6 % (6.7M, n = 540)'}
PROG = [  # label, wave, class, learned %, matched scripted %, test set
    ('SAC from scratch', 'W1', 'auto', 0.0, 77.8, '5.1M, n=324'),
    ('SAC + demo replay', 'W1', 'auto', 10.5, 77.8, '5.1M, n=324'),
    ('Residual SAC', 'W2', 'residual', 49.4, 77.8, '5.1M, n=324'),
    ('Conservative residual SAC', 'W3', 'residual', 69.1, 76.7, '6.1M, n=540'),
    ('Curriculum SAC', 'W4', 'auto', 18.9, 77.8, '6.6M, n=540'),
    ('BC + unrestricted SAC fine-tuning', 'W4', 'bcrl', 48.7, 77.8, '6.6M, n=540'),
    ('Behaviour cloning (BC)', 'W4', 'imit', 74.6, 77.8, '6.6M, n=540'),
    ('BC ensemble', 'W5', 'imit', None, None, '6.7M, n=540'),
    ('E6 BC-anchored RL', 'W5', 'bcrl', None, None, '6.7M, n=540')]


def progression(s):
    al = s[s.approach == 'all'].set_index('policy').gt_success * 100
    rows = []
    for lab, w, k, v, sc, ts in PROG:
        if lab == 'BC ensemble':
            v, sc = al['e5_ens'], al['scripted']
        if lab.startswith('E6'):
            v, sc = al['e6'], al['scripted']
        rows.append((lab, w, k, float(v), float(sc), ts))
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    for i, (lab, w, k, v, sc, ts) in enumerate(rows):
        ax.plot([v, sc], [i, i], color=GRID, lw=3, zorder=0, solid_capstyle='round')
        ax.plot(sc, i, 'D', color=CLS['expert'], ms=4.5, zorder=2)
        ax.plot(v, i, 'o', color=CLS[k], ms=6.5, zorder=3, mec='white', mew=0.7)
        d = v - sc
        ax.text(101.5, i, f'{v:.1f}% vs {sc:.1f}%  ({d:+.1f} pp)', va='center', fontsize=7, color=INK,
                fontweight='bold' if lab.startswith('E6') else 'normal')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f'{r[1]}  {r[0]}' for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-3, 100)
    ax.set_xlabel('Full-task GT success on each wave\'s own fresh test set (%); grey diamond = scripted expert on the same episodes')
    ax.set_title('Learning progression: best learned controllers vs the matched scripted expert (absolute rates differ between test sets)',
                 loc='left')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', color=CLS[k], label=l, ms=6) for k, l in
         (('auto', 'autonomous / curriculum RL'), ('residual', 'residual RL on scripted controller'), ('imit', 'imitation (BC)'),
          ('bcrl', 'BC + RL fine-tuning'))] + [Line2D([], [], marker='D', ls='', color=CLS['expert'], label='scripted expert', ms=5)]
    ax.legend(handles=h, fontsize=6.5, frameon=False, loc='lower left', ncol=1, bbox_to_anchor=(0.0, 0.0))
    clean(ax)
    save(fig, 'fig_progression')
    pd.DataFrame(rows, columns=['method', 'wave', 'class', 'learned_pct', 'scripted_pct', 'test_set']).to_csv(FD / 'evidence' / 'progression_table.csv', index=False)


def failures(e):
    cats = [('never_lifted', 'never lifted', '#8c8b85'), ('object_pushed_before_close', 'pushed before closing', '#eb6834'),
            ('no_contact', 'no contact', '#c9c7bf'), ('lifted_then_dropped', 'lifted, then dropped', '#2a78d6'),
            ('held_without_pad_pair', 'held without thumb–pad pair', '#e87ba4')]
    f = e[~e.gt_success.astype(bool)]
    tab = f.groupby(['policy', 'outcome']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    left = np.zeros(len(ORDER))
    for key, lab, col in cats:
        v = np.array([tab.loc[p, key] if key in tab.columns else 0 for p in ORDER])
        ax.barh(range(len(ORDER)), v, left=left, color=col, label=lab, height=0.68, edgecolor='white', linewidth=0.8)
        left += v
    other = np.array([len(f[f.policy == p]) for p in ORDER]) - left
    assert (other == 0).all(), other
    for i, p in enumerate(ORDER):
        ax.text(left[i] + 1.5, i, f'{int(left[i])}', va='center', fontsize=7, color=INK)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NAMES[p] for p in ORDER], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Failed episodes out of 540 (by outcome)')
    ax.set_xlim(0, 135)
    ax.legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.38, -0.2), ncol=2, columnspacing=1.0,
              handlelength=1.2)
    clean(ax)
    save(fig, 'fig_failures')
    tab.to_csv(FD / 'evidence' / 'failure_outcomes_by_policy.csv')


def box(ax, x, y, w, h, text, fc, ec, fs=7.2, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.025', fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, fontweight='bold' if bold else 'normal',
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, col=INK2, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1, ls=ls, shrinkA=1, shrinkB=1))


def architecture():
    fig, ax = plt.subplots(figsize=(7.16, 3.05))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1)
    ax.axis('off')
    P, PE = '#fde8dc', '#eb6834'      # privileged simulator information
    L, LE = '#dff3ea', '#1baf7a'      # learned
    S, SE = '#eef0f3', '#5f5e58'      # physics / control
    X, XE = '#e7effa', '#2a78d6'      # expert / data
    fs = 6.5
    y0, h0 = 0.50, 0.33                                      # control-loop row
    box(ax, 0.005, y0, 0.15, h0, 'MuJoCo physics\n2.5 ms step\n20 substeps / action\nhand, object, table', S, SE, fs)
    box(ax, 0.195, y0, 0.175, h0, 'PRIVILEGED inputs\nobject pose + extents\nGT pad contacts\n(simulator truth)', P, PE, fs)
    box(ax, 0.41, y0, 0.175, h0, '130-D observation\nfingers 48 · wrist 15\ntips 12 · object 15\ngrasp 9 · contact 8\nprev. action 22 · vis. 1',
        S, SE, fs)
    box(ax, 0.625, y0, 0.15, h0, 'Actor policy\nMLP 256×256\na = tanh μ(s)\n22-D action', L, LE, fs, bold=True)
    box(ax, 0.815, y0, 0.18, h0, 'Hand controller\n16 finger Δ-targets\n6-DoF mocap target\n+ weld constraint', S, SE, fs)
    for xa, xb in ((0.155, 0.195), (0.37, 0.41), (0.585, 0.625), (0.775, 0.815)):
        arrow(ax, xa, y0 + h0 / 2, xb, y0 + h0 / 2)
    ax.plot([0.905, 0.905, 0.08, 0.08], [y0 + h0, 0.93, 0.93, y0 + h0 + 0.012], color=INK2, lw=1.1)
    arrow(ax, 0.08, 0.90, 0.08, y0 + h0 + 0.004)
    ax.text(0.49, 0.955, 'bounded action every 50 ms; the physics moves the hand through the weld constraint', ha='center',
            fontsize=fs, color=INK2)
    y1, h1 = 0.08, 0.27                                      # not-learned / training row
    box(ax, 0.41, y1, 0.175, h1, 'Analytic grasp planner\ntop / 45° / 90° poses\n(PRIVILEGED pose)', X, XE, fs)
    box(ax, 0.625, y1, 0.15, h1, 'Scripted expert\nphase machine\n(demos, baseline)', X, XE, fs)
    box(ax, 0.815, y1, 0.18, h1, 'Offline / RL training\nBC · SAC ·\nAWAC-style E6', L, LE, fs)
    arrow(ax, 0.4975, y1 + h1, 0.4975, y0, col=XE, ls=':')
    arrow(ax, 0.585, y1 + h1 / 2, 0.625, y1 + h1 / 2, col=XE)
    arrow(ax, 0.775, y1 + h1 / 2, 0.815, y1 + h1 / 2, col=XE, ls='--')
    arrow(ax, 0.86, y1 + h1, 0.72, y0, col=LE, ls='--')
    ax.text(0.505, 0.42, 'grasp-pose\nfeatures', fontsize=5.8, color=XE, va='center')
    ax.text(0.79, 0.415, 'weights', fontsize=5.8, color=LE, va='center', ha='right')
    ax.text(0.005, y1 + h1 - 0.01, 'Evaluation only (never in the actor input):\nGT success = COM ≥ 10 cm for 3 s with the thumb\n'
            'pad + one more pad ≥ 0.2 N.\nCameras / colour segmentation / D0 exist in V4 but\nare NOT used by the final policies '
            '(obs_mode =\nprivileged, contact_source = gt_oracle).', fontsize=6.0, color=INK2, va='top', linespacing=1.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=P, ec=PE, label='privileged simulator information'), Patch(fc=L, ec=LE, label='learned component'),
                       Patch(fc=X, ec=XE, label='scripted / analytic (not learned)'), Patch(fc=S, ec=SE, label='physics, interface, control')],
              fontsize=6.2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=4)
    save(fig, 'fig_architecture')


def main():
    (FD / 'evidence').mkdir(exist_ok=True)
    s = pd.read_csv(FIN / 'analysis' / 'summary.csv')
    pt = pd.read_csv(FIN / 'analysis' / 'paired_all.csv')
    e = pd.read_csv(FIN / 'episodes.csv')
    final_results(s)
    paired(pt)
    progression(s)
    failures(e)
    architecture()
    (FD / 'evidence' / 'figure_sources.json').write_text(json.dumps({
        'fig_final_results': 'wave5/final_test/final/analysis/summary.csv',
        'fig_paired': 'wave5/final_test/final/analysis/paired_all.csv (group = all)',
        'fig_failures': 'wave5/final_test/final/episodes.csv (gt_success = False, by outcome)',
        'fig_progression': WAVES_SOURCE, 'fig_architecture': 'diagram of env.py / controller.py / scripted.py / grasp_planner.py'}, indent=1))
    print('figures written to', FIG, 'and', ASSET)


if __name__ == '__main__':
    main()
