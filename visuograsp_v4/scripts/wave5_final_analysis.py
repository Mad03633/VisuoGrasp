"""Wave 5 final-test analysis (reads final_test/final/episodes.csv; nothing is re-run).

Paired comparisons on the identical 540 fresh episodes (seeds 6.7M): McNemar exact + paired bootstrap CI.
  imitation contribution : e5g vs bc4, e5_ens vs bc4
  RL contribution        : e6 vs e5c_ep10 (its initialisation), e6b vs e5g (its initialisation)
  vs expert              : every learned policy vs scripted
Also per family and approach, outcome counts, numerical-failure rates, Holm-adjusted p over the 12 overall
comparisons, and the decision rule of final_test/final_decision_rule.json (written before the results). PRIVILEGED setting.

    python -m visuograsp_v4.scripts.wave5_final_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.scripts.evaluate_policy import wilson
from visuograsp_v4.scripts.paired_compare import paired_stats, paired_table

FIN = OUTPUT_ROOT / 'rl' / 'wave5' / 'final_test' / 'final'
ORDER = ['scripted', 'bc4', 'e5g', 'e5_ens', 'e5c_ep10', 'e6', 'e6b']
LEARNED = ['bc4', 'e5g', 'e5_ens', 'e5c_ep10', 'e6', 'e6b']
SIMPLICITY = {'bc4': 0, 'e5g': 0, 'e5c_ep10': 0, 'e5_ens': 1, 'e6': 2, 'e6b': 2}
PAIRS = [('e5g', 'bc4', 'imitation: improved BC vs Wave 4 BC'), ('e5_ens', 'bc4', 'imitation: BC ensemble vs Wave 4 BC'),
         ('e6', 'e5c_ep10', 'RL: E6 vs its BC initialisation'), ('e6b', 'e5g', 'RL: E6b vs its BC initialisation'),
         ('bc4', 'scripted', 'vs expert'), ('e5g', 'scripted', 'vs expert'), ('e5_ens', 'scripted', 'vs expert'),
         ('e5c_ep10', 'scripted', 'vs expert'), ('e6', 'scripted', 'vs expert'), ('e6b', 'scripted', 'vs expert'),
         ('e6', 'e5_ens', 'best RL vs best imitation'), ('e6', 'bc4', 'E6 vs Wave 4 BC')]


def wl(k, n):
    lo, hi = wilson(k, n)
    return max(0.0, lo), min(1.0, hi)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def verify(e: pd.DataFrame) -> dict:
    from visuograsp_v4.scripts.wave2_final_test import grid as test_grid
    keys = {(int(g[0]), g[4]) for g in test_grid(V4Config(), 10, 6_700_000)}
    per = {}
    for p in ORDER:
        k = list(zip(e[e.policy == p].seed.astype(int), e[e.policy == p].approach))
        per[p] = {'rows': len(k), 'unique': len(set(k)), 'missing': len(keys - set(k)), 'extra': len(set(k) - keys)}
    uniq = len(e.drop_duplicates(['policy', 'seed', 'approach']))
    ok = (len(e) == 3780 and uniq == 3780 and set(e.policy) == set(ORDER)
          and all(v['rows'] == 540 and v['unique'] == 540 and v['missing'] == 0 and v['extra'] == 0 for v in per.values()))
    rep = {'rows': len(e), 'unique_policy_seed_approach': uniq, 'grid_keys': len(keys), 'per_policy': per, 'ok': bool(ok)}
    if not ok:
        raise SystemExit(f'integrity check failed: {json.dumps(rep)}')
    return rep


def decide(e, s, pt):
    """Apply final_decision_rule.json (pre-written) to the results."""
    al = s[s.approach == 'all'].set_index('policy')
    ob = s[s.approach == 'oblique'].set_index('policy')
    rank = sorted(LEARNED, key=lambda p: (-al.loc[p, 'gt_success_n'], -ob.loc[p, 'gt_success_n'], al.loc[p, 'sim_failures'], SIMPLICITY[p]))
    best = rank[0]
    g = lambda n: e[e.policy == n].sort_values(['seed', 'approach']).reset_index(drop=True)
    vs = {}
    for o in [p for p in ORDER if p != best]:
        a, b = g(best), g(o)
        assert (a.seed.values == b.seed.values).all() and (a.approach.values == b.approach.values).all()
        st = paired_stats(a.gt_success, b.gt_success)
        if o == 'scripted':
            verdict = 'exceeds' if st['diff_lo'] > 0 else ('below' if st['diff_hi'] < 0 else 'not significantly different')
        else:
            verdict = ('significantly stronger' if st['mcnemar_exact_p'] < 0.05 and st['diff_a_minus_b'] > 0 else
                       'significantly weaker' if st['mcnemar_exact_p'] < 0.05 else 'not statistically distinguishable')
        vs[o] = {'diff': st['diff_a_minus_b'], 'diff_ci': [st['diff_lo'], st['diff_hi']], 'only_best': st['only_a'],
                 'only_other': st['only_b'], 'mcnemar_exact_p': st['mcnemar_exact_p'], 'verdict': verdict}

    def contrib(a, b):
        r = pt[(pt.A == a) & (pt.B == b) & (pt.group == 'all')].iloc[0]
        d, p = float(r.diff_a_minus_b), float(r.mcnemar_exact_p)
        v = ('beneficial' if d > 0 and p < 0.05 else 'degraded' if d < 0 and p < 0.05 else
             'inconclusive (positive, p >= 0.05)' if d > 0 else 'inconclusive (no gain)')
        return {'diff': d, 'diff_ci': [float(r.diff_lo), float(r.diff_hi)], 'only_a': int(r.only_a), 'only_b': int(r.only_b),
                'mcnemar_exact_p': p, 'holm_p': float(r.holm_p), 'verdict': v}
    k, n = int(al.loc[best, 'gt_success_n']), int(al.loc[best, 'episodes'])
    return {'rule_file': 'final_test/final_decision_rule.json', 'ranking_learned': rank,
            'ranking_gt_success': {p: float(al.loc[p, 'gt_success']) for p in rank},
            'strongest_learned_policy': best, 'gt_success': k / n, 'wilson': list(wl(k, n)),
            'top': float(s[(s.policy == best) & (s.approach == 'top')].gt_success.iloc[0]),
            'oblique': float(ob.loc[best, 'gt_success']), 'best_vs_others': vs,
            'rl_contribution': {'e6_vs_e5c_ep10': contrib('e6', 'e5c_ep10'), 'e6b_vs_e5g': contrib('e6b', 'e5g')},
            'imitation_contribution': {'e5g_vs_bc4': contrib('e5g', 'bc4'), 'e5_ens_vs_bc4': contrib('e5_ens', 'bc4')},
            'target_80': {'point_estimate_reached': k / n >= 0.80, 'wilson_lower_bound_reached': wl(k, n)[0] >= 0.80}}


def main() -> None:
    e = pd.read_csv(FIN / 'episodes.csv')
    out = FIN / 'analysis'
    out.mkdir(exist_ok=True)
    integ = verify(e)
    (out / 'integrity.json').write_text(json.dumps(integ, indent=1))
    g = lambda n: e[e.policy == n]
    rows = []
    for a, b, what in PAIRS:
        t = paired_table(g(a), g(b), a, b, by=('approach', 'family'))
        t.insert(2, 'comparison', what)
        rows.append(t)
    pt = pd.concat(rows, ignore_index=True)
    m = pt.group == 'all'
    pt['holm_p'] = np.nan
    pt.loc[m, 'holm_p'] = holm(pt.loc[m, 'mcnemar_exact_p'])
    pt.to_csv(out / 'paired_all.csv', index=False)
    summ = []
    for p, gg in e.groupby('policy'):
        for ap, h in [('all', gg)] + list(gg.groupby('approach')):
            k, n = int(h.gt_success.sum()), len(h)
            lo, hi = wl(k, n)
            f = int(h.sim_failure.sum())
            flo, fhi = wl(f, n)
            summ.append({'policy': p, 'approach': ap, 'episodes': n, 'gt_success_n': k, 'gt_success': k / n, 'wilson_lo': lo,
                         'wilson_hi': hi, 'sim_failures': f, 'sim_failure_rate': f / n, 'sim_fail_wilson_lo': flo,
                         'sim_fail_wilson_hi': fhi, 'object_runaway': int(h.object_runaway.sum()),
                         'object_off_table': int(h.object_off_table.sum())})
    s = pd.DataFrame(summ)
    s.to_csv(out / 'summary.csv', index=False)
    fr = []
    for (p, f, ap), h in [((p, f, 'all'), h) for (p, f), h in e.groupby(['policy', 'family'])] + list(e.groupby(['policy', 'family', 'approach'])):
        k, n = int(h.gt_success.sum()), len(h)
        lo, hi = wl(k, n)
        fr.append({'policy': p, 'family': f, 'approach': ap, 'episodes': n, 'gt_success_n': k, 'gt_success': k / n,
                   'wilson_lo': lo, 'wilson_hi': hi, 'sim_failures': int(h.sim_failure.sum())})
    pd.DataFrame(fr).to_csv(out / 'by_family_long.csv', index=False)
    fam = e.groupby(['policy', 'family']).gt_success.mean().unstack().round(3)
    fam.to_csv(out / 'by_family.csv')
    oc = e.groupby('policy').outcome.value_counts().unstack(fill_value=0)
    oc.to_csv(out / 'outcomes.csv')
    e.groupby(['policy', 'approach']).outcome.value_counts().unstack(fill_value=0).to_csv(out / 'outcomes_by_approach.csv')
    dec = decide(e, s, pt)
    (out / 'decision.json').write_text(json.dumps(dec, indent=1))
    pd.set_option('display.width', 250)
    print('integrity ok:', integ['ok'], integ['rows'], 'rows,', integ['unique_policy_seed_approach'], 'unique')
    print(s.pivot(index='policy', columns='approach', values='gt_success').loc[ORDER].round(3).to_string())
    print(s[s.approach == 'all'].set_index('policy').loc[ORDER][['gt_success_n', 'wilson_lo', 'wilson_hi', 'sim_failures', 'sim_fail_wilson_hi']].round(3).to_string())
    print(pt[pt.group.isin(['all', 'approach=top', 'approach=oblique'])][['A', 'B', 'group', 'rate_a', 'rate_b', 'diff_a_minus_b',
                                                                        'diff_lo', 'diff_hi', 'only_a', 'only_b', 'mcnemar_exact_p', 'holm_p']].round(4).to_string(index=False))
    print(fam.loc[ORDER].to_string())
    print(oc.loc[ORDER].to_string())
    print(json.dumps(dec, indent=1))
    (out / 'meta.json').write_text(json.dumps({'episodes_per_policy': int(len(g('scripted'))), 'seed_base': 6_700_000,
                                               'note': 'fresh final set, used once; selection frozen before; run interrupted once '
                                                       'by memory protection and resumed (resume_log.txt)'}, indent=1))


if __name__ == '__main__':
    main()
