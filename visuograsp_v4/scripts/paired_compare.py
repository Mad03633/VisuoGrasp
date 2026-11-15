"""Paired comparison of two policies evaluated on identical episodes (same seeds and initial states).

Per approach and overall: success rates with Wilson CIs, the 2x2 table (both / only A / only B /
neither), exact McNemar p-value (two-sided binomial test on the discordant pairs) and a 95 %
paired-bootstrap CI of the difference A - B (10,000 resamples of episodes, fixed RNG seed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from visuograsp_v4.scripts.evaluate_policy import wilson


def _wilson(k, n):
    lo, hi = wilson(k, n)
    return max(0.0, lo), min(1.0, hi)


def paired_stats(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    la, ha = _wilson(int(a.sum()), n)
    lb, hb = _wilson(int(b.sum()), n)
    return {'episodes': n, 'rate_a': float(a.mean()), 'a_lo': la, 'a_hi': ha, 'rate_b': float(b.mean()), 'b_lo': lb,
            'b_hi': hb, 'diff_a_minus_b': float(a.mean() - b.mean()), 'diff_lo': float(np.percentile(diffs, 2.5)),
            'diff_hi': float(np.percentile(diffs, 97.5)), 'both': int((a & b).sum()), 'only_a': only_a,
            'only_b': only_b, 'neither': int((~a & ~b).sum()), 'mcnemar_exact_p': float(p)}


def paired_table(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str, by=('approach',),
                 col: str = 'gt_success') -> pd.DataFrame:
    m = df_a[['seed', 'approach', 'family', col]].merge(df_b[['seed', 'approach', col]], on=['seed', 'approach'],
                                                        suffixes=('_a', '_b'))
    rows = [{'A': label_a, 'B': label_b, 'group': 'all', **paired_stats(m[f'{col}_a'], m[f'{col}_b'])}]
    for key in by:
        for g, sub in m.groupby(key):
            rows.append({'A': label_a, 'B': label_b, 'group': f'{key}={g}', **paired_stats(sub[f'{col}_a'], sub[f'{col}_b'])})
    return pd.DataFrame(rows)
