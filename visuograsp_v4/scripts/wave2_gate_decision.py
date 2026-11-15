"""Apply the pre-registered Wave 2 gate rule at 75k (outputs/rl/wave2/wave2_plan_and_decision_rule.json).

Reads independent_validation/<tag>/episodes.csv of the residual run (gate checkpoint, best_model,
zero_residual on the same 100 episodes, seeds 8_500_000+), computes paired statistics against the
zero-residual (= scripted) baseline and prints / writes the decision.

    python -m visuograsp_v4.scripts.wave2_gate_decision --run residual_sac_seed11 --tag gate75k [--apply]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts.paired_compare import paired_table


def substantially_worse(row) -> bool:
    d, p = row['diff_a_minus_b'], row['mcnemar_exact_p']
    return d <= -0.20 or (d <= -0.10 and p < 0.05)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='residual_sac_seed11')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gate-model', default='checkpoints/gate_step_75000')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    run = OUTPUT_ROOT / 'rl' / 'wave2' / args.run
    ep = pd.read_csv(run / 'independent_validation' / args.tag / 'episodes.csv')
    zero = ep[ep.model == 'zero_residual']
    tables, worse, corrections = [], {}, {}
    for m in (args.gate_model, 'best_model'):
        sub = ep[ep.model == m]
        if sub.empty:
            continue
        t = paired_table(sub, zero, m, 'zero_residual')
        tables.append(t)
        worse[m] = substantially_worse(t.iloc[0])
        corrections[m] = {'mean_abs_residual': float(sub.res_abs_mean.mean()),
                          'mean_pos_offset_cm': float(sub.res_pos_offset_cm.mean()),
                          'mean_rot_offset_rad': float(sub.res_rot_offset_rad.mean()),
                          'mean_finger_offset_rad': float(sub.res_finger_offset_rad.mean()),
                          'outcome_changed_vs_zero': int((sub.set_index(['seed', 'approach']).outcome
                                                         != zero.set_index(['seed', 'approach']).outcome).sum())}
    tab = pd.concat(tables, ignore_index=True)
    tab.to_csv(run / 'independent_validation' / args.tag / 'paired_vs_zero.csv', index=False)
    numerics_bad = (run / 'stop_reason.json').exists() or (
        (run / 'control_log.csv').exists() and pd.read_csv(run / 'control_log.csv').event.eq('q_above_hard_limit').any())
    if numerics_bad:
        decision, basis = 'stop', 'numerical instability flagged'
    elif worse and all(worse.values()):
        decision, basis = 'stop', 'substantially worse than the zero-residual baseline (paired)'
    else:
        decision, basis = 'continue', 'not substantially worse than the zero-residual baseline'
    pd.set_option('display.width', 250)
    cols = ['A', 'group', 'episodes', 'rate_a', 'rate_b', 'diff_a_minus_b', 'diff_lo', 'diff_hi', 'only_a', 'only_b',
            'mcnemar_exact_p']
    print(tab[cols].round(3).to_string(index=False))
    print('substantially_worse:', worse)
    print('corrections:', json.dumps(corrections, indent=1))
    print(f'DECISION: {decision.upper()} ({basis})')
    if args.apply:
        if not (run / 'gate_waiting.json').exists():
            print('run is not waiting at the gate; nothing written')
            return
        (run / 'gate_decision.json').write_text(json.dumps({'decision': decision, 'reason': basis,
                                                            'rule': 'wave2_plan_and_decision_rule.json',
                                                            'substantially_worse': worse, 'corrections': corrections,
                                                            'paired': tab.round(5).to_dict(orient='records')},
                                                           indent=1, default=str), encoding='utf-8')
        print('gate_decision.json written')


if __name__ == '__main__':
    main()
