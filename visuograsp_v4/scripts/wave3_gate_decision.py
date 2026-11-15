"""Apply the pre-registered Wave 3 gate rule (outputs/rl/wave3/wave3_plan_and_decision_rule.json).

Independent validation (seeds 8_500_000+, 200 paired episodes): gate checkpoint, best_model,
zero residual (= scripted) and the chaos control (r ~ U(-1e-3, 1e-3), physically negligible).

    python -m visuograsp_v4.scripts.wave3_gate_decision --tag gate25k --stage 25000 [--apply]
    python -m visuograsp_v4.scripts.wave3_gate_decision --tag gate50k --stage 50000 --previous gate25k [--apply]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT
from visuograsp_v4.scripts.paired_compare import paired_table

RUN = OUTPUT_ROOT / 'rl' / 'wave3' / 'conservative_residual_sac_seed11'


def evaluate(tag: str, gate_model: str) -> tuple[pd.DataFrame, dict]:
    ep = pd.read_csv(RUN / 'independent_validation' / tag / 'episodes.csv')
    zero = ep[ep.model == 'zero_residual']
    tabs, info = [], {}
    for m in (gate_model, 'best_model', 'perturbed_1e-3'):
        sub = ep[ep.model == m]
        if sub.empty:
            continue
        t = paired_table(sub, zero, m, 'zero_residual')
        tabs.append(t)
        a = t.iloc[0]
        info[m] = {'rate': a.rate_a, 'zero_rate': a.rate_b, 'diff': a.diff_a_minus_b, 'diff_lo': a.diff_lo,
                   'diff_hi': a.diff_hi, 'improved': int(a.only_a), 'broken': int(a.only_b),
                   'p': a.mcnemar_exact_p, 'mean_abs_residual': float(sub.res_abs_mean.mean()),
                   'pos_offset_mm': 10 * float(sub.res_pos_offset_cm.mean()),
                   'rot_offset_rad': float(sub.res_rot_offset_rad.mean()),
                   'finger_offset_rad': float(sub.res_finger_offset_rad.mean())}
    return pd.concat(tabs, ignore_index=True), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--previous', default=None, help='tag of the previous gate (50k stage)')
    ap.add_argument('--previous-stage', type=int, default=None)
    ap.add_argument('--first-stage', type=int, default=25_000, help='stage that uses the 25k rule')
    ap.add_argument('--run', default='conservative_residual_sac_seed11')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    global RUN
    RUN = OUTPUT_ROOT / 'rl' / 'wave3' / args.run
    gm = f'checkpoints/gate_step_{args.stage}'
    tab, info = evaluate(args.tag, gm)
    tab.to_csv(RUN / 'independent_validation' / args.tag / 'paired_vs_zero.csv', index=False)
    g, ctrl = info[gm], info.get('perturbed_1e-3', {'improved': 0, 'broken': 0})
    numerics_bad = (RUN / 'stop_reason.json').exists() or (
        (RUN / 'control_log.csv').exists() and pd.read_csv(RUN / 'control_log.csv').event.eq('q_above_hard_limit').any())
    clearly_worse = g['diff'] <= -0.05 or (g['diff'] < 0 and g['p'] < 0.05)
    useful = g['improved'] >= 3 and g['improved'] > ctrl['improved']
    if numerics_bad:
        decision, basis = 'stop', 'numerical instability'
    elif clearly_worse:
        decision, basis = 'stop', 'clearly underperforms the zero-residual baseline'
    elif args.stage == args.first_stage:
        if g['diff'] >= 0 and useful:
            decision, basis = 'continue', 'comparable or better, with more fixed episodes than the chaos control'
        else:
            decision, basis = 'stop', 'no evidence of useful corrections beyond the chaos control'
    else:
        prev = None
        if args.previous:
            ps = args.previous_stage or args.stage - 25_000
            _, pinfo = evaluate(args.previous, f'checkpoints/gate_step_{ps}')
            prev = pinfo[f'checkpoints/gate_step_{ps}']['diff']
        if g['diff'] > 0 and useful and (prev is None or g['diff'] >= prev):
            decision, basis = 'continue', 'improvement over zero residual, not smaller than at the previous gate'
        else:
            decision, basis = 'stop', 'no growing improvement over the zero residual'
    pd.set_option('display.width', 250)
    cols = ['A', 'group', 'episodes', 'rate_a', 'rate_b', 'diff_a_minus_b', 'diff_lo', 'diff_hi', 'only_a', 'only_b',
            'mcnemar_exact_p']
    print(tab[cols].round(3).to_string(index=False))
    print(json.dumps(info, indent=1, default=float))
    print(f'clearly_worse={clearly_worse} useful={useful} numerics_bad={numerics_bad}')
    print(f'DECISION at {args.stage}: {decision.upper()} ({basis})')
    if args.apply:
        if not (RUN / 'gate_waiting.json').exists():
            print('run is not waiting at a gate; nothing written')
            return
        (RUN / 'gate_decision.json').write_text(json.dumps({'decision': decision, 'reason': basis, 'stage': args.stage,
                                                            'rule': 'wave3_plan_and_decision_rule.json', 'evaluation': info},
                                                           indent=1, default=float), encoding='utf-8')
        print('gate_decision.json written')


if __name__ == '__main__':
    main()
