"""Apply the pre-registered Wave 1 gate rule (outputs/rl/wave1/wave1_plan_and_decision_rule.json).

Reads independent_validation/<tag>/summary.csv of each run (checkpoints 50k/100k/150k and the
in-training-best model) and the run's control log; prints the evaluation of every criterion and,
with --apply, writes gate_decision.json into each run folder (the waiting process then continues
or stops). The rule was fixed before any Wave 1 result existed.

    python -m visuograsp_v4.scripts.wave1_gate_decision --tag gate150k            # dry run
    python -m visuograsp_v4.scripts.wave1_gate_decision --tag gate150k --apply
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from visuograsp_v4.config import OUTPUT_ROOT

RUNS = ('sac_scratch_s11', 'sac_demo_s11')


def decide(run_dir, tag: str, gate_step: int = 150_000) -> dict:
    s = pd.read_csv(run_dir / 'independent_validation' / tag / 'summary.csv')
    get = lambda model, ap, col: float(s[(s.model == model) & (s.approach == ap)][col].iloc[0])
    m50, m100, m150 = 'checkpoints/step_50000', 'checkpoints/step_100000', f'checkpoints/step_{gate_step}'
    res = {'tag': tag, 'numbers': {}}
    for m in (m50, m100, m150, 'best_model'):
        if (s.model == m).any():
            res['numbers'][m] = {ap: {'gt': get(m, ap, 'gt_success'), 'lo': get(m, ap, 'wilson_lo'),
                                      'hi': get(m, ap, 'wilson_hi'), 'partial': get(m, ap, 'partial_lift_rate'),
                                      'n': int(get(m, ap, 'episodes'))} for ap in ('all', 'top', 'oblique')}
    n = res['numbers']
    numerics_bad = False
    ctrl = run_dir / 'control_log.csv'
    if ctrl.exists():
        ev = pd.read_csv(ctrl).event.astype(str)
        numerics_bad = bool(ev.isin(['q_above_hard_limit']).any() or (run_dir / 'stop_reason.json').exists())
    thr = {m: max(n[m][ap]['gt'] for ap in ('all', 'top', 'oblique')) for m in (m150, 'best_model') if m in n}
    res['threshold_max_gt_150k_or_best'] = thr
    res['threshold_met'] = any(v >= 0.20 for v in thr.values())
    gt_trend = n[m150]['all']['gt'] >= 0.05 and n[m150]['all']['gt'] > n[m100]['all']['gt']
    part_trend = (n[m150]['all']['partial'] >= 0.10 and n[m150]['all']['partial'] > n[m50]['all']['partial']
                  and n[m150]['all']['partial'] > n[m100]['all']['partial'])
    res.update(trend_gt=bool(gt_trend), trend_partial=bool(part_trend), numerics_bad=numerics_bad)
    if numerics_bad:
        res['decision'], res['basis'] = 'stop', 'numerical instability flagged'
    elif res['threshold_met']:
        res['decision'], res['basis'] = 'continue', 'independent-validation GT >= 20% (overall or one approach)'
    elif gt_trend or part_trend:
        res['decision'], res['basis'] = 'continue', 'trend criterion (' + ('GT' if gt_trend else 'partial lift') + ')'
    else:
        res['decision'], res['basis'] = 'stop', 'below 20% and no learning trend on the independent set'
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    for name in RUNS:
        run = OUTPUT_ROOT / 'rl' / 'wave1' / name
        res = decide(run, args.tag)
        print(f'== {name}: {res["decision"].upper()} ({res["basis"]})')
        for m, d in res['numbers'].items():
            print(f'   {m:24s} ' + '  '.join(f'{ap} GT {v["gt"]:.2f} [{v["lo"]:.2f},{v["hi"]:.2f}] part {v["partial"]:.2f} (n={v["n"]})'
                                              for ap, v in d.items()))
        print(f'   threshold_met={res["threshold_met"]} trend_gt={res["trend_gt"]} trend_partial={res["trend_partial"]} '
              f'numerics_bad={res["numerics_bad"]}')
        if args.apply:
            if not (run / 'gate_waiting.json').exists():
                print('   (run is not waiting at the gate; nothing written)')
                continue
            dec = {'decision': res['decision'], 'reason': res['basis'], 'rule': 'wave1_plan_and_decision_rule.json',
                   'evaluation': res}
            (run / 'gate_decision.json').write_text(json.dumps(dec, indent=1), encoding='utf-8')
            print('   gate_decision.json written')


if __name__ == '__main__':
    main()
