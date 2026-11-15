"""Record successful and failed trajectories of a trained V4 policy (overview | side | wrist).

Episodes are picked from an independent-validation episodes.csv (seeds 8_500_000+): per
approach the first success and the first failure of the most frequent failure outcome (by seed).
Each episode is re-run with the same seed and the deterministic policy; the recomputed GT outcome
is compared with the CSV (column ``reproduced``) so a video is never labelled from memory.
torch runs with ONE thread, as in validate_checkpoints.py: with more threads the batch-1 policy
output differs by ~1e-7 (BLAS reduction order) and contact-rich episodes diverge chaotically.

    python -m visuograsp_v4.scripts.record_policy_videos --run wave1/sac_demo_s11 --model best_model.zip --iv-tag final
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from visuograsp_v4 import rotations as rot
from visuograsp_v4.config import OUTPUT_ROOT, V4Config
from visuograsp_v4.video import Mp4Writer, overlay, render_panels


def pick(df: pd.DataFrame, per_approach: int) -> list[tuple]:
    picks = []
    for ap, g in df.sort_values('seed').groupby('approach'):
        succ = g[g.gt_success.astype(bool)]
        fail = g[~g.gt_success.astype(bool)]
        for _, r in succ.head(per_approach).iterrows():
            picks.append(('success', r))
        if not fail.empty:
            top_fail = fail.outcome.value_counts().index[0]
            for _, r in fail[fail.outcome == top_fail].head(per_approach).iterrows():
                picks.append(('failure', r))
            other = fail[fail.outcome != top_fail]
            if not other.empty:                       # one example of a second failure mode
                picks.append(('failure', other.iloc[0]))
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', required=True, help='model file relative to the run folder')
    ap.add_argument('--iv-tag', required=True, help='independent_validation/<tag>/episodes.csv to pick from')
    ap.add_argument('--per-approach', type=int, default=1)
    args = ap.parse_args()
    run = OUTPUT_ROOT / 'rl' / args.run
    model_path = run / args.model
    label = Path(args.model).stem if Path(args.model).parent == Path('.') else f'{Path(args.model).parent.name}/{Path(args.model).stem}'
    ep = pd.read_csv(run / 'independent_validation' / args.iv_tag / 'episodes.csv')
    ep = ep[ep.model == label]
    if ep.empty:
        raise SystemExit(f'no episodes for model {label} in tag {args.iv_tag}')
    os.chdir(OUTPUT_ROOT)
    import torch
    torch.set_num_threads(1)                      # must match validate_checkpoints.py (reproducibility)
    from visuograsp_v4.agents import load_sac
    from visuograsp_v4.env import VisuoGraspV4Env
    from visuograsp_v4.evaluation import run_policy_episode
    model = load_sac(model_path)
    env = VisuoGraspV4Env(V4Config(), obs_mode='privileged', contact_source='gt_oracle', approaches=('top', 'oblique'))
    out = run / 'videos' / f'{args.iv_tag}_{label.replace("/", "_")}'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, r in pick(ep, args.per_approach):
        name = f'{r.approach}_{kind}_{r.outcome}_seed{int(r.seed)}'
        writer = Mp4Writer(out / f'{name}.mp4', fps=10)
        state = {'k': 0}

        def act(obs, writer=writer, r=r):
            s = env.sim
            if state['k'] % 2 == 0:
                t = s.tracker
                pads = ''.join(str(int(v)) for v in s.contact_summary()['pads'])
                img = render_panels(s)
                writer.add(overlay(img, [f'SAC {args.run.split("/")[-1]} {label} | {r.approach} | {r.family}/{r.resting_mode} | seed {int(r.seed)}',
                                         f'step {state["k"]}  lift {100 * s.object_lift():.1f} cm  GT pads {pads}  hold {t.gt_hold}/60',
                                         f'GT success {int(t.gt_success)}  tilt {rot.finger_tilt_deg(s.ctrl.hand_quat()):.0f} deg  PRIVILEGED state']))
            state['k'] += 1
            return model.predict(obs, deterministic=True)[0]

        row = run_policy_episode(env, act, int(r.seed), r.family, r.variant, r.resting_mode, r.approach)
        writer.close()
        rep = bool(row['gt_success']) == bool(r.gt_success) and row['outcome'] == r.outcome \
            and abs(row['max_lift_cm'] - r.max_lift_cm) < 1e-6
        rows.append({'file': f'{name}.mp4', 'kind': kind, 'approach': r.approach, 'seed': int(r.seed), 'family': r.family,
                     'variant': r.variant, 'resting_mode': r.resting_mode, 'outcome_csv': r.outcome,
                     'outcome_rerun': row['outcome'], 'gt_success_rerun': bool(row['gt_success']),
                     'max_lift_cm_rerun': row['max_lift_cm'], 'reproduced': rep, 'frames': writer.frames,
                     'torch_threads': torch.get_num_threads()})
        print(f'[video] {name}: rerun outcome {row["outcome"]} reproduced={rep}', flush=True)
    pd.DataFrame(rows).to_csv(out / 'manifest.csv', index=False)
    print('written to', out)


if __name__ == '__main__':
    main()
