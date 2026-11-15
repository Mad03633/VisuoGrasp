"""Write DEMO_MEDIA_MANIFEST.md with absolute Windows paths; every path is checked for existence."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
FD = ROOT / 'visuograsp_v4' / 'outputs' / 'final_defense'
RR = FD / 'media' / 'rerendered'
H264 = FD / 'media' / 'selected_videos' / 'h264'
GIF = FD / 'media' / 'gifs'
SHOT = FD / 'media' / 'screenshots'
V33 = ROOT / 'visuograsp_full_outputs' / 'cv_v2' / 'drl_runs_v3_final' / 'videos' / 'v33_final_demos'
W5V = ROOT / 'visuograsp_v4' / 'outputs' / 'rl' / 'wave5' / 'videos' / 'e6'
DEMO = ROOT / 'visuograsp_v4' / 'outputs' / 'demos' / 'scripted_elliptic_imp10_noslip5_dt2.5ms_sr6'
m = pd.read_csv(RR / 'rerender_manifest.csv').set_index('clip')

# id, title, controller, family, approach, outcome text, existing original MP4 (or None), re-rendered clip (or None),
# GIF (or None), screenshot, slides, report
ITEMS = [
    ('D1', 'V3.3 Level A grasp (previous system)', 'V3.3 SAC M2 (D2 vision-confirmed contact), seed 33', 'box', 'top-down (fixed vertical wrist)',
     'GT success 1, CV success 1 (README V3.3)', V33 / 'A_train_success.mp4', None, 'gif_v33_level_a_success.gif',
     SHOT / 'v33_A_train_success__t3.2s_side.png', 'Slide 2 (GIF)', 'Fig. 1(a)'),
    ('D2', 'E6 oblique bottle grasp', 'E6 BC-anchored RL (frozen e6_best_90k)', 'bottle', 'oblique 45°', None, None,
     'e6_oblique_bottle_6715100', 'gif_e6_oblique_bottle.gif', SHOT / 'e6_oblique_bottle_6715100__contact_side.png',
     'Slide 1 (still), Slide 2 (still), Slide 4 (GIF)', 'Fig. 1(b), Fig. 1 bottom'),
    ('D3', 'Scripted side 90° grasp (not learned)', 'Scripted expert (PRIVILEGED planner)', 'box (tall)', 'side 90°', None,
     DEMO / 'side_box_tall.mp4', 'scripted_demo_side_box_tall_7000007', 'gif_scripted_side_box_tall.gif',
     SHOT / 'scripted_demo_side_box_tall_7000007__contact_side.png', 'Slide 2 (still), Slide 4 (GIF)', 'Fig. 1(c)'),
    ('D4', 'Scripted expert vs E6 on the same episode', 'Scripted expert (top) vs E6 (bottom)', 'box (flat)', 'oblique 45°', None,
     W5V / 'side_by_side_learned_wins_oblique_box_seed6706109.mp4', 'e6_oblique_box_6706109', 'gif_scripted_vs_e6_oblique_box_6706109.gif',
     SHOT / 'e6_oblique_box_6706109__lift_side.png', 'Slide 10 (GIF)', 'Fig. 6'),
    ('D5', 'E6 top-down cylinder grasp', 'E6 BC-anchored RL', 'cylinder', 'top-down', None, None, 'e6_top_cylinder_6709000',
     'gif_e6_top_cylinder.gif', SHOT / 'e6_top_cylinder_6709000__lift_side.png', 'Slide 12 (GIF), Slide 3 (strip)', 'Fig. 1 bottom'),
    ('D6', 'E6 failure: oblique flat ring, no contact', 'E6 BC-anchored RL', 'ring', 'oblique 45°', None, None,
     'e6_fail_ring_oblique_6719100', 'gif_e6_fail_ring_oblique.gif', SHOT / 'e6_fail_ring_oblique_6719100__end_overview.png',
     'Slide 11 (GIF)', 'Fig. 7 (right)'),
    ('D7', 'E6 top-down box grasp', 'E6 BC-anchored RL', 'box (cube)', 'top-down', None,
     W5V / 'top_success_top_box_seed6700000.mp4', 'e6_top_box_6700000', None, SHOT / 'e6_top_box_6700000__lift_side.png',
     'Slide 3 (strip)', 'Fig. 1 bottom'),
    ('D8', 'E6 oblique L-shape grasp', 'E6 BC-anchored RL', 'L-shape', 'oblique 45°', None, None, 'e6_oblique_lshape_6721100',
     'gif_e6_oblique_lshape.gif', SHOT / 'e6_oblique_lshape_6721100__lift_side.png', 'Slide 3 (strip); GIF spare', 'Fig. 1 bottom'),
    ('D9', 'E6 top-down ring grasp', 'E6 BC-anchored RL', 'ring', 'top-down', None, None, 'e6_top_ring_6719000', 'gif_e6_top_ring.gif',
     SHOT / 'e6_top_ring_6719000__lift_side.png', 'Slide 3 (strip); GIF spare', 'Fig. 1 bottom'),
    ('D10', 'E6 failure: palm pushes the box', 'E6 BC-anchored RL', 'box (flat)', 'oblique 45°', None, None,
     'e6_fail_pushed_oblique_box_6706102', 'gif_e6_fail_pushed_oblique_box.gif', SHOT / 'e6_fail_pushed_oblique_box_6706102__contact_side.png',
     'GIF spare', 'Fig. 7 (left)'),
    ('D11', 'E6 failure: never lifted', 'E6 BC-anchored RL', 'box (tall, lying on side)', 'oblique 45°', None,
     W5V / 'failure_never_lifted_oblique_box_seed6704100.mp4', 'e6_fail_never_lifted_oblique_box_6704100', None,
     SHOT / 'e6_fail_never_lifted_oblique_box_6704100__end_side.png', '—', 'Fig. 7 (middle)'),
]


def chk(p):
    if p is None:
        return '—'
    p = Path(p)
    assert p.exists(), p
    return f'`{p}`'


def main():
    L = ['# Demo media manifest — VisuoGrasp V4 final defense', '',
         '* All V4 clips show **PRIVILEGED state-based control**: the simulator object pose and ground-truth contacts are inputs. '
         'They are not vision-based.',
         '* **"Re-rendered" clips** replay an already-evaluated episode with the frozen policy at 640×480 per camera (deterministic, '
         'torch 1 thread). Each one reproduced the GT outcome stored in the frozen record; see the outcome column and '
         '`media/rerendered/rerender_manifest.csv`.',
         '* **GIFs** cover approach → contact → lift → hold at the speed noted in the GIF corner (2× or 3×). The MP4s play in real time.',
         '* **H.264 copies** of every MP4 are in `media/selected_videos/h264/` and play in any player. The originals use MPEG-4 Part 2.',
         '* Every path below was checked for existence when this file was written.', '']
    for it in ITEMS:
        i, title, ctrl, fam, ap, outc, orig, clip, gif, shot, slides, rep = it
        if clip:
            r = m.loc[clip]
            outc = (f"GT success {int(r.gt_success_rerun)} ({r.outcome_rerun}); frozen record: {r.outcome_frozen} "
                    f"[{r.frozen_source}], reproduced = {bool(r.reproduced)}; seed {int(r.seed)}")
        L += [f'## {i}. {title}', '',
              f'* **Controller / model:** {ctrl}',
              f'* **Object family:** {fam}',
              f'* **Approach direction:** {ap}',
              f'* **GT success / failure:** {outc}',
              f'* **Original MP4 (pre-existing project video):** {chk(orig)}',
              f'* **Re-rendered MP4 (640×480 per camera, real time):** {chk(RR / clip / f"{clip}_all.mp4") if clip else "—"}',
              f'* **H.264 copy for playback:** {chk(H264 / ((Path(orig).stem if orig else clip + "_all") + "_h264.mp4")) if (orig or clip) else "—"}',
              f'* **Generated GIF:** {chk(GIF / gif) if gif else "—"}',
              f'* **Representative screenshot:** {chk(shot)}',
              f'* **Slide:** {slides}',
              f'* **Report section / figure:** {rep}', '']
    L += ['## Embedded in the PPTX', '',
          '| Slide | Embedded animated GIF | MP4 fallback (hyperlink on the slide) |', '|---|---|---|',
          f'| 2 | gif_v33_level_a_success.gif (46 frames) | {chk(H264 / "A_train_success_h264.mp4")} |',
          f'| 4 | gif_e6_oblique_bottle.gif (50 frames) | {chk(H264 / "e6_oblique_bottle_6715100_all_h264.mp4")} |',
          f'| 4 | gif_scripted_side_box_tall.gif (42 frames) | {chk(H264 / "scripted_demo_side_box_tall_7000007_all_h264.mp4")} |',
          f'| 10 | gif_scripted_vs_e6_oblique_box_6706109.gif (56 frames) | {chk(H264 / "side_by_side_learned_wins_oblique_box_seed6706109_h264.mp4")} |',
          f'| 11 | gif_e6_fail_ring_oblique.gif (54 frames) | {chk(H264 / "e6_fail_ring_oblique_6719100_all_h264.mp4")} |',
          f'| 12 | gif_e6_top_cylinder.gif (56 frames) | {chk(H264 / "e6_top_cylinder_6709000_all_h264.mp4")} |', '',
          '**Verification:**',
          '* The PPTX package contains the six GIF files with all frames and an infinite loop.',
          '* The picture elements reference these GIFs directly.',
          '* PowerPoint renders all slides (COM export).',
          '* Playback in slideshow mode could not be observed automatically: the screenshot attempt captured another window.',
          '* Open the deck in Slide Show once before the defense. If a GIF does not move, click the MP4 link under it.', '']
    (FD / 'DEMO_MEDIA_MANIFEST.md').write_text('\n'.join(L), encoding='utf-8')
    print('written', FD / 'DEMO_MEDIA_MANIFEST.md')


if __name__ == '__main__':
    main()
