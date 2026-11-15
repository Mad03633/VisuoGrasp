"""Final-defense media: copies of the selected original MP4s, animated GIFs, screenshots and multi-panel report
figures. Sources: existing project videos (V3.3 demos, Wave 5 videos, scripted V4 demos) and the re-rendered,
reproduction-checked clips in media/rerendered (record_defense_clips.py). Nothing in the original folders is changed.

    python visuograsp_v4/outputs/final_defense/build/make_media.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
FD = ROOT / 'visuograsp_v4' / 'outputs' / 'final_defense'
RR = FD / 'media' / 'rerendered'
GIF = FD / 'media' / 'gifs'
SHOT = FD / 'media' / 'screenshots'
ORIG = FD / 'media' / 'selected_videos' / 'original'
FIG = FD / 'report' / 'figures'
ASSET = FD / 'presentation' / 'assets'
V33 = ROOT / 'visuograsp_full_outputs' / 'cv_v2' / 'drl_runs_v3_final' / 'videos' / 'v33_final_demos'
W5V = ROOT / 'visuograsp_v4' / 'outputs' / 'rl' / 'wave5' / 'videos'
DEMO = ROOT / 'visuograsp_v4' / 'outputs' / 'demos' / 'scripted_elliptic_imp10_noslip5_dt2.5ms_sr6'
FFMPEG = shutil.which('ffmpeg') or r'C:\ffmpeg\bin\ffmpeg.exe'
FONT = r'C:\Windows\Fonts\arial.ttf'
FONTB = r'C:\Windows\Fonts\arialbd.ttf'
for d in (GIF, SHOT, ORIG, FIG, ASSET):
    d.mkdir(parents=True, exist_ok=True)

ORIGINALS = [V33 / 'A_train_success.mp4', V33 / 'B_train_success.mp4', V33 / 'A_train_failure.mp4',
             W5V / 'e6' / 'top_success_top_box_seed6700000.mp4', W5V / 'e6' / 'oblique_success_oblique_box_seed6700100.mp4',
             W5V / 'e6' / 'side_by_side_learned_wins_oblique_box_seed6706109.mp4',
             W5V / 'e6' / 'side_by_side_scripted_wins_top_box_seed6704000.mp4',
             W5V / 'e6' / 'failure_never_lifted_oblique_box_seed6704100.mp4',
             DEMO / 'side_box_tall.mp4', DEMO / 'top_ring_flat.mp4', DEMO / 'oblique_bottle.mp4']


def manifest():
    m = pd.read_csv(RR / 'rerender_manifest.csv')
    assert m.reproduced.all(), 'a re-rendered clip did not reproduce its frozen outcome'
    return {r.clip: dict(r._asdict(), keyframes=json.loads(r.keyframes)) for r in m.itertuples()}


def read_frames(path, idx, crop=None):
    c = cv2.VideoCapture(str(path))
    out = []
    for i in idx:
        c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = c.read()
        if not ok:
            break
        if crop is not None:
            x0, y0, x1, y1 = crop
            f = f[y0:y1, x0:x1]
        out.append(f)
    return out


def nframes(path):
    return int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))


def tag(img, text, pos='tr'):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = w - tw - 14 if pos == 'tr' else 10
    cv2.rectangle(img, (x - 6, 8), (x + tw + 6, 14 + th + 8), (30, 30, 30), -1)
    cv2.putText(img, text, (x, 14 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def write_gif(frames, out, width):
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            cv2.imwrite(str(Path(td) / f'f{i:04d}.png'), f)
        vf = (f'fps=10,scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];'
              f'[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle')
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-framerate', '10', '-i', str(Path(td) / 'f%04d.png'), '-vf', vf,
                        '-loop', '0', str(out)], check=True)
    return out


def segment(k, n, pad_before=8, pad_after=14):
    if 'hold' in k:
        return max(0, k.get('approach', k['contact']) - pad_before), min(n - 1, k['hold'] + pad_after)
    return 0, n - 1


def gif_from_clip(name, m, out, width=900, speed=2, label=''):
    p = RR / name / f'{name}_all.mp4'
    n = nframes(p)
    a, b = segment(m[name]['keyframes'], n)
    fr = read_frames(p, range(a, b + 1, speed), crop=(0, 0, 1280, 576))
    fr = [tag(f, f'{speed}x speed' if speed > 1 else 'real time') for f in fr]
    if label:
        fr = [tag(f, label, 'tl') for f in fr]
    write_gif(fr, out, width)
    return {'source': p, 'frames_from': a, 'frames_to': b, 'speed': speed, 'gif_frames': len(fr), 'seconds': len(fr) / 10}


def gif_stacked(top, bottom, m, out, width=800, speed=2):
    pt, pb = RR / top / f'{top}_all.mp4', RR / bottom / f'{bottom}_all.mp4'
    k = m[bottom]['keyframes']
    a, b = max(0, k['approach'] - 8), min(nframes(pb) - 1, k['hold'] + 14)
    ft = read_frames(pt, range(a, b + 1, speed), crop=(0, 0, 1280, 576))
    fb = read_frames(pb, range(a, b + 1, speed), crop=(0, 0, 1280, 576))
    fr = [tag(np.vstack((x, y)), f'{speed}x speed') for x, y in zip(ft, fb)]
    write_gif(fr, out, width)
    return {'source': [pt, pb], 'frames_from': a, 'frames_to': b, 'speed': speed, 'gif_frames': len(fr), 'seconds': len(fr) / 10}


def gif_v33(path, out, width=560, speed=2):
    n = nframes(path)
    fr = [tag(f, f'{speed}x speed') for f in read_frames(path, range(0, n, speed))]
    write_gif(fr, out, width)
    return {'source': path, 'frames_from': 0, 'frames_to': n - 1, 'speed': speed, 'gif_frames': len(fr), 'seconds': len(fr) / 10}


# ---------------------------------------------------------------- multi-panel figures
def tile(img_bgr, label, w=400, h=300, sub=None):
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONTB, 17)
    fs = ImageFont.truetype(FONT, 15)
    bh = 30 if sub is None else 50
    d.rectangle([0, h - bh, w, h], fill=(20, 22, 26))
    d.text((8, h - bh + 5), label, font=f, fill=(255, 255, 255))
    if sub:
        d.text((8, h - bh + 27), sub, font=fs, fill=(205, 205, 205))
    return np.array(im)


def grid(tiles, ncol, gap=6):
    h, w = tiles[0].shape[:2]
    rows = []
    for i in range(0, len(tiles), ncol):
        r = tiles[i:i + ncol]
        r += [np.full_like(tiles[0], 255)] * (ncol - len(r))
        rows.append(np.hstack(sum([[t, np.full((h, gap, 3), 255, np.uint8)] for t in r], [])[:-1]))
    out = sum([[r, np.full((gap, rows[0].shape[1], 3), 255, np.uint8)] for r in rows], [])[:-1]
    return np.vstack(out)


def key(clip, ev, cam):
    return cv2.imread(str(RR / clip / f'key_{ev}_{cam}.png'))


def over_crop(img):
    return img[30:330, 120:520]


def mid(clip, frame, cam='side'):
    x0 = {'overview': 0, 'side': 640, 'wrist': 1280}[cam]
    return read_frames(RR / clip / f'{clip}_all.mp4', [frame], crop=(x0, 0, x0 + 640, 480))[0]


def save_fig(arr_rgb, name):
    im = Image.fromarray(arr_rgb)
    im.save(FIG / f'{name}.jpg', quality=92)
    im.save(ASSET / f'{name}.jpg', quality=92)


def figures(m):
    t = lambda c, e: f"t = {m[c]['keyframes'][e] / 10:.1f} s"
    v33 = read_frames(V33 / 'A_train_success.mp4', [32], crop=(0, 26, 320, 320))[0]
    save_fig(grid([tile(v33, '(a) V3.3: fixed vertical wrist', sub='V3.3 SAC M2, Level A, box'),
                   tile(key('e6_oblique_bottle_6715100', 'contact', 'side'), '(b) V4: oblique 45° wrist', sub='V4 E6 policy, bottle'),
                   tile(key('scripted_demo_side_box_tall_7000007', 'contact', 'side'), '(c) V4: side 90° wrist',
                        sub='V4 scripted expert (not learned), box')], 3), 'fig_hand_v33_v4')
    objs = [('e6_top_box_6700000', 'box · top-down'), ('e6_top_cylinder_6709000', 'cylinder · top-down'),
            ('e6_oblique_bottle_6715100', 'bottle · oblique 45°'), ('e6_oblique_lshape_6721100', 'L-shape · oblique 45°'),
            ('e6_top_ring_6719000', 'ring · top-down')]
    save_fig(grid([tile(key(c, 'lift', 'side'), l, sub='E6, GT success (lift ≥ 5 cm)') for c, l in objs], 5), 'fig_objects_e6')
    c = 'e6_oblique_bottle_6715100'
    save_fig(grid([tile(key(c, 'approach', 'side'), '1  approach', sub=t(c, 'approach')),
                   tile(key(c, 'contact', 'side'), '2  thumb + pad contact', sub=t(c, 'contact')),
                   tile(key(c, 'lift', 'side'), '3  lift (5 cm)', sub=t(c, 'lift')),
                   tile(over_crop(key(c, 'hold', 'overview')), '4  held 3 s at ≥ 10 cm', sub=t(c, 'hold') + ' (overview camera)')], 4),
             'fig_e6_sequence')
    s, e = 'scripted_oblique_box_6706109', 'e6_oblique_box_6706109'
    save_fig(grid([tile(key(s, 'approach', 'side'), 'Scripted: approach', sub=t(s, 'approach')),
                   tile(key(s, 'contact', 'side'), 'Scripted: contact', sub=t(s, 'contact')),
                   tile(mid(s, 70), 'Scripted: object pushed', sub='t = 7.0 s, never lifted (failure)'),
                   tile(over_crop(key(s, 'end', 'overview')), 'Scripted: end', sub='GT failure (pushed before closing)'),
                   tile(key(e, 'approach', 'side'), 'E6: approach', sub=t(e, 'approach')),
                   tile(key(e, 'contact', 'side'), 'E6: contact', sub=t(e, 'contact')),
                   tile(key(e, 'lift', 'side'), 'E6: lift', sub=t(e, 'lift')),
                   tile(over_crop(key(e, 'hold', 'overview')), 'E6: held 3 s', sub=t(e, 'hold') + ', GT success')], 4),
             'fig_scripted_vs_e6')
    save_fig(grid([tile(mid('e6_fail_pushed_oblique_box_6706102', 34), 'E6: palm pushes the box', sub='t = 3.4 s, oblique, seed 6706102'),
                   tile(mid('e6_fail_never_lifted_oblique_box_6704100', 50), 'E6: hand leaves without object',
                        sub='t = 5.0 s, never lifted, seed 6704100'),
                   tile(mid('e6_fail_ring_oblique_6719100', 34), 'E6: fingers miss the flat ring',
                        sub='t = 3.4 s; oblique ring 0/20 for all')], 3), 'fig_failures_frames')
    hero = key('e6_oblique_bottle_6715100', 'lift', 'side')
    cv2.imwrite(str(ASSET / 'hero_e6_oblique_bottle_lift.png'), hero)
    cv2.imwrite(str(ASSET / 'v33_side_frame.png'), v33)


def screenshots(m):
    rows = []
    for clip, info in m.items():
        for ev in info['keyframes']:
            for cam in ('side', 'overview'):
                src = RR / clip / f'key_{ev}_{cam}.png'
                dst = SHOT / f'{clip}__{ev}_{cam}.png'
                shutil.copy2(src, dst)
                rows.append({'clip': clip, 'event': ev, 'camera': cam, 'frame': info['keyframes'][ev], 'file': str(dst)})
    for nm, f in (('v33_A_train_success__t3.2s_side.png', read_frames(V33 / 'A_train_success.mp4', [32], crop=(0, 26, 320, 320))[0]),
                  ('v33_B_train_success__t6.0s_side.png', read_frames(V33 / 'B_train_success.mp4', [60], crop=(0, 26, 320, 320))[0]),
                  ('wave5_e6_side_by_side_6706109__frame100_original.png',
                   read_frames(W5V / 'e6' / 'side_by_side_learned_wins_oblique_box_seed6706109.mp4', [100])[0])):
        cv2.imwrite(str(SHOT / nm), f)
        rows.append({'clip': nm.split('__')[0], 'event': nm.split('__')[1][:-4], 'camera': 'side', 'frame': None, 'file': str(SHOT / nm)})
    pd.DataFrame(rows).to_csv(SHOT / 'screenshots_index.csv', index=False)


def main():
    m = manifest()
    for p in ORIGINALS:
        assert p.exists(), p
        sub = 'v33' if 'visuograsp_full_outputs' in str(p) else ('wave5_e6' if 'wave5' in str(p) else 'scripted_demos')
        (ORIG / sub).mkdir(exist_ok=True)
        shutil.copy2(p, ORIG / sub / p.name)
    info = {}
    info['gif_e6_oblique_bottle.gif'] = gif_from_clip('e6_oblique_bottle_6715100', m, GIF / 'gif_e6_oblique_bottle.gif')
    info['gif_e6_top_cylinder.gif'] = gif_from_clip('e6_top_cylinder_6709000', m, GIF / 'gif_e6_top_cylinder.gif')
    info['gif_e6_oblique_lshape.gif'] = gif_from_clip('e6_oblique_lshape_6721100', m, GIF / 'gif_e6_oblique_lshape.gif')
    info['gif_e6_top_ring.gif'] = gif_from_clip('e6_top_ring_6719000', m, GIF / 'gif_e6_top_ring.gif')
    info['gif_scripted_vs_e6_oblique_box_6706109.gif'] = gif_stacked('scripted_oblique_box_6706109', 'e6_oblique_box_6706109', m,
                                                                     GIF / 'gif_scripted_vs_e6_oblique_box_6706109.gif')
    info['gif_scripted_side_box_tall.gif'] = gif_from_clip('scripted_demo_side_box_tall_7000007', m, GIF / 'gif_scripted_side_box_tall.gif')
    info['gif_e6_fail_ring_oblique.gif'] = gif_from_clip('e6_fail_ring_oblique_6719100', m, GIF / 'gif_e6_fail_ring_oblique.gif', speed=3)
    info['gif_e6_fail_pushed_oblique_box.gif'] = gif_from_clip('e6_fail_pushed_oblique_box_6706102', m,
                                                               GIF / 'gif_e6_fail_pushed_oblique_box.gif', speed=3)
    info['gif_v33_level_a_success.gif'] = gif_v33(V33 / 'A_train_success.mp4', GIF / 'gif_v33_level_a_success.gif')
    for k, v in info.items():
        g = Image.open(GIF / k)
        v.update(gif_frames_checked=getattr(g, 'n_frames', 1), size_mb=round((GIF / k).stat().st_size / 2 ** 20, 2), wh=g.size)
        v['source'] = [str(x) for x in v['source']] if isinstance(v['source'], list) else str(v['source'])
        print(k, {x: v[x] for x in ('gif_frames_checked', 'seconds', 'speed', 'size_mb', 'wh')})
    (GIF / 'gif_index.json').write_text(json.dumps(info, indent=1))
    figures(m)
    screenshots(m)
    print('media written to', FD / 'media')


if __name__ == '__main__':
    main()
