// VisuoGrasp V4 final-defense deck (pptxgenjs). Numbers come from evidence/VERIFIED_RESULTS.md (frozen Wave 5 files).
// Build:  set NODE_PATH=<dir with pptxgenjs, jszip>;  node build_deck.js
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');

const FD = path.resolve(__dirname, '..');
const A = (p) => path.join(FD, 'presentation', 'assets', p);
const G = (p) => path.join(FD, 'media', 'gifs', p);
const RR = (p) => path.join(FD, 'media', 'rerendered', p);
const OUT = path.join(FD, 'presentation', 'VisuoGrasp_V4_Final_Presentation.pptx');

const THEME = {
  name: 'VisuoGrasp', headFontFace: 'Calibri', bodyFontFace: 'Calibri',
  colors: { dk1: '1F2A37', lt1: 'FFFFFF', dk2: '0F1C2E', lt2: 'F1F4F7', accent1: '1BAF7A', accent2: '2A78D6',
            accent3: 'EB6834', accent4: '6B6A64', accent5: 'EDA100', accent6: '8FD3F4', hlink: '2A78D6', folHlink: '6B6A64' },
};
const C = { navy: '0F1C2E', ink: '1F2A37', ink2: '5F6B7A', mute: '8A94A3', line: 'D9DEE5', bg2: 'F1F4F7',
            green: '1BAF7A', blue: '2A78D6', orange: 'EB6834', gray: '6B6A64', amber: 'EDA100', ice: '8FD3F4', white: 'FFFFFF' };
const W = 13.333, H = 7.5, M = 0.6;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'VisuoGrasp V4 - Final Defense';
pres.author = 'Madiyar Bolatov, Amanbek Faizolla, Alisher Amangeldy';
pres.company = 'Astana IT University';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

pres.defineSlideMaster({
  title: 'DARK', background: { color: C.navy },
  objects: [],
});
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: C.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.32, w: W - 2 * M, h: 0.85, fontSize: 32, bold: true,
        color: C.navy, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
    { text: { text: 'VisuoGrasp V4  ·  MuJoCo simulation  ·  all learned results use PRIVILEGED state (simulator object pose + GT contacts), not vision',
        options: { x: M, y: H - 0.42, w: 10.6, h: 0.3, fontSize: 10, color: C.mute, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: H - 0.42, w: 0.6, h: 0.3, fontSize: 10, color: C.mute, align: 'right' },
});

const NOTES = [];
function notes(slide, n, title, text) { slide.addNotes(text); NOTES.push({ n, title, text }); }

function pill(slide, x, y, text, color, w) {
  const ww = w || (0.12 + 0.085 * text.length);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: ww, h: 0.32, fill: { color }, line: { color }, rectRadius: 0.16 });
  slide.addText(text, { x, y, w: ww, h: 0.32, fontSize: 11, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0,
    isTextBox: true });
  return ww;
}
let sizeOf = require('image-size');
if (typeof sizeOf !== 'function') sizeOf = sizeOf.imageSize;
function img(slide, file, x, y, w, pxW, pxH, extra) {
  if (!fs.existsSync(file)) throw new Error('missing ' + file);
  const d = sizeOf(fs.readFileSync(file));
  const h = w * d.height / d.width;
  slide.addImage(Object.assign({ path: file, x, y, w, h }, extra || {}));
  return h;
}
function caption(slide, text, x, y, w, opts) {
  slide.addText(text, Object.assign({ x, y, w, h: 0.5, fontSize: 12, color: C.ink2, margin: 0, valign: 'top', isTextBox: true }, opts || {}));
}
function mp4link(slide, src, x, y, w, color) {
  const file = path.join(FD, 'media', 'selected_videos', 'h264', path.basename(src).replace(/\.mp4$/, '_h264.mp4'));
  if (!fs.existsSync(file)) throw new Error('missing ' + file);
  const url = 'file:///' + file.split(path.sep).join('/');
  slide.addText([{ text: '▶ MP4 fallback (real time): ' + path.basename(file), options: { hyperlink: { url, tooltip: file } } }],
    { x, y, w, h: 0.28, fontSize: 10, color: color || C.blue, margin: 0, isTextBox: true });
}
const V33MP4 = path.resolve(FD, '..', '..', '..', 'visuograsp_full_outputs', 'cv_v2', 'drl_runs_v3_final', 'videos', 'v33_final_demos', 'A_train_success.mp4');
const W5MP4 = path.resolve(FD, '..', 'rl', 'wave5', 'videos', 'e6', 'side_by_side_learned_wins_oblique_box_seed6706109.mp4');
function card(slide, x, y, w, h, fill, line) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill || C.bg2 }, line: { color: line || fill || C.bg2, width: 1 },
    rectRadius: 0.12 });
}
function arrow(slide, x1, y1, x2, y2, color, dash) {
  slide.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: color || C.ink2, width: 1.75, endArrowType: 'triangle', dashType: dash || 'solid' } });
}

// ---------------------------------------------------------------- 1 Title
{
  const s = pres.addSlide({ masterName: 'DARK' });
  s.addText('MSc project defense  ·  Astana IT University  ·  16 November 2026', { x: M + 0.1, y: 1.0, w: 6.4, h: 0.4, fontSize: 14,
    color: C.ice, margin: 0, isTextBox: true });
  s.addText('VisuoGrasp V4', { x: M + 0.1, y: 1.55, w: 6.4, h: 1.0, fontSize: 54, bold: true, color: C.white, margin: 0, isTextBox: true });
  s.addText('6-DoF Robotic Grasping with Imitation and Reinforcement Learning', { x: M + 0.1, y: 2.6, w: 6.2, h: 1.1, fontSize: 26,
    color: 'DCE6F0', margin: 0, valign: 'top', isTextBox: true });
  s.addText('Madiyar Bolatov  ·  Amanbek Faizolla  ·  Alisher Amangeldy', { x: M + 0.1, y: 4.25, w: 6.4, h: 0.4, fontSize: 16,
    color: C.white, margin: 0, isTextBox: true });
  s.addText([{ text: '82.0 %', options: { fontSize: 30, bold: true, color: C.green } },
             { text: '  GT grasp success of the best learned policy (E6) on 540 unseen episodes', options: { fontSize: 14, color: 'DCE6F0' } }],
    { x: M + 0.1, y: 5.05, w: 6.3, h: 0.9, margin: 0, valign: 'middle', isTextBox: true });
  const h = img(s, RR('e6_oblique_bottle_6715100/key_contact_side.png'), 7.2, 1.0, 5.55, 640, 480,
    { rounding: false });
  s.addText('Real MuJoCo frame: learned E6 policy closing on a bottle from a 45° approach (simulator object pose and contacts as input)',
    { x: 7.2, y: 1.0 + h + 0.1, w: 5.55, h: 0.6, fontSize: 11, color: 'B8C4D1', margin: 0, valign: 'top', isTextBox: true });
  notes(s, 1, 'Title', 'Good afternoon. We are presenting VisuoGrasp V4, our final project on six-degree-of-freedom robotic grasping. ' +
    'The picture on the right is a real frame from our MuJoCo simulation: our best learned policy, called E6, is closing a four-finger hand on a bottle ' +
    'from a tilted, 45-degree approach. In the next ten minutes we will show three things: how we upgraded the robot from vertical-only grasping to a ' +
    'physically controlled 6-DoF wrist, which learning methods we tried and why most of them did not beat the expert, and the final result: 82 percent ' +
    'ground-truth grasp success on 540 unseen test episodes. One honest point up front: the policies receive the object pose and contacts from the ' +
    'simulator, so this is state-based control, not vision-based control.');
}

// ---------------------------------------------------------------- 2 Problem and motivation
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('From vertical-only grasping to 6-DoF approaches', { placeholder: 'title' });
  pill(s, M, 1.38, 'V3.3', C.gray, 0.75);
  s.addText('fixed vertical wrist', { x: M + 0.85, y: 1.38, w: 4, h: 0.32, fontSize: 14, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
  const h1 = img(s, G('gif_v33_level_a_success.gif'), M, 1.85, 5.3, 560, 420);
  caption(s, 'V3.3 Level A, SAC with vision-confirmed contact (M2): fingers only, scripted lift. 2× speed.', M, 1.85 + h1 + 0.08, 5.3);
  mp4link(s, V33MP4, M, 1.85 + h1 + 0.6, 5.3);
  const x2 = 6.35;
  pill(s, x2, 1.38, 'V4', C.green, 0.6);
  s.addText('free 6-DoF wrist: position and orientation', { x: x2 + 0.7, y: 1.38, w: 5.5, h: 0.32, fontSize: 14, color: C.ink, margin: 0,
    valign: 'middle', isTextBox: true });
  const wI = 3.1;
  const hI = img(s, RR('e6_oblique_bottle_6715100/key_contact_side.png'), x2, 1.85, wI, 640, 480);
  img(s, RR('scripted_demo_side_box_tall_7000007/key_contact_side.png'), x2 + wI + 0.18, 1.85, wI, 640, 480);
  caption(s, '45° oblique (learned E6 policy)', x2, 1.85 + hI + 0.08, wI);
  caption(s, '90° side (scripted expert, not learned)', x2 + wI + 0.18, 1.85 + hI + 0.08, wI);
  s.addText([
    { text: 'V3.3: 16 finger joints + at most 3-DoF translation; the hand always points down', options: { bullet: true, breakLine: true } },
    { text: 'Tilted approaches are impossible without wrist rotation', options: { bullet: true, breakLine: true } },
    { text: 'V4 goal: one policy that approaches, orients, closes, lifts and holds', options: { bullet: true } },
  ], { x: x2, y: 4.85, w: 6.4, h: 1.5, fontSize: 16, color: C.ink, margin: 0, valign: 'top', paraSpaceAfter: 6, isTextBox: true });
  notes(s, 2, 'Problem and motivation', 'Our previous version, V3.3, is on the left. It used the same four-finger, sixteen-joint hand, but the wrist ' +
    'could at most translate; the hand always pointed straight down. That is fine for a box on a table, but it cannot grasp an object from the side or at an ' +
    'angle, which many real objects require. V3.3 also focused on vision-confirmed contact; its best learned policy reached about 63 percent on its own task. ' +
    'For V4 we kept the hand but gave it a free six-degree-of-freedom wrist. On the right you see a 45-degree grasp by our learned policy and a 90-degree ' +
    'side grasp. The side grasp is performed by the scripted expert; we did not train a learned policy for side grasps. The task is now much harder: the ' +
    'policy must approach, orient the hand, close the fingers, lift and hold the object.');
}

// ---------------------------------------------------------------- 3 What changed in V4
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('What changed in V4', { placeholder: 'title' });
  const stats = [['6-DoF', 'wrist pose', 'free hand body pulled by a mocap target through a weld constraint: physics, not teleporting', C.green],
                 ['22', 'continuous actions', '16 finger Δ-targets + 3 translations + 3 rotations, bounded per 50 ms step', C.blue],
                 ['130-D', 'observation', 'fingers, wrist, fingertips, object pose, grasp targets, contacts, previous action', C.orange],
                 ['5 × 3', 'objects × approaches', 'box, cylinder, bottle, ring, L-shape; top-down, 45°, 90° (learned: top + 45°)', C.gray]];
  const cw = (W - 2 * M - 3 * 0.25) / 4;
  stats.forEach((st, i) => {
    const x = M + i * (cw + 0.25);
    card(s, x, 1.4, cw, 2.15, C.bg2);
    s.addText(st[0], { x: x + 0.2, y: 1.5, w: cw - 0.4, h: 0.8, fontSize: 40, bold: true, color: st[3], margin: 0, isTextBox: true });
    s.addText(st[1], { x: x + 0.2, y: 2.28, w: cw - 0.4, h: 0.35, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(st[2], { x: x + 0.2, y: 2.65, w: cw - 0.4, h: 0.85, fontSize: 12, color: C.ink2, margin: 0, valign: 'top', isTextBox: true });
  });
  const h = img(s, A('fig_objects_e6.jpg'), M, 3.85, W - 2 * M, 2006, 300);
  caption(s, 'Real frames from the final test: the learned E6 policy lifting all five object families (box, cylinder, bottle, L-shape, ring).',
    M, 3.85 + h + 0.08, W - 2 * M);
  s.addText('Also new: true centre of mass and inertia, randomised size, mass (50–300 g) and friction (0.5–1.2), Python package instead of notebook patches.',
    { x: M, y: 6.4, w: W - 2 * M, h: 0.4, fontSize: 13, color: C.ink, margin: 0, isTextBox: true });
  notes(s, 3, 'What changed in V4', 'Four numbers summarise the upgrade. First, a six-degree-of-freedom wrist. The hand is a free rigid body; a ' +
    'virtual target, a so-called mocap body, is moved by the policy and a weld constraint pulls the hand towards it. So contacts push back and the hand ' +
    'carries the object weight; nothing is teleported. Second, the action is 22-dimensional: 16 finger target increments plus three translations and three ' +
    'rotations, each bounded per 50-millisecond step. Third, the observation is 130-dimensional. Fourth, five object families and three approach ' +
    'directions; the learned policies were trained and tested on top-down and 45-degree grasps. The strip shows our learned policy lifting every object ' +
    'family. We also modelled true centre of mass and inertia and rebuilt the code as a tested Python package.');
}

// ---------------------------------------------------------------- 4 Robotic environment
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('The robotic environment in motion', { placeholder: 'title' });
  const gw = (W - 2 * M - 0.3) / 2;
  pill(s, M, 1.35, 'LEARNED  ·  E6 policy  ·  45° oblique', C.green, 3.6);
  pill(s, M + gw + 0.3, 1.35, 'SCRIPTED EXPERT  ·  90° side approach', C.gray, 3.6);
  const h = img(s, G('gif_e6_oblique_bottle.gif'), M, 1.8, gw, 900, 405);
  img(s, G('gif_scripted_side_box_tall.gif'), M + gw + 0.3, 1.8, gw, 900, 405);
  caption(s, 'Bottle, final-test seed 6715100, GT success. Overview and side cameras, 2× speed.', M, 1.8 + h + 0.06, gw);
  caption(s, 'Tall box, demo seed 7000007, GT success. Side grasps were never learned. 2× speed.', M + gw + 0.3, 1.8 + h + 0.06, gw);
  mp4link(s, RR('e6_oblique_bottle_6715100/e6_oblique_bottle_6715100_all.mp4'), M, 1.8 + h + 0.4, gw);
  mp4link(s, RR('scripted_demo_side_box_tall_7000007/scripted_demo_side_box_tall_7000007_all.mp4'), M + gw + 0.3, 1.8 + h + 0.4, gw);
  const facts = [['Physics', 'MuJoCo, 2.5 ms steps, 20 Hz control, 16 s episodes (320 actions)'],
                 ['Randomised', '4 size bins, mass 50–300 g, friction 0.5–1.2, resting pose and yaw'],
                 ['Success (ground truth)', 'object centre ≥ 10 cm up for 3 s, thumb pad + one more pad pressing ≥ 0.2 N']];
  const fw = (W - 2 * M - 2 * 0.25) / 3;
  facts.forEach((f, i) => {
    const x = M + i * (fw + 0.25);
    card(s, x, 5.25, fw, 1.4, C.bg2);
    s.addText(f[0], { x: x + 0.2, y: 5.33, w: fw - 0.4, h: 0.35, fontSize: 16, bold: true, color: C.navy, margin: 0, isTextBox: true });
    s.addText(f[1], { x: x + 0.2, y: 5.7, w: fw - 0.4, h: 0.85, fontSize: 14, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
  notes(s, 4, 'Robotic environment', 'Here the environment is in motion. On the left, our learned policy E6 grasps a bottle from 45 degrees on an ' +
    'unseen test episode: approach, contact, lift and a three-second hold. On the right, the scripted expert performs a side grasp of a tall box; we show it ' +
    'to illustrate the third approach direction, but it is scripted, not learned. Both clips play at twice real speed. The physics runs at 2.5 milliseconds per ' +
    'step, the policy acts at 20 hertz, and an episode lasts 16 seconds. Object size, mass, friction and resting pose are randomised every episode. Success ' +
    'is judged only by simulator ground truth, using the same definition as in the course specification: the object must stay at least 10 centimetres up for ' +
    'three seconds while the thumb and another finger press on it.');
}

// ---------------------------------------------------------------- 5 System architecture
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('System architecture: what the policy sees and controls', { placeholder: 'title' });
  const y = 1.75, h = 1.55, bw = 2.12, gap = 0.36;
  const boxes = [['MuJoCo physics', '2.5 ms step\n20 substeps per action', C.bg2, C.gray],
                 ['PRIVILEGED input', 'object pose + extents\nGT pad contacts', 'FDE8DC', C.orange],
                 ['130-D observation', 'proprioception, wrist,\nobject, grasp targets', C.bg2, C.gray],
                 ['Actor policy (learned)', 'MLP 256×256\n22-D action', 'DFF3EA', C.green],
                 ['Hand controller', '16 finger Δ-targets\n6-DoF mocap + weld', C.bg2, C.gray]];
  boxes.forEach((b, i) => {
    const x = M + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h, fill: { color: b[2] }, line: { color: b[3], width: 2 }, rectRadius: 0.12 });
    s.addText(b[0], { x: x + 0.1, y: y + 0.15, w: bw - 0.2, h: 0.4, fontSize: 15, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
    s.addText(b[1], { x: x + 0.1, y: y + 0.6, w: bw - 0.2, h: 0.8, fontSize: 13, color: C.ink2, align: 'center', margin: 0, isTextBox: true });
    if (i < 4) arrow(s, x + bw + 0.03, y + h / 2, x + bw + gap - 0.03, y + h / 2);
  });
  // feedback loop above
  const xl = M + bw / 2, xr = M + 4 * (bw + gap) + bw / 2;
  s.addShape(pres.shapes.LINE, { x: xl, y: 1.42, w: xr - xl, h: 0, line: { color: C.ink2, width: 1.75 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: 1.42, w: 0, h: y - 1.42, line: { color: C.ink2, width: 1.75 } });
  arrow(s, xl, 1.42, xl, y - 0.02);
  s.addText('action every 50 ms: the physics moves the hand through the weld constraint', { x: xl + 0.25, y: 1.12, w: 7.0, h: 0.28, fontSize: 11,
    color: C.ink2, margin: 0, isTextBox: true });
  const y2 = 4.05, h2 = 1.25;
  const lower = [[2, 'Analytic grasp planner', 'top / 45° / 90° poses\nfrom the PRIVILEGED pose', 'E7EFFA', C.blue],
                 [3, 'Scripted expert', 'demonstrations,\nbaseline (not learned)', 'E7EFFA', C.blue],
                 [4, 'Training', 'BC · SAC · E6 (AWAC-style)', 'DFF3EA', C.green]];
  lower.forEach((b) => {
    const x = M + b[0] * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: y2, w: bw, h: h2, fill: { color: b[3] }, line: { color: b[4], width: 2 }, rectRadius: 0.12 });
    s.addText(b[1], { x: x + 0.1, y: y2 + 0.12, w: bw - 0.2, h: 0.4, fontSize: 15, bold: true, color: C.ink, align: 'center', margin: 0, isTextBox: true });
    s.addText(b[2], { x: x + 0.1, y: y2 + 0.52, w: bw - 0.2, h: 0.65, fontSize: 12, color: C.ink2, align: 'center', margin: 0, isTextBox: true });
  });
  const xp = M + 2 * (bw + gap), xs = M + 3 * (bw + gap), xt = M + 4 * (bw + gap);
  arrow(s, xp + bw / 2, y2 - 0.03, xp + bw / 2, y + h + 0.03, C.blue, 'dash');
  arrow(s, xp + bw + 0.03, y2 + h2 / 2, xs - 0.03, y2 + h2 / 2, C.blue);
  arrow(s, xs + bw + 0.03, y2 + h2 / 2, xt - 0.03, y2 + h2 / 2, C.blue);
  arrow(s, xt + bw / 2, y2 - 0.03, xs + bw / 2 + 0.3, y + h + 0.03, C.green, 'dash');
  card(s, M, y2, 2 * bw + gap, h2, 'FFF6EC', C.orange);
  s.addText([{ text: 'Not used by the final policies', options: { bold: true, color: C.orange, breakLine: true } },
             { text: 'RGB-D cameras, colour segmentation and D0 contact exist in V4, but every reported policy uses simulator pose + GT contacts.',
               options: { color: C.ink } }],
    { x: M + 0.18, y: y2 + 0.1, w: 2 * bw + gap - 0.36, h: h2 - 0.2, fontSize: 13, margin: 0, valign: 'middle', isTextBox: true });
  let px = M;
  [['privileged simulator information', C.orange], ['learned', C.green], ['scripted / analytic', C.blue], ['physics / interface', C.gray]].forEach((l) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: px, y: 5.75, w: 0.3, h: 0.22, fill: { color: l[1] }, line: { color: l[1] }, rectRadius: 0.05 });
    s.addText(l[0], { x: px + 0.38, y: 5.7, w: 2.9, h: 0.32, fontSize: 12, color: C.ink2, margin: 0, valign: 'middle', isTextBox: true });
    px += 0.38 + 0.085 * l[0].length + 0.45;
  });
  notes(s, 5, 'System architecture', 'This is the information flow. MuJoCo simulates the physics. From the simulator we take privileged information, ' +
    'marked in orange: the exact object pose and size, and ground-truth contact flags on the four finger pads. Together with joint positions, velocities ' +
    'and the wrist pose this forms the 130-dimensional observation. The actor, in green, is the only learned component at run time; it outputs 22 numbers. ' +
    'The hand controller turns them into finger targets and a new wrist target, and the physics moves the hand. In blue, the analytic grasp planner computes ' +
    'the target grasp pose from the object pose, and the scripted expert uses it to generate demonstrations. The orange box is important: our V4 code has ' +
    'cameras, colour segmentation and a geometric contact detector, but none of the reported policies uses them.');
}

// ---------------------------------------------------------------- 6 RL formulation
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('Reinforcement-learning formulation', { placeholder: 'title' });
  const items = [['State  s', '130-D, normalised, clipped to ±10; object pose and contacts are PRIVILEGED'],
                 ['Action  a', '[−1, 1]²²: 16 finger Δ-targets, 3 translations, 3 rotations'],
                 ['Transition', 'MuJoCo, 20 substeps per action; end: drop, object off table, numerical failure, or 320 steps'],
                 ['Return', 'γ = 0.99; SAC critics use 10-step returns; twin Q-networks with LayerNorm']];
  const cw = (W - 2 * M - 3 * 0.25) / 4;
  items.forEach((it, i) => {
    const x = M + i * (cw + 0.25);
    card(s, x, 1.4, cw, 1.75, C.bg2);
    s.addText(it[0], { x: x + 0.2, y: 1.5, w: cw - 0.4, h: 0.4, fontSize: 18, bold: true, color: C.navy, margin: 0, isTextBox: true });
    s.addText(it[1], { x: x + 0.2, y: 1.95, w: cw - 0.4, h: 1.1, fontSize: 13, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
  s.addText('Reward v4r1 (potential-based shaping + sparse terms)', { x: M, y: 3.4, w: 8, h: 0.35, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true });
  img(s, A('eq_reward.png'), M, 3.85, 11.4, 2729, 103);
  img(s, A('eq_phi.png'), M, 4.4, 9.1, 2161, 104);
  s.addText('Wrist and finger update per action (quaternion composition, then 4 safety projections)', { x: M, y: 5.05, w: 10, h: 0.35, fontSize: 15,
    bold: true, color: C.ink, margin: 0, isTextBox: true });
  img(s, A('eq_wrist.png'), M, 5.5, 9.6, 2322, 156);
  notes(s, 6, 'RL formulation', 'Formally, the task is a Markov decision process. The state is the 130-dimensional observation; the action is a vector ' +
    'of 22 numbers between minus one and one. One action is held for 20 physics substeps, and an episode ends after 320 actions or earlier if the object ' +
    'drops, leaves the table, or the simulation becomes numerically unstable. The reward has two parts. Most of it is potential-based shaping: a potential ' +
    'Phi rewards being close to the planned grasp pose, aligned with it, in contact, and lifting the object, and penalises pushing the object before it is ' +
    'lifted. Because it is a difference of potentials, it guides learning without changing which policy is optimal. On top come a success bonus of 10, a ' +
    'drop penalty, and small penalties for table contact and large actions. The wrist update composes the current orientation quaternion with a small rotation ' +
    'and renormalises it, which avoids gimbal lock.');
}

// ---------------------------------------------------------------- 7 Learning methods
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('Learning methods: from pure RL to expert-anchored RL', { placeholder: 'title' });
  const cards = [['Autonomous SAC', 'reward only (from scratch, demo replay, curriculum)', '0 – 18.9 %', 'expert 77.8 %', C.orange, 'RL'],
                 ['Residual SAC', 'learns corrections added to the scripted controller', '49.4 / 69.1 %', 'expert 77.8 / 76.7 %', C.amber, 'RL'],
                 ['Behaviour cloning', 'supervised regression on expert actions', '74.6 % (W4)', 'expert 77.8 %', C.blue, 'IMITATION'],
                 ['BC ensemble', 'mean action of 6 BC actors with different input masks', '80.7 %', 'expert 80.6 %', C.blue, 'IMITATION'],
                 ['E6 BC-anchored RL', 'BC actor fine-tuned with advantage-weighted regression + BC anchor', '82.0 %', 'expert 80.6 %', C.green, 'BC + RL']];
  const cw = (W - 2 * M - 4 * 0.2) / 5;
  cards.forEach((c, i) => {
    const x = M + i * (cw + 0.2);
    card(s, x, 1.4, cw, 3.25, i === 4 ? 'E8F7F0' : C.bg2, i === 4 ? C.green : C.bg2);
    pill(s, x + 0.18, 1.55, c[5], c[4], 0.25 + 0.1 * c[5].length);
    s.addText(c[0], { x: x + 0.18, y: 2.0, w: cw - 0.36, h: 0.65, fontSize: 17, bold: true, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
    s.addText(c[1], { x: x + 0.18, y: 2.65, w: cw - 0.36, h: 0.95, fontSize: 12, color: C.ink2, margin: 0, valign: 'top', isTextBox: true });
    s.addText(c[2], { x: x + 0.18, y: 3.62, w: cw - 0.36, h: 0.55, fontSize: 24, bold: true, color: c[4], margin: 0, isTextBox: true });
    s.addText(c[3], { x: x + 0.18, y: 4.15, w: cw - 0.36, h: 0.35, fontSize: 12, color: C.ink2, margin: 0, isTextBox: true });
    if (i < 4) arrow(s, x + cw + 0.02, 3.0, x + cw + 0.18, 3.0, C.mute);
  });
  s.addText('GT success on each wave\'s fresh test set; the expert value is measured on the same episodes.', { x: M, y: 4.75, w: W - 2 * M, h: 0.3,
    fontSize: 12, color: C.ink2, margin: 0, isTextBox: true });
  s.addText('E6 actor objective (as implemented in wave5.py)', { x: M, y: 5.2, w: 8, h: 0.35, fontSize: 15, bold: true, color: C.ink, margin: 0, isTextBox: true });
  img(s, A('eq_e6.png'), M, 5.62, 10.2, 2474, 135);
  notes(s, 7, 'Learning methods', 'We compared five families of methods, from left to right with increasing use of the expert. Pure SAC learns from the ' +
    'reward alone; even with demonstrations in the replay buffer or a curriculum it reached at most 19 percent. Residual SAC learns corrections on top of the ' +
    'scripted controller; the corrections made it worse. Behaviour cloning is supervised learning, not reinforcement learning: it regresses the expert ' +
    'actions and reached 74.6 percent. An ensemble of six behaviour-cloning actors matched the expert. E6 starts from a behaviour-cloning actor and fine-tunes ' +
    'it with an AWAC-style objective: it regresses towards actions that the critic rates better than the current policy, weighted by the exponential of the ' +
    'advantage, plus a behaviour-cloning anchor on expert data. The actor never follows the critic gradient directly, so it cannot drift far from the expert.');
}

// ---------------------------------------------------------------- 8 Experimental development
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('Five experimental waves: each failure shaped the next design', { placeholder: 'title' });
  const waves = [['W1', 'Autonomous SAC', '0 / 10.5 %', 'vs 77.8 %', 'Reward alone is not enough: objects are pushed away', C.orange],
                 ['W2', 'Residual SAC', '49.4 %', 'vs 77.8 %', 'Free corrections break a good controller', C.amber],
                 ['W3', 'Conservative residual', '69.1 %', 'vs 76.7 %', 'Smaller harm, still no gain', C.amber],
                 ['W4', 'Curriculum SAC · BC', '18.9 · 74.6 %', 'vs 77.8 %', 'Imitation works; SAC fine-tuning hurts BC (48.7 %)', C.blue],
                 ['W5', 'BC ensemble · E6', '80.7 · 82.0 %', 'vs 80.6 %', 'Anchored RL improves its BC start (+2.6 pp)', C.green]];
  const yl = 2.55, step = (W - 2 * M) / 5;
  s.addShape(pres.shapes.LINE, { x: M + step / 2, y: yl, w: 4 * step, h: 0, line: { color: C.line, width: 3 } });
  waves.forEach((w, i) => {
    const cx = M + step * i + step / 2;
    s.addShape(pres.shapes.OVAL, { x: cx - 0.38, y: yl - 0.38, w: 0.76, h: 0.76, fill: { color: w[5] }, line: { color: C.white, width: 2 } });
    s.addText(w[0], { x: cx - 0.38, y: yl - 0.38, w: 0.76, h: 0.76, fontSize: 18, bold: true, color: C.white, align: 'center', valign: 'middle', margin: 0,
      isTextBox: true });
    s.addText(w[1], { x: cx - step / 2 + 0.1, y: 1.45, w: step - 0.2, h: 0.6, fontSize: 15, bold: true, color: C.ink, align: 'center', valign: 'bottom',
      margin: 0, isTextBox: true });
    s.addText(w[2], { x: cx - step / 2 + 0.05, y: 3.1, w: step - 0.1, h: 0.6, fontSize: 26, bold: true, color: w[5], align: 'center', margin: 0, isTextBox: true });
    s.addText('learned ' + w[3] + ' expert', { x: cx - step / 2 + 0.05, y: 3.68, w: step - 0.1, h: 0.32, fontSize: 12, color: C.ink2, align: 'center',
      margin: 0, isTextBox: true });
    card(s, cx - step / 2 + 0.1, 4.2, step - 0.2, 1.25, C.bg2);
    s.addText(w[4], { x: cx - step / 2 + 0.22, y: 4.25, w: step - 0.44, h: 1.15, fontSize: 14, color: C.ink, align: 'center', valign: 'middle', margin: 0,
      isTextBox: true });
  });
  s.addText('Each wave used its own unseen test seeds (5.1M, 6.1M, 6.6M, 6.7M); compare each learned result with the expert on the same episodes, not across waves.',
    { x: M, y: 5.75, w: W - 2 * M, h: 0.55, fontSize: 13, color: C.ink2, margin: 0, isTextBox: true });
  notes(s, 8, 'Experimental development', 'We ran five waves, and each one was stopped or redesigned by a pre-registered rule. Wave 1: SAC from scratch ' +
    'learned nothing useful; the hand mostly pushed the object away. Wave 2: we let SAC add corrections to the scripted controller; the corrections cut ' +
    'success from 78 to 49 percent. Wave 3: smaller, conservative corrections reduced the harm but still did not help. Wave 4: a curriculum made autonomous SAC ' +
    'better than in Wave 1, but only 19 percent; behaviour cloning of the expert reached 75 percent, and ordinary SAC fine-tuning of that clone made it worse. ' +
    'That led to Wave 5: better imitation and an RL update that is anchored to the expert. Each wave used fresh test seeds, so the fair comparison is always ' +
    'against the expert on the same episodes.');
}

// ---------------------------------------------------------------- 9 Final performance
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('Final test: 7 controllers on the same 540 unseen episodes', { placeholder: 'title' });
  const labels = ['E6b BC-anchored RL', 'E6 BC-anchored RL', 'E5c (E6 initialisation)', 'BC ensemble', 'Improved BC (E5g)', 'BC Wave 4', 'Scripted expert'];
  const vals = [78.5, 82.0, 79.4, 80.7, 77.4, 77.4, 80.6];
  // bar chart drawn with shapes so that the decimal point does not depend on the viewer's locale (PowerPoint charts use it)
  const cols = ['7FD1B0', C.green, '8DB8EA', C.blue, '8DB8EA', '8DB8EA', C.gray];
  const order = [6, 5, 4, 3, 2, 1, 0];
  const x0 = M + 2.55, bwMax = 3.9, lo = 60, hi = 90, y0 = 1.85, rowH = 0.6;
  const X = (v) => x0 + bwMax * (v - lo) / (hi - lo);
  s.addText('GT grasp success, all 540 episodes (axis starts at 60 %)', { x: M, y: 1.3, w: 7.2, h: 0.35, fontSize: 13, color: C.ink2, margin: 0,
    isTextBox: true });
  [60, 70, 80, 90].forEach((g) => {
    s.addShape(pres.shapes.LINE, { x: X(g), y: y0 - 0.1, w: 0, h: rowH * 7 + 0.05, line: { color: 'E4E7EB', width: 0.75 } });
    s.addText(g + '%', { x: X(g) - 0.4, y: y0 + rowH * 7 - 0.02, w: 0.8, h: 0.3, fontSize: 12, color: C.ink2, align: 'center', margin: 0, isTextBox: true });
  });
  order.forEach((i, r) => {
    const y = y0 + r * rowH;
    s.addText(labels[i], { x: M, y, w: 2.4, h: rowH - 0.12, fontSize: 13, color: C.ink, align: 'right', valign: 'middle', margin: 0, isTextBox: true,
      bold: i === 1 });
    s.addShape(pres.shapes.RECTANGLE, { x: x0, y: y + 0.04, w: X(vals[i]) - x0, h: rowH - 0.2, fill: { color: cols[i] }, line: { color: cols[i] } });
    s.addText(vals[i].toFixed(1) + ' %', { x: X(hi) + 0.15, y, w: 1.0, h: rowH - 0.12, fontSize: 14, bold: i === 1, color: C.ink, valign: 'middle',
      margin: 0, isTextBox: true });
  });
  s.addShape(pres.shapes.LINE, { x: X(80.6), y: y0 - 0.1, w: 0, h: rowH * 7 + 0.05, line: { color: C.gray, width: 1.25, dashType: 'dash' } });
  s.addText('expert', { x: X(80.6) - 0.5, y: y0 - 0.42, w: 1.0, h: 0.28, fontSize: 11, color: C.gray, align: 'center', margin: 0, isTextBox: true });
  const x = 8.4, cw = W - M - x;
  card(s, x, 1.35, cw, 2.35, 'E8F7F0', C.green);
  s.addText('E6 BC-anchored RL', { x: x + 0.25, y: 1.45, w: cw - 0.5, h: 0.4, fontSize: 16, bold: true, color: C.ink, margin: 0, isTextBox: true });
  s.addText('82.0 %', { x: x + 0.25, y: 1.85, w: cw - 0.5, h: 0.85, fontSize: 48, bold: true, color: C.green, margin: 0, isTextBox: true });
  s.addText('Wilson 95 % CI [78.6, 85.0]   ·   0 numerical failures', { x: x + 0.25, y: 2.75, w: cw - 0.5, h: 0.35, fontSize: 13, color: C.ink, margin: 0, isTextBox: true });
  s.addText('Expert 80.6 % [77.0, 83.7] · BC ensemble 80.7 % [77.2, 83.8]', { x: x + 0.25, y: 3.12, w: cw - 0.5, h: 0.5, fontSize: 12, color: C.ink2,
    margin: 0, isTextBox: true });
  card(s, x, 3.95, cw, 2.3, C.bg2);
  s.addText([{ text: 'Oblique 45° is the hard case', options: { bold: true, breakLine: true } },
             { text: 'E6 68.1 % · expert 64.4 % · BC W4 58.9 %', options: { breakLine: true } },
             { text: 'Top-down: 93–97 % for every controller', options: { bold: true, breakLine: true } },
             { text: 'CIs overlap → decisions need paired tests', options: { color: C.orange, bold: true } }],
    { x: x + 0.25, y: 4.05, w: cw - 0.5, h: 2.1, fontSize: 14, color: C.ink, margin: 0, valign: 'top', paraSpaceAfter: 6, isTextBox: true });
  notes(s, 9, 'Final performance', 'This is the main result. After freezing every model, we evaluated all seven controllers once on the same 540 ' +
    'unseen episodes. E6 reached the highest observed success, 82.0 percent, with a 95 percent Wilson interval from 78.6 to 85. The scripted expert reached ' +
    '80.6 and the behaviour-cloning ensemble 80.7. Notice that the axis starts at 60 percent, and that the confidence intervals overlap, so from this chart ' +
    'alone we cannot rank these three. Top-down grasps are almost solved by every controller, at 93 to 97 percent. The differences come from 45-degree ' +
    'grasps, where E6 reaches 68 percent compared with 64 for the expert. To decide whether differences are real, we use paired tests on identical ' +
    'episodes, on the next slide.');
}

// ---------------------------------------------------------------- 10 Contribution of RL
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('RL fine-tuning significantly improved its own BC start', { placeholder: 'title' });
  const bx = M, by = 1.45;
  s.addText('BC initialisation (E5c)', { x: bx, y: by, w: 3.0, h: 0.35, fontSize: 14, color: C.ink2, margin: 0, isTextBox: true });
  s.addText('79.4 %', { x: bx, y: by + 0.35, w: 3.0, h: 0.85, fontSize: 44, bold: true, color: C.blue, margin: 0, isTextBox: true });
  arrow(s, bx + 2.75, by + 0.78, bx + 3.45, by + 0.78, C.ink2);
  s.addText('after E6 fine-tuning', { x: bx + 3.6, y: by, w: 3.0, h: 0.35, fontSize: 14, color: C.ink2, margin: 0, isTextBox: true });
  s.addText('82.0 %', { x: bx + 3.6, y: by + 0.35, w: 3.0, h: 0.85, fontSize: 44, bold: true, color: C.green, margin: 0, isTextBox: true });
  const st = [['+2.6 pp', 'paired difference\n[+0.9, +4.3]'], ['18 / 4', 'episodes won / lost\nby E6 vs its start'], ['p = 0.039', 'Holm-corrected\nexact McNemar']];
  const sw = 2.05;
  st.forEach((x, i) => {
    const xx = bx + i * (sw + 0.15);
    card(s, xx, 2.85, sw, 1.45, 'E8F7F0', C.green);
    s.addText(x[0], { x: xx + 0.15, y: 2.92, w: sw - 0.3, h: 0.6, fontSize: 26, bold: true, color: C.green, margin: 0, isTextBox: true });
    s.addText(x[1], { x: xx + 0.15, y: 3.5, w: sw - 0.3, h: 0.75, fontSize: 12, color: C.ink, margin: 0, valign: 'top', isTextBox: true });
  });
  s.addText('Gain is in 45° grasps: +5.6 pp (16 won / 1 lost); fewer objects pushed before closing.', { x: bx, y: 4.45, w: 6.5, h: 0.5, fontSize: 14,
    color: C.ink, margin: 0, isTextBox: true });
  card(s, bx, 5.05, 6.45, 1.3, 'FFF6EC', C.orange);
  s.addText([{ text: 'What this does NOT show', options: { bold: true, color: C.orange, breakLine: true } },
             { text: 'E6 vs expert +1.5 pp, p = 0.096 · vs BC ensemble p = 0.19', options: { breakLine: true } },
             { text: 'Replication E6b: +1.1 pp, p = 0.39 · one training seed per run' }],
    { x: bx + 0.2, y: 5.1, w: 6.1, h: 1.2, fontSize: 13, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
  const gx = 7.45, gw = W - M - gx;
  const gh = img(s, G('gif_scripted_vs_e6_oblique_box_6706109.gif'), gx, 1.3, gw, 800, 720);
  caption(s, 'Same unseen episode (seed 6706109): the scripted expert pushes the box (top); E6 grasps, lifts and holds it (bottom). 2× speed.',
    gx, 1.3 + gh + 0.06, gw, { fontSize: 11 });
  mp4link(s, W5MP4, gx, 1.3 + gh + 0.5, gw);
  notes(s, 10, 'Contribution of RL', 'Here is the contribution of reinforcement learning. E6 started from a behaviour-cloning actor, E5c, that scored ' +
    '79.4 percent. After fine-tuning it scored 82.0, a paired difference of 2.6 points. On the same episodes, E6 succeeded on 18 episodes where its ' +
    'starting policy failed, and failed on only 4 where it had succeeded. The exact McNemar test, corrected for the 12 comparisons we had listed in advance, ' +
    'gives p equals 0.039, below 0.05. Almost all of the gain is in 45-degree grasps, mainly because E6 pushes the object less before closing; the clip ' +
    'shows one such episode. What it does not show: E6 is not significantly better than the expert or the ensemble, and our second run, E6b, did not ' +
    'reproduce a significant gain. With one training seed per run, we report this as a real effect for E6, not a general law.');
}

// ---------------------------------------------------------------- 11 Limitations and future work
{
  const s = pres.addSlide({ masterName: 'CONTENT' });
  s.addText('Limitations and future work', { placeholder: 'title' });
  const rows = [['Privileged state', 'policy sees simulator pose + GT contacts', 'train V4 with vision: D2 contact for the moving wrist camera'],
                ['One training seed', 'per RL configuration; E6b did not replicate', '3–5 seeds per configuration, as the course asks'],
                ['No win over the expert', '+1.5 pp, p = 0.096', 'longer anchored RL, more diverse demonstrations'],
                ['Oblique flat ring', '0 / 20 for every controller incl. the expert', 'better grasp synthesis for thin objects'],
                ['Side grasps', 'only scripted (61 % on feasible tall objects)', 'learn the 90° approach']];
  const y0 = 1.4, rh = 0.92;
  s.addText('Limitation', { x: M, y: y0 - 0.05, w: 2.4, h: 0.3, fontSize: 12, bold: true, color: C.mute, margin: 0, isTextBox: true });
  s.addText('Next step', { x: M + 4.05, y: y0 - 0.05, w: 3.0, h: 0.3, fontSize: 12, bold: true, color: C.mute, margin: 0, isTextBox: true });
  rows.forEach((r, i) => {
    const y = y0 + 0.32 + i * rh;
    card(s, M, y, 7.3, rh - 0.14, C.bg2);
    s.addText(r[0], { x: M + 0.15, y: y + 0.05, w: 3.7, h: 0.35, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
    s.addText(r[1], { x: M + 0.15, y: y + 0.38, w: 3.7, h: 0.35, fontSize: 11, color: C.ink2, margin: 0, isTextBox: true });
    s.addText(r[2], { x: M + 4.05, y: y + 0.05, w: 3.15, h: rh - 0.24, fontSize: 12, color: C.navy, margin: 0, valign: 'middle', isTextBox: true });
  });
  const gx = 8.25, gw = W - M - gx;
  pill(s, gx, 1.35, 'FAILURE  ·  E6  ·  oblique ring', C.orange, 3.2);
  const gh = img(s, G('gif_e6_fail_ring_oblique.gif'), gx, 1.8, gw, 900, 405);
  caption(s, 'Fingers close above the flat ring and never touch it (seed 6719100, no contact). 3× speed.', gx, 1.8 + gh + 0.06, gw, { fontSize: 11 });
  mp4link(s, RR('e6_fail_ring_oblique_6719100/e6_fail_ring_oblique_6719100_all.mp4'), gx, 1.8 + gh + 0.48, gw);
  card(s, gx, 4.55, gw, 1.75, C.bg2);
  s.addText([{ text: 'Why E6 fails (97 of 540 episodes)', options: { bold: true, breakLine: true } },
             { text: 'never lifted 46 · pushed before closing 38', options: { breakLine: true } },
             { text: 'no contact 7 · dropped 3 · no thumb–pad pair 3', options: { breakLine: true } },
             { text: 'Expert: 105 failures, same two main modes', options: { color: C.ink2 } }],
    { x: gx + 0.2, y: 4.62, w: gw - 0.4, h: 1.6, fontSize: 13, color: C.ink, margin: 0, valign: 'middle', isTextBox: true });
  notes(s, 11, 'Limitations', 'We want to be clear about the limits. First and most important: the policy receives the object pose and ground-truth ' +
    'contacts from the simulator. Our V4 code contains cameras and a geometric contact detector, but we did not train a vision-based V4 policy; the next step ' +
    'is to bring the D2 contact detector from V3.3 to the moving wrist camera. Second, each RL configuration used one training seed, and the second run ' +
    'did not reproduce the gain. Third, E6 is not significantly better than the expert. Fourth, a flat ring approached at 45 degrees is never grasped by any ' +
    'controller, including the expert, as the clip shows; this is a limitation of the grasp planner that all learned policies imitate. Finally, side grasps ' +
    'are only scripted. The box below the clip shows that failures are dominated by never lifting and by pushing the object before closing, for E6 and the expert alike.');
}

// ---------------------------------------------------------------- 12 Conclusion
{
  const s = pres.addSlide({ masterName: 'DARK' });
  s.addText('Conclusion', { x: M + 0.1, y: 0.55, w: 6, h: 0.8, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true });
  const items = [['Engineering', 'From vertical-only grasping to a physically controlled 6-DoF wrist with 22-D actions and five object families.', C.ice],
                 ['Learning', 'Pure, residual and curriculum SAC did not reach the expert; imitation plus expert-anchored RL did.', C.ice],
                 ['Result', 'E6: 82.0 % GT [78.6, 85.0] on 540 unseen episodes; +2.6 pp over its BC start (Holm p = 0.039); not significantly above the expert (p = 0.096).', C.green]];
  items.forEach((it, i) => {
    const y = 1.6 + i * 1.45;
    s.addShape(pres.shapes.OVAL, { x: M + 0.1, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: it[2] }, line: { color: it[2] } });
    s.addText(String(i + 1), { x: M + 0.1, y: y + 0.05, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.navy, align: 'center', valign: 'middle',
      margin: 0, isTextBox: true });
    s.addText(it[0], { x: M + 0.9, y, w: 5.5, h: 0.4, fontSize: 19, bold: true, color: C.white, margin: 0, isTextBox: true });
    s.addText(it[1], { x: M + 0.9, y: y + 0.42, w: 5.6, h: 0.95, fontSize: 14, color: 'DCE6F0', margin: 0, valign: 'top', isTextBox: true });
  });
  const gx = 7.3, gw = W - M - gx;
  const gh = img(s, G('gif_e6_top_cylinder.gif'), gx, 1.7, gw, 900, 405);
  s.addText('E6 policy, top-down cylinder grasp on an unseen test episode (seed 6709000), GT success, 2× speed. PRIVILEGED state input.',
    { x: gx, y: 1.7 + gh + 0.08, w: gw, h: 0.6, fontSize: 11, color: 'B8C4D1', margin: 0, valign: 'top', isTextBox: true });
  mp4link(s, RR('e6_top_cylinder_6709000/e6_top_cylinder_6709000_all.mp4'), gx, 1.7 + gh + 0.68, gw, C.ice);
  s.addText('Thank you — questions?', { x: gx, y: 5.6, w: gw, h: 0.6, fontSize: 26, bold: true, color: C.white, margin: 0, isTextBox: true });
  notes(s, 12, 'Conclusion', 'To conclude, three messages. Engineering: we turned a vertical-only grasping setup into a physically controlled ' +
    'six-degree-of-freedom wrist with a 22-dimensional action space and five object families. Learning: reinforcement learning on its own, as residual ' +
    'corrections, or with a curriculum did not reach the expert; what worked was imitation of the expert combined with a conservative, expert-anchored RL ' +
    'update. Result: E6 reached 82 percent ground-truth success on unseen episodes and significantly improved its own behaviour-cloning start, but it is ' +
    'not significantly better than the expert, and it relies on privileged simulator state. The clip shows E6 grasping a cylinder on an unseen episode. ' +
    'Thank you; we are happy to take questions.');
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(process.env.PPTX_SKILL_THEME);
  await applyTheme(OUT, THEME);
  const md = ['# Slide speaking notes — VisuoGrasp V4 final defense', '',
    'Target: about 10 minutes (≈ 50 s per slide). The same text is stored as speaker notes in the PPTX.',
    'Numbers: `evidence/VERIFIED_RESULTS.md`. Every learned result uses PRIVILEGED state (simulator object pose + GT contacts).', ''];
  NOTES.forEach((n) => { md.push(`## Slide ${n.n} — ${n.title}`, '', n.text, ''); });
  fs.writeFileSync(path.join(FD, 'defense', 'SLIDE_SPEAKING_NOTES.md'), md.join('\n'));
  console.log('written', OUT);
})();
