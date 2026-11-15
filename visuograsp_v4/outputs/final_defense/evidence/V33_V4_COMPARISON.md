# V3.3 versus V4 (verified against the repository)

Sources:
* V3.3: `README.md` (project root), `visuograsp_full_outputs/`, the V3.3 demo videos.
* V4: `visuograsp_v4/README_V4.md`, `config.py`, `env.py`, `controller.py`, `scene.py`, `objects.py`,
  `perception.py`.

| Aspect | V3.3 (frozen, untouched) | V4 (final project) |
|---|---|---|
| Hand | Allegro-style 4 fingers, 16 hinge joints (flexion), position servos | Same 4-finger, 16-joint geometry (V3.3 finger kinematics; no abduction joints) |
| Wrist | Level A: none (scripted lift). Level B: 3-DoF translation; fixed vertical orientation | Free 6-DoF hand body with gravity compensation, driven by a mocap target + weld constraint (solref 10 ms); per step ≤ 1 cm per world axis and ≤ 0.06 rad per hand axis, quaternion composition + normalisation, 4 safety projections |
| Action | 16 finger targets (A) / 16 + 3 (B, scripted-prior residual) | 22-D: 16 finger Δ-targets (0.055 rad) + 3 translation + 3 rotation |
| Actor observation | Proprioception + CV contact (D2) + object estimate (M2) | 130-D (see report Table II); final policies use PRIVILEGED object pose and GT pad contacts |
| Contact signal | D0 / D1 / D2 (temporal CNN–GRU) vision-confirmed contact (M1/M2), GT oracle as M0 baseline | GT oracle (`contact_source = gt_oracle`) for all learned V4 policies; D0 geometric contact and colour segmentation exist in `perception.py`, but no D2 for the moving wrist camera and no V4 policy trained with vision |
| Approaches | Top-down only | Top-down, oblique 45°, side 90° in the planner; learned policies trained and tested on top-down + oblique only |
| Objects | Train: sphere, box, cylinder, ellipsoid, capsule, ring; unseen: cone, hex prism, L-shape, mug, bottle | Box (cube / tall / flat), cylinder, bottle, ring, L-shape; 4 scale bins (0.85–1.15); mass 0.05–0.30 kg; friction 0.5–1.2; several resting modes + random yaw |
| Object dynamics | Isotropic inertia at the body origin | Uniform-density geoms: true centre of mass and inertia |
| Physics | elliptic cone, impratio 10, noslip 5, dt 2.5 ms; finger armature 5e-5 | Same solver settings; finger armature 1e-3 (5e-5 caused object "explosions" in squeezed grasps: 3/486 episodes with noslip) |
| Success (GT) | COM ≥ 10 cm for 3 s with thumb + one other fingertip in contact | Same: COM ≥ 10 cm for 60 frames (3 s) with the thumb pad and one more pad ≥ 0.2 N |
| Code | Notebook with monkey-patching (`main.ipynb`) | Python package `visuograsp_v4/` (Gymnasium env, frozen dataclass configs, scripts, smoke tests, integrity manifest for V3.3) |
| Learning | SAC (LayerNorm critics, 10-step returns), 3 seeds × 100k, M0 / M2 / Blind | SAC, residual SAC, curriculum SAC, BC, BC ensemble, BC-regularised SAC, AWAC-style BC-anchored fine-tuning (Waves 1–5) |
| Best GT result | Level A SAC M2 62.6 % (scripted 86.7 %), nominal set | E6 82.0 % (scripted 80.6 %) on the V4 task, PRIVILEGED state |

**Note.** V3.3 and V4 success rates come from different tasks (fixed vertical grasp versus full
6-DoF approach; different objects and contact signals) and must not be compared directly.
