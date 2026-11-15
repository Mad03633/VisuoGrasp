# VisuoGrasp V4 — 6-DoF physical foundation (work in progress)

V4 is a new, isolated implementation. **V3.3 (`main.ipynb`, `visuograsp_full_outputs/`, archived
notebooks, report and presentation) is frozen and untouched**; this is verified by an integrity
manifest (`outputs/integrity/v33_manifest_baseline.json`, 651 files, checked by the smoke tests).

Status (2026-11-14): physical 6-DoF hand, approach-pose generator, object set, scripted
diagnostic baseline, Gymnasium interface and smoke tests are implemented. **No RL policy has been
trained in V4.** The scripted baseline uses the simulator's object pose (PRIVILEGED planning).

## Layout

| File | Content |
|---|---|
| `config.py` | All physical / control / task parameters as frozen dataclasses (`V4Config`) |
| `rotations.py` | Quaternion helpers (MuJoCo `[w,x,y,z]`, normalised, slerp, 6-D rotation) |
| `objects.py` | Box, cylinder, bottle, ring, L-shape; aspect variants; resting modes; support functions |
| `scene.py` | MJCF generation and `SceneCache` (each family/variant/scale compiled once per process) |
| `controller.py` | `HandController`: bounded 22-D action → mocap wrist target + 16 finger servo targets |
| `sim.py` | Placement + settling, stepping, ground-truth contacts, success/drop/failure bookkeeping |
| `grasp_planner.py` | Top / oblique (45°) / side (90°) grasp, pre-grasp, transit and lift poses |
| `scripted.py` | Scripted grasp phase machine (transit → descend → approach → close → lift → hold) |
| `perception.py` | Colour segmentation, RGB-D back-projection for moving cameras, PCA box, D0 contact |
| `env.py` | `VisuoGraspV4Env` (Gymnasium): 22-D action, 130-D actor observation, 23-D privileged state |
| `evaluation.py`, `video.py` | Episode runner with ground-truth metrics; MP4 writer (overview / side / wrist) |
| `scripts/` | `integrity_manifest`, `run_smoke_tests`, `run_scripted_baseline`, `record_demos` |

## Model and controller

* Hand: free rigid body (`hand_free`, 6 DoF) with `gravcomp`; palm box 11×11×2.4 cm; the four
  V3.3 fingers (16 hinge joints, flexion only, position servos kp 2.5/1.8, ±3 N·m).
* Wrist actuation: a mocap body `hand_target` is welded to the hand (`solref 0.01 1`). The hand
  follows through the constraint solver (contacts push back, no teleporting during steps).
  Per step the target moves by ≤ 1 cm per axis (world) and rotates by ≤ 0.06 rad per axis (hand
  frame) with quaternion composition + normalisation, then four projections: workspace box,
  finger-tilt limit (110°), anti-windup lead limit (2.5 cm / 0.30 rad), table clearance
  (lowest palm corner or finger capsule, current curl, ≥ 4 mm above the table).
  Inside a control step the target is interpolated over the 20 physics substeps.
* Model sizes: nq 30, nv 28, nu 16 (finger servos), nmocap 1, neq 1 (weld). Action dim 22.
* Objects: uniform-density geoms (true centre of mass and inertia; V3.3 used an isotropic inertia
  at the body origin); mass 0.05–0.30 kg and sliding friction 0.5–1.2 per episode without
  recompilation; four cached scale bins (0.85–1.15) × aspect variants; resting modes
  (upright / side / end / lying / flat / standing) + random yaw; exact support-function placement
  1.5 mm above the table followed by a 0.5 s settling check (speed, penetration, hand contact).

### Physics change versus V3.3 (important for the report)

V4 keeps the V3.3 solver settings (elliptic cone, impratio 10, noslip 5, dt 2.5 ms) but raises the
finger-joint **armature from 5e-5 to 1e-3 kg·m²** (reflected rotor inertia of a geared finger motor,
N²·J ≈ 100²·1e-7). With 5e-5, squeezed grasps occasionally exploded (objects launched at
10⁵–10¹⁰ m/s, sometimes without any MuJoCo warning): 3/486 episodes with noslip 5 and 30/486 without
noslip. With 1e-3 and noslip 5, the 33 previously failing episodes ran without failure (with 1e-3
and no noslip, 8/33 still failed). Those runs are kept in
`outputs/scripted/_diagnostic_prefix_armature5e-5_seeds5000000/` and their seeds are no longer used
for evaluation. A runaway detector (object speed > 5 m/s or |position| > 2 m) now marks any such
event as a simulation failure.

## Ground-truth success definitions

* `gt_success` (primary, same as V3.3 / project spec): COM lifted ≥ 10 cm for 60 consecutive
  frames (3 s) while the thumb **pad** and at least one other **pad** press with ≥ 0.2 N.
* `gt_success_any_phalanx` (secondary, V4): same, but thumb and another finger may touch with any
  phalanx (power grasps in side approaches often hold the object with middle phalanges).

## Commands (PowerShell, project root, CPU only)

```powershell
python -m visuograsp_v4.scripts.integrity_manifest verify          # V3.3 untouched?
python -m visuograsp_v4.scripts.run_smoke_tests                   # ~1 min, 15 tests
python -m visuograsp_v4.scripts.run_scripted_baseline --episodes 6 --noslip 5   # ~4 min
python -m visuograsp_v4.scripts.run_scripted_baseline --episodes 6 --noslip 0   # sensitivity
python -m visuograsp_v4.scripts.record_demos                      # ~1 min, 10 MP4 files
```

Seeds: development/tuning 9xx_xxx and 5_000_000+ (diagnosis); scripted-baseline evaluation
5_100_000 + grid·1000 + approach·100 + episode; demos 7_000_001–7_000_010; smoke tests fixed.

## RL preparation (no long training has been run)

| Item | File / value |
|---|---|
| SAC settings | `config.SACConfig` (LayerNorm critics, n-step 10, lr 3e-4, batch 256, UTD 1, gamma 0.99) |
| Policy class | `agents.LayerNormSACPolicy` (importable, no `__main__` pickling problem) |
| Reward | `config.RewardConfig` v4r1 (see `env.py` header); audit: `scripts/reward_audit.py` |
| Training | `scripts/train_sac.py --arm scratch|demo --seed S --total-steps N` |
| Test evaluation | `scripts/evaluate_policy.py --model ...` (paired with the scripted baseline) |
| Throughput | `scripts/benchmark_throughput.py` -> `outputs/benchmark/throughput.json` |

Seeds: demo prefill 3_000_000 + 10_000·seed + k (training), validation / model selection
8_000_000+, test 5_100_000+ (the scripted-baseline grid; initial states identical, smoke-tested).
Folders starting with `check_` are pipeline checks (4k steps), not results.

## Known limitations

* Side grasps need objects taller than about 8 cm: the 11 cm palm and the finger layout keep the
  palm centre ≥ 6.8 cm above the table. Shorter objects can only be pinched at the top edge
  (flagged `plan_feasible = False`).
* Fingers have no abduction joints (V3.3 finger kinematics); closing all fingers in free space
  makes fingers 1–3 collide near the hand axis (correct physics, not a defect).
* Perception is still colour-threshold segmentation (red objects only). Object centre from the
  PCA box of the visible point cloud: mean error 0.6–2.2 cm, max 2.5 cm at reset (smoke test).
  No orientation-aware learned segmentation and no D2 detector for the new viewpoints yet.
* Scripted baseline: open-loop finger closing, PRIVILEGED object pose for planning; not a policy.
* `noslip_iterations = 5` is kept from V3.3; without it more squeezed grasps fail.
* Contacts use `condim = 4` (V3.3): no rolling friction, so a dropped cylinder or bottle lying on
  its side rolls without slowing down and can leave the table (outcome `object_left_table`, which
  is a task failure, not a numerical failure).
