# Demo media manifest — VisuoGrasp V4 final defense

* All V4 clips show **PRIVILEGED state-based control**: the simulator object pose and ground-truth contacts are inputs. They are not vision-based.
* **"Re-rendered" clips** replay an already-evaluated episode with the frozen policy at 640×480 per camera (deterministic, torch 1 thread). Each one reproduced the GT outcome stored in the frozen record; see the outcome column and `media/rerendered/rerender_manifest.csv`.
* **GIFs** cover approach → contact → lift → hold at the speed noted in the GIF corner (2× or 3×). The MP4s play in real time.
* **H.264 copies** of every MP4 are in `media/selected_videos/h264/` and play in any player. The originals use MPEG-4 Part 2.
* Every path below was checked for existence when this file was written.

## D1. V3.3 Level A grasp (previous system)

* **Controller / model:** V3.3 SAC M2 (D2 vision-confirmed contact), seed 33
* **Object family:** box
* **Approach direction:** top-down (fixed vertical wrist)
* **GT success / failure:** GT success 1, CV success 1 (README V3.3)
* **Original MP4 (pre-existing project video):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_full_outputs\cv_v2\drl_runs_v3_final\videos\v33_final_demos\A_train_success.mp4`
* **Re-rendered MP4 (640×480 per camera, real time):** —
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\A_train_success_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_v33_level_a_success.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\v33_A_train_success__t3.2s_side.png`
* **Slide:** Slide 2 (GIF)
* **Report section / figure:** Fig. 1(a)

## D2. E6 oblique bottle grasp

* **Controller / model:** E6 BC-anchored RL (frozen e6_best_90k)
* **Object family:** bottle
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6715100
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_oblique_bottle_6715100\e6_oblique_bottle_6715100_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_oblique_bottle_6715100_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_oblique_bottle.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_oblique_bottle_6715100__contact_side.png`
* **Slide:** Slide 1 (still), Slide 2 (still), Slide 4 (GIF)
* **Report section / figure:** Fig. 1(b), Fig. 1 bottom

## D3. Scripted side 90° grasp (not learned)

* **Controller / model:** Scripted expert (PRIVILEGED planner)
* **Object family:** box (tall)
* **Approach direction:** side 90°
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [demos/scripted_elliptic_imp10_noslip5_dt2.5ms_sr6/demo_manifest.csv], reproduced = True; seed 7000007
* **Original MP4 (pre-existing project video):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\demos\scripted_elliptic_imp10_noslip5_dt2.5ms_sr6\side_box_tall.mp4`
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\scripted_demo_side_box_tall_7000007\scripted_demo_side_box_tall_7000007_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\side_box_tall_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_scripted_side_box_tall.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\scripted_demo_side_box_tall_7000007__contact_side.png`
* **Slide:** Slide 2 (still), Slide 4 (GIF)
* **Report section / figure:** Fig. 1(c)

## D4. Scripted expert vs E6 on the same episode

* **Controller / model:** Scripted expert (top) vs E6 (bottom)
* **Object family:** box (flat)
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6706109
* **Original MP4 (pre-existing project video):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\rl\wave5\videos\e6\side_by_side_learned_wins_oblique_box_seed6706109.mp4`
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_oblique_box_6706109\e6_oblique_box_6706109_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\side_by_side_learned_wins_oblique_box_seed6706109_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_scripted_vs_e6_oblique_box_6706109.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_oblique_box_6706109__lift_side.png`
* **Slide:** Slide 10 (GIF)
* **Report section / figure:** Fig. 6

## D5. E6 top-down cylinder grasp

* **Controller / model:** E6 BC-anchored RL
* **Object family:** cylinder
* **Approach direction:** top-down
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6709000
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_top_cylinder_6709000\e6_top_cylinder_6709000_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_top_cylinder_6709000_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_top_cylinder.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_top_cylinder_6709000__lift_side.png`
* **Slide:** Slide 12 (GIF), Slide 3 (strip)
* **Report section / figure:** Fig. 1 bottom

## D6. E6 failure: oblique flat ring, no contact

* **Controller / model:** E6 BC-anchored RL
* **Object family:** ring
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 0 (no_contact); frozen record: no_contact [wave5/final_test/final/episodes.csv], reproduced = True; seed 6719100
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_fail_ring_oblique_6719100\e6_fail_ring_oblique_6719100_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_fail_ring_oblique_6719100_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_fail_ring_oblique.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_fail_ring_oblique_6719100__end_overview.png`
* **Slide:** Slide 11 (GIF)
* **Report section / figure:** Fig. 7 (right)

## D7. E6 top-down box grasp

* **Controller / model:** E6 BC-anchored RL
* **Object family:** box (cube)
* **Approach direction:** top-down
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6700000
* **Original MP4 (pre-existing project video):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\rl\wave5\videos\e6\top_success_top_box_seed6700000.mp4`
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_top_box_6700000\e6_top_box_6700000_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\top_success_top_box_seed6700000_h264.mp4`
* **Generated GIF:** —
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_top_box_6700000__lift_side.png`
* **Slide:** Slide 3 (strip)
* **Report section / figure:** Fig. 1 bottom

## D8. E6 oblique L-shape grasp

* **Controller / model:** E6 BC-anchored RL
* **Object family:** L-shape
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6721100
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_oblique_lshape_6721100\e6_oblique_lshape_6721100_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_oblique_lshape_6721100_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_oblique_lshape.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_oblique_lshape_6721100__lift_side.png`
* **Slide:** Slide 3 (strip); GIF spare
* **Report section / figure:** Fig. 1 bottom

## D9. E6 top-down ring grasp

* **Controller / model:** E6 BC-anchored RL
* **Object family:** ring
* **Approach direction:** top-down
* **GT success / failure:** GT success 1 (success_retained); frozen record: success_retained [wave5/final_test/final/episodes.csv], reproduced = True; seed 6719000
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_top_ring_6719000\e6_top_ring_6719000_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_top_ring_6719000_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_top_ring.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_top_ring_6719000__lift_side.png`
* **Slide:** Slide 3 (strip); GIF spare
* **Report section / figure:** Fig. 1 bottom

## D10. E6 failure: palm pushes the box

* **Controller / model:** E6 BC-anchored RL
* **Object family:** box (flat)
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 0 (object_pushed_before_close); frozen record: object_pushed_before_close [wave5/final_test/final/episodes.csv], reproduced = True; seed 6706102
* **Original MP4 (pre-existing project video):** —
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_fail_pushed_oblique_box_6706102\e6_fail_pushed_oblique_box_6706102_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_fail_pushed_oblique_box_6706102_all_h264.mp4`
* **Generated GIF:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\gifs\gif_e6_fail_pushed_oblique_box.gif`
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_fail_pushed_oblique_box_6706102__contact_side.png`
* **Slide:** GIF spare
* **Report section / figure:** Fig. 7 (left)

## D11. E6 failure: never lifted

* **Controller / model:** E6 BC-anchored RL
* **Object family:** box (tall, lying on side)
* **Approach direction:** oblique 45°
* **GT success / failure:** GT success 0 (never_lifted); frozen record: never_lifted [wave5/final_test/final/episodes.csv], reproduced = True; seed 6704100
* **Original MP4 (pre-existing project video):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\rl\wave5\videos\e6\failure_never_lifted_oblique_box_seed6704100.mp4`
* **Re-rendered MP4 (640×480 per camera, real time):** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\rerendered\e6_fail_never_lifted_oblique_box_6704100\e6_fail_never_lifted_oblique_box_6704100_all.mp4`
* **H.264 copy for playback:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\failure_never_lifted_oblique_box_seed6704100_h264.mp4`
* **Generated GIF:** —
* **Representative screenshot:** `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\screenshots\e6_fail_never_lifted_oblique_box_6704100__end_side.png`
* **Slide:** —
* **Report section / figure:** Fig. 7 (middle)

## Embedded in the PPTX

| Slide | Embedded animated GIF | MP4 fallback (hyperlink on the slide) |
|---|---|---|
| 2 | gif_v33_level_a_success.gif (46 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\A_train_success_h264.mp4` |
| 4 | gif_e6_oblique_bottle.gif (50 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_oblique_bottle_6715100_all_h264.mp4` |
| 4 | gif_scripted_side_box_tall.gif (42 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\scripted_demo_side_box_tall_7000007_all_h264.mp4` |
| 10 | gif_scripted_vs_e6_oblique_box_6706109.gif (56 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\side_by_side_learned_wins_oblique_box_seed6706109_h264.mp4` |
| 11 | gif_e6_fail_ring_oblique.gif (54 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_fail_ring_oblique_6719100_all_h264.mp4` |
| 12 | gif_e6_top_cylinder.gif (56 frames) | `C:\Users\user\Desktop\AMRL\Project_2_2\visuograsp_v4\outputs\final_defense\media\selected_videos\h264\e6_top_cylinder_6709000_all_h264.mp4` |

**Verification:**
* The PPTX package contains the six GIF files with all frames and an infinite loop.
* The picture elements reference these GIFs directly.
* PowerPoint renders all slides (COM export).
* Playback in slideshow mode could not be observed automatically: the screenshot attempt captured another window.
* Open the deck in Slide Show once before the defense. If a GIF does not move, click the MP4 link under it.
