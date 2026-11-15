# Slide speaking notes — VisuoGrasp V4 final defense

Target: about 10 minutes (≈ 50 s per slide). The same text is stored as speaker notes in the PPTX.
Numbers: `evidence/VERIFIED_RESULTS.md`. Every learned result uses PRIVILEGED state (simulator object pose + GT contacts).

## Slide 1 — Title

Good afternoon. We are presenting VisuoGrasp V4, our final project on six-degree-of-freedom robotic grasping. The picture on the right is a real frame from our MuJoCo simulation: our best learned policy, called E6, is closing a four-finger hand on a bottle from a tilted, 45-degree approach. In the next ten minutes we will show three things: how we upgraded the robot from vertical-only grasping to a physically controlled 6-DoF wrist, which learning methods we tried and why most of them did not beat the expert, and the final result: 82 percent ground-truth grasp success on 540 unseen test episodes. One honest point up front: the policies receive the object pose and contacts from the simulator, so this is state-based control, not vision-based control.

## Slide 2 — Problem and motivation

Our previous version, V3.3, is on the left. It used the same four-finger, sixteen-joint hand, but the wrist could at most translate; the hand always pointed straight down. That is fine for a box on a table, but it cannot grasp an object from the side or at an angle, which many real objects require. V3.3 also focused on vision-confirmed contact; its best learned policy reached about 63 percent on its own task. For V4 we kept the hand but gave it a free six-degree-of-freedom wrist. On the right you see a 45-degree grasp by our learned policy and a 90-degree side grasp. The side grasp is performed by the scripted expert; we did not train a learned policy for side grasps. The task is now much harder: the policy must approach, orient the hand, close the fingers, lift and hold the object.

## Slide 3 — What changed in V4

Four numbers summarise the upgrade. First, a six-degree-of-freedom wrist. The hand is a free rigid body; a virtual target, a so-called mocap body, is moved by the policy and a weld constraint pulls the hand towards it. So contacts push back and the hand carries the object weight; nothing is teleported. Second, the action is 22-dimensional: 16 finger target increments plus three translations and three rotations, each bounded per 50-millisecond step. Third, the observation is 130-dimensional. Fourth, five object families and three approach directions; the learned policies were trained and tested on top-down and 45-degree grasps. The strip shows our learned policy lifting every object family. We also modelled true centre of mass and inertia and rebuilt the code as a tested Python package.

## Slide 4 — Robotic environment

Here the environment is in motion. On the left, our learned policy E6 grasps a bottle from 45 degrees on an unseen test episode: approach, contact, lift and a three-second hold. On the right, the scripted expert performs a side grasp of a tall box; we show it to illustrate the third approach direction, but it is scripted, not learned. Both clips play at twice real speed. The physics runs at 2.5 milliseconds per step, the policy acts at 20 hertz, and an episode lasts 16 seconds. Object size, mass, friction and resting pose are randomised every episode. Success is judged only by simulator ground truth, using the same definition as in the course specification: the object must stay at least 10 centimetres up for three seconds while the thumb and another finger press on it.

## Slide 5 — System architecture

This is the information flow. MuJoCo simulates the physics. From the simulator we take privileged information, marked in orange: the exact object pose and size, and ground-truth contact flags on the four finger pads. Together with joint positions, velocities and the wrist pose this forms the 130-dimensional observation. The actor, in green, is the only learned component at run time; it outputs 22 numbers. The hand controller turns them into finger targets and a new wrist target, and the physics moves the hand. In blue, the analytic grasp planner computes the target grasp pose from the object pose, and the scripted expert uses it to generate demonstrations. The orange box is important: our V4 code has cameras, colour segmentation and a geometric contact detector, but none of the reported policies uses them.

## Slide 6 — RL formulation

Formally, the task is a Markov decision process. The state is the 130-dimensional observation; the action is a vector of 22 numbers between minus one and one. One action is held for 20 physics substeps, and an episode ends after 320 actions or earlier if the object drops, leaves the table, or the simulation becomes numerically unstable. The reward has two parts. Most of it is potential-based shaping: a potential Phi rewards being close to the planned grasp pose, aligned with it, in contact, and lifting the object, and penalises pushing the object before it is lifted. Because it is a difference of potentials, it guides learning without changing which policy is optimal. On top come a success bonus of 10, a drop penalty, and small penalties for table contact and large actions. The wrist update composes the current orientation quaternion with a small rotation and renormalises it, which avoids gimbal lock.

## Slide 7 — Learning methods

We compared five families of methods, from left to right with increasing use of the expert. Pure SAC learns from the reward alone; even with demonstrations in the replay buffer or a curriculum it reached at most 19 percent. Residual SAC learns corrections on top of the scripted controller; the corrections made it worse. Behaviour cloning is supervised learning, not reinforcement learning: it regresses the expert actions and reached 74.6 percent. An ensemble of six behaviour-cloning actors matched the expert. E6 starts from a behaviour-cloning actor and fine-tunes it with an AWAC-style objective: it regresses towards actions that the critic rates better than the current policy, weighted by the exponential of the advantage, plus a behaviour-cloning anchor on expert data. The actor never follows the critic gradient directly, so it cannot drift far from the expert.

## Slide 8 — Experimental development

We ran five waves, and each one was stopped or redesigned by a pre-registered rule. Wave 1: SAC from scratch learned nothing useful; the hand mostly pushed the object away. Wave 2: we let SAC add corrections to the scripted controller; the corrections cut success from 78 to 49 percent. Wave 3: smaller, conservative corrections reduced the harm but still did not help. Wave 4: a curriculum made autonomous SAC better than in Wave 1, but only 19 percent; behaviour cloning of the expert reached 75 percent, and ordinary SAC fine-tuning of that clone made it worse. That led to Wave 5: better imitation and an RL update that is anchored to the expert. Each wave used fresh test seeds, so the fair comparison is always against the expert on the same episodes.

## Slide 9 — Final performance

This is the main result. After freezing every model, we evaluated all seven controllers once on the same 540 unseen episodes. E6 reached the highest observed success, 82.0 percent, with a 95 percent Wilson interval from 78.6 to 85. The scripted expert reached 80.6 and the behaviour-cloning ensemble 80.7. Notice that the axis starts at 60 percent, and that the confidence intervals overlap, so from this chart alone we cannot rank these three. Top-down grasps are almost solved by every controller, at 93 to 97 percent. The differences come from 45-degree grasps, where E6 reaches 68 percent compared with 64 for the expert. To decide whether differences are real, we use paired tests on identical episodes, on the next slide.

## Slide 10 — Contribution of RL

Here is the contribution of reinforcement learning. E6 started from a behaviour-cloning actor, E5c, that scored 79.4 percent. After fine-tuning it scored 82.0, a paired difference of 2.6 points. On the same episodes, E6 succeeded on 18 episodes where its starting policy failed, and failed on only 4 where it had succeeded. The exact McNemar test, corrected for the 12 comparisons we had listed in advance, gives p equals 0.039, below 0.05. Almost all of the gain is in 45-degree grasps, mainly because E6 pushes the object less before closing; the clip shows one such episode. What it does not show: E6 is not significantly better than the expert or the ensemble, and our second run, E6b, did not reproduce a significant gain. With one training seed per run, we report this as a real effect for E6, not a general law.

## Slide 11 — Limitations

We want to be clear about the limits. First and most important: the policy receives the object pose and ground-truth contacts from the simulator. Our V4 code contains cameras and a geometric contact detector, but we did not train a vision-based V4 policy; the next step is to bring the D2 contact detector from V3.3 to the moving wrist camera. Second, each RL configuration used one training seed, and the second run did not reproduce the gain. Third, E6 is not significantly better than the expert. Fourth, a flat ring approached at 45 degrees is never grasped by any controller, including the expert, as the clip shows; this is a limitation of the grasp planner that all learned policies imitate. Finally, side grasps are only scripted. The box below the clip shows that failures are dominated by never lifting and by pushing the object before closing, for E6 and the expert alike.

## Slide 12 — Conclusion

To conclude, three messages. Engineering: we turned a vertical-only grasping setup into a physically controlled six-degree-of-freedom wrist with a 22-dimensional action space and five object families. Learning: reinforcement learning on its own, as residual corrections, or with a curriculum did not reach the expert; what worked was imitation of the expert combined with a conservative, expert-anchored RL update. Result: E6 reached 82 percent ground-truth success on unseen episodes and significantly improved its own behaviour-cloning start, but it is not significantly better than the expert, and it relies on privileged simulator state. The clip shows E6 grasping a cylinder on an unseen episode. Thank you; we are happy to take questions.
