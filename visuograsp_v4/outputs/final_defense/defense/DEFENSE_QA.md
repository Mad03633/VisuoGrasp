# VisuoGrasp V4 — technical defense Q&A

All numbers come from `evidence/VERIFIED_RESULTS.md`, which reads the frozen files.
RU lines give short Russian explanations of the harder concepts.

## Numbers to remember

| Item | Value |
|---|---|
| Action / observation | 22-D / 130-D |
| Finger joints | 16 (4 per finger, flexion only) |
| Wrist | free 6-DoF body driven by mocap + weld |
| Timing | physics 2.5 ms; control 20 Hz (20 substeps); episode 320 steps = 16 s; γ = 0.99; n-step 10 |
| Success (GT) | object centre ≥ 10 cm above rest for 3 s (60 steps); thumb pad + one more pad ≥ 0.2 N |
| Final test | 540 unseen episodes (seeds 6.7M); 7 controllers on identical episodes; used once |
| E6 | **82.0 % [78.6, 85.0]**; top-down 95.9 %, oblique 68.1 %, 0 numerical failures |
| Scripted expert | 80.6 % [77.0, 83.7] |
| BC ensemble | 80.7 % [77.2, 83.8] |
| BC Wave 4 | 77.4 % |
| E5c (E6 init.) | 79.4 % |
| E6b | 78.5 % |
| E6 − its BC init. | +2.6 pp [+0.9, +4.3], 18 / 4 discordant, McNemar p = 0.0043, **Holm p = 0.039** |
| E6 − expert | +1.5 pp [0.0, +3.1], 13 / 5, p = 0.096 (Holm 0.58): **not significant** |
| E6 − BC ensemble | +1.3 pp, p = 0.19: not significant |
| E6b − its init. | +1.1 pp, 20 / 14, p = 0.39: not significant |
| Earlier waves (each on its own test set) | W1 SAC 0 % / 10.5 %; W2 residual 49.4 %; W3 conservative residual 69.1 %; W4 curriculum SAC 18.9 %, BC 74.6 %, BC+SAC 48.7 % |

---

## A. Robotic system

**Why four fingers?**
* The course specification names an Allegro-style four-finger, 16-DoF hand as the primary platform.
* A thumb opposing three fingers is enough for stable opposition grasps of our object sizes, and it
  keeps the action space manageable.
* We kept the exact V3.3 finger kinematics so that the only major change in V4 is the wrist.

**How many joints and degrees of freedom?**

| Body | Joints / DoF | Actuation |
|---|---|---|
| Fingers | 16 hinge joints, 4 per finger, flexion only, no abduction | position servos |
| Hand (wrist) | one free joint: 3 translational + 3 rotational DoF | not a motor: a mocap target pulls it through a weld constraint |
| Object | free joint | — |

* MuJoCo model size: nq = 30 (7 hand + 16 fingers + 7 object), nv = 28, 16 actuators, 1 mocap
  body, 1 weld constraint.
* The action has 22 dimensions: 16 finger target increments, 3 wrist translations and 3 wrist rotations.
* There is no robot arm. There are no shoulder or elbow joints: the wrist pose is commanded directly.

**What changed between V3.3 and V4?**

| Aspect | V3.3 | V4 |
|---|---|---|
| Wrist | none (Level A) or 3-DoF translation (Level B); always vertical | full 6-DoF |
| Action | 16 or 19 | 22 |
| Approaches | top-down only | top-down, 45° oblique, 90° side (side is scripted only) |
| Objects | sphere, box, cylinder, ellipsoid, capsule, ring (train) + unseen families | box (cube / tall / flat), cylinder, bottle, ring, L-shape |
| Mass properties | isotropic inertia at the body origin | true centre of mass and inertia |
| Finger armature | 5e-5 (object "explosions" in squeezed grasps) | 1e-3 |
| Code | notebook with monkey-patching | Python package (Gymnasium env, configs, smoke tests, integrity manifest) |
| Contact signal | vision-confirmed (D2) | privileged GT contacts in the final policies |

**How does 6-DoF wrist control work?**
* Each step, the policy moves a target pose:
  * position by up to 1 cm per world axis;
  * orientation by up to 0.06 rad per hand axis, as a quaternion composition with the rotation
    vector, then renormalised.
* Four safety projections follow:
  * workspace box;
  * finger-tilt limit (110°);
  * anti-windup: the target may lead the real hand by at most 2.5 cm / 0.30 rad;
  * table clearance: the lowest palm or finger point stays ≥ 4 mm above the table.
* The target is interpolated over the 20 physics substeps. The physics, not the code, moves the hand.

**Why quaternions?**
* Unlike Euler angles they have no gimbal lock.
* Composing rotations is a single multiplication, and renormalising removes numerical drift.
* MuJoCo stores orientations as unit quaternions [w, x, y, z].
* For the network we encode the wrist orientation as a continuous 6-D rotation representation
  (two matrix columns), because quaternions have a sign ambiguity (q and −q are the same rotation).

> RU: кватернион — четыре числа, задающие поворот без «складывания рамок» (gimbal lock); повороты
> складываются умножением, после чего кватернион нормируют.

**What is mocap + weld control?**
* A mocap body is a massless, kinematic marker that code can place anywhere.
* A weld equality constraint is a soft spring-damper (time constant 10 ms, critical damping) that
  pulls the free hand towards the marker.
* Because the hand is a real dynamic body, it collides, can be blocked by the table or the object,
  carries the object's weight, and never teleports.

> RU: mocap — невидимая «цель», которую двигает код; weld — мягкая пружина, тянущая настоящую руку к
> цели, поэтому рука подчиняется физике.

**What is MuJoCo responsible for?**
* MuJoCo does rigid-body dynamics, collision detection, frictional contact with an elliptic cone,
  constraint solving (weld, joint limits), the position-servo actuators and camera rendering.
* Our code does the rest: action-to-target mapping, rewards, the success check, the grasp planner,
  the scripted expert, logging and statistics.

**What are the main physical limitations?**
* No abduction joints.
* Side grasps need objects taller than about 8 cm.
* Contacts use condim 4, so there is no rolling friction.
* Idealised servos; no sensor noise in the privileged setting; simulation only.
* Rare numerical "runaway" events: 4 of 3,780 final-test episodes, all after success had been
  reached. They are detected (object speed > 5 m/s) and reported.

## B. Reinforcement learning

**What is SAC?**
* Soft Actor-Critic is an off-policy actor-critic method that maximises reward plus policy entropy.
* Critics learn Q(s, a) by temporal-difference learning from a replay buffer.
* The actor is a tanh-squashed Gaussian, trained to maximise min Q − α log π.
* The temperature α is tuned automatically towards a target entropy of −22 (one per action
  dimension).

**Why twin critics?**
* Maximising over a learned Q-function over-estimates values.
* Taking the minimum of two independently trained critics, both in the target and in the actor
  loss, reduces this bias (clipped double-Q, as in TD3).
* We also put LayerNorm in every critic hidden layer, which was the V3.3 stabiliser against growing Q-values.

**What does the actor output?**
* A mean μ(s) and a log standard deviation for each of the 22 action dimensions.
* During training the action is tanh(μ + σ·ε).
* For evaluation we use the deterministic action tanh(μ).
* In E6 the standard deviation is fixed at 0.05.

**States, actions, rewards, transitions?**

| Element | Definition |
|---|---|
| State | 130-D observation (below) |
| Action | 22-D vector in [−1, 1] |
| Reward | v4r1: potential-based shaping γΦ(s′) − Φ(s); Φ = reach + alignment + contact + 3·lift − push |
| Reward bonuses / penalties | +0.1 per step while held at 10 cm; +10 once at success; −3 once at a drop; small table-contact and effort penalties; −5 on a numerical failure |
| Transition | MuJoCo, 20 substeps per action |
| Termination | drop, object off the table, numerical failure, or 320 steps |

> RU: потенциальная награда (shaping) — разность «потенциала» до и после шага; она подсказывает
> направление, но не меняет оптимальную политику.

**What is the observation dimension?**

130, made up of:

| Block | Dim |
|---|---|
| Finger positions, velocities and targets | 48 |
| Wrist pose and velocities | 15 |
| Fingertip positions | 12 |
| Object centre, axes, extents, centre in hand frame | 15 (privileged) |
| Grasp-pose targets and errors | 9 |
| GT pad contacts and their confirmed version | 8 (privileged) |
| Previous action | 22 |
| Visibility flag | 1 |

**What is the action dimension?** 22.

**What is the discount factor?** γ = 0.99.

**What is an n-step return?**
* Instead of bootstrapping after one step, the target sums 10 real rewards and then bootstraps:
  R = Σ_{k<n} γ^k r_{t+k} + γ^n max-entropy Q(s_{t+n}), truncated at the episode end.
* It propagates the delayed success signal faster.
* The cost is some bias, because the sampled rewards come from an older policy.

> RU: n-шаговый возврат — суммируем 10 настоящих наград и только потом подставляем оценку критика;
> так сигнал об успехе быстрее доходит до ранних шагов.

**Why did SAC fail when trained from scratch?**
* The task is a long chain: approach with the right orientation, close, lift, hold for 3 s, in a
  22-D action space.
* Random exploration mostly pushes the object away before any useful contact, so the +10 success
  bonus is almost never seen.
* 150–300k steps is small for this horizon.
* A curriculum that starts episodes near the grasp pose helped (18.9 % vs 0–10.5 %) but did not
  get close to the expert.

**Why did residual SAC degrade the scripted controller?**
* The expert is a precise set-point tracker whose success depends on timing.
* Early exploratory residuals perturb it: more pushing, misaligned closing.
* The critic then learns from these degraded states, and the actor follows the critic's gradient.
* Even small, zero-centred residuals with a penalty (Wave 3) lowered success: 69.1 % vs 76.7 %.

## C. Imitation and advanced learning

**What is Behaviour Cloning?**
* Supervised regression of the policy output onto expert actions: minimise mean ||tanh μ(s) − a*||².
* It uses no reward and no environment interaction, so it is not reinforcement learning.

**How does the expert generate demonstrations?**
* An analytic planner computes top-down / oblique grasp, pre-grasp and lift poses from the
  simulator object pose (privileged).
* A phase machine (transit → descend → approach → close → lift → hold) drives the hand towards
  these poses.
* We recorded 600 clean episodes and 600 DART episodes. In DART, noise σ = 0.05 is executed but
  the clean expert action is stored as the label, so BC learns to recover from small deviations.
* Only successful episodes are used as BC training pairs: 218,755.

**What is the copycat problem?**
* The observation contains the previous action and joint/wrist velocities, which are almost equal
  to the next expert action.
* BC learns to copy its own last command: low training error, but in closed loop it cannot switch
  phase (for example, start lifting) and lifts too slowly.
* E5a: DEV success fell from 29 % to 2 % while the supervised error kept falling.
* Fix: mask these inputs, or randomly drop / perturb them during training.

> RU: «копирующий агент» — сеть учится повторять собственное предыдущее действие, а не реагировать
> на состояние; ошибка на данных маленькая, а в замкнутом контуре робот «застревает».

**What is Curriculum Learning?**
* Training on easier versions first.
* Our episodes start from scripted hand-over states: first at the grasp pose, then at the
  pre-grasp, then at the transit pose.
* The stage advances when stage success reaches 50 %.

**What is a BC Ensemble?**
* The average action of six independently trained BC actors with different input masking.
* Their errors are partly independent, so averaging removes individual mistakes.
* It improved on Wave 4 BC by +3.3 pp (Holm p = 0.029), mainly on oblique grasps. It is still
  imitation, not RL.

**What is AWAC?**
* Advantage-Weighted Actor-Critic: a critic is learned by TD.
* The actor does weighted regression onto actions from the data, with weight exp(advantage / τ).
* Good actions are imitated more, bad ones less, and the actor stays within the data distribution.

> RU: AWAC — «взвешенное подражание»: актор копирует действия из буфера, но с весом, растущим с
> преимуществом (advantage) по оценке критика.

**What is BC-anchored RL fine-tuning (E6)?**
* Our variant of AWAC, starting from a BC actor (E5c epoch 10).
* Critics:
  * 10-step targets, no entropy term (α = 0);
  * target values clipped to [−25, 35];
  * a 25 % demonstration share in every batch.
* Actor:
  * weighted regression with w = exp(A / (τ·std A)), clipped at 20 and normalised, plus a constant
    BC term (λ = 0.5) on expert pairs;
  * frozen for the first 20k updates;
  * fixed exploration σ = 0.05.
* It never follows ∂Q/∂a.

**Why did E6 improve?**
* It only re-weights executed actions and stays anchored to expert labels. The action drift from
  the BC actor stayed below 3e-4.
* The change concentrated where the expert and BC are weakest: oblique grasps, +5.6 pp, 16 vs 1
  discordant.
* In 13 of its 18 wins the BC start had pushed the object before closing. Oblique pushes fell from
  45 to 35.

**Why did E6b fail to reproduce the improvement?**
* E6b used the same method from a different start (E5g, full observation with history dropout),
  which is weaker top-down.
* Its changes went both ways: 20 wins, 14 losses (p = 0.39).
* With one training seed per configuration we cannot tell whether this is the initialisation or
  ordinary run-to-run variance. So the RL gain is real for E6 but not shown to be general.

## D. Evaluation

**What counts as a successful grasp?**
* Ground truth from the simulator: the object's centre of mass stays at least 10 cm above its
  resting height for 3 s (60 consecutive steps).
* During that time the thumb pad and at least one other pad press with at least 0.2 N.
* It is used only for evaluation and selection, never as an actor input.

**How are validation and test episodes separated?**

| Purpose | Seeds |
|---|---|
| Demonstrations | 3.6M / 3.7M |
| Epoch and in-run selection | DEV_A, 8.8M |
| Cross-method selection | DEV_B, 8.9M, 200 episodes |
| Final test | 6.7M, never used before; run once after all models and the reading rule were frozen |
| Earlier waves' test sets | 5.1M, 6.1M, 6.6M; never reused |

**What is a confidence interval?**
* A range that, by construction, contains the true success probability in 95 % of repeated experiments.
* We use the Wilson interval, which behaves well near 0 % or 100 %.
* E6: 82.0 % [78.6, 85.0].

**What is the paired McNemar test?**
* Both controllers run on the same 540 episodes, so only the discordant episodes are informative:
  those that one controller wins and the other loses.
* If the controllers were equally good, each discordant episode would be a fair coin flip.
* The exact test is a binomial test on these counts, for example 18 vs 4.

> RU: тест Макнемара сравнивает две политики на одних и тех же эпизодах и смотрит только на
> эпизоды, где они разошлись; если бы политики были равны, такие эпизоды делились бы примерно 50/50.

**Why is p = 0.039 significant?**
* p is the probability of a split at least as extreme as 18:4 if fine-tuning had no effect.
* The raw p is 0.0043. After Holm correction for the 12 comparisons listed in advance, it is
  0.039 < 0.05.
* So, at the 5 % level, the improvement is unlikely to be chance for this run.
* It does not prove that the method always helps (see E6b).

> RU: поправка Холма — мы сделали 12 сравнений, поэтому порог строже; после поправки p = 0.039 < 0.05.

**Why does 82.0 % not prove superiority over 80.6 %?**
* The paired difference is 8 episodes out of 540 (13 vs 5 discordant): p = 0.096, Holm 0.58.
* The bootstrap CI of the difference is [0.0, +3.1] pp, which includes zero.
* A difference this small is compatible with chance.

**Why is a single training seed a limitation?**
* RL results vary strongly between random seeds.
* One run can be lucky or unlucky; E6b is an example.
* The course asks for at least 3–5 seeds, which we did not have the compute and time for in Wave 5.
* Our paired tests measure the variability over test episodes, not over training runs.

**Which parts use privileged information?**
* **Actor observation:** exact object pose and extents; ground-truth pad contacts.
* **Grasp-pose targets:** computed from the true object pose.
* **Reward:** lift height, object push and table force come from the simulator.
* **Scripted expert:** plans from the true pose.
* **Critics:** they see the same observation.
* **Ground-truth success:** used for evaluation only, which is legitimate.

**What is needed for genuine vision-based grasping?**
* Train and evaluate with `obs_mode='vision'`: the object estimate comes from RGB-D colour
  segmentation and a PCA bounding box. This exists in V4 but is untested in learning.
* Contacts from a vision detector (port V3.3's D2 CNN–GRU to the moving wrist camera, with
  temporal confirmation).
* Learned segmentation for arbitrary objects, sensor noise and domain randomisation.
* Multiple seeds and, ideally, a sim-to-real test.

---

## E. Likely follow-up questions

**Why not PPO or TD3?**
* SAC was the V3.3 baseline with proven stabilisers (LayerNorm critics, n-step returns), and
  off-policy learning reuses scarce samples (about 50 environment steps/s on CPU).
* E6 already borrows TD3+BC's idea of a BC term in the actor loss.

**Is 540 episodes enough?**
* For absolute rates near 80 %, the Wilson half-width is about ±3.4 pp.
* Pairing makes comparisons much sharper than unpaired CIs: E6 vs its initialisation is
  significant although their CIs overlap.

**Was the final test re-run or tuned after seeing it?**
* No. Policies, checkpoints and the reading rule were frozen first.
* The run was interrupted once by memory protection and resumed without changes. The earlier
  episode records are byte-identical, and the final file has 3,780 unique records.

**Why fixed σ and α = 0 in E6?**
* Entropy pressure and a learned σ were what moved the Wave 4 BC+SAC actor away from the expert.
* E6 explores with a small fixed noise and gets its improvement from advantage weighting only.

**Is 82 % a guaranteed success rate?**
* No. It is the observed rate on 540 episodes.
* The 95 % interval is 78.6–85.0 %, so "at least 80 %" is not established.

**Is the policy autonomous?**
* At run time, yes: no scripted controller acts.
* But it receives privileged state and imitates a scripted expert's grasp poses, which are
  provided as targets in the observation.
