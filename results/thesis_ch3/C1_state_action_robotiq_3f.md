# C1 · Robotiq 3-Finger — state / action per dimension

Source: `rl_inverse_kinematics/robotiq_3f/gym/envs/kelin/move_env.py` (registered `kelin-v0` → `MoveUr5RuthEnv`, `gym/envs/__init__.py:9-13`, `max_episode_steps=100`). File mtime 2022-05-04 19:43 — i.e. **after** seed--0 training finished (last checkpoint 12:30 the same day); it is the evaluation version (`step()` returns `distance` instead of `r`, `reset()` returns `(obs, idx)`, see C2). The state/action layout is the same as what the checkpoints expect (`GaussianPolicy.2300.pt`: `linear1.weight` (256, 15), `mean_linear` (6, 256)).
PyBullet joint indices (DFS order of `urdf/ur5_plus_robotiq_3f.urdf`): 1-6 UR5, 10 palm_finger_1_joint, 11 finger_1_joint_1, 15 palm_finger_2_joint, 16 finger_2_joint_1, 21 finger_middle_joint_1 (finger joints 2/3 are `fixed` in this URDF except finger_middle_joint_3).

## Observation (15 = 1 + 5 + 9)

`get_obs()` (line 492-494) = `concat(ur5_values[6], motors[5], goal[9])` = 20; `reset()`/`step()` return `[5:]` (line 314, 371) → 15. No normalisation. `observation_space = Box(0,1)^6` (line 244), `goal_space = Box(0,1)^9` (line 245) → `obs_dim = 6+9 = 15` in train.py:59.

| idx | meaning | source | unit |
|---:|---|---|---|
| 0 | UR5 wrist_3 angle | `getJointState(1,6)[0]` (line 115-117) | rad |
| 1 | finger 1 proximal flexion `finger_1_joint_1` | `getJointState(1,11)[0]` (line 118) | rad |
| 2 | finger 1 palm (scissor) joint `palm_finger_1_joint` | `getJointState(1,10)[0]` | rad |
| 3 | finger 2 proximal flexion `finger_2_joint_1` | `getJointState(1,16)[0]` | rad |
| 4 | finger 2 palm (scissor) joint `palm_finger_2_joint` | `getJointState(1,15)[0]` | rad |
| 5 | middle finger flexion `finger_middle_joint_1` | `getJointState(1,21)[0]` | rad |
| 6-14 | goal fingertips 1,2,3 (x,y,z) in the nominal ee frame (same `T_be`, `p=[0.464,-0.11,0.66]`, line 386-390) | `self.goal` | m |

Previous action: **not included** (input dim 15 confirmed from the checkpoint).

## Action (6)

`action_space = Box(-1,1)^6` (line 243); `a = clip(a,-1,1)·0.5` (line 256-259, `max_delta_action=0.5` line 247); all components are **increments** added to the current joint readings (line 262, 295) and then clipped; a clip costs −100 reward each (in `r`, which this file does not return).

| idx | drives (`motor_control_rtq`, line 50-53, force 1000) | clip range | lines |
|---:|---|---|---|
| 0 | UR5 `wrist_3_joint` | none (−100 if any UR5 angle outside ±π, line 318-320) | 295, 299 |
| 1 | `finger_1_joint_1` | [0, π/2] | 264-269 |
| 2 | `palm_finger_1_joint` | [−π/2, π/2] (URDF limit is only [−0.21, 0.2967]) | 282-287 |
| 3 | `finger_2_joint_1` | [0, π/2] | 270-275 |
| 4 | `palm_finger_2_joint` | [−π/2, π/2] (URDF limit [−0.2967, 0.21]) | 288-293 |
| 5 | `finger_middle_joint_1` | [0, π/2] (URDF limit [0, 0.9]) | 276-281 |

Convergence loop after the command: until `Σ(real−target)² < 1e-3` over the 6 commanded joints or 100 sub-steps (line 301-309). Fingertip links used for the reward: `finger_1_tip, finger_2_tip, finger_middle_tip` (line 506).
