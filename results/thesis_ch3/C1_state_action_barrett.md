# C1 · BarrettHand — state / action per dimension

Source: `rl_inverse_kinematics/barrett/gym/envs/kelin/move_env.py` (registered `kelin-v0` → `MoveUr5RuthEnv`, `gym/envs/__init__.py:6-10`, `max_episode_steps=100`). File mtime 2022-05-06 09:28, i.e. after seed--1 finished (09:26) — evaluation version (`step()` returns `distance`, `reset()` returns `(obs, idx)`). Checkpoints expect the same layout (`GaussianPolicy.3050.pt`: `linear1.weight` (256, 18), `mean_linear` (9, 256)).
PyBullet joint indices (DFS order of `urdf/ur5_plus_barrett.urdf`): 1-6 UR5; 11/12/13 finger_1 prox/med/dist; 15/16/17 finger_2 prox/med/dist; 19/20 finger_3 med/dist (finger 3 has no prox/spread joint).

## Observation (18 = 1 + 8 + 9)

`get_obs()` (line 516-518) = `concat(ur5_values[6], motors[8], goal[9])` = 23; `[5:]` returned (line 338, 395) → 18. No normalisation. `observation_space = Box(0,1)^9`, `goal_space = Box(0,1)^9` (line 248-249) → `obs_dim = 18` (train.py:59).

| idx | meaning | source (line 118-123) | unit |
|---:|---|---|---|
| 0 | UR5 wrist_3 angle | `getJointState(1,6)[0]` | rad |
| 1 | finger 1 spread `finger_1/prox_joint` | `getJointState(1,11)[0]` | rad |
| 2 | finger 1 flexion `finger_1/med_joint` | `getJointState(1,12)[0]` | rad |
| 3 | finger 1 distal `finger_1/dist_joint` | `getJointState(1,13)[0]` | rad |
| 4 | finger 2 spread `finger_2/prox_joint` | `getJointState(1,15)[0]` | rad |
| 5 | finger 2 flexion `finger_2/med_joint` | `getJointState(1,16)[0]` | rad |
| 6 | finger 2 distal `finger_2/dist_joint` | `getJointState(1,17)[0]` | rad |
| 7 | finger 3 flexion `finger_3/med_joint` | `getJointState(1,19)[0]` | rad |
| 8 | finger 3 distal `finger_3/dist_joint` | `getJointState(1,20)[0]` | rad |
| 9-17 | goal fingertips 1,2,3 (x,y,z) in the nominal ee frame; **before** the transform the base-frame target is lifted by +0.1 m in z (`goal_b = goal_b + [0,0,0.1,...]`, line 404) | `self.goal` | m |

Previous action: **not included** (input dim 18 confirmed from the checkpoint).

## Action (9)

`action_space = Box(-1,1)^9` (line 247); `a = clip(a,-1,1)·0.5` (line 260-263); all **increments** on the current readings (line 266, 319), clipped with −100 reward per clip (`r`, not returned by this file version).

| idx | drives (`motor_control_rtq`, line 50-53, force 1000) | clip range | lines |
|---:|---|---|---|
| 0 | UR5 `wrist_3_joint` | none (−100 if any UR5 angle outside ±π, line 342-344) | 319, 323 |
| 1 | `finger_1/prox_joint` (spread) | [0, π/2] | 268-273 |
| 2 | `finger_1/med_joint` | [0, π/2] (URDF limit [0, 2.44]) | 274-279 |
| 3 | `finger_1/dist_joint` | [0, 0.8] (URDF [0, 0.838]) | 280-285 |
| 4 | `finger_2/prox_joint` (spread) | [0, π/2] | 287-292 |
| 5 | `finger_2/med_joint` | [0, π/2] | 293-298 |
| 6 | `finger_2/dist_joint` | [0, 0.8] | 299-304 |
| 7 | `finger_3/med_joint` | [0, π/2] | 306-311 |
| 8 | `finger_3/dist_joint` | [0, 0.8] | 312-317 |

Note: the convergence loop compares only the first 6 commanded values (`target` has 6 entries, `real[5:11]`, line 322-333) — the last three Barrett joints are commanded but not waited for. The distal joints are driven independently (the real BarrettHand couples med/dist through a breakaway clutch; the simulated hand is fully actuated with 8 DOF). `point_transform()` is **not** called in `reset()` (line 393 commented) → no wrist perturbation; the goal offset is a fixed +0.1 m in world z instead of 0.1 m along the contact normal. Fingertip links for the reward: `wam/bhand/finger_{1,2,3}/tip_link` (line 530).
