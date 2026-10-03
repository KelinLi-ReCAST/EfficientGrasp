# C1 · RUTH — state / action per dimension

Source: `rl_inverse_kinematics/ruth/gym/envs/kelin/move_env.py` (registered as `kelin-v0`, class `MoveUr5RuthEnv`, via `gym/envs/__init__.py:9-13` and `gym/envs/kelin/__init__.py:5`; `max_episode_steps=100`).
Joint indices refer to PyBullet's depth-first URDF order of `urdf/ur5_plus_RUTH.urdf` (0 world_joint, 1-6 UR5, 7 ee_fixed, 8 ruth_ur5_joint, 9 tcp_joint, 10 Joint_Link_1, 11 Joint_Link_2, 12-14 Phal_1A/B/C, 15 Joint_Link_3, 16 Joint_Link_4, 17-19 Phal_2A/B/C, 20-22 Phal_3A/B/C; verified by DFS of the URDF tree, `evaluation/thesis_ch3/_draft_C.md` §C1).

## Observation (13 = 1 + 3 + 9)

`get_obs()` (line 579-581) builds `concat(ur5_values[6], RUTH_motors[3], goal[9])` = 18 values; `reset()` (line 463) and `step()` (line 383) return `obs[5:]` → 13 values. No normalisation anywhere.

| idx | meaning | source | unit |
|---:|---|---|---|
| 0 | UR5 wrist_3 joint angle | `pybullet.getJointState(1,6)[0]` (line 200-202, `ur5_values[5]`) | rad |
| 1 | RUTH base motor 1 (five-bar link 1) | `getJointState(1,10)[0]` = Joint_Link_1 (line 203) | rad |
| 2 | RUTH base motor 2 (five-bar link 3) | `getJointState(1,15)[0]` = Joint_Link_3 (line 203) | rad |
| 3 | RUTH finger-closure motor (tendon), reconstructed | `-(getJointState(1,13)[0] + 0.12745044 + 0.55)` = inverse of the Phal_1B command in `motor_control_ruth` (line 140, 204) | rad |
| 4-6 | goal fingertip 1 (x,y,z) | `self.goal[0:3]` = contact point expressed in a *nominal* end-effector frame: `T_be = inv(T_eb)` with fixed `p=[0.464,-0.11,0.66]`, `R=Euler(0,π/2,0)` (line 477-496) | m |
| 7-9 | goal fingertip 2 (x,y,z) | `self.goal[3:6]` | m |
| 10-12 | goal fingertip 3 (x,y,z) | `self.goal[6:9]` | m |

* `observation_space` is declared as `Box([0]*4,[1]*4)` (line 325) and `goal_space` as `Box([0]*9,[1]*9)` (line 326); train.py uses `obs_dim = 4 + 9 = 13` (train.py:59). The declared bounds are never enforced (actual values are rad / m, can be negative).
* **The previous action is NOT part of the observation** (thesis claim not supported by code). The policy input is exactly these 13 values; checkpoint `GaussianPolicy.2885.pt` has `linear1.weight` of shape (256, 13) (checked with torch, see `_draft_C.md`).
* Note the goal (idx 4-12) is the *unperturbed* ee-frame target, whereas the reward target `goal_b` (line 548) is recomputed from the perturbed ee pose in `point_transform()` and shifted by `0.1·n̂` (contact-plane normal); this constant offset is not observed.

## Action (4)

`action_space = Box(-1, 1)^4` (line 324). `step()` (line 335-369): `a = clip(a, -1, 1) * max_delta_action` with `max_delta_action = 0.5` (line 328, 337, 340) → every component is an **increment of at most ±0.5 rad per step added to the current joint reading**; the result is clipped to the ranges below (with −100 reward each time a clip happens) and sent as an absolute position target with `POSITION_CONTROL`.

| idx | drives | update rule | clip range after update | penalty on clip |
|---:|---|---|---|---|
| 0 | UR5 `wrist_3_joint` (joint 6) | `ur5_values[5] += 0.5·a0` (line 364, 368), force 1000 | none (only −100 if any UR5 angle leaves ±π, line 386-388) | — |
| 1 | RUTH motor 1 `Joint_Link_1` | `RUTH_motors[0] += 0.5·a1` (line 343) | [−1, π/2] (line 345-350) | −100 |
| 2 | RUTH motor 2 `Joint_Link_3` | `RUTH_motors[1] += 0.5·a2` | [−π/2, 1] (line 351-356) | −100 |
| 3 | RUTH finger closure (one value → all 9 phalanx joints via `motor_control_ruth`, line 130-145) | `RUTH_motors[2] += 0.5·a3` | [−0.5, 0.12] (line 357-362) | −100 |

After the command the simulation is stepped until the 4 commanded joints are within `Σ(real−target)² < 1e-3` or 100 sub-steps (line 370-379).

The 5 arm joints shoulder_pan … wrist_2 are **not** controlled by the policy: they are set to `_ur5_init` at reset (line 318, 447) and then wrist_1/wrist_2 are perturbed once by U(−π/6, π/6) in `point_transform()` (line 524-525).
