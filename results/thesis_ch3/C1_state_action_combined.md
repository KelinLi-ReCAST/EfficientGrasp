# C1 · State / action summary, all three grippers

Per-gripper detail with line numbers: `C1_state_action_ruth.md`, `C1_state_action_robotiq_3f.md`, `C1_state_action_barrett.md`.
All three envs are the same class `MoveUr5RuthEnv` (`rl_inverse_kinematics/<g>/gym/envs/kelin/move_env.py`, registered as `kelin-v0`, `max_episode_steps=100`); only the URDF, the joint lists and the clip ranges differ.

| | RUTH | Robotiq 3-Finger | BarrettHand |
|---|---|---|---|
| obs dim (returned by env / expected by checkpoint) | 13 / 13 | 15 / 15 | 18 / 18 |
| obs composition | 1 wrist_3 angle + 3 hand motors + 9 goal coords | 1 + 5 + 9 | 1 + 8 + 9 |
| obs[0] | UR5 wrist_3 (rad) | same | same |
| hand-state part (rad) | Joint_Link_1, Joint_Link_3, tendon value reconstructed as −(q_Phal_1B+0.6775) | f1_joint_1, palm_f1, f2_joint_1, palm_f2, middle_joint_1 | f1 prox/med/dist, f2 prox/med/dist, f3 med/dist |
| goal part | 3 fingertip targets (m) in a *fixed nominal* ee frame (`T_be`, p=[0.464,−0.11,0.66], R=Euler(0,π/2,0)) | same | same, after +0.1 m z-shift of the base-frame target |
| previous action in obs? | **no** | **no** | **no** |
| normalisation | none | none | none |
| declared `observation_space` | Box(0,1)^4 (+ goal_space Box(0,1)^9) — not enforced, only used for `obs_dim` | Box(0,1)^6 (+9) | Box(0,1)^9 (+9) |
| action dim | 4 | 6 | 9 |
| action semantics | `Δ = clip(a,−1,1)·0.5` rad **added to current reading**, then clipped, sent as absolute position target (force 1000) | same | same |
| a[0] | wrist_3 increment, no clip (−100 if |q|>π) | same | same |
| hand action clips | m1 [−1, π/2], m2 [−π/2, 1], tendon [−0.5, 0.12] | f1 [0,π/2], palm1 [−π/2,π/2], f2 [0,π/2], palm2 [−π/2,π/2], mid [0,π/2] | prox [0,π/2], med [0,π/2], dist [0,0.8] (×2 fingers), f3 med [0,π/2], dist [0,0.8] |
| clip penalty | −100 per clipped component | same | same |
| arm joints 1-5 | fixed at `_ur5_init`; wrist_1/wrist_2 perturbed U(±π/6) at reset (`point_transform`) | same | fixed, **no perturbation** (`point_transform` commented out) |
| target offset for reward | +0.1 m along contact-plane normal (upward) | same | +0.1 m in world z |
| fingertip links for reward | Phal_1C/2C/3C (link CoM) | finger_1_tip, finger_2_tip, finger_middle_tip | finger_{1,2,3}/tip_link |
| `step()` returns | `(obs, r, d, {})` (shaped reward) | `(obs, distance, d, {})` — eval version | `(obs, distance, d, {})` — eval version |
| `reset()` returns | `obs` | `(obs, idx)` | `(obs, idx)` |

Policy network (model.py:179-223): 3 hidden layers × 256, softplus, Gaussian head with tanh squashing, `action_scale=1, action_bias=0` (checkpoints confirm). Q-networks: 2 × (2 hidden × 256, ReLU), input = obs ⊕ action (17 / 21 / 27).
