# C2 · Reward, termination, horizon, hyper-parameters (per gripper)

Files: `E` = `rl_inverse_kinematics/<g>/gym/envs/kelin/move_env.py`, `T` = `rl_inverse_kinematics/<g>/train.py`, `S` = `sac.py`, `M` = `model.py`, `R` = `gym/envs/__init__.py`, `cfg` = `log/kelin-v0/seed--*/config.json` (`vas` block). train.py / sac.py / model.py / replay_memory.py are byte-identical across the three grippers except `--seed` default (RUTH 4, others 1; T:170) and a RUTH-only `reward_history` list (T:95-136, unused).

**Caveat on provenance.** None of the three `move_env.py` files is the version that was running during training: RUTH seeds were trained 25 Jan–1 Feb 2022 but `ruth/.../move_env.py` is dated 18 Aug 2022 (earliest surviving copy `move_env _train.py` 7 Feb 2022); `robotiq_3f/.../move_env.py` (4 May 19:43) and `barrett/.../move_env.py` (6 May 09:28) post-date the end of their training runs and are the *evaluation* versions, in which `step()` returns `distance` (sum of fingertip distances) instead of the reward `r`. That training used a shaped reward of the RUTH kind is confirmed by `progress.txt` (`Reward` column negative, e.g. barrett seed--0 epoch 0: −1457, epoch 3054: −612; ruth seed--1: −677 → −187), which would be impossible with the positive `distance`.

| item | RUTH (E = ruth) | Robotiq 3F (E = robotiq_3f) | BarrettHand (E = barrett) |
|---|---|---|---|
| per-finger distance | `dist_j = ‖p_j − goal_b_j‖·1000` (mm, Euclidean, **not squared**), `p_j` = fingertip link CoM (E:393-395) | same (E:327-329) | same (E:351-353) |
| dense term | `r −= Σ_j dist_j` (E:422) | `r −= Σ dist` (E:337) | `r −= Σ dist` (E:361) |
| per-finger sparse term | +100 if `dist_j < 15` mm else −100, each finger (E:410-416) | same (E:330-333) | same (E:354-357) |
| success bonus | +1000 and `done` if `Σ_j dist_j < 15` mm (E:418-420); extra +100 if all three `dist_j < 15` (E:425-428); `good_count>20` branch is dead code (local var reset each step, E:409, 429) | +1000 and `done` if `Σ dist < 15` (E:334-336) | same (E:358-360) |
| joint-limit penalty | −100 per clipped hand motor (E:345-362), −100 per UR5 joint outside ±π (E:386-388) | same (E:264-293, 318-320) | same (E:268-317, 342-344) |
| value returned by `step()` | `r` (E:444) | `distance = Σ dist` (E:338, 352) — eval version | `distance` (E:362, 376) — eval version |
| done | `Σ dist < 15` mm (env) **or** 100 steps (`TimeLimit`, R:9-13 `max_episode_steps=100`; `gym/wrappers/time_limit.py:17-19`) | same (R:9-13) | same (R:6-10) |
| max steps / episode | 100 (R) ; `train.py` uses `env._max_episode_steps` only for the bootstrap mask (T:108) | 100 | 100 |
| inner sim loop per step | ≤100 `stepSimulation` until commanded joints within 1e-3 (E:370-379) | ≤100 (E:301-309) | ≤100 (E:325-333) |
| episodes / epoch | 10 (`--episodes`, T:209; cfg `episodes: 10`) | 10 | 10 |
| targets / epoch | one target per episode → 10 per epoch, drawn **with replacement** by `random.choice` from `contact_list.npy` = **21 fixed row indices** of `contact_points.npy` (485376×9) (E:466-471); plus a random wrist_1/wrist_2 perturbation U(±π/6) per episode (E:521-526) | same 21 indices (hard-coded path `/home/kelin/workspace_kelin/RAL-IROS2022/train_robotiq/contact_list.npy`, E:375); perturbation on | same 21 indices (same hard-coded robotiq path, E:399); **no** perturbation (E:393) |
| γ | 0.99 (T:176; cfg) | 0.99 | 0.99 |
| α | CLI/cfg `alpha: 0.8` **but** `automatic_entropy_tuning: True` (T:185, cfg) → `log_alpha` initialised to 0 (α=1.0) and learned with Adam(lr) toward target entropy −dim(A) (S:30-33, 88-95). 0.8 is used only in the very first update. `progress.txt` Alpha: seed--1 1.16 → 11.0, seed--2 3.6 → 13.3 | same code; cfg identical | same; barrett seed--0 Alpha 1.03 → 1.00, seed--1 1.09 → 8.3 |
| τ | 0.005 (T:178, S:103 `soft_update`) | 0.005 | 0.005 |
| lr | 0.003 (T:180; cfg) for actor, critics and α | 0.003 | 0.003 |
| batch | 256 (T:187; cfg) | 256 | 256 |
| replay | 10 000 000 (T:203; cfg), FIFO list (replay_memory.py) | 10 000 000 | 10 000 000 |
| updates / env step | 1 (T:194, 119) once `len(memory) > 256`; **no** random warm-up (`--start_steps 10000` parsed, T:199, but never used) | same | same |
| target update interval | 1 (T:201; S:102) | 1 | 1 |
| policy net | 13→256→256→256→(μ,logσ)(4), softplus, tanh-squash, logσ∈[−20,2] (M:179-223) | 15→…→6 | 18→…→9 |
| Q nets | 2 × (17→256→256→1, ReLU) (M:149-176) | 21→… | 27→… |
| seeds | dirs seed--0,1,2,3 (`--seed` only names the log dir: `torch.manual_seed`, `np.random.seed`, `env.seed` are commented out, T:49-52); seed--2 was warm-started from seed--1 (`ctdir` in cfg) | seed--0, seed--1 | seed--0, seed--1 |
| epochs (cfg `epochs`) | 20480 requested; actually reached 2885 / 2475 / 5865 / 3365 (last checkpoint; save every 5 epochs, `save_freq=5`, T:158) | 20480 requested; reached 2300 (checkpoints every 100 only survive) / 145 | 20480 requested; reached 3050 / 2905 |
| wall time | seed--1: 2477 epochs in 64 412 s (progress.txt) | — (progress.txt empty) | seed--0: 3054 epochs in 65 259 s; seed--1: 2907 in 48 384 s |

Checkpoint policy/Q input sizes verified with torch (`/home/kelin/.claude/jobs/231d0744/tmp/inspect_ckpt.py`).

## Thesis claims vs. code

| thesis claim | code | verdict |
|---|---|---|
| `r = −Σ_i ‖p_i − p_i^goal‖²` | `r = −Σ_i ‖p_i−goal_i‖·1000` (mm, not squared) ± 100 per finger (15 mm threshold) + 1000 success (Σ<15 mm) + 100 all-fingers bonus − 100 per joint-limit clip | **mismatch** (shaped, non-squared, in mm) |
| SAC α = 0.8 | 0.8 is the CLI default but α is auto-tuned from 1.0 (and grows to ≈8–13) | **mismatch** (α not fixed) |
| lr = 0.003, τ = 0.005, γ = 0.99 | 0.003 / 0.005 / 0.99 | match |
| each epoch 10 target sets, up to 100 episodes per target | each epoch 10 episodes, each with one target (sampled from only 21 indices) and ≤100 **steps** | **terminology mismatch** (episodes↔steps); the pool of targets is 21 rows, not 485 376 |
| previous action appended to policy input | not in observation; policy input dims 13/15/18 = joints + goal | **mismatch** |
| policy trained at fixed end-effector orientation | arm joints 1–5 fixed, but wrist_1/wrist_2 randomly perturbed ±30° per episode for RUTH and Robotiq (`point_transform`); Barrett: fixed | partly (Barrett only) |
| two-step approach (arm pose at distance d along plane normal, then gripper) | training: target shifted 0.1 m along the normal (RUTH/Robotiq) or +0.1 m z (Barrett); demo.py: `calc_target_pos(..., distance=0.3)` → d = 0.3 m then `rl_step` policy steps (ruth demo.py:385-420), note demo applies `a` without the 0.5 scaling | consistent in structure; d differs between training (0.1) and demo (0.3) |
| action dims 4 / 6 / 9 | 4 / 6 / 9 | match |
| RUTH observation 13 | 13 | match (composition 1+3+9) |
