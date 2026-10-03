# 第三章核查 · C 部分：RL 逆运动学（SAC）—— 状态/动作、奖励与超参数、Fig 3.7 重绘、数值 IK 对比

工作目录 `/home/kelin/EfficientGrasp`（分支 `thesis-ch3-verification`）；原始副本 `/home/kelin/RAL-IROS2022` 只读。
所有脚本在 `evaluation/thesis_ch3/`，所有输出在 `results/thesis_ch3/`。未改动任何既有代码文件。

产出文件一览

| 文件 | 内容 |
|---|---|
| `results/thesis_ch3/C1_state_action_{ruth,robotiq_3f,barrett}.md`, `C1_state_action_combined.md` | C1 状态/动作逐维表 |
| `results/thesis_ch3/C2_reward_hparams.md` | C2 奖励/终止/超参数表 + 论文说法对照 |
| `evaluation/thesis_ch3/plot_rl_error.py` → `results/thesis_ch3/C3_rl_error_table.{md,csv}`, `fig_rl_error.{pdf,png}` | C3 原始日志统计与重绘 |
| `evaluation/thesis_ch3/c4_numeric_ik_vs_rl.py` → `results/thesis_ch3/C4_rl_vs_numeric_ik.md`, `C4_numeric_ik_raw.csv` | C4 数值 IK 运行结果 |
| `/home/kelin/.claude/jobs/231d0744/tmp/inspect_ckpt.py` | 用 torch 读取 checkpoint 网络形状（只读） |

---

## C1 · 状态 / 动作

### 注册与文件版本
- 三个夹爪的 `gym/envs/__init__.py` 都把 `kelin-v0` 注册到 `gym.envs.kelin:MoveUr5RuthEnv`，`max_episode_steps=100`（ruth/robotiq 第 9-13 行，barrett 第 6-10 行）；`gym/envs/kelin/__init__.py:5` 从 `move_env.py` 导入。`move_env_2.py`、`move_env _train.py` 未被注册（后两者在 robotiq/barrett 目录下其实是 RUTH 版本的拷贝）。
- 文件时间（`ls --time-style=long-iso`，原始副本相同）：`ruth/.../move_env.py` 2022-08-18；`robotiq_3f/.../move_env.py` 2022-05-04 19:43；`barrett/.../move_env.py` 2022-05-06 09:28。RUTH 四个 seed 训练于 2022-01-25 ~ 02-01，robotiq seed--0 训练 05-03 19:30 ~ 05-04 12:30，barrett seed--0 05-04 21:05 ~ 05-05 15:30、seed--1 05-05 20:00 ~ 05-06 09:26。**因此仓库中没有任何一个 move_env.py 是训练时运行的版本**；robotiq/barrett 的是评估版（`step()` 返回 `distance` 而非 `r`，`reset()` 返回 `(obs, idx)`，与 `val.py` 的 `o, idx = env.reset()` 配套）。
- PyBullet 关节编号按 URDF 树的深度优先顺序（自行按 URDF parent/child 重建，与代码中 `getJointState(1, 1..6)` 读 UR5 六个关节、`10/15/13`（RUTH）、`11/10/16/15/21`（Robotiq）、`11,12,13,15,16,17,19,20`（Barrett）一致）。

### 观测（无任何归一化）
`get_obs()` = `concat(ur5 六关节角, 手部电机读数, goal 九坐标)`，`reset()`/`step()` 返回 `[5:]`，即**丢掉前五个 UR5 关节**：

| | RUTH | Robotiq | Barrett |
|---|---|---|---|
| 维度 | **13 = 1 + 3 + 9** | 15 = 1 + 5 + 9 | 18 = 1 + 8 + 9 |
| [0] | wrist_3 角 (rad) | 同 | 同 |
| 手部 | Joint_Link_1、Joint_Link_3 角；腱电机值由 `-(q_Phal_1B + 0.12745 + 0.55)` 反推（E:203-204） | f1_joint_1, palm_f1, f2_joint_1, palm_f2, middle_joint_1 | f1 prox/med/dist, f2 prox/med/dist, f3 med/dist |
| goal | 3 个指尖目标 (m)，用**固定名义末端位姿** `p=[0.464,-0.11,0.66]`, `R=Euler(0,π/2,0)` 变换到末端系（E:477-496） | 同 | 同（但基座系目标先 +0.1 m z，E:404） |

- `observation_space` 声明为 `Box(0,1)^4/6/9`，`goal_space=Box(0,1)^9`，只用于 `train.py:59` 计算 `obs_dim`，边界从不检查。
- **论文"上一步动作拼入策略输入"的说法在代码中不成立**：checkpoint `GaussianPolicy.*.pt` 的 `linear1.weight` 形状为 (256,13)/(256,15)/(256,18)（`inspect_ckpt.py`，`/home/kelin/venvs/lerobot/bin/python`），恰等于关节+目标维度。
- 观测中的 goal 是未扰动的名义末端系坐标；奖励用的 `goal_b` 则由 `point_transform()` 用扰动后的真实末端位姿重算并沿接触平面法向 +0.1 m（E:521-548）。

### 动作
`action_space=Box(-1,1)^{4/6/9}`；`a ← clip(a,-1,1)·0.5`（`max_delta_action=0.5`，E:328/247/251）；**每一维都是加到当前读数上的增量**（≤0.5 rad/步），再按下表裁剪，裁剪一次 −100，然后以 `POSITION_CONTROL, force=1000` 发绝对目标；之后最多 100 个 `stepSimulation` 直到 `Σ(real−target)²<1e-3`。

| | a[0] | 手部各维裁剪范围 |
|---|---|---|
| RUTH | wrist_3 增量，无裁剪（任一 UR5 关节超 ±π 则 −100） | m1 [−1, π/2], m2 [−π/2, 1], 腱 [−0.5, 0.12]（E:345-362） |
| Robotiq | 同 | f1 [0,π/2], palm1 [−π/2,π/2], f2 [0,π/2], palm2 [−π/2,π/2], mid [0,π/2]（E:264-293；URDF 中 palm 关节限位仅 ±0.21/0.30 rad，finger_joint_1 [0,0.9]） |
| Barrett | 同 | prox [0,π/2], med [0,π/2], dist [0,0.8] ×2 指；f3 med [0,π/2], dist [0,0.8]（E:268-317）；收敛循环只比较前 6 个指令值（E:322-333） |

UR5 其余 5 个关节不受策略控制：reset 时置 `_ur5_init`，RUTH/Robotiq 再对 wrist_1、wrist_2 各加 U(−π/6, π/6) 随机扰动（`point_transform`，E:524-525）；Barrett 的 `point_transform()` 被注释（E:393），姿态固定。

---

## C2 · 奖励、终止、回合长度、超参数（详表见 `C2_reward_hparams.md`）

**奖励（以 RUTH `move_env.py:390-431` 为准；robotiq/barrett 同构，但评估版把 `distance` 作为返回值）**

```
dist_j = ‖p_j − goal_b_j‖ · 1000        # mm，欧氏距离，不平方；p_j 为指尖 link CoM（get_pos）
r  = −Σ_j dist_j                        # 稠密项
   + Σ_j (+100 若 dist_j < 15 mm，否则 −100)
   + 1000 且 done=True 若 Σ_j dist_j < 15 mm   # 三指之和 < 15 mm，即平均每指 < 5 mm
   + 100 若三指都 < 15 mm（RUTH 版独有；good_count>20 分支是死代码）
   − 100 × (被裁剪的手部电机数) − 100 × (超 ±π 的 UR5 关节数)
```
- 论文 "r = −Σ‖p_i − p_i^goal‖²" 与代码不符：代码是 **非平方、毫米单位的稠密项 + 稀疏奖惩**。
- 终止：`Σdist<15 mm` 或 `TimeLimit` 100 步（`gym/wrappers/time_limit.py:17-19`）。`val.py` 评估时强制跑满 100 步（`d = pit>=env._max_episode_steps`，val.py:91）。
- 每 epoch 10 个 episode（`--episodes 10`，train.py:209），每个 episode 一个目标：`random.choice(contact_list.npy)`，而 `contact_list.npy` 只有 **21 个行号**（E:466-471；`np.load` 得 shape (21,)：`[460435 38669 385634 471437 141518 58666 364150 57173 310487 227177 66987 290595 7132 315339 278559 341429 162503 384592 444935 159485 253238]`）。即**训练目标池只有 21 组接触点**（RUTH/Robotiq 另加随机腕部扰动），而不是 485 376 行全部。注意 `contact_list.npy` 时间戳 2022-02-06，晚于 RUTH 训练，RUTH 训练时的目标采样方式 未找到（训练版 env 未保存）。
- 论文 "每 epoch 10 个目标，每目标最多 100 个 episode"：代码是 10 个 episode/epoch、每 episode ≤100 **步**。
- 超参数（`train.py` 默认值 = 全部 8 个 `config.json`）：γ 0.99、τ 0.005、lr 0.003、batch 256、replay 1e7、hidden 256、updates_per_step 1、target_update_interval 1、`start_steps 10000` **被解析但从未使用**（无随机暖启动）、`episodes 10`、`epochs 20480`（实际跑到的最后 checkpoint：ruth 2885/2475/5865/3365，robotiq 2300/145，barrett 3050/2905；每 5 epoch 存一次）。
- **α**：`--alpha 0.8` 但 `automatic_entropy_tuning=True`（train.py:185，config 一致）；`sac.py:30-33` 把 `log_alpha` 初始化为 0（α=1），用 Adam(lr=0.003) 自动调节；0.8 只在第一次更新时用到。`progress.txt` 的 Alpha 列：ruth seed--1 1.16→11.0、seed--2 3.65→13.3、barrett seed--0 1.03→1.00、seed--1 1.09→8.30。**论文 "α=0.8 固定" 不成立。**
- 网络：策略 3 隐层×256、softplus、tanh 压缩（model.py:179-223）；双 Q 各 2 隐层×256 ReLU（model.py:149-176）。
- seeds：`torch.manual_seed / np.random.seed / env.seed` 全被注释（train.py:49-52），`--seed` 只决定日志目录名；ruth seed--2 由 seed--1 热启动（config `ctdir`）。
- 2022-06 的 `revision_lr_ablation/`（lr 0.01/0.005/0.003，α 初值 0.2，2048 epochs）与论文无直接对应，仅记录。
- 与三夹爪差异：Barrett 无腕部扰动、目标偏移为世界 z +0.1 m（而非法向）；Barrett/Robotiq 的 `contact_list.npy` 路径硬编码到 `.../train_robotiq/`；其余相同。

---

## C3 · Fig 3.7 重绘（`evaluation/thesis_ch3/plot_rl_error.py`）

### 保存量到底是什么（`val.py:75-104`，三份只差 GUI/DIRECT、保存文件名与硬编码 seed/itr）
```python
distance += r            # r = env.step() 返回值；评估版 env 返回 distance = Σ_j ‖p_j−goal_b_j‖·1000 (mm)
d = (pit>=env._max_episode_steps)   # 固定 100 步
avg_dist = distance/pit/3           # ⇒ 一个 episode 内 100 步的"每指平均欧氏误差 (mm)"的时间平均
contact_list.append(avg_dist); np.save('epoch_200', contact_list)   # 100 个 episode → (100,)
```
- 单位 **mm**（`*1000` 在 env 内），**不平方**，是**三指平均**而非三指之和，且是**整个 episode 100 步的时间平均**（包含起始远离目标的若干步），因此是末步误差的上界；末步误差无法从日志恢复（ruth `val.py:103` 的 `error.append(r)` 没保存）。
- 全部日志值为正且落在 8–135 mm，说明 RUTH 的日志也是用返回 `distance` 的 env 版本评估的（若用返回 `r` 的版本，值会是负数）。

### 原始统计（`C3_rl_error_table.md`，命令 `python3 evaluation/thesis_ch3/plot_rl_error.py`）

| gripper | file | n | mean | median | p5 | p95 | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ruth | epoch_0 | 100 | 84.25 | 85.17 | 45.8 | 129.5 | 41.34 | 135.41 |
| ruth | epoch_10 | 100 | 41.56 | 29.76 | 19.3 | 79.7 | 14.16 | 82.51 |
| ruth | epoch_100 | 100 | 36.94 | 28.68 | 17.8 | 71.7 | 14.10 | 80.33 |
| ruth | epoch_200（**n=20**） | 20 | 25.31 | 20.04 | 10.6 | 49.2 | 10.27 | 50.21 |
| ruth | epoch_500 | 100 | 43.06 | 43.35 | 20.4 | 80.2 | 9.88 | 84.23 |
| ruth | epoch_1000 | 100 | 26.06 | 23.02 | 9.2 | 47.1 | 8.60 | 57.67 |
| ruth | epoch_1500 | 100 | 39.86 | 42.27 | 23.3 | 56.7 | 13.49 | 59.70 |
| ruth | epoch_2000 | 100 | 18.28 | 16.65 | 11.5 | 34.3 | 10.53 | 35.77 |
| ruth | epoch_2500 | 100 | 17.90 | 16.87 | 12.1 | 33.7 | 9.46 | 34.61 |
| ruth | **epoch_2885** | 100 | 15.54 | 15.23 | 8.8 | 24.1 | 8.30 | 29.43 |
| robotiq_3f | epoch_0 | 100 | 42.93 | 39.09 | 27.2 | 64.8 | 25.33 | 72.68 |
| robotiq_3f | epoch_00（论文用） | 100 | 48.18 | 45.87 | 30.0 | 70.4 | 27.79 | 82.32 |
| robotiq_3f | epoch_1（= barrett/epoch_1，md5 c8b197b1…，同一 mtime） | 100 | 45.61 | 41.54 | 28.5 | 70.5 | 25.98 | 85.62 |
| robotiq_3f | epoch_10 | 100 | 19.85 | 16.84 | 10.0 | 42.0 | 9.35 | 43.37 |
| robotiq_3f | epoch_100 | 100 | 17.48 | 16.36 | 9.3 | 31.9 | 7.32 | 36.84 |
| robotiq_3f | epoch_200 | 100 | 36.27 | 36.59 | 13.8 | 62.6 | 10.68 | 76.13 |
| robotiq_3f | epoch_500 | 100 | 33.29 | 30.83 | 20.5 | 49.6 | 18.02 | 85.60 |
| robotiq_3f | epoch_1000 | 100 | 28.55 | 23.81 | 15.6 | 56.0 | 10.87 | 58.95 |
| robotiq_3f | epoch_1500 | 100 | 17.93 | 17.46 | 8.3 | 29.2 | 8.11 | 39.07 |
| robotiq_3f | epoch_2000 | 100 | 18.62 | 18.77 | 8.4 | 28.0 | 8.06 | 38.80 |
| robotiq_3f | epoch_2500 | 100 | 17.55 | 16.52 | 8.3 | 36.5 | 7.64 | 38.40 |
| robotiq_3f | **epoch_2885** | 100 | 18.93 | 18.99 | 8.7 | 37.4 | 8.17 | 38.57 |
| barrett | epoch_0 | 100 | 75.42 | 76.09 | 43.1 | 113.1 | 40.00 | 115.08 |
| barrett | epoch_1（robotiq 文件的拷贝） | 100 | 45.61 | 41.54 | 28.5 | 70.5 | 25.98 | 85.62 |
| barrett | epoch_10 | 100 | 38.77 | 33.05 | 22.4 | 68.6 | 21.12 | 83.08 |
| barrett | epoch_100 | 100 | 24.98 | 22.10 | 14.2 | 44.7 | 12.42 | 46.59 |
| barrett | epoch_200 | 100 | 18.05 | 18.61 | 11.2 | 26.0 | 10.92 | 27.72 |
| barrett | epoch_500 | 100 | 13.77 | 12.16 | 10.0 | 22.4 | 9.98 | 23.92 |
| barrett | epoch1000（无下划线） | 100 | 12.03 | 11.19 | 9.3 | 17.2 | 8.79 | 24.98 |
| barrett | epoch_1000（下划线，≈35） | 100 | 35.48 | 36.52 | 19.9 | 42.8 | 14.93 | 59.67 |
| barrett | epoch1500（无下划线） | 100 | 12.41 | 11.05 | 9.1 | 24.3 | 8.68 | 27.57 |
| barrett | epoch_1500（下划线，≈36） | 100 | 36.33 | 36.67 | 24.8 | 43.9 | 21.50 | 47.16 |
| barrett | epoch_2000 | 100 | 11.44 | 11.09 | 9.8 | 18.0 | 9.49 | 18.30 |
| barrett | epoch_2500 | 100 | 11.50 | 11.27 | 9.4 | 16.5 | 8.99 | 16.99 |
| barrett | **epoch_2885** | 100 | 10.88 | 10.72 | 9.1 | 12.7 | 8.91 | 17.34 |

（完整的 p5/p95 等见 `C3_rl_error_table.csv`。）

### 结论性数字
- 末 epoch（文件名 2885）每指平均误差：**RUTH 15.5 mm（中位 15.2）、Robotiq 18.9（19.0）、Barrett 10.9（10.7）**。按正确单位解释，"**within 5 mm per finger**" **在这些日志里不成立**：三夹爪没有任何一个 episode 的平均误差 < 5 mm（RUTH 最小 8.3，Robotiq 8.2，Barrett 8.9 mm）；<15 mm 的 episode 比例分别为 45 %、29 %、96 %。若论文的 5 mm 指的是环境成功阈值（Σ<15 mm ⇔ 平均每指 <5 mm），应改写为"成功判据"，而非"达到的精度"。
- 降幅（均值，首→末文件）：**RUTH 84.25→15.54 = 81.6 %**（中位数 82.1 %）——与论文 "81 %" 一致（且只对 RUTH 成立）；Robotiq 48.18→18.93 = 60.7 %；Barrett 75.42→10.88 = 85.6 %。
- 旧绘图脚本 `evaluation/simulation/RL_eval.py` 的两个问题可解释论文图里 RUTH 末端≈5 mm 的来源：(i) 它用 `sum/100` 求均值（第 60-61 行），而 `ruth/epoch_200.npy` 只有 20 个值 → 该点被算成 **5.06 mm**；(ii) 随后 `pppp.sort(reverse=True)`（第 62 行）把 10 个均值按降序重排再画，于是 5.06 落到最右端（标签 "2885"），曲线也被人为变成单调。重排后的 RUTH 序列：84.25, 43.06, 41.57, 39.86, 36.94, 26.06, 18.28, 17.90, 15.54, **5.06**。Robotiq 末点 17.48、Barrett 10.88。min/max 带还被 0.3–0.5 的手调系数压缩（第 71-72、104-105、134-135 行）。
- 新图 `fig_rl_error.{pdf,png}`：左图按真实 epoch（对数轴）画中位数与 5–95 % 带（n≠100 处标注 n=20）；右图并列变体文件（robotiq epoch_0/00/1、barrett 带/不带下划线 1000、1500）。曲线并不单调（ruth 500/1500、robotiq 200/500、barrett 1000/1500 回升），见下节。

### 文件来源与异常说明（mtime 来自 `ls --time-style=full-iso`）
- **ruth**：epoch_0/1000/2885/100 生成于 2022-02-07 14:56–15:46（训练结束 6 天后，评估版 env 即 `move_env _train.py` 2022-02-07 16:28 的前身）；epoch_10/2000/2500/1500/500 生成于 2022-05-03；**epoch_200（20 条）生成于 2022-06-12 20:22**，`ruth/val.py`（mtime 20:24）末态写着 `fpath='log/kelin-v0/seed--0'`, `args.itr=1000`, `np.save('epoch_200')`, `--episodes` 默认 100 → 该文件是用 `-n 20` 跑的，用的 checkpoint 无法确定（文件名 200 与代码里的 1000 矛盾）。此外 ruth 日志非单调（500: 43.1, 1500: 39.9 高于 1000: 26.1），5 月的 5 个点与 2 月的 4 个点可能来自不同 seed（4 个 seed 中只有 seed--0/2/3 有 2885）—— 无法确定。`.mat` 四份与同名 `.npy` 数值一致。
- **robotiq**：全部生成于 2022-05-04 12:47–20:05。seed--0 最后 checkpoint 为 **2300**（12:30 结束；目录里只剩每 100 epoch 一个的 24 个 checkpoint），seed--1 16:22 才开始且只到 145。因此 `robotiq/epoch_2500.npy`（15:15）与 `epoch_2885.npy`（16:07）**不可能对应 2500/2885 的 checkpoint**（`load_policy` 对不存在的 itr 会抛 FileNotFoundError），最可能是用 `itr='last'`=2300 或其它已有 checkpoint 评估后按 RUTH 的标签命名的；具体 无法确定。epoch_1500/2000/2500/2885 的均值 17.9/18.6/17.5/18.9 几乎相同也支持"同一阶段策略反复评估"。`epoch_0`（18:26）与 `epoch_00`（20:05，`val.py` 末态 `itr=0`, `np.save('epoch_00')`）是 epoch-0 策略的两次评估（42.9 vs 48.2 mm，随机目标/扰动不同），论文用的是 epoch_00。`epoch_1.npy` 与 `barrett/epoch_1.npy` 字节相同、mtime 相同（13:01，早于 barrett 训练开始 21:05）→ barrett/epoch_1 是复制过来的 robotiq 文件，不是 Barrett 的评估；两者均未被论文图使用。
- **barrett ≈35 vs ≈12 mm 两组**：
  - 下划线 `epoch_1000`（05-05 18:48）、`epoch_1500`（05-05 19:16）生成时只有 **seed--0** 训练完成（15:30），seed--1 20:00 才开始 → 这两份 ≈35–36 mm 的文件只能来自 seed--0。
  - 05-06 09:39–11:12 连续生成 epoch_2885, 2000, 2500, 0, 10, 100, 200, 500（75.4→38.8→25.0→18.0→13.8→…→11.4/11.5/10.9，单调且平滑），紧接着 11:23/11:35 生成无下划线 `epoch1000`（12.0）、`epoch1500`（12.4）；`barrett/val.py` 末态硬编码 `fpath='log/kelin-v0/seed--1'`, `itr=1500`, `np.save('epoch1500')`（mtime 05-06 11:25）。seed--1 于 09:26 结束，13 分钟后评估开始。seed--0/seed--1 的 `progress.txt` 末 epoch 平均奖励分别为 −612 与 +224，说明 seed--1 明显更好；`barrett/RL_gripper.py`（05-07 20:14）与 `demo.py` 载入的也是 **seed--1**（`fpath=.../train_barrett/log/kelin-v0/seed--1`，`itr='last'`=2905）。
  - 因而最合理的解释：**无下划线系列（≈12 mm）= seed--1，与 05-06 的其余下划线文件同源；下划线 `epoch_1000/1500`（≈35 mm）= 更差的 seed--0**。论文图（`RL_eval.py:36-46`）读取的是下划线文件，所以它的 Barrett 曲线实际上**混合了两个 seed**（1000/1500 两点来自 seed--0），旧脚本的降序排序掩盖了由此产生的回升。严格来说 05-06 那批下划线文件属于哪个 seed 没有直接记录（val.py 被反复改写），以上为基于时间戳、val.py 末态、RL_gripper.py 与 progress.txt 的推断。
- Barrett `epoch_2885` 的 checkpoint 两个 seed 都存在（3050、2905 ≥ 2885，每 5 存一次），robotiq 的则不存在（见上）。

---

## C4 · 数值 IK（`ruth_grasping_kinematics.py`）与 RL 的对比（`C4_rl_vs_numeric_ik.md`）

### 求解器做什么（`gripper_representation/feature_extraction/ruth_grasping_kinematics.py`，与 `rl_inverse_kinematics/*/ruth_grasping_kinematics.py` md5 相同）
- 输入三个接触点（**毫米**；杆长 l1=28.5、l2=70、fingerLength=100，第 67-72 行；历史调用处 `compute_ruth_pose(cp*1000, …)`，如 `ruth/gym/envs/kelin/grasping_with_RUTH_2.py:196`）。
- 第 13-45 行：按"哪两条边最接近等长"重排，使 CP1 为**等腰三角形顶点**（文件首行注释：assume contact points form isosceles triangle），并翻转使法向朝下；第 47-54 行建立接触平面坐标系 R（x 沿 CP2→CP3，z 为平面法向），之后**所有计算在该平面内进行（平面假设）**。
- 第 59-177 行：对"物体角度"做 100 格网格搜索，内层对五连杆电机角 t∈[0,π] 以 1° 步长搜索，且**强制对称 θ1=t, θ2=π−t**（第 118-119 行），目标是让指根到接触点距离 `|P2−CP3|` 与 `|P3−CP1|` 相等（第 152 行）。这是网格搜索，不是 fsolve。
- 第 181-193 行：唯一的 `fsolve`，**1 个未知数** θ1s（手指弯曲角），**1 个方程** `P9s_x(θ)+hori_bend=0`，初值 0；手指为平面两段模型（44.7 mm 与 35 mm，关节耦合为 θ、2θ），`hori_bend` 取三指所需水平伸展的**最大值**（第 190 行）——即三指同一弯曲角（单腱电机假设）。
- 返回 `(TCP_Point, theta1_calc, R, RT)`：手臂 TCP 位置（mm，接触点坐标系→世界）、一个弯曲角、旋转矩阵。**不返回五连杆电机角 `opt_t`，也没有任何前向运动学**。

### 运行结果（`python3 evaluation/thesis_ch3/c4_numeric_ik_vs_rl.py`，seed 0；含 θ 范围与不收敛样本细节见 `C4_rl_vs_numeric_ik.md`）

| 目标集 | n | 异常 | NaN | fsolve ier==1 | 中位 θ | 所需伸展中位/p95 (mm) | 三指伸展差 均值/中位/p95 (mm) |
|---|---:|---:|---:|---:|---:|---|---|
| `gripper_representation/ruth_workspace_sampling/contact_points.npy`（8649 行抽 1000） | 1000 | 0 | 0 | 1000 | 8.0° | 22.6 / 60.7 | 3.9 / 0.3 / 38.9 |
| `rl_inverse_kinematics/ruth/contact_points.npy`（485376 行抽 1000，RL 训练目标分布） | 1000 | 0 | 0 | 999 | 3.6° | 14.1 / 31.4 | 0.3 / 0.3 / 0.7 |
| 同上，`contact_list.npy` 的 21 个训练目标 | 21 | 0 | 0 | 21 | 3.2° | 13.3 / 17.0 | 0.2 / 0.2 / 0.5 |

- 唯一不收敛样本（行 332345）所需伸展 147 mm，超出平面手指模型的最大伸展（≈67 mm），残差 79 mm。
- 20–29 % 的解 θ 为微小负值（−1~−2°），对应接触点比伸直手指还近（物理上不可达的"反向弯曲"），fsolve 仍报收敛。
- RL 对照（C3，ruth epoch_2885）：均值 15.54 mm、中位 15.23 mm、p95 24.14 mm（n=100）。

### 对比状态：**未完成（部分）**
无法进行同口径的指尖误差对比，原因：
1. 求解器**没有 RUTH 的前向运动学**（五连杆 + 三指三维），只给出 TCP、一个弯曲角和旋转矩阵，连搜索到的电机角都不输出；指尖位置只能在 PyBullet 中读取（`move_env.py: get_pos()`），而本机无 PyBullet。不编造前向模型。
2. RL 误差是全臂+手 PyBullet 模型、带随机腕部扰动、100 步时间平均的量；数值解是静态平面模型。
3. 能报告的只有求解器自身的可行性/收敛指标（上表）：在 RL 训练目标分布上 99.9 % 收敛、三指伸展差中位 0.3 mm（因为这些目标本来就是同一腱值下采样得到的近等腰三角形）；在 8649 行工作空间样本上则有 p95≈39 mm 的伸展差，说明一旦目标三角形不等腰，单腱模型无法同时满足三指。
因此论文若要并列"解析/数值 IK vs RL"，只能比较"可解率/模型假设适用性"，不能比较毫米误差。

---

## 与论文说法的对照汇总

| 论文说法 | 核查结果 |
|---|---|
| r = −Σ‖p_i − p_i^goal‖² | 不符：非平方、mm、含 ±100/+1000/−100 稀疏项 |
| "after 2885 epochs the average error is reduced by 81 %" | RUTH 均值 84.25→15.54 mm = 81.6 %，成立（仅 RUTH；Robotiq 60.7 %，Barrett 85.6 %） |
| "within 5 mm per finger across all three grippers" | 不成立：末 epoch 每指平均 15.5 / 18.9 / 10.9 mm；"5 mm"来自旧脚本把 20 条数据除以 100 的错误（5.06）并降序排序后落到末端，或来自成功阈值 Σ<15 mm 的换算 |
| α=0.8, lr=0.003, τ=0.005, γ=0.99 | lr/τ/γ 一致；α 自动调节（初值 1，终值 1–13），0.8 仅为 CLI 默认 |
| 每 epoch 10 个目标集、每目标最多 100 episode | 每 epoch 10 episode、每 episode ≤100 步；目标池仅 21 组 |
| 上一步动作拼入策略输入 | 不符（输入维 13/15/18 无动作） |
| 固定末端姿态训练 | 仅 Barrett；RUTH/Robotiq 每 episode 腕部 ±30° 随机扰动 |
| 两步法（先臂后手） | 结构一致；训练偏移 0.1 m（法向 / z），demo 用 0.3 m；demo 未乘 0.5 的动作缩放 |
| 动作维 4/6/9，RUTH 观测 13 | 一致 |

## 未找到 / 无法确定 清单
- 训练时实际运行的 `move_env.py`（三夹爪）；RUTH 训练时的目标采样（`contact_list.npy` 晚于训练生成）。查找位置：`rl_inverse_kinematics/*/gym/envs/kelin/`、`/home/kelin/RAL-IROS2022/train_*/gym/envs/kelin/`、`RAL-IROS2022/workspace_generation`、`workspace_xian/Hindsight-Experience-Replay`。
- ruth epoch_200（n=20）与 2022-05-03 批次所用的 seed/checkpoint；robotiq epoch_2500/2885 所用 checkpoint；barrett 05-06 下划线批次的 seed（推断为 seed--1）。
- RUTH 前向运动学（任何解析形式）：仅存在于 URDF+PyBullet。
