# 第三章事实核验 — A 组 (RUTH 工作空间采样 / 576 子采样 / 自编码器训练集 / 各夹爪特征)

仓库: /home/kelin/EfficientGrasp (分支 thesis-ch3-verification); 原始副本 /home/kelin/RAL-IROS2022 (只读, 非 git 仓库: `git -C /home/kelin/RAL-IROS2022 rev-parse` -> "fatal: not a git repository"; 其中仅 workspace_xian/Hindsight-Experience-Replay/.git 是上游 HER 项目的仓库).
所有脚本在 evaluation/thesis_ch3/, 所有输出在 results/thesis_ch3/. 以下路径若无说明均相对 /home/kelin/EfficientGrasp.

---
## A1 · RUTH 工作空间 contact_points.npy 的采样网格

**数据**: gripper_representation/ruth_workspace_sampling/contact_points.npy, (8649, 9) float64, mtime 2022-02-07 20:14:24 (与 /home/kelin/RAL-IROS2022/workspace_generation/contact_points.npy 同 md5 5d357cb2…, 同 mtime).

**去重** (`python3 evaluation/thesis_ch3/a1_grid_structure.py`, 输出 results/thesis_ch3/A1_grid_structure.txt):

| 取整位数 | `np.unique(np.round(W,d),axis=0).shape[0]` |
|---|---|
| 6 | 8649 |
| **4** | **8649** |
| 3 | 8616 |
| 2 | 1323 |

即 `np.unique(np.round(W,4),axis=0).shape[0] = 8649`: 四位小数 (0.1 mm) 下 8649 行全部互异.

**采样脚本**: gripper_representation/ruth_workspace_sampling/workspace.py (= RAL-IROS2022/workspace_generation/workspace.py, md5 相同, mtime 2022-02-20 19:45 — 注意晚于数据文件 13 天, 因此只能说"现存版本"的循环与数据吻合).
- L518 `ro = gym.make("kelin-v0", version="GUI")`; 该 env 注册于 rl_inverse_kinematics/early_prototype/gym/envs/__init__.py L10-12 (`id='kelin-v0', entry_point='gym.envs.kelin:MoveUr5RuthEnv'`), gym/envs/kelin/__init__.py L5 `from gym.envs.kelin.move_env import MoveUr5RuthEnv`. 因此 **实际使用的是 move_env.py 中的类**, 而不是 workspace.py 文件内部也定义的同名 MoveUr5RuthEnv (L210-482, 带 ±1 动作裁剪和电机限幅, 未被 gym.make 使用). RAL-IROS2022/workspace_generation/gym/envs/kelin/move_env.py 与 early_prototype 版逐字节相同 (diff -q 无差异), mtime 2022-02-20 21:33 (同样晚于数据).
- 主循环 L533-550:
  ```
  L533 for j in range(round((np.pi*2)/0.2)):      # round(31.416) = 31
  L535   for k in range(round((np.pi*2)/0.2)):    # 31
  L537     for l in range(round((0.4+0.5)/0.1)):  # round(9.0) = 9
  L540       ruth_1 = 0.08*j ; L541 ruth_2 = 0.08*k ; L542 ruth_3 = 0.1*l
  L543       a = np.array([0,ruth_1,ruth_2,ruth_3])   # a[0] 为 UR5 wrist_3 增量 = 0
  L544       o, r, d, _ = ro.step(a) ; L545 list.append(o) ; L546 ro.reset()
  L550 np.save('contact_points.npy', ooo)
  ```
  **31 × 31 × 9 = 8649**. 8649 = 93² 只是数值巧合; 真实网格是三个电机 31×31×9, 行索引 = j*279 + k*9 + l.
- 数据验证 (A1_grid_structure.txt): 按 (31,31,9) reshape 后, 沿 j、k、l 三轴相邻样本的指尖位移均为平滑小量 (j: 0.0021–0.0033 m/步, k: 0.0023–0.0032 m/步, l: 0.0078–0.0100 m/步, 30 步内无饱和/重复); 按 (93,93) reshape 则第二轴每 3 步出现 0.070 m 的跳变 (其余 0.009 m), 第一轴每步 0.063–0.074 m — 证明 93×93 不是真实结构.

**step()/reset() 中的实际处理** (rl_inverse_kinematics/early_prototype/gym/envs/kelin/move_env.py):
- L336 `action = np.clip(action, -10, 10)`: 对 0–2.4 rad 的指令无影响; **move_env.py 的 step() 中没有任何电机限幅** (workspace.py 内嵌副本 L343-360 的 [-1,π/2],[-π/2,1],[-0.5,0.12] 限幅不生效).
- L342 `RUTH_motors = action[1:] + RUTH_motors`: 增量叠加在 reset 后读取的关节状态上; L316 `self._RUTH_init = [np.pi/2, -np.pi/2, -0.5]` 是 reset() 的电机目标 (L372-373), reset 循环因目标向量顺序不匹配固定跑 100 个仿真步 (L383 `count==100`).
- step 返回 L369 `return pos, r, d, {}`, pos = get_pos() = 三个指尖 (Phal_1C/2C/3C) 质心 xyz, 共 9 维 (L439-450) — 这就是 contact_points.npy 每行的 9 列.
- 名义指令网格 (相对 reset 姿态的增量): motor1 (Joint_Link_1): 31 个值, 0..2.40 rad, 步长 0.08; motor2 (Joint_Link_3): 31 个值, 0..2.40 rad, 步长 0.08; 手指腱电机 (motor3, 经 motor_control_ruth L130-145 映射到 Phal_*B/*C 关节): 9 个值, 0..0.80, 步长 0.1. **第三个电机不是固定的** — 它取 9 个值.
- 名义绝对目标 = [π/2+0.08j, −π/2+0.08k, −0.5+0.1l]. URDF 关节限位 (early_prototype/gym/envs/kelin/urdf/ur5_plus_RUTH.urdf L390 Joint_Link_1 [-1.5708, 3.1416], L656 Joint_Link_3 [-3.1416, 1.5708]) 下 motor1 目标在 j≥20 时超过上限 π, 但数据中 j=20..30 仍有 0.0024–0.0027 m/步 的连续位移, 没有饱和迹象. 因此 **各电机的实际关节角没有被记录在 contact_points.npy 中, 无法从数据反推**; 只能确认指令网格 31×31×9. 若需精确关节角, 需在装有 pybullet 的环境中重跑 workspace.py (系统 python3 无 pybullet; urdf 路径 L239 为绝对路径 "/gym/envs/kelin/urdf/..." 需修补).
- 旁证: 若 gym.make 解析到的是 workspace.py 内嵌的带限幅类, 指令会被裁到 [-1,1] 且 motor3 裁到 0.12, 只能产生 14×14×3 = 588 种不同配置, 与 8649 行全部互异矛盾.

**另一个 RUTH 扫描脚本**: gripper_representation/feature_extraction/workspace.py (创建 2022-02-07 17:45, mtime 18:44) 直接用 pybullet GUI 扫 32×32×6 (L228-237: ruth_1 = −1+0.08i, ruth_2 = −π/2+0.08j, ruth_3 = −0.41+0.1k), **不保存任何文件**, 与 8649 不符, 不是数据来源.

**与 RL 目标集的区分**: rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/contact_points.npy 及 RAL-IROS2022/train_*/contact_points.npy 为 (485376, 9), md5 f3f4b364, 2022-01-17, 485376 = 79×32×32×6 (含 UR5 wrist 扫描), 指尖 z≈0.35 m (另一臂姿), 被 move_env.py L466 用作 RL 的目标采样池, **与工作空间编码无关**, 且 8649 文件的行不在其中.

---
## A2 · 576 行子采样

**搜索**: `grep -rn "576" --include=*.py --include=*.ipynb /home/kelin/EfficientGrasp /home/kelin/RAL-IROS2022 | grep -v encoder` — 去掉 gym 自带文件中的常数 (0.4576751…, 0.7119514…) 后, 命中的只有 6 份同一脚本 `workspace_visual.py` (md5 609acebd…, 1219 B, mtime 2022-02-07 20:51): rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/workspace_visual.py 与 RAL-IROS2022/train_{ruth,robotiq,barrett}/workspace_visual.py. /home/kelin 其余位置 (排除 anaconda3/venv/snap/.cache/.local 等) 仅命中 ~/Downloads/recast-candidate-kit-controls/.venv 里 numpy/scipy 的测试文件, 无关. 未发现 .ipynb.

**脚本内容** (rl_inverse_kinematics/ruth/workspace_visual.py):
```
L13 a=np.load('/home/kelin/workspace_kelin/RL_IK/contact_points.npy')   # 该路径现已不存在
L15 b=np.zeros((576,9))
L17 vertices_index = np.linspace(0,len(a)-1,576,dtype=int)
L18-19 for i in range(576): b[i,:]=a[vertices_index[i],:]
L20 np.save('ws1',b)
```
= 等间隔抽样: 名义步长 8648/575 = **15.04** (任务描述中的 15.03 应更正为 15.04), 整数索引步长 15 (552 次) 与 16 (23 次), 索引 0..8648.

**复现**: `python3 evaluation/thesis_ch3/subsample_576.py` (头注释: 复现 workspace_visual.py 的抽样步骤), 输出 results/thesis_ch3/A2_ws1_reproduced.npy 与 A2_subsample_576.txt:
- `np.array_equal(reproduced, ruthArrays/ws1.npy) = True`, max abs diff 0.0, **md5 完全相同 84e3ed31590268ccadc0ae1a0ffea857**.
- ws1 的 576 行在 contact_points.npy 中 1e-6 容差内全部找到: 576/576 = 1.0000.

**同一文件的拷贝**: gripper_representation/feature_extraction/ruthArrays/ws1.npy (2022-02-07 20:52) 与 rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/ws1.npy、RAL-IROS2022/train_{ruth,robotiq,barrett}/ws1.npy (2022-02-08 12:51) 全部 md5 84e3ed31… — robotiq_3f/、barrett/ 下的 ws1.npy 都是 RUTH 的子采样 (见 A4).

---
## A3 · 自编码器训练集

### (a) 工作空间数组目录统计
命令: `python3 evaluation/thesis_ch3/count_workspace_arrays.py` (mmap 只读 shape, 18.6 s), 输出 results/thesis_ch3/A3_workspace_arrays_stats.{md,csv}.
目录定位: `find /home/kelin -maxdepth 7 -type d \( -name WorkspaceArrays -o -name coupRotWorkspaceArrays -o -name fourbarWorkspaceArrays -o -name singRotWorkspaceArrays \) -not -path "*/Trash/*"` ->
/home/kelin/RAL-IROS2022/WorkspaceArrays, /home/kelin/EfficientGrasp/gripper_representation/data/WorkspaceArrays, /home/kelin/RAL-IROS2022/feature_extraction/coupRotWorkspaceArrays, /home/kelin/EfficientGrasp/gripper_representation/feature_extraction/coupRotWorkspaceArrays. **fourbarWorkspaceArrays、singRotWorkspaceArrays: 未找到** (整个 /home/kelin 深度 7 内不存在).

| 目录 | 文件数 | 编号 | 空 (0,9) | 非空 | 行数分布 {rows: n} | mtime |
|---|---|---|---|---|---|---|
| gripper_representation/data/WorkspaceArrays (= RAL-IROS2022/WorkspaceArrays, 同 44751 个文件) | 44751 | ws1..ws44751 无缺号 | **14112** | **30639** | 0: 14112, 432: 1008, 648: 2551, 864: 11760, 1458: 1008, 2187: 2552, 2916: 11760 | 2022-02-07 16:56 ~ 18:24 |
| gripper_representation/feature_extraction/coupRotWorkspaceArrays | 3005 | ws39..ws96759 (缺号 93716, 即从更大集合中挑出) | 0 | 3005 | **576: 3005** | 2021-08-21 19:02 ~ 2022-01-13 13:35 |
| fourbarWorkspaceArrays | 未找到 | | | | | |
| singRotWorkspaceArrays | 未找到 | | | | | |
| ruthArrays | 1 (ws1.npy) | | 0 | 1 | 576: 1 | 2022-02-07 20:52 |

所有数组列数均为 9. **WorkspaceArrays 中没有任何一个数组是 576 行** (行数 ∈ {0,432,648,864,1458,2187,2916}); 只有 coupRot 与 ruthArrays 是 576 行.

workspace_generator.py (gripper_representation/feature_extraction/workspace_generator.py, mtime 2022-02-07 16:56 = WorkspaceArrays 第一个文件的时间) 只实现五杆机构 (L113 `five_bar = True`), 写出路径固定为 `/home/kelin/workspace_kelin/previous_work/WorkspaceArrays/ws{count}.npy` (L136, L143, L151); four_bar / three_RRR / sing_rot / coup_rot 分支全部是注释掉的 `pass` 存根 (L153-160). 因此 **该脚本不会写 fourbar/singRot/coupRot 目录**; coupRotWorkspaceArrays 的生成脚本在两个仓库中 未找到 (`grep -rn "coupRot\|singRot\|fourbar" --include=*.py` 仅命中 encoder.py 的注释块).
行数解释: five_bar_workspace() 遍历 ≤4 个五杆构型 (L57-59, 无解时 `continue` L63), 每构型 fin_ori_meth=0 时 27 个指向组合 × finger_bend(mult=1) (全驱动 2³=8 行 / 欠驱动 27 行), fin_ori_meth=1 时 finger_bend(mult=27) (全驱动 6³=216 行 / 欠驱动 729 行) -> 每数组 {0,1,2,3,4}×{216,729} = {0,216,432,648,864,…,729,1458,2187,2916}, 与统计一致 (0 行 = 4 个构型全无解). 现脚本参数组合数为 3·3·3·3·2·2·(6+18+54) = 25272 ≠ 44751, 说明实际生成时参数范围与现存脚本不同 (无法确定).

### (b) encoder.py 如何加载训练数据 (gripper_representation/feature_extraction/encoder.py, mtime 2022-02-07 21:32)
- L26-45: 四目录循环 (WorkspaceArrays / coupRotWorkspaceArrays / fourbarWorkspaceArrays / singRotWorkspaceArrays) **整段被注释**.
- L46 `ROOT_DIR = os.path.join(BASE_DIR,'ruthArrays/')`; L47 只取以 "ws" 开头的文件 -> 现存版本的训练集 = **ruthArrays/ws1.npy 一个数组**.
- L52-54 `obj_pcs = np.load(env_dir); if len(obj_pcs!=0): in_gripper_list.append(obj_pcs)`: `len(obj_pcs!=0)` 等于行数, 行数为 0 的空数组被跳过 (写法有误但效果是过滤 (0,9)); L55-56 `if len(in_gripper_list)%2000 == 0: break` -> 每目录最多 2000 个数组.
- **没有任何子采样代码** (无 linspace / random / 截断): L204 `data1[j,:,:] = data` 直接赋给 (batch,576,9) 的缓冲区, 行数 ≠ 576 的数组会触发广播错误. 因此 WorkspaceArrays (无 576 行数组) **不可能被现存 encoder.py 直接训练**; 可直接喂入的只有 576 行的 coupRot 与 ruthArrays.
- 超参: L18 learning_rate = 0.001, L19 training_epochs = 10000, L20 batch_size = 1, L21 display_step = 1 (每个 epoch 存一次 ckpt, L231-232), L176 AdamOptimizer. L182 `data1 = np.zeros((batch_size,576,9))`; L210-212 每步 feed 一个 (1,576,9).
- 损失 = 耦合 Chamfer: L158-163 把 (576,9) 按列 split 成 3 个 (576,3) 并 stack 成 (3,576,3); distance_matrix L110-121 对三个手指分别算 576×576 欧氏距离并**相加** (L119), 即两条"三指尖构型"之间的距离 = Σ_f ||p_f − q_f||; av_dist L130-133 取行最小再平均; av_dist_sum L143-145 双向求和. 正是三指耦合的 Chamfer 距离.
- 网络: 编码器 6 层 conv1d(k=1) 64-64-128-128-256-256 + BN, max_pool_1d(576) -> 256 维 (L67-80); 解码器 FC 512-1024-5184 reshape (576,9) (L84-89). 参数量 ≈6.1 M, 与 9999model.ckpt.data 24.43 MB (float32) 吻合; Saver 在优化器之前创建 (L169 vs L176), 所以 ckpt 不含 Adam 槽变量 (`strings` 9999model.ckpt.index 无 Adam/beta1 等名字).
- 保存路径 L93 `/home/kelin/workspace_kelin/previous_work/feature_extraction/saved_models/workspace/{epoch}model.ckpt`; L174 `tf.summary.FileWriter("logs/", sess.graph)` 在优化器之前写图, 且 L229-230 add_summary 被注释 -> **event 文件里只有图, 没有标量**.

### (c) 训练运行推断 (TF event 文件 + 时间戳)
命令: `python3 evaluation/thesis_ch3/a3_parse_tf_events.py` (自写 TFRecord/protobuf 解码, 无 TF 依赖), 输出 results/thesis_ch3/A3_tf_events.csv.
- 43 个 event 文件, 每个恰好 3 条记录 (file_version, graph_def, meta_graph_def), **0 条 summary / 0 个 step** -> 无法从日志得到步数或每 epoch 步数.
- 图有两类: 928 节点 (含 Chamfer 节点 split/Tile/Min/Mean, 无 Adam — 与 encoder.py 在优化器前写图一致) = encoder.py 启动, 共 32 次; 699 节点 (仅 gripper_encoder/gripper_decoder/init) = encoder_test.py 启动 (其 L94 同样写 logs/), 共 11 次.
- 时间线 (本地时间):
  - 2022-01-14 18:17–18:30 encoder.py ×6; 2022-01-19 16:34–16:55 encoder.py ×5 (coupRot 数组最后修改 2022-01-13; 这些早期运行的数据/ckpt 未保存).
  - 2022-02-07 16:56–18:24 生成 WorkspaceArrays; 20:14 contact_points.npy; 20:26–20:53 encoder.py ×15; 20:51 workspace_visual.py, 20:52 ws1.npy; 21:04–21:21 encoder_test.py ×6; 21:22–21:32:52 encoder.py ×6, encoder.py 最终 mtime 21:32; **21:32:52 最后一次 encoder.py 启动 -> 22:15 写出 9999model.ckpt** (checkpoint 文件列出 9000..9999 共 1000 个, 对应 max_to_keep=1000, 每 epoch 存一次); 22:16:06、23:03、23:06 encoder_test.py; 2022-02-08 14:31:07/14:31:53 encoder_test.py ×2 -> ruthArrays/{mean,max,min}.npy mtime 2022-02-08 14:31:55, encoder_test.py mtime 14:31.
- **9999 = epoch 编号** (0 起算的第 10000 个 epoch, L198 `for epoch in range(training_epochs)`, L232 `save_model_stage(epoch)`), 不是 global step.
- 训练集规模估计: 21:32:52 -> 22:15 ≈ 42.5 min / 10000 epoch ≈ 0.26 s/epoch. batch_size=1 时每 epoch 步数 = 数组数; 若用 2000 个 coupRot 数组则需 2×10⁷ 步, 42 分钟内不可能 (每步 576×576×3 的距离矩阵 + 反传). 0.26 s/epoch 与 **每 epoch 1 步 (1 个数组)** 的 GPU 开销一致. 结合 (b) 中现存代码只读 ruthArrays, 推断 9999model.ckpt 是 **只用 ruthArrays/ws1.npy 这 1 个 576×9 数组训练 10000 epoch (= 10000 步)** 的结果. 这是基于时间与代码的推断, 日志中无直接步数记录.
- RAL-IROS2022 不是 git 仓库, encoder.py 无版本历史; 其 mtime 21:32 恰在最后一次训练启动 (21:32:52) 之前 55 s, 现存文件即训练所用版本的概率很高, 但 20:26–21:28 的 20 余次启动所用的中间版本已不可恢复.

### (d) 结论
- 可确认的最终自编码器 (9999model.ckpt, 2022-02-07 22:15, 被 encoder_test.py 和下游 ruthArrays/mean.npy 使用) 的训练目录 = gripper_representation/feature_extraction/ruthArrays/, 有效数组 = 1 (ws1.npy, 576×9), 总计 1.
- WorkspaceArrays: 44751 个文件, 30639 个非空、14112 个空; 行数均非 576, 现存 encoder.py 无法直接使用; 如果曾经用过, 必然经过一个未保存的子采样步骤 (未找到). coupRotWorkspaceArrays: 3005 个 576×9 的有效数组 (encoder.py 的 2000 上限意味着最多用 2000 个), 其生成脚本 未找到, 其被用于 2022-01-14/19 训练的可能性只能由日期推测. fourbar/singRot: 目录与脚本均 未找到.
- 无法确定的: 2022-01 的早期训练用了哪些数组; coupRot 数组的生成方法; 2021-08/09 生成 Data_DB 特征时的训练集与 checkpoint (见 A4).

---
## A4 · 各夹爪工作空间特征

详细表: results/thesis_ch3/A4_gripper_workspace_features.md.

文件检查 (shape/dtype/mtime/md5):
| 文件 | shape | dtype | mtime | md5 |
|---|---|---|---|---|
| Data_DB/ruth/workspace.npy | (1,256) | float32 | 2021-09-10 23:33:02 | b840a651847bd9f0a4d23f63eb0b006b |
| Data_DB/robotiq_3f/workspace.npy | (1,256) | float32 | 2021-08-30 21:07:12 | 78d88f0e9ea765ae292545d46244bebc |
| Data_DB/kinova_kg3/workspace.npy | (1,256) | float32 | 2021-08-30 21:08:38 | 7729470dbfc415bc68c8f4826c02a394 |
| Data_DB/bh_282/workspace.npy (== data/grippers/bh_282/workspace.npy) | (1,256) | float32 | 2021-08-30 21:09:22 | 222f063d25597721b09cd13d1206dabb |
| data/grippers/robotiq_3f/ | 无 workspace.npy (仅 mean/max/min 2021-01-18 + 33 个 2048×3 点云) | | | |
| ruthArrays/mean.npy = max.npy = min.npy | (1,256) | float32 | 2022-02-08 14:31:55 | a12baf3ddb7bd8e15fcd33e92d0ce0c3 |
(Data_DB 前缀 = contact_point_selection/UniGrasp/data/gripper_features/Data_DB; RAL-IROS2022/UniGrasp 下同名文件 mtime 相同.)

- ruthArrays 的 mean/max/min 三者逐字节相同: encoder_test.py L106-137 对目录内所有 "ws*" 数组取特征后 mean/max/min, 目录只有 1 个数组, 故三者相等; 它们是 ws1.npy (576×9) 经 9999model.ckpt 编码的 256 维特征 (L181 restore(9999), L184-185 gripper_dir=…/ruthArrays, gripper_name="ws").
- **`np.allclose(ruthArrays/mean.npy, Data_DB/ruth/workspace.npy) = False`** (max abs diff 0.274); 与 robotiq_3f/kinova_kg3/bh_282 的 workspace.npy 也都不等 (0.312 / 0.201 / 0.513). 57 个 (1,256) 特征文件两两比对, 仅有的相等对是 Data_DB/bh_282/{mean,max,min} == Data_DB/{mean,max,min}、Data_DB/bh_282/workspace == grippers/bh_282/workspace、Data_DB/robotiq_3f/{mean,max,min} == grippers/robotiq_3f/{mean,max,min}.
- Data_DB 四个 workspace.npy (2021-08-30 / 09-10) **早于**本仓库所有 RUTH 工作空间数组 (2022-02-07)、coupRot 数组的大部分、以及 9999model.ckpt; 生成它们的输入数组与 checkpoint 均 未找到 (feature_extraction/saved_models/ 目录 2022-01-14 才创建; encoder.py/encoder_test.py 文件头注明创建于 2021-08-08 / 08-18, 与这些日期相符, 但当时版本不可恢复).
- (N,9) 数组全量扫描 (两个仓库 435 个 .npy, 排除 WorkspaceArrays/物体点云): 仅有 RUTH 的 (8649,9)、(576,9) 六份同 md5 拷贝、(485376,9) RL 目标集六份拷贝、若干 (10,9) 接触点结果. **不存在 Robotiq / Barrett / Kinova 的工作空间数组**. rl_inverse_kinematics/{robotiq_3f,barrett}/ws1.npy 与 RUTH ws1.npy 逐字节相同 (md5 84e3ed31…) — 它们是 RUTH 数据的误拷贝, 不能作为 Robotiq/Barrett 的工作空间.
- 非 RUTH 采样脚本: rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/gym/envs/kelin/workspace_generation.py 三份相同 (md5 7a928a3f, 2022-02-07 16:40), 内容是 RUTH 环境 (L239 加载 ur5_plus_RUTH.urdf), __main__ (L518-533) 只跑 500 个随机动作, 不保存任何数组; robotiq_3f/barrett 的 move_env.py 分别加载 ur5_plus_robotiq_3f.urdf / ur5_plus_barrett.urdf (L155/L159) 但没有工作空间扫描代码; UniGrasp/gripper_representation/gripper_feature_extraction.py 是原版 UniGrasp 的 2048×3 点云编码器 (L42, restore(2248)), 产出 mean/max/min 而非 workspace.npy. 结论: Robotiq/Barrett/Kinova 的工作空间采样脚本 **未找到**.
- 下游使用: unigrasp.py L548-566 推理时按 gripper_id 12/11/13 -> bh_282/robotiq_3f/**ruth** 读取 Data_DB/<name>/workspace.npy (256 维); kinova_kg3/workspace.npy 未被引用. train_pssn.py L645-659 则读取 mean/max/min 拼成 768 维 (gripper_id 13 -> kinova_kg3), 与推理代码输入维度不一致 — 需在 PSSN 核验项中核对所用 checkpoint 的输入维度.

汇总表 (gripper | 采样脚本 | 网格 | 行数 | 工作空间数组 | 特征文件 | 链接验证):
| RUTH (2022) | ruth_workspace_sampling/workspace.py L533-550 | 31×31×9 | 8649 -> 576 | ruth_workspace_sampling/contact_points.npy; feature_extraction/ruthArrays/ws1.npy | feature_extraction/ruthArrays/mean.npy | 是 (子采样逐字节复现; 特征=encoder_test.py 代码链路+mtime, 数值未重跑) |
| RUTH (Data_DB) | 未找到 | 未找到 | 未找到 | 未找到 | Data_DB/ruth/workspace.npy | 否 (≠ ruthArrays/mean.npy) |
| Robotiq 3F | 未找到 | 未找到 | 未找到 | 未找到 (robotiq_3f/ws1.npy 是 RUTH 的) | Data_DB/robotiq_3f/workspace.npy | 否 |
| Barrett BH-282 | 未找到 | 未找到 | 未找到 | 未找到 (barrett/ws1.npy 是 RUTH 的) | Data_DB/bh_282/workspace.npy | 否 |
| Kinova KG-3 | 未找到 | 未找到 | 未找到 | 未找到 | Data_DB/kinova_kg3/workspace.npy (推理未用) | 否 |

---
## 产出文件清单
- evaluation/thesis_ch3/a1_grid_structure.py -> results/thesis_ch3/A1_grid_structure.txt
- evaluation/thesis_ch3/subsample_576.py -> results/thesis_ch3/A2_ws1_reproduced.npy, A2_subsample_576.txt
- evaluation/thesis_ch3/count_workspace_arrays.py -> results/thesis_ch3/A3_workspace_arrays_stats.{md,csv}
- evaluation/thesis_ch3/a3_parse_tf_events.py -> results/thesis_ch3/A3_tf_events.csv
- results/thesis_ch3/A4_gripper_workspace_features.md (表)
- 本文件 results/thesis_ch3/details_A.md
