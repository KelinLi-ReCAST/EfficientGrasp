# 第三章 3.3 节 · 本机核查报告

对应清单：`~/Desktop/local-verification-tasks-3.3.md`。核查对象：`~/EfficientGrasp`（分支 `thesis-ch3-verification`）与原始副本 `~/RAL-IROS2022`（只读，非 git 仓库，无版本历史）。
所有新脚本在 `evaluation/thesis_ch3/`，所有产物在本目录。未改动任何既有代码文件。
逐条的完整依据（含行号、命令、全部中间数字）见本目录 `details_A.md`（A1–A4）、`details_B.md`（B1–B5）、`details_C.md`（C1–C4）；本文件只给结论。

路径缩写：`PSS/` = `contact_point_selection/UniGrasp/point_set_selection/`；`Data_DB/` = `contact_point_selection/UniGrasp/data/gripper_features/Data_DB/`；`E:` = 对应夹爪的 `gym/envs/kelin/move_env.py`。

---

## 一页总览：需要改正文的事实

| 正文现有说法 | 核查结果 | 条目 |
|---|---|---|
| RUTH 工作空间 8649 个样本 | 成立。网格为 **31 × 31 × 9**（两个手掌电机各 31 档、手指腱电机 9 档），不是 93×93 | A1 |
| 576 行工作空间 | 由 8649 行**等间隔抽样**得到（步长 ≈15.04），逐字节复现 | A2 |
| 自编码器训练集 `[NUMBER]` | 现存检查点 `9999model.ckpt` 可确认的训练集 = **1 个数组**（`ruthArrays/ws1.npy`）。44751 个 WorkspaceArrays 无一是 576 行，现存代码读不了 | A3 |
| 各夹爪工作空间特征 | Robotiq / Barrett / Kinova 的工作空间**数组与采样脚本均未找到**；它们目录下的 `ws1.npy` 是 RUTH 的误拷贝 | A4 |
| k₁=1024, k₂=512, k₃=512 | 代码、6 个 TF 日志的图常量、3 个检查点一致：**三阶段全是 1024**；k₂、k₃ 是对点对/三元组整体取 top-k | B1 |
| 最终按 PSSN 概率排序 | 成立：**仅第三阶段掩码后的 softmax 概率**（`PSS/unigrasp.py:404-406`），无阈值、无 GQS 过滤 | B1 |
| 600 GB / 150 GB | 本机无数据；官方压缩包 236.6 GB + 50.6 GB。**测试样本 N = 283**（所有 1332 个保存值与 42 个表值均为 k/283） | B2 |
| Adam lr 2e-3 | 2e-3 只是法向预测头；PSSN 三阶段为 5e-5 / 1e-4(或 5e-5) / 1e-3 | B3 |
| 超参敏感性 | **未做** | B3 |
| UniGrasp 输入 32 / 8 / 254 或 256 | 磁盘与 UniGrasp 论文一致：2^D 边界 + 1 中心 = **Robotiq 33、Kinova 9、BarrettHand 257**；"254" 无来源 | B4 |
| 9.85 % | **无法复现**。18 格平均提升 10.62 pp（相对 13.9 %）；最接近的是 Barrett Top-1 均值 9.83 | B5 |
| RUTH 观测 13 维含上一步动作 | 13 = wrist_3 角 1 + 手部电机 3 + 目标坐标 9；**不含上一步动作**（检查点首层权重 (256,13) 证实） | C1 |
| r = −Σ‖p−p*‖² | 代码为 **−Σ‖p−p*‖ (mm，不平方)** + 每指 ±100（15 mm 阈值）+ 1000 终止奖励 − 100/限位 | C2 |
| α = 0.8 固定 | 自动熵调节，初值 1，训练中升至 1–13；0.8 仅 CLI 默认 | C2 |
| 每 epoch 10 个目标、每目标 ≤100 episode | 每 epoch 10 个 episode、每 episode ≤100 **步**；训练目标池只有 **21 组**接触点 | C2 |
| 固定末端姿态 | 仅 Barrett；RUTH/Robotiq 每 episode 腕部 ±30° 随机扰动 | C1 |
| 误差降低 81 % | RUTH 84.3→15.5 mm = **81.6 %**，成立（仅 RUTH；Robotiq 60.7 %，Barrett 85.6 %） | C3 |
| 三夹爪均达 5 mm/指 | **不成立**。末 epoch 每指平均 RUTH 15.5、Robotiq 18.9、Barrett 10.9 mm，无任何 episode <5 mm。旧图的 5 mm 是 `RL_eval.py` 把 20 条数据除以 100 再降序排列造成的 | C3 |
| 数值 IK vs RL 精度 | **无法同口径比较**：解析求解器无前向运动学。只能报可解率：RL 目标分布上 99.9 % 收敛 | C4 |

---

## A · 工作空间与自编码器

### A1 · fig_workspace_ruth 与采样网格
**结论**
- 图：`fig_workspace_ruth.pdf / .png`（左俯视、右 3D，三指三色，mm，质心为原点）。脚本输出 `8649 rows, 8649 distinct rows`。
- `np.unique(np.round(W,4),axis=0).shape[0] = 8649`（取 6、4 位小数均 8649；3 位 8616；2 位 1323）。
- 网格 = **31 × 31 × 9**：`workspace.py` L533-546 三重循环 `j,k ∈ range(round(2π/0.2))=31`，`l ∈ range(round(0.9/0.1))=9`，指令 `ruth_1=0.08j, ruth_2=0.08k, ruth_3=0.1l`，行索引 `j·279+k·9+l`。手指腱电机取 9 个值，不是固定的。
- 数据自证：按 (31,31,9) 排列，三轴相邻位移平滑（j 2.0 mm/步、k 0.4 mm/步、l 9.2 mm/步，最大 3.3/1.3/9.5 mm）；按 93×93 排列第二轴每 3 列出现 70 mm 跳变。93² 只是巧合。
- `gym.make('kelin-v0')` 解析到 `early_prototype/gym/envs/kelin/move_env.py`，其 `step()` 只有 `np.clip(action,-10,10)`，**无电机限幅**；`workspace.py` 文件内另有一份带限幅的同名类，未被使用（若使用只能得 588 种配置）。

**依据** `python3 evaluation/thesis_ch3/make_fig3_workspace.py gripper_representation/ruth_workspace_sampling/contact_points.npy results/thesis_ch3/fig_workspace_ruth`；`python3 evaluation/thesis_ch3/a1_grid_structure.py` → `A1_grid_structure.txt`。

**不确定** 各电机实际关节角未记录在数据里，无法反推（URDF 限位下 motor1 在 j≥20 应饱和，但数据无饱和迹象）。`workspace.py` 与 `move_env.py` 的 mtime（2022-02-20）晚于数据（2022-02-07），只能说"现存版本的循环与数据吻合"。

### A2 · 576 行的来源
**结论** 等间隔抽样：`np.linspace(0, 8648, 576, dtype=int)`，名义步长 8648/575 = **15.04**（整数步 15×552 次 + 16×23 次）。来源脚本 `rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/workspace_visual.py` L17（6 份同 md5 拷贝）。复现脚本 `evaluation/thesis_ch3/subsample_576.py` 的输出与 `ruthArrays/ws1.npy` **md5 完全相同**（84e3ed31…）；576/576 行在 8649 行中 1e-6 内匹配。

**依据** `grep -rn "576" --include=*.py --include=*.ipynb ~/EfficientGrasp ~/RAL-IROS2022 | grep -v encoder`；`python3 evaluation/thesis_ch3/subsample_576.py` → `A2_subsample_576.txt`, `A2_ws1_reproduced.npy`。全家目录无其他相关脚本或 notebook。

### A3 · 自编码器实际训练集
**结论**
- 目录统计（`A3_workspace_arrays_stats.md`）：

| 目录 | 文件 | 空 (0,9) | 非空 | 行数分布 |
|---|---:|---:|---:|---|
| `WorkspaceArrays` | 44751 | 14112 | 30639 | 432:1008, 648:2551, 864:11760, 1458:1008, 2187:2552, 2916:11760 |
| `coupRotWorkspaceArrays` | 3005 | 0 | 3005 | 576:3005 |
| `fourbarWorkspaceArrays` | **未找到** | | | |
| `singRotWorkspaceArrays` | **未找到** | | | |
| `ruthArrays` | 1 | 0 | 1 | 576:1 |

- `encoder.py`：读四目录的循环 L26-45 **整段被注释**，L46-47 只读 `ruthArrays/ws*`；**没有任何子采样代码**，L204 直接赋给 (1,576,9) 缓冲区，非 576 行的数组会报错 → WorkspaceArrays 不可能被现存代码训练。超参：lr 0.001、10000 epoch、batch 1、Adam、每 epoch 存 ckpt。损失为耦合 Chamfer（L158-163 按列 split 三指，L113-119 三指距离相加，L143-145 双向）。
- TF 日志 43 个 event 文件只含 graph，**0 个标量 / step**（`add_summary` 被注释）。时间线：2022-02-07 21:32:52 最后一次 `encoder.py` 启动 → 22:15 写出 `9999model.ckpt`，42.5 min / 10000 epoch ≈ 0.26 s/epoch，与每 epoch 1 步（1 个数组）一致；2000 个数组需 2×10⁷ 步，不可能。9999 是 epoch 编号。
- **结论：现存 `9999model.ckpt` 的训练集 = `ruthArrays/ws1.npy` 1 个 576×9 数组，训练 10000 epoch。** 正文 `[NUMBER]` 若指这个检查点，只能填 1；若要写 44751 或 3005，必须说明那是另一次未保存的训练。
- `workspace_generator.py` 只实现五杆机构，只写 `WorkspaceArrays`；four_bar/sing_rot/coup_rot 分支是注释掉的 `pass`。`coupRotWorkspaceArrays` 的生成脚本未找到。

**依据** `python3 evaluation/thesis_ch3/count_workspace_arrays.py`；`python3 evaluation/thesis_ch3/a3_parse_tf_events.py` → `A3_tf_events.csv`；`ls --time-style=full-iso` 时间线见 `details_A.md` A3(c)。

**不确定** 2022-01-14/19 的早期训练（11 次启动）用了哪些数组、ckpt 在哪；2021-08/09 生成 `Data_DB` 特征时的训练集与检查点。均无法确定。

### A4 · 各夹爪 workspace 数组与特征
**结论**

| 夹爪 | 采样脚本 | 网格 | 行数 | 工作空间数组 | 特征文件 (1,256) | 链接验证 |
|---|---|---|---|---|---|---|
| RUTH（2022-02） | `ruth_workspace_sampling/workspace.py` | 31×31×9 | 8649 → 576 | `ruth_workspace_sampling/contact_points.npy`, `ruthArrays/ws1.npy` | `ruthArrays/mean.npy`(=max=min) | 是（抽样逐字节复现；特征按 `encoder_test.py` 链路与 mtime） |
| RUTH（Data_DB, 2021-09-10） | 未找到 | 未找到 | 未找到 | 未找到 | `Data_DB/ruth/workspace.npy` | **否**：与 `ruthArrays/mean.npy` 不等（max diff 0.274） |
| Robotiq-3F | 未找到 | 未找到 | 未找到 | 未找到（`robotiq_3f/ws1.npy` 是 RUTH 的） | `Data_DB/robotiq_3f/workspace.npy`（2021-08-30） | 否 |
| BarrettHand | 未找到 | 未找到 | 未找到 | 未找到（`barrett/ws1.npy` 是 RUTH 的） | `Data_DB/bh_282/workspace.npy` | 否 |
| Kinova KG-3 | 未找到 | 未找到 | 未找到 | 未找到 | `Data_DB/kinova_kg3/workspace.npy`（推理未引用） | 否 |

- 两仓库 435 个 .npy 全量扫描：**不存在任何 Robotiq/Barrett/Kinova 的 (N,9) 数组**。`rl_inverse_kinematics/{robotiq_3f,barrett}/ws1.npy` 与 RUTH 的 `ws1.npy` md5 相同，是误拷贝。
- 三份 `gym/envs/kelin/workspace_generation.py` 相同且是 RUTH 环境，`__main__` 只跑随机动作不保存；robotiq/barrett 的 `move_env.py` 无扫描代码。
- `Data_DB` 四个特征（2021-08/09）早于本仓库所有 RUTH 数组和检查点，其输入与编码器均未找到。
- 旁注：`unigrasp.py` L548-566 推理用 256 维 `workspace.npy`（id 13 → ruth），`train_pssn.py` L645-659 用 768 维 mean/max/min（id 13 → kinova_kg3），见 B2。

**依据** `A4_gripper_workspace_features.md`（shape/dtype/mtime/md5 全表及两两比对）。

---

## B · PSSN 与接触点选择

### B1 · 第二、三阶段选点准则
**结论**（伪代码，行号为 `PSS/unigrasp.py`；网络定义 stage 2 `tf_models/point_quality_v6.py`，stage 3 `point_quality_v12.py`）
```
Stage 1  物体点云 [2048,3] → PointNet++(s4)；夹爪特征 g 注入 3 处，最终 conv 到 64 维 tile 到各点，
         与 64 维逐点特征拼成 128 维 → conv128-conv64-conv2 → softmax p1          (L85-105)
         S1 = top_k(p1, 1024)；候选池 P2 = top_k(p1, 1024)（同一集合）             (L110-112)
Stage 2  对每个 i∈S1, j∈P2：concat[feat2(j), feat2(i)]（独立 PointNet++，只拼第一点的学习特征，
         无坐标/相对向量）→ conv128-IN-conv64-IN-conv2 → softmax p2(i,j)          (v6:7-33; L128-175)
         p2 *= [n_i·n_j < −0.5]（预测法向反向掩码）                                 (L162-169)
         S2 = top_k(p2 over 1024×1024 个点对, 1024)                                  (L192-193)
Stage 3  对 S2 前 1024 对 (i,j) 与每个候选 m：concat[F3(i),F3(j)](128) + feat3(m)(64) = 192
         → conv192-IN-conv64-IN-conv2 → softmax p3(i,j,m)                          (L236-272, 384-386; v12:40-64)
         p3 *= final_mask（法向符号模式 thre 0.6、最大内角 <120°、最短边 >10 mm、
         法向-边符号、摩擦锥 μ=0.42）                                                (L293-382)
         S3 = top_k(p3, 1024)；最终名次 = p3 本身，Top-1 = S3[0]                     (L390, 404-406, 640-642)
```
- **无概率阈值、无 GQS 过滤**；唯一距离过滤是最短边 >1 cm。
- 标签：stage-1 正样本 = 出现在 ≥1 个有效三点组（`*_fullest_v3.npy`）中的点；stage-2 = 出现在有效三点组中的点对（权重 = 出现次数，已逐对核对相等）；stage-3 `_s3_gt` 由 `point_set_selection_test.py:1004-1026` 构造。损失：stage 1 稀疏交叉熵，stage 2/3 ListNet 型。
- Top-1/Top-10 带 **5 mm KD-tree 邻域放宽**（`point_set_selection_test.py:728-758` 等），继承自 UniGrasp。
- 与正文出入：k 全为 1024（`TOP_K=TOP_K2=1024`；6 个 event 文件 GraphDef 中 4 个 TopKV2 的 k 均 1024；上游论文亦 1024）。"512" 只出现在 2021-01 下载的预计算文件 `data/objects/1812/*_s1_topk.npy` 形状 (512,)。第一阶段逐点分类输入是 64+64，不是 64+256。

**依据** `B1_selection_rule.md`（完整伪代码与行号）；`python3 evaluation/thesis_ch3/b3_graphdef_topk.py`。

### B2 · 训练/测试样本数
**结论**
- 数据根路径硬编码 `/media/robin-lab/Jim/{train,test}_data`（`PSS/data_preparing.py:118`, `data_preparing_test.py:143`），**本机不存在**；本地只有 `data/objects/1812` 一个物体（37 文件，无 bh282 标签）。官方压缩包 `train_data.tar.gz` 236.6 GB、`test_data.tar.gz` 50.6 GB（`curl -sIL` Content-Length）；"600/150 GB" 作为解压后大小合理但无法核实。UniGrasp 论文：1000 模型 → 3275 实例 × 8 视角，12 种夹爪，80/20。
- 夹爪 ID：11 = robotiq_3f，12 = bh_282，13 = kinova_kg3（`train_pssn.py:635-643`）；推理 `unigrasp.py:555-557` 把 13 映射为 **ruth**（RUTH 复用 Kinova 标签）。每 batch `random.choice([11,12,13])`。
- **重要**：保存的三份训练脚本输入都是 768 维 mean/max/min，`workspace.npy` 加载行被注释（`train_pssn.py:655-657`）；但 `220model.ckpt` 首层 `Conv1D/W [1,1,256,512]`（189/221 为 768）且 2021-09-02/03 的 event 文件 Placeholder 为 [-1,256] → 256 维 EfficientGrasp 训练确实发生过，**该版本脚本未保存**。
- **Table 3.2 分母 N = 283**：`results/pssn_accuracy/` 9 个文件 1332 个数值全部 = k/283（误差 <3e-14），论文 42 格亦然（99.3=281/283, 88.7=251/283, 54.8=155/283…）。瑕疵：282/283=99.65 在表中一处写 99.6、一处写 99.7。
- 保存的文件是 74 行 × 2 列的 per-checkpoint 曲线（epoch 1,4,…,220），每文件一个夹爪一种模型，**不是逐样本**，也不分夹爪列。21 个 (Top-1,Top-10) 对里只有 8 个能在某文件某行同时命中，且 epoch 分散（58、124、163、199），不是末行也不是最大行 → Robotiq/Kinova 大多数格子与 "I BarrettHand" 的原始曲线 **未找到**；表中每格取自哪个 checkpoint 无法说明。
- 训练样本数：**未找到**（脚本只 print，无日志）。按压缩包比例推断量级 ≈ 283×4，仅为推断。

**依据** `python3 evaluation/thesis_ch3/b2_pssn_accuracy_stats.py` → `B2_pssn_accuracy_dump.md`；`python3 evaluation/thesis_ch3/b3_ckpt_index_shapes.py` → `B3_ckpt_variables.md`。

### B3 · 超参敏感性
**结论：未做。** 查遍 `PSS/*.py`（7 个脚本）、`tf_models/`、6 个 event 文件的 GraphDef 常量、3 个检查点形状：k 均 1024，M 均 2048；学习率：法向头 2e-3，stage 1 5e-5（测试脚本 1e-4），stage 2 1e-4 或 5e-5，stage 3 1e-3（6 个 event 文件 Adam 常量一致 2e-3/5e-5/5e-5/1e-3）。脚本间的 lr 差异是版本差异，无配套结果。唯一对照是 256 维 vs 768 维夹爪特征（即 E vs U）。日志：仅 `loss_stage1` 一个标量，256 维 run 180 epoch、768 维 run 132 epoch，其余 4 个是中止的启动。epoch 上限 `if epoch == 220: break`。

**依据** `B3_hyperparameter_grep.md`、`B3_tf_event_runs.md`、`B3_scalars_*.csv`。

### B4 · BarrettHand 254 还是 256
**结论** 磁盘上逐构型点云：`Data_DB/bh_282/bh282_0..256` = **257** 个，`robotiq_3f_0..32` = **33**，`kinova_kg3_0..8` = **9**，无缺号，均 (2048,3) 且内容互异。与 UniGrasp 论文规则 2^D 边界构型 + 1 中心构型吻合，与 URDF 非固定关节数一致（`barrett_hand_280.urdf` 8、`robotiq_3f_test.urdf` 5、`kinova_kg3.urdf` 3）。**正文应写 33 / 9 / 257**，或写 32 / 8 / 256 并注明另加 1 个中心构型。"254" 无任何来源。

**依据** `python3 evaluation/thesis_ch3/b4_gripper_config_counts.py` → `B4_gripper_config_counts.md`。

### B5 · 9.85 % 怎么算的
**结论：无法复现。** 候选聚合（`B5_table32_aggregates.md`）：18 格 E−U 均值 **10.62 pp**，相对提升均值 13.93 %，中位数 9.75；Top-1 均值 12.97，Top-10 均值 8.28；Stage 1/2/3 均值 4.57 / 13.03 / 14.27；按夹爪 Robotiq / Kinova / Barrett 8.77 / 15.15 / 7.95；Barrett Top-1 均值 **9.83**（精确分数 9.78）；Stage-3 Top-10 中位数 9.90。穷举 18 格所有子集，落在 9.85±0.05 的都无合理解释。两仓库 grep "9.85" 无相关来源。建议改用可复现的 10.6 pp 并写明定义。

**依据** `python3 evaluation/thesis_ch3/b5_table32_aggregates.py`。

---

## C · RL 逆运动学

### C1 · 状态 / 动作逐维
**结论**（三张表 `C1_state_action_{ruth,robotiq_3f,barrett}.md`，合并表 `C1_state_action_combined.md`）
- 三夹爪共用类 `MoveUr5RuthEnv`（`kelin-v0`，`max_episode_steps=100`）。观测 = `get_obs()[5:]` = [wrist_3 角] + 手部关节读数 + 9 个目标坐标（m，变换到**固定名义末端系** p=[0.464,−0.11,0.66]、R=Euler(0,π/2,0)），无归一化：**RUTH 13 = 1+3+9**，Robotiq 15 = 1+5+9，Barrett 18 = 1+8+9。声明的 `observation_space=Box(0,1)^{4/6/9}` 只用来算 obs_dim。
- **上一步动作不在策略输入中**：检查点 `GaussianPolicy.*.pt` 的 `linear1.weight` 形状 (256,13)/(256,15)/(256,18)。
- 动作：`Box(-1,1)^{4/6/9}`，`a ← clip(a,−1,1)·0.5`，每维是加到当前读数上的**增量**（≤0.5 rad/步），裁剪后以 `POSITION_CONTROL` 发绝对目标；a[0] = wrist_3。裁剪范围 RUTH m1 [−1,π/2]、m2 [−π/2,1]、腱 [−0.5,0.12]；Robotiq/Barrett 见表。每裁剪一次 −100。
- UR5 关节 1–5 不受控；RUTH/Robotiq 每 episode 对 wrist_1/wrist_2 加 U(−π/6,π/6) 扰动（`point_transform`），Barrett 无扰动。

**不确定** 三份 `move_env.py` 的 mtime 都晚于对应训练（RUTH 训练 2022-01-25~02-01，文件 08-18；robotiq/barrett 文件是评估版，`step()` 返回 `distance`）→ **训练时运行的 env 版本均未保存**。`progress.txt` 中奖励为负，证明训练用的是带惩罚的 shaped reward。

### C2 · 奖励、终止、视界、超参
**结论**（`C2_reward_hparams.md`；RUTH `E:390-431`）
```
dist_j = ‖p_j − goal_j‖ × 1000  (mm, 欧氏, 不平方; p_j 为指尖 link 质心)
r = −Σ_j dist_j
    + Σ_j (+100 if dist_j < 15 mm else −100)
    + 1000 且 done   if Σ_j dist_j < 15 mm      (三指之和 <15 ⇔ 平均每指 <5 mm)
    + 100            if 三指都 <15 mm            (RUTH 版独有)
    − 100 × 被裁剪的手部电机数 − 100 × 超 ±π 的 UR5 关节数
```
| 项 | 值 | 来源 |
|---|---|---|
| 终止 | Σdist <15 mm 或 100 步（TimeLimit） | `gym/wrappers/time_limit.py` |
| episode/epoch | 10 | `train.py:209` |
| 目标 | 每 episode 1 个：`random.choice(contact_list.npy)`，**仅 21 个行号** | `E:466-471` |
| γ / τ / lr / batch / replay | 0.99 / 0.005 / 0.003 / 256 / 1e7 | `train.py` 默认 = 8 个 `config.json` |
| α | `--alpha 0.8` 但 `automatic_entropy_tuning=True`，log_alpha 初值 0（α=1），Adam 自动调节；`progress.txt` 中 α 终值 1.0–13.3 | `train.py:185`, `sac.py:30-33` |
| 网络 | 策略 3×256 softplus + tanh；双 Q 各 2×256 ReLU | `model.py:149-223` |
| updates/step, target_update | 1, 1 | config.json |
| start_steps | 10000，**被解析但未使用**（无随机暖启动） | `train.py` |
| seed | torch/np/env 的 seed 调用全被注释，`--seed` 只决定目录名；ruth seed--2 由 seed--1 热启动 | `train.py:49-52` |
| 最后 checkpoint | ruth 2885/2475/5865/3365；robotiq 2300/145；barrett 3050/2905（每 5 epoch 存一次） | `log/kelin-v0/seed--*/` |
- 正文"每 epoch 10 个目标、每目标最多 100 episode"是 episode↔步 的混淆。
- `contact_list.npy` 时间戳 2022-02-06 晚于 RUTH 训练，RUTH 训练时的目标采样方式未找到。

### C3 · Fig 3.7 重出
**结论**
- `val.py:75-104` 保存的量 = 一个 episode 内 100 步的 `Σ_j dist_j / 3` 的时间平均，即**每指平均欧氏误差 (mm)，整个 episode 时间平均**（含起始远离目标的步），是末步误差的上界；末步误差无法恢复。
- 新图 `fig_rl_error.{pdf,png}`，数表 `C3_rl_error_table.{md,csv}`（每文件 n/mean/median/std/p5/p95/min/max）。末 epoch（文件名 2885，n=100）：

| 夹爪 | mean | median | p95 | min | <15 mm 的 episode |
|---|---:|---:|---:|---:|---:|
| RUTH | 15.54 | 15.23 | 24.1 | 8.30 | 45 % |
| Robotiq-3F | 18.93 | 18.99 | 37.4 | 8.17 | 29 % |
| BarrettHand | 10.88 | 10.72 | 12.7 | 8.91 | 96 % |

- 降幅（均值，首→末文件）：RUTH 84.25→15.54 = **81.6 %**；Robotiq 48.18→18.93 = 60.7 %；Barrett 75.42→10.88 = 85.6 %。"81 %" 只对 RUTH 成立。
- "within 5 mm per finger across all three grippers" **不成立**：无任何 episode <5 mm。旧脚本 `evaluation/simulation/RL_eval.py` 第 60-62 行用 `sum/100` 求均值而 `ruth/epoch_200.npy` 只有 20 条 → 5.06 mm，再 `sort(reverse=True)` 把它排到最右端（标签 2885），曲线被人为变成单调；min/max 带另乘 0.3–0.5 手调系数。若正文想保留"5 mm"，只能表述为环境的**成功判据**（Σ<15 mm ⇔ 平均每指 <5 mm），不是达到的精度。
- 曲线并不单调：RUTH 500/1500、Robotiq 200/500、Barrett 1000/1500 回升。
- Barrett 两套日志：下划线 `epoch_1000/1500`（≈35–36 mm，05-05 18:48/19:16）生成时只有 seed--0 训练完 → 只能是 seed--0；05-06 的其余下划线文件与无下划线 `epoch1000/1500`（≈12 mm）在 seed--1 结束 13 分钟后连续生成，`val.py` 末态硬编码 `seed--1`，`RL_gripper.py`/`demo.py` 也加载 seed--1（末 epoch 平均奖励 +224 vs seed--0 −612）→ **论文用的下划线系列在 1000/1500 两点混入了 seed--0**，被排序掩盖；无下划线系列是 seed--1 的同源评估。
- 其他异常：`robotiq/epoch_2500/2885` 不可能对应真实 checkpoint（seed--0 止于 2300，seed--1 止于 145），所用 checkpoint 无法确定；`barrett/epoch_1.npy` 是 `robotiq/epoch_1.npy` 的字节拷贝（早于 Barrett 训练开始）；`ruth/epoch_200`（n=20，2022-06-12）所用 checkpoint 与代码中的 `itr=1000` 矛盾，无法确定。

**依据** `python3 evaluation/thesis_ch3/plot_rl_error.py`；文件 mtime 时间线见 `details_C.md` C3。

### C4 · 与数值 IK 的精度对比
**结论：未完成（只能报可解率，不能报毫米误差）。**
- `ruth_grasping_kinematics.py`（6 份同 md5）：假设接触点构成**等腰三角形**且计算在接触平面内；对物体角度做 100 格网格搜索，内层对五连杆电机角 1° 步长搜索并**强制对称** θ1=t, θ2=π−t；唯一的 `fsolve` 只有 1 个未知数（三指共用的弯曲角，平面两段手指 44.7+35 mm，取三指所需伸展的最大值）。返回 TCP 位置、一个弯曲角、旋转矩阵；**不返回电机角，没有前向运动学**。
- 运行结果（`c4_numeric_ik_vs_rl.py`，seed 0 → `C4_rl_vs_numeric_ik.md`, `C4_numeric_ik_raw.csv`）：

| 目标集 | n | fsolve 收敛 | 三指所需伸展差 中位 / p95 (mm) |
|---|---:|---:|---|
| `ruth_workspace_sampling/contact_points.npy` 抽 1000 | 1000 | 1000 | 0.3 / 38.9 |
| RL 目标池 `ruth/contact_points.npy` 抽 1000 | 1000 | 999 | 0.3 / 0.7 |
| 21 个训练目标 | 21 | 21 | 0.2 / 0.5 |

  唯一不收敛样本所需伸展 147 mm > 模型上限 ≈67 mm；20–29 % 的解 θ 为微小负值（反向弯曲，物理不可达）。
- 不能同口径比较的原因：无 RUTH 前向运动学（只在 URDF+PyBullet 中，本机无 PyBullet）；RL 误差是全臂+手仿真、带腕部扰动、100 步时间平均的量。MH p56 只能回答"数值解在等腰/共面假设下可解率 99.9 %，一旦三角形不等腰（工作空间样本 p95 伸展差 39 mm）单腱模型无法同时满足三指"，RL 末 epoch 每指 15.5 mm。

---

## 补进仓库的内容
- `evaluation/thesis_ch3/`：`make_fig3_workspace.py`, `a1_grid_structure.py`, `subsample_576.py`, `count_workspace_arrays.py`, `a3_parse_tf_events.py`, `b2_pssn_accuracy_stats.py`, `b3_parse_tf_events.py`, `b3_graphdef_topk.py`, `b3_ckpt_index_shapes.py`, `b4_gripper_config_counts.py`, `b5_table32_aggregates.py`, `plot_rl_error.py`, `c4_numeric_ik_vs_rl.py`（纯 numpy/scipy，无 TF/torch/PyBullet 依赖；`plot_rl_error.py` 需要 `results/rl_ik_error/` 数据）。
- `results/thesis_ch3/`：本报告、`details_{A,B,C}.md`、`fig_workspace_ruth.{pdf,png}`、`fig_rl_error.{pdf,png}`、各 A*/B*/C* 表格与 CSV、`A2_ws1_reproduced.npy`。
- `gripper_representation/ruth_workspace_sampling/contact_points.npy`（623 KB）加入 git。
- `docs/file_reference.md`：按 A3 更正自编码器训练集描述，补 A2 抽样步骤，标注 robotiq_3f/barrett 下 `ws1.npy` 为 RUTH 误拷贝。

## 缺件清单（本机未找到，需从别处补或在正文中声明）
1. Robotiq-3F / BarrettHand / Kinova 的工作空间数组与采样脚本（A4）。
2. 生成 `Data_DB/*/workspace.npy` 的 2021 年编码器检查点与输入数组（A4）；`coupRotWorkspaceArrays` 的生成脚本；`fourbar/singRot` 目录（A3）。
3. 2022-01 早期自编码器训练的数据与检查点（A3）。
4. 256 维输入版的 PSSN 训练脚本（只剩 768 维快照）与写 TF summary 的版本（B2/B3）。
5. UniGrasp 训练/测试数据集本体（`/media/robin-lab/Jim/`）；Table 3.2 中 Robotiq/Kinova 大部分格子与 "I BarrettHand" 行的原始精度曲线；训练样本数（B2）。
6. 9.85 % 的计算来源（B5）。
7. 训练时实际运行的三份 `move_env.py`；RUTH 训练时的目标采样方式（C1/C2）。
8. `ruth/epoch_200`（n=20）、`robotiq/epoch_2500/2885`、barrett 05-06 批次所用的 seed/checkpoint（C3）。
9. RUTH 解析前向运动学（C4）。
