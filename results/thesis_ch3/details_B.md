# 第三章核查草稿 B：PSSN 选点规则 / 样本数 / 超参 / 输入构型数 / 9.85 %

仓库：`/home/kelin/EfficientGrasp`（分支 thesis-ch3-verification）。原始副本 `/home/kelin/RAL-IROS2022` 只读对照。
所有脚本在 `evaluation/thesis_ch3/b*_*.py`，输出在 `results/thesis_ch3/B*.md`。系统 python3 无 TensorFlow，
TF event 文件 / checkpoint `.index` 用自写的 protobuf / SSTable 解析器读取（`b3_parse_tf_events.py`, `b3_graphdef_topk.py`, `b3_ckpt_index_shapes.py`）。

路径缩写：`PSS/` = `contact_point_selection/UniGrasp/point_set_selection/`，`DATA/` = `contact_point_selection/UniGrasp/data/`。

---

## B1 · 第 2、3 阶段的选点规则

完整伪代码与行号见 `results/thesis_ch3/B1_selection_rule.md`。要点：

| 问题 | 结论 | 依据 |
|---|---|---|
| 导入的网络定义 | stage 2 用 `tf_models/point_quality_v6.py`，stage 3 用 `tf_models/point_quality_v12.py`（别名 `two_point_quality_stage3`） | `PSS/unigrasp.py:19-20`, `PSS/train_pssn.py:21-22` |
| Stage 1 输入 | 物体点云 [B,2048,3] 进 PointNet++（`pointnet4/models/pointnet2_sem_seg_s4.py`），夹爪特征 g（EfficientGrasp 256 维 / UniGrasp 768 维）在 3 处注入：原始 g 拼到 l5 层、conv 到 256 维拼到 l3 层、conv 到 64 维 tile 到 2048 点后与 64 维逐点特征拼接 → conv128 → conv64 → conv2 → softmax。逐点分类头输入是 **64+64=128 维**，不是 "64+256" | `s4.py:46-54, 62-68, 76, 90-94`; `unigrasp.py:85, 98, 105` |
| k1 | `top_k(p1, k=TOP_K=1024)` 得 S1；同时 `top_k(p1, k=TOP_K2=1024)` 得候选池 P2（同一集合） | `unigrasp.py:66-67, 110, 112` |
| Stage 2 条件方式 | 对 S1 中每个 i、P2 中每个 j：`concat[feat(j) 64 维, feat(i) 64 维]`（来自独立权重的 stage2/pointnet2，`pointnet2_sem_seg_s6.py`）→ conv128-IN-conv64-IN-conv2 → softmax。**只拼第一点的学习特征**；不拼坐标/相对向量/法向 | `point_quality_v6.py:7-33`; `unigrasp.py:128-129, 140-141, 173, 175` |
| k2 | 对 1024×1024 **个点对**整体取 `top_k(k=TOP_K2=1024)`，不是"给定首点选 512 个点"；取 top-k 前先乘 antipodal 掩码 `n_i·n_j < -0.5`（预测法向） | `unigrasp.py:162-169, 192-193` |
| Stage 3 条件方式 | 取 S2 前 1024 对，对特征 = `concat[F3(i), F3(j)]` 128 维；再与候选点 m 的 64 维特征拼成 192 维 → conv192-IN-conv64-IN-conv2 → softmax，得 1024×1024 个三元组分数 | `unigrasp.py:236-243, 271-272, 384, 386`; `point_quality_v12.py:40-64` |
| k3 / 最终排序 | `p3 *= final_mask`（5 个几何掩码：法向符号模式 thre 0.6、最大内角 <120°、最短边 >0.01 m、法向-边符号检查、摩擦锥 μ=0.42）后 `top_k(k=1024)`；**最终名次 = stage-3 掩码后的 softmax 概率**，不是三阶段概率乘积，也不是 log 之和；Top-1 即 S3[0] | `unigrasp.py:293-304, 316-344, 348-362, 368-382, 390, 404-406, 640-642` |
| 阈值 | 无概率阈值；无 GQS/抓取质量过滤；唯一的距离过滤是最短边 >1 cm；stage 2 有法向 antipodal 掩码（<-0.5），stage 3 有上述几何掩码 | 同上 |
| 标签（正样本定义） | `*_fullest_v3.npy` = 有效三点组列表（力闭合+可达+无碰撞，UniGrasp 论文 Sec. IV）；stage-1 标签 = 出现在 ≥1 个有效三点组中的点（598 个正样本 ⊂ v3 中 604 点，权重=出现次数）；stage-2 标签 = 出现在有效三点组中的点对（157450 对 ⊂ v3 导出的 160866 对，权重=出现于几个三点组，共同对上完全相等）；stage-3 标签 `_s3_gt.npy` = (i_k, j_k, m) 任一排列在 v3 中 | 数据核对命令见下；`point_set_selection_test.py:1004-1026`；`train_pssn.py:660-673, 821-830, 938-949` |
| 损失 | stage 1 稀疏 softmax 交叉熵（`train_pssn.py:125,128`）；stage 2/3 为 ListNet 型 `-mean(label·w·log softmax(p))`（`train_pssn.py:193-196, 404-408`）；stage-2 训练标签额外把法向点积 > -0.7 的对置零（`train_pssn.py:825-829`） | |
| 评测指标 | Top-1/Top-10 均带 **5 mm KD-tree 邻域放宽**（预测点 5 mm 内有正样本即算对），与上游 UniGrasp 论文 Sec. V-B.1 相同 | `point_set_selection_test.py:728-758, 876-913, 1111-1158` |

标签关系核对命令（在 `DATA/objects/1812/` 下运行，结果：`v3 idx not in stage1: 6, stage1 not in v3: 0; pairs: v3 only 3416, stage2 only 0, both 157450; w2==count on common pairs: True`）：
```
python3 - <<'EOF'
import numpy as np, glob
v=np.load(glob.glob('*_robotiq_3f_fullest_v3.npy')[0]); s1=np.load(glob.glob('*robotiq_3f_fullest_tmp_label_stage1.npy')[0]); s2=np.load(glob.glob('*robotiq_3f_fullest_tmp_label_stage2.npy')[0]); w2=np.load(glob.glob('*robotiq_3f_fullest_tmp_label_stage2_w.npy')[0])
t=v[:,:3].astype(int); idx=set(np.unique(t)); pos=set(np.where(s1>0)[0]); print(len(idx-pos),len(pos-idx))
P=np.zeros((2048,2048),bool); C=np.zeros((2048,2048))
for a,b in [(0,1),(0,2),(1,2)]: P[t[:,a],t[:,b]]=1; P[t[:,b],t[:,a]]=1; np.add.at(C,(t[:,a],t[:,b]),1); np.add.at(C,(t[:,b],t[:,a]),1)
S=s2>0; print(int((P&~S).sum()),int((S&~P).sum()),int((P&S).sum()), np.array_equal(w2[S&P],C[S&P]))
EOF
```

**与论文/正文的出入**
1. 论文写 k1=1024, k2=512, k3=512：代码中 **所有阶段都是 1024**（`TOP_K=TOP_K2=1024`，stage-3 `top_k(k=1024)`；6 个 TF event 文件内嵌 GraphDef 中 `stage1/TopKV2, stage1/TopKV2_1, stage2/TopKV2, stage3/TopKV2` 的 k 常量全部为 1024，见 `B3_tf_event_runs.md`）。上游 UniGrasp 论文 Sec. V-B 也写 K1=K2=K3=K4=1024。唯一出现 512 的地方是 `DATA/objects/1812/{11,12,13}_s1_topk.npy` 形状 (512,)（2021-01 随上游数据下载的预计算文件，说明上游某个早期版本 TOP_K=512），但该形状与当前代码 `reshape [-1, TOP_K*TOP_K2]` 不兼容。
2. "k2 = 给定第一点选 512 个点" 的表述不对：k2 是对全部点对取 top-k。

---

## B2 · 样本数

### (a) 数据集与本机数据

| 项目 | 结论 | 依据 |
|---|---|---|
| 数据根路径（硬编码） | 训练 `/media/robin-lab/Jim/train_data`，测试 `/media/robin-lab/Jim/test_data`；本机 **不存在**（`/media/kelin` 为空，`df` 只有一块 528 GB 根盘） | `PSS/data_preparing.py:118`, `PSS/data_preparing_test.py:143`, `PSS/train_pssn.py:41`, `PSS/point_set_selection_test.py:40` |
| 样本列表构造 | `data_preparing.py` 把 train_data 下含 `*grasp4_new.npz` 的目录全部列为样本，`Train_Val_Test(..., splitting=[1.0,0,0])` → 全部进 `_train`；测试同理把 test_data 全部目录列为 `_train`（测试脚本遍历的也是 `_train`）。**没有对象 ID 列表**，完全靠目录枚举 | `data_preparing.py:118-132`, `data_preparing_test.py:143-155`, `Train_Val_Test.py:162-178`, `point_set_selection_test.py:489` |
| 本机数据 | `DATA/objects/` 只有 1 个目录 `1812`（37 个文件，448 MB）；`DATA/ObjectPointClouds/` 19 个 YCB 点云；`DATA/real_world/` 9 个 .npy。`find /home/kelin -maxdepth 5 -type d -iname "*unigrasp*" -o -iname "Data_DB*" -o -iname "train_data*" -o -iname "test_data*"` 只命中仓库内目录和 `RAL-IROS2022/UniGrasp`（后者 `data/objects` 同样只有 1812） | `ls DATA/objects`, `du -sh DATA/*` |
| 1812 目录内容 | 见 `B2_pssn_accuracy_dump.md` (a)：2048 点云 + 法向 (`_pcn_new_normal.npz.npy` (1569,6)，`_pc_above_table.npy` 1569 个桌面以上点索引)、二指夹爪标签 `_par_grasp{1,2,3,4,5,7,8}*.npz` (`par` (N,6,3))、三指标签 `_robotiq_3f_fullest_v3.npy` (2976285,4)、`_kinova_3f_fullest_v3.npy` (1837720,4)、stage1/stage2 标签及权重、预计算的 `{11,12,13}_s1_top1024/_s1_topk/_s2_top1024/_s3_gt.npy`。**没有 bh282 的标签文件**（代码期望 `_par_bh282tmp_label_stage1.npy` / `_par_bh282.npy`，`train_pssn.py:636, 929`） | `python3 evaluation/thesis_ch3/b2_pssn_accuracy_stats.py` |
| 官方数据规模 | UniGrasp README 只给链接，不给大小。`curl -sIL` 得 `train_data.tar.gz` Content-Length 236,647,551,097 B（**236.6 GB 压缩**，2020-09-14），`test_data.tar.gz` 50,590,343,876 B（**50.6 GB 压缩**）。论文 "600 GB / 150 GB" 应是解压后大小，比例 4:1 与 UniGrasp 论文 80/20 划分一致，压缩比 2.5–3 倍也合理，但 **无法直接核实**（本机没有数据） | `curl -sIL http://download.cs.stanford.edu/juno/UniGrasp/{train,test}_data.tar.gz` |
| UniGrasp 论文的数据描述 | 1000 个 Bullet 物体模型 ×≤5 个尺寸 = 3275 个实例，8 个视角渲染，12 种夹爪（9 二指 + 3 三指），80/20 划分，Adam lr 1e-4，逐阶段训练 | arXiv 1910.10900 Sec. IV, V（`pdftotext` 到 `~/.claude/jobs/231d0744/tmp/unigrasp.txt` 行 284-312） |

### (b) EfficientGrasp 训练使用的夹爪 ID

| ID | 名称（训练脚本） | 名称（推理 unigrasp.py） | 标签文件后缀 | 依据 |
|---|---|---|---|---|
| 11 | `robotiq_3f` | `robotiq_3f` | `_par_robotiq_3f_fullest_tmp_label_stage{1,2}.npy`, `_robotiq_3f_fullest_v3.npy` | `train_pssn.py:638-639, 807-809, 931-933`; `unigrasp.py:552-554` |
| 12 | `bh_282` | `bh_282` | `_par_bh282tmp_label_stage{1,2}.npy`, `_par_bh282.npy` | `train_pssn.py:635-637, 804-806, 928-930`; `unigrasp.py:549-551` |
| 13 | `kinova_kg3` | **`ruth`** | `_par_kinova_3f_fullest_tmp_labelkinova_stage{1,2}.npy`, `_kinova_3f_fullest_v3.npy` | `train_pssn.py:641-643, 810-812, 934-936`; `unigrasp.py:555-557` |

- 训练每个 batch 从 `[11,12,13]` 中无放回抽样：`gripper_index = np.random.choice(np.array([11,12,13]), gripper_size, replace=False)`（`train_pssn.py:549`；`point_set_selection.py:540`；`unigrasp_train.py:548` 为 `[12,13,11]`）。`train_pssn.py` 中 `nnn=1` → 每 batch 1 个夹爪 1 个样本；`point_set_selection.py`/`unigrasp_train.py` 中 `nnn=3` → 每 batch 同一物体 × 3 个夹爪。
- 上游 UniGrasp 的训练用 `[1,2,3,4,5,11]`（GitHub master `point_set_selection.py`，WebFetch 核对）。
- **注意**：保存下来的三份训练脚本（`train_pssn.py:82`, `point_set_selection.py:80`, `unigrasp_train.py:88`）输入占位符都是 `[None, 256*3]`，并加载 `mean/max/min.npy`（UniGrasp 式 768 维）；加载 `workspace.npy` 的三行在 `train_pssn.py:655-657` 被注释掉。只有推理脚本 `unigrasp.py:75, 561-566` 用 256 维 `workspace.npy`。但 **checkpoint 与 TF 日志证明 256 维版本确实训练过**：`220model.ckpt` 中 `stage{1,2,3}/pointnet2/Conv1D/W` 形状 `[1,1,256,512]`、`fa_layer0/conv_0/weights [1,1,1792,1024]`（189/221model 为 768 与 2304），且 2021-09-02/03 的 4 个 event 文件 GraphDef 中 `Placeholder [-1,256]`（`B3_ckpt_variables.md`, `B3_tf_event_runs.md`）。即训练脚本的 EfficientGrasp 版本（workspace 输入）**未以原样保存**，当前 `train_pssn.py` 是切回 768 维对照实验后的快照。

### (c) Table 3.2 的分母

`results/pssn_accuracy/`（`python3 evaluation/thesis_ch3/b2_pssn_accuracy_stats.py`，输出 `B2_pssn_accuracy_dump.md`）：

| 文件 | 形状 | 内容 | 备注 |
|---|---|---|---|
| `stage1_acc.mat` | (73,2) | 每行 (Top-1, Top-10) | 2021-09-09 02:51，73 行（少 1 行） |
| `stage1_ours.mat` | (74,2) | 同上 | 09-09 13:35 |
| `stage1_unigrasp.mat` = `stage1_acc.npy` | (74,2) | 同上 | 09-09 16:52，两者逐元素相同 |
| `stage2_ours.mat` = `stage2_acc.npy` | (74,2) | 同上 | 09-09 04:11 |
| `stage3_ours.mat` | (74,2) | 同上 | 09-09 12:45 |
| `stage3_unigrasp.mat` = `stage3_acc.npy` | (74,2) | 同上 | 09-10 19:39 |

- 行 = checkpoint：`point_set_selection_test.py:1233-1239`（注释掉的 main）`acc=np.zeros((74,2)); for i in range(220): if i%3==0: restore_stage3(i+1); acc[i//3]=test(0)` → 74 个 checkpoint（epoch 1,4,…,220），**不是逐样本**，也**不按夹爪分列**（每次运行固定一个夹爪，快照里 `gripper_index=np.array([12])`，`point_set_selection_test.py:539`；`test_with_gt.py:535` 为 `[11]`）。这与论文 Fig. 3.6（精度随 epoch 曲线）对应。
- **分母 N = 283**：9 个文件共 1332 个数值全部等于 k/283（最大误差 2.8e-14，最小满足的 N 就是 283，其次 566、849…）。即每条曲线每个点是在 **283 个测试样本**（test_data 下 283 个目录，每目录一个物体视角点云，`batch_size=1`，`num_batch = 283`）上算的 0/1 平均。
- 论文 Table 3.2 全部 42 个数值也都能写成 k/283（见 `B2_pssn_accuracy_dump.md` (c) 表），例如 99.3=281/283, 88.7=251/283, 54.8=155/283, 77.7=220/283。唯一的小瑕疵：282/283=99.65 %，在 "U Robotiq-3F S1 Top-10" 写成 99.6，在 "E BarrettHand S2 Top-10" 写成 99.7，同一分数两种舍入。
- 但 **Table 3.2 的具体格子大多不能在保存的曲线里找到**：21 个 (Top-1,Top-10) 对里只有 8 个在某个文件的某行同时命中，而且命中的 epoch 分散（58、124、163、199…），不是末行(220)也不是最大值行（`B2_pssn_accuracy_dump.md` (d)）。保存的 9 个文件每个只对应一个夹爪、一种模型，Table 3.2 需要 7 行×3 阶段 = 至少 7 组运行，因此 **Robotiq/Kinova 的大多数格子以及 "I BarrettHand" 行的原始曲线未保存**（未找到；查过 `results/pssn_accuracy/`、`RAL-IROS2022/UniGrasp/point_set_selection/`（同样 9 个文件 + 空的 `exp_test.txt`、空 `logging/`）、`RAL-IROS2022/UniGrasp/point_set_selection/logs/`）。
- 每格样本数汇总：

| 行 | S1 Top-1/Top-10 | S2 | S3 | 分母 | 原始曲线文件 |
|---|---|---|---|---|---|
| E Robotiq-3F | 99.3/100 | 97.5/98.9 | 88.7/96.1 | 283（由 k/283 推断） | 未找到（S1 对在 stage1_acc.mat 第45/54/56/67 行、stage1_ours.mat 第19 行出现；S2 对在 stage2_ours.mat 第46/57/68 行出现；S3 未找到） |
| U Robotiq-3F | 91.9/99.6 | 83.4/90.8 | 76/86.2 | 283 | 未找到 |
| E Kinova-3F | 97.9/100 | 87.6/97.2 | 75.6/89.8 | 283 | 未找到 |
| U Kinova-3F | 86.9/98.2 | 66.4/80.2 | 54.8/70.7 | 283 | 未找到 |
| E BarrettHand | 100/100 | 96.5/99.7 | 91.9/96.5 | 283 | S1 在 stage1_acc.mat 第41 行(epoch 124)；S2/S3 未找到（stage3_ours.mat 最高 94.35/98.59，末行 92.58/97.53） |
| U BarrettHand | 94.3/98.9 | 86.9/91.5 | 77.7/87.6 | 283 | S3 = stage3_unigrasp.mat 第54 行(epoch 163)；S1/S2 未找到 |
| I BarrettHand | 92.2/98.6 | 85.9/90.8 | 77.4/86.9 | 283 | S1 = stage1_unigrasp.mat 第19/45/56 行；S3 = stage3_unigrasp.mat 第66 行(epoch 199)；S2 未找到 |

- 训练样本数：**未找到**（脚本只 `print("train num …")`，无日志保存；`exp_test.txt` 为空；TF event 只有每 epoch 一个 `loss_stage1` 标量）。可以说的是：按压缩包大小比例（236.6 GB : 50.6 GB ≈ 4.7 : 1）和 UniGrasp 80/20 划分，训练目录数量级约为 283×4 ≈ 1100–1300，这是推断不是证据。

---

## B3 · 超参数敏感性

### 脚本中的取值（`B3_hyperparameter_grep.md`）

| 参数 | train_pssn.py | point_set_selection.py | unigrasp_train.py | *_test*.py / raw_point_cloud.py | unigrasp.py | 上游 UniGrasp master |
|---|---|---|---|---|---|---|
| TOP_K / TOP_K2 | 1024 / 1024 (71,73) | 1024/1024 (69,71) | 1024/1024 (77,79) | 1024/1024 | 1024/1024 (66-67) | 1024/1024 |
| stage-3 top_k | k=1024 (414) | 1024 (412) | 1024 (420) | 1024 | 1024 (406) | – |
| M | 2048（所有 placeholder） | 2048 | 2048 | 2048 | 2048 | 2048 |
| batch_size | nnn*1 = 1 (51,57) | 3 (49,55) | 3 (57,63) | 1 | 1 | – |
| lr 法向头 `train_op_nor` | 2e-3 (101) | 2e-3 (99) | 2e-3 (107) | 2e-3 | 2e-3 (94) | 2e-3 |
| lr stage 1 | 5e-5 (129) | 5e-5 (127) | 5e-5 (135) | 1e-4 | 1e-4 (121) | 5e-5 |
| lr stage 2 | 1e-4 (198) | 5e-5 (196) | 5e-5 (204) | 1e-4 | 1e-4 (190) | 5e-5 |
| lr stage 3 | 1e-3 (410) | 1e-3 (408) | 1e-3 (416) | 1e-3 | 1e-3 (402) | 1e-3 |
| epochs | `num_epochs=1e6`，`if epoch == 220: break` (56, 1073)；`data=np.zeros((220,10))` (498) | 同 | 同 | – | – | – |
| 夹爪特征维度 | 768 (82) | 768 (80) | 768 (88) | 768 | **256** (75) | 768 |
| 实际训练的阶段 | 只有 `train_op_stage3` 在 `sess.run` 中 (958)，stage1/2 的 train_op 行被注释 (732, 843)；main `restore_stage3_v2(221)` 恢复 stage1+2 后训 stage 3 (1110) | 只 stage 3 (1115) | stage 1+nor (727)、stage 2 (837)、stage 3 (951/1122) 全部 | – | – | – |

- 论文的 "Adam lr 2e-3" 只对应 **法向预测头** `train_op_nor`（`train_pssn.py:101`）；PSSN 三个阶段的学习率是 5e-5 / 1e-4(或 5e-5) / 1e-3。"220 epochs per stage" 与 `if epoch == 220: break` 一致，但快照只能证明 stage 3 是这样跑的；stage 1 的 TF 日志为 180 和 132 个 epoch（下）。

### TF 日志（`PSS/logs/`，129 MB，`B3_tf_event_runs.md`）

| event 文件 | 记录的墙钟时间 | 标量 | step 范围 | GraphDef 中 Placeholder 维度 | TopKV2 的 k | Adam lr 常量 |
|---|---|---|---|---|---|---|
| 1630611164 | 2021-09-02 20:32 → 09-03 16:15 | `loss_stage1` ×181 | 0..180（≈394 s/epoch） | **[-1,256]**（EfficientGrasp） | 1024,1024,1024,1024 | 2e-3, 5e-5, 5e-5, 1e-3 |
| 1630704176 | 09-03 22:22 | 无 | – | [-1,256] | 同 | 同 |
| 1630704490 | 09-03 22:28 | 无 | – | [-1,256] | 同 | 同 |
| 1630704605 | 09-03 22:30 → 22:45 | `loss_stage1` ×4 | 0..3 | [-1,256] | 同 | 同 |
| 1630705672 | 09-03 22:47 → 22:57 | `loss_stage1` ×1 | – | **[-1,768]**（UniGrasp 对照） | 同 | 同 |
| 1630708190 | 09-03 23:29 → 09-04 08:03 | `loss_stage1` ×133 | 0..132（≈232 s/epoch） | [-1,768] | 同 | 同 |

- 全部 6 个 run 的图结构相同（34964 节点，graph_def 10,759,481 B），只有夹爪特征维度不同；`loss_stage1` 曲线：256 维 run 0.72→末 10 个 epoch 均值 0.487（最小 0.284）；768 维 run 0.68→0.447（最小 0.356）。每 epoch 一个标量，无 stage-2/3、无精度标量。
- 现存脚本（两个副本）都没有 `tf.summary`/`FileWriter`（`grep -rn "tf.summary\|FileWriter" PSS/*.py RAL-IROS2022/UniGrasp/point_set_selection/*.py` 为空），写日志的脚本版本未保存。
- checkpoint（`B3_ckpt_variables.md`）：`189model` / `221model`（768 维输入，316.7 MB，2021-09-10 07:04 / 08:36）、`220model`（256 维输入，282.1 MB，09-10 22:20）；三者图结构相同（441 个非 Adam 变量）。stage-2 相关网络权重 `stage2/Conv1D{,_1,_2}/W` = [128→128],[128→64],[64→2]，stage-3 = [192→192],[192→64],[64→2]，与 v6/v12 定义一致。

### 结论：**未做**
查遍 `PSS/*.py`（7 个脚本）、`PSS/tf_models/*.py`、`PSS/logs/` 6 个 event 文件的 GraphDef 常量、3 个 checkpoint 的变量形状、`RAL-IROS2022/UniGrasp/point_set_selection/`（内容相同）：没有任何一次运行使用 k≠1024、M≠2048 或不同于上表的学习率；学习率在各脚本间的差异（stage-1 5e-5 vs 1e-4，stage-2 1e-4 vs 5e-5）是脚本版本差异，没有配套结果。唯一的"对照"是 256 维 vs 768 维夹爪特征（即 E vs U 本身）。

---

## B4 · UniGrasp 的输入构型数

`python3 evaluation/thesis_ch3/b4_gripper_config_counts.py` → `B4_gripper_config_counts.md`：

| 夹爪 | 目录 | 逐构型点云文件 | 索引范围 | 缺号 | 形状 | 内容去重 |
|---|---|---|---|---|---|---|
| BarrettHand | `DATA/gripper_features/Data_DB/bh_282/bh282_<i>.npy` | **257** | 0..256 | 无 | (2048,3) float64 | 257 个 md5 全不同 |
| Robotiq-3F | `.../robotiq_3f/robotiq_3f_<i>.npy` | **33** | 0..32 | 无 | (2048,3) | 33 个全不同 |
| Kinova KG-3 | `.../kinova_kg3/kinova_kg3_<i>.npy` | **9** | 0..8 | 无 | (2048,3) | 9 个全不同 |
| （副本）`DATA/grippers/bh_282/` | | 6 | 0..5 | | | 与 Data_DB 同名文件逐元素相同 |
| （副本）`DATA/grippers/robotiq_3f/` | | 33 | 0..32 | | | 与 Data_DB 相同 |

每个目录另有 `mean.npy / max.npy / min.npy / workspace.npy`，均 (1,256) float32；`ruth/` 只有 `workspace.npy`。二指夹爪 G1–G9/franka/sawyer/robotiq_2f 各只有 open/middle/close 3 个构型。

来源与"预期数目"：
- `gripper_representation/feature_extraction/encoder_test.py:106-141`：`test(gripper_dir, gripper_name)` 读取目录内 **所有以 gripper_name 开头的 .npy**，过 encoder 后对特征取 mean/max/min 存成 `mean/max/min.npy`。没有写死构型数，也没有生成构型的循环。
- 构型点云的生成脚本不在仓库内（UniGrasp README 第 34-35 行指向外部仓库 `linsats/Python-Parser-for-Robotic-Gripper`；本机 `find` 未找到该仓库）。
- UniGrasp 论文 Sec. III-A.2：D 个关节 → 2^D 个"boundary configurations"（每关节取上/下限）+ 1 个"central configuration"（中值），"3DoF 的例子共 9 个点云"。与磁盘完全吻合，且与 URDF 的非固定关节数一致（`contact_point_selection/UniGrasp/gripper_urdf/`（原副本 `RAL-IROS2022/UniGrasp/gripper_urdf/` 相同），命令 `grep -c 'type="revolute"\|type="prismatic"' <urdf>`）：`kinova_kg3.urdf` 3 个 → 2^3+1 = **9**；`robotiq_3f_test.urdf` 5 个 → 2^5+1 = **33**（`robotiq_3f.urdf` 为 26 个关节的未简化模型）；`barrett_hand_280.urdf` 8 个 → 2^8+1 = **257**。

**结论**：论文中 "32 / 8 / 254 或 256" 应统一为：UniGrasp 需要 2^D 个边界构型 + 1 个中心构型，即 Robotiq-3F **33**（32 边界 + 1 中心）、Kinova **9**（8+1）、BarrettHand **257**（256+1）。"254" 没有任何来源支持；如果论文只想说边界构型，则 32 / 8 / 256，并注明还有 1 个中心构型。

---

## B5 · 9.85 % 这个数

`python3 evaluation/thesis_ch3/b5_table32_aggregates.py` → `B5_table32_aggregates.md`（硬编码 Table 3.2；`results/pssn_accuracy` 不含该表的完整数值，见 B2(c)）。

| 候选聚合 | 值 (%) |
|---|---|
| 18 个 E−U 差的均值（3 夹爪×3 阶段×2 指标） | **10.62** |
| 18 个相对提升 (E−U)/U 的均值 | 13.93 |
| (mean E − mean U)/mean U | 12.56 |
| 18 个差的中位数 | 9.75 |
| 仅 Top-1（9 个）均值 / 仅 Top-10 均值 | 12.97 / 8.28 |
| 仅 Stage-1 / Stage-2 / Stage-3（各 6 个）均值 | 4.57 / 13.03 / 14.27 |
| Stage-3 Top-1 / Top-10（各 3 个） | 15.90 / 12.63（中位数 14.2 / **9.90**） |
| 按夹爪：Robotiq / Kinova / Barrett（各 6 个）均值 | 8.77 / 15.15 / 7.95 |
| 按夹爪 Top-1（各 3 个）：Robotiq / Kinova / Barrett | 11.40 / 17.67 / **9.83**（用精确分数 k/283 则为 9.78） |
| Barrett 用 I 代替 U：mean(E−I) 6 个 / Top-1 3 个 | 8.80 / 10.97 |
| 18 格以 I 代替 Barrett 的 U | 10.91 |
| 24 个差（E−U 18 + E−I 6）均值 | 10.17 |
| 相对误差减少 mean(1−(100−E)/(100−U)) | 79.9 |

穷举 18 个差的所有子集均值：落在 9.85±0.05 的最小子集是单格 "Robotiq-3F S3 Top-10"（9.9）等，均无合理解释。

**结论：无法复现**。没有任何自然的聚合给出 9.85；最接近的是 BarrettHand 三阶段 Top-1 的平均提升 9.83（精确分数 9.78）与 Stage-3 Top-10 的中位数 9.90。整表 18 格的平均提升是 **10.6 个百分点**（相对 13.9 %），建议论文改用这个可复现的数并写明定义，或删除 9.85。本机两份仓库的 .py/.md/.txt/.tex 中 grep "9.85" 没有任何与该表相关的来源。

---

## 其他需要在正文中修正/说明的点（汇总）

1. k 值：全阶段 1024（不是 1024/512/512）；k2、k3 是对点对/三元组整体 top-k。
2. 学习率：2e-3 只是法向头；PSSN 各阶段 5e-5 / 1e-4(5e-5) / 1e-3。
3. 第一阶段逐点分类输入为 64+64 维（夹爪特征经 7 层 conv 压到 64 维后 tile），不是 64+256。
4. 最终排序 = stage-3 掩码后概率；Top-1/Top-10 带 5 mm 邻域放宽（继承自 UniGrasp）。
5. 测试集 283 个样本（由全部 1332 个保存值和 42 个表值均为 k/283 推出）；训练集数目未找到。
6. 输入构型数：33 / 9 / 257（= 2^D+1）。
7. 9.85 % 无法复现；18 格平均提升 10.6 个百分点。
8. ID 13 在训练脚本里是 `kinova_kg3`，在 `unigrasp.py` 里映射为 `ruth`（RUTH 复用 Kinova 标签），正文若提到 RUTH 的 PSSN 应说明这一点。
9. 保存下来的 Table 3.2 原始曲线只覆盖部分格子（见 B2(c) 表），且与表中数值的对应 epoch 不一致，无法说明表中每格取自哪个 checkpoint（末 epoch？最佳 epoch？）。

## 生成的文件
- 脚本：`evaluation/thesis_ch3/b2_pssn_accuracy_stats.py`, `b3_parse_tf_events.py`, `b3_graphdef_topk.py`, `b3_ckpt_index_shapes.py`, `b4_gripper_config_counts.py`, `b5_table32_aggregates.py`
- 结果：`results/thesis_ch3/B1_selection_rule.md`, `B2_pssn_accuracy_dump.md`, `B3_tf_event_runs.md`（含 GraphDef 常量节）, `B3_scalars_*.csv`, `B3_ckpt_variables.md`, `B3_hyperparameter_grep.md`, `B4_gripper_config_counts.md`, `B5_table32_aggregates.md`
- 临时：`~/.claude/jobs/231d0744/tmp/unigrasp_1910.10900.pdf`, `unigrasp.txt`（上游论文全文，pdftotext）
