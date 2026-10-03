# B1 PSSN 三阶段选点规则（代码级伪代码，含 file:line）

代码根目录：`contact_point_selection/UniGrasp/point_set_selection/`。
推理图在 `unigrasp.py`（EfficientGrasp，256 维 workspace 特征），训练图在 `train_pssn.py`（行号整体比 unigrasp.py 大 8 左右）。
两个文件均 `from tf_models.point_quality_v6 import two_point_quality`（stage 2）、
`from tf_models.point_quality_v12 import two_point_quality as two_point_quality_stage3`（stage 3）
（unigrasp.py:19-20, train_pssn.py:21-22）。

常量：`TOP_K = 1024`, `TOP_K2 = 1024`（unigrasp.py:66-67, train_pssn.py:71-73）。M=2048（所有 placeholder `[None,2048,3]`）。

## Stage 1（第一点）

```
F0 = PointNet++_s4(obj_pc [B,2048,3], gripper_feat g)          # unigrasp.py:85, pointnet2_sem_seg_s4.py:20-95
    g 注入三处：原始 g 拼到 l5 (4 点) 特征 -> fa_layer0            # s4.py:62-64
               conv(g)->256 维 拼到 l3 (64 点) 特征 -> fa_layer2   # s4.py:46-48, 67-68
               conv(g)->64 维 tile 到 2048 点, 与 64 维逐点特征拼接 -> conv128 -> conv64 -> conv2  # s4.py:51-54, 76, 90-94
    (逐点分类头的输入是 64(点)+64(夹爪) = 128 维, 不是 64+256)
logits1 [B,2048,2]; p1 = softmax(logits1)[:,:,1]                # unigrasp.py:98,105
S1  = top_k(p1, k=TOP_K=1024)   (sorted)                         # unigrasp.py:110  -> 第一点候选
P2  = top_k(p1, k=TOP_K2=1024)                                   # unigrasp.py:112  -> 第二/三点候选池 (与 S1 相同集合)
同一网络另有法向量头 nor [B,2048,3]                                 # s4.py:84-86
```
- 没有概率阈值；只有 top-k。
- 损失（训练）：`loss_stage1 = sparse_softmax_cross_entropy(labels=gq_label, logits)`（train_pssn.py:125,128）；ListNet 版 `loss_ListNet_Loss_s1` 已定义但未用（train_pssn.py:123）。

## Stage 2（第二点，条件于第一点）

```
F2 = PointNet++_s6(obj_pc, g)['feats']  (独立权重 stage2/pointnet2, 64 维/点)   # unigrasp.py:128-129
feat_S1 = gather(F2, S1)   [B,1024,64]                                          # unigrasp.py:141
feat_P2 = gather(F2, P2)   [B,1024,64]                                          # unigrasp.py:140
for i in S1, for j in P2:  x_ij = concat[feat_P2[j], feat_S1[i]]  (128 维)      # point_quality_v6.py:23-27
    logits2_ij = conv1d(128)->IN->conv1d(64)->IN->conv1d(2)                     # point_quality_v6.py:7-16
p2 = softmax(logits2)[:,:,1]   展平为 [B, 1024*1024] 对                          # unigrasp.py:173,175
antipodal 掩码: n_i·n_j < -0.5 (预测法向)  -> p2 *= mask                        # unigrasp.py:162-169,192
S2 = top_k(p2, k=TOP_K2=1024)  -> 1024 个 (i,j) 对, 索引 = i*1024 + j           # unigrasp.py:193; 解码见 600-602
```
- 条件方式：**只拼接第一点的 64 维学习特征**，不拼坐标、不拼相对向量、不拼法向（法向只用于掩码）。
- k2 是对 **1024×1024 个点对** 整体取 top-1024，不是"给定第一点取 top-k 点"。
- 损失：ListNet 型 `-mean(label * w * log(exp(p2)/sum exp(p2)))`（train_pssn.py:193-196）；注意是对 softmax 概率再做 softmax 的变体。

## Stage 3（第三点，条件于点对）

```
F3 = PointNet++_s6(obj_pc, g)['feats']  (独立权重 stage3/pointnet2)              # unigrasp.py:200-201
pairfeat_k = concat[F3[i_k], F3[j_k]]  (128 维), k 取 S2 的前 TOP_K=1024 对       # unigrasp.py:236-243, 271
feat_P3    = gather(F3, P2)  [B,1024,64]                                        # unigrasp.py:272
for k in S2, for m in P2: x_km = concat[feat_P3[m], pairfeat_k] (192 维)         # point_quality_v12.py:54-58
    logits3_km = conv1d(192)->IN->conv1d(64)->IN->conv1d(2)                     # point_quality_v12.py:40-47
p3 = softmax(logits3)[:,:,1]  [B, 1024*1024] 三元组                             # unigrasp.py:384,386
几何掩码 final_mask = type_flag * angle_flag * flag_side * check_flag * fc_check  # unigrasp.py:390
    type_flag : 三法向两两点积符号模式, 阈值 0.6                                  # unigrasp.py:293-304
    angle_flag: 三角形最大内角 < 120° (min cos > -0.5)                            # unigrasp.py:368-382
    flag_side : 最短边 > 0.01 m                                                   # unigrasp.py:354-362
    check_flag: 法向与边向量符号检查                                              # unigrasp.py:348-352
    fc_check  : 摩擦锥检查, mu=0.42, fc_thre = sin(atan 0.42)                     # unigrasp.py:316-344
p3 *= final_mask                                                                 # unigrasp.py:404
S3 = top_k(p3, k=1024)  -> 三元组 (i,j,m); 解码 unigrasp.py:627-638
Top-1 = S3[0]                                                                    # unigrasp.py:640-642
```
- **最终排序只用 stage-3 的（掩码后）softmax 概率** `out_corr_score_stage3_tf`，不是各阶段概率乘积，也不是 log 和（unigrasp.py:404-406）。
- 没有概率阈值；没有 GQS/抓取质量过滤；距离过滤只有 `min_side > 0.01 m`。
- 损失：ListNet 型，用未掩码标签（train_pssn.py:401-408）；`final_mask_gt_tf` 只计算不参与损失（train_pssn.py:399）。

## 标签构造（什么算正样本）

数据文件（以 `data/objects/1812/` 为例，`b2_pssn_accuracy_stats.py` 列出形状）：
- `*_robotiq_3f_fullest_v3.npy` (2976285,4)、`*_kinova_3f_fullest_v3.npy` (1837720,4)：**有效三点组列表**（3 个点索引 + 常数 0.001）。按 UniGrasp 论文 Sec. IV：力闭合 (Q>1e-4) + 可达 + 无碰撞。
- `*_tmp_label_stage1.npy` (2048,) 0/1：出现在至少一个有效三点组中的点（验证：598 个正样本 ⊂ v3 中 604 个点）；`*_stage1_w.npy` = 该点出现次数（相关系数 0.99999）。
- `*_tmp_label_stage2.npy` (2048,2048) 对称 0/1：出现在至少一个有效三点组中的点对（157450 个非零，全部 ⊂ v3 导出的 160866 对）；`*_stage2_w.npy` = 该对出现于多少个三点组（在共同对上完全相等）。
- `*_s3_gt.npy` (1024*1024,)：由 `point_set_selection_test.py:1004-1026` 生成：对 S2 的每个对 k 和候选池每个点 m，若 (i_k, j_k, m) 的任意排列在 v3 列表中则为 1（6 个排列都写入 `tmp_label`，1009-1014）。

训练时的使用：
- Stage 1：`stage1_gq_label[0:n_above] = label_stage1[pc_above_table]`（train_pssn.py:660-673），只保留桌面以上点；权重归一化后 +0.5（672）但 CE 损失不使用权重。
- Stage 2：对 (S1 的第 k 点, 候选池) 收集 `label_stage2`（train_pssn.py:821-824），再把 **法向点积 > -0.7 的对强制置 0**（825-829），权重归一化 +0.5（830）。
- Stage 3：直接读预生成的 `<gripper_id>_s3_gt.npy`、`_s1_top1024.npy`、`_s1_topk.npy`、`_s2_top1024.npy`（train_pssn.py:938-949），即 stage-3 训练是在 **冻结的 stage-1/2 输出索引** 上做的（主程序 `restore_stage3_v2(221)` 只恢复 stage1+stage2 变量，1110）。

## 评测指标（Table 3.2 的 Top-1 / Top-10 定义）

`point_set_selection_test.py`：
- Stage 1 Top-1：S1[0] 为正样本，否则在 **5 mm 半径 KD-tree 邻域** 内任一点为正也算对（728-743）；Top-10：S1[0:10] 中任一满足（744-758）。
- Stage 2 Top-1/Top-10：对 S2[0] / S2[0:10]，同样加 5 mm 邻域放宽（876-913）。
- Stage 3 Top-1/Top-10：对 S3[0] / S3[0:10]，同样 5 mm 邻域（1111-1158）。
- 每个样本计 0/1，除以 `num_batch*batch_size`（1217-1222）。5 mm 放宽与上游 UniGrasp 论文 Sec. V-B.1 相同。
