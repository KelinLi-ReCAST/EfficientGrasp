# A4 各夹爪工作空间特征溯源表

生成依据: `python3 - <<EOF ...` 中的 shape/md5/mtime 检查 (见 _draft_A.md A4 节的命令), 以及
`find /home/kelin/EfficientGrasp /home/kelin/RAL-IROS2022 -name "*.npy"` + 逐文件读取 header 的 (N,9) 扫描.

| gripper | 采样脚本 | 电机网格 | 工作空间行数 | 工作空间数组路径 | 特征文件路径 (推理用, unigrasp.py L561) | 特征 mtime / md5 | 链接验证 |
|---|---|---|---|---|---|---|---|
| RUTH (2022 重做) | gripper_representation/ruth_workspace_sampling/workspace.py L533-550 (env: rl_inverse_kinematics/early_prototype/gym/envs/kelin/move_env.py) | 31 x 31 x 9 (0.08 / 0.08 / 0.1 rad 步长, 指令增量) | 8649 -> linspace 抽 576 | gripper_representation/ruth_workspace_sampling/contact_points.npy (8649,9); gripper_representation/feature_extraction/ruthArrays/ws1.npy (576,9) | gripper_representation/feature_extraction/ruthArrays/{mean,max,min}.npy (三者 md5 相同 a12baf3d…) | 2022-02-08 14:31 / a12baf3ddb7bd8e15fcd33e92d0ce0c3 | **是**: ws1.npy 由 contact_points.npy 复现(逐字节相同); mean.npy 由 encoder_test.py(restore(9999), gripper_dir=ruthArrays) 生成 — 代码链路 + mtime 吻合, 但未重新跑 TF 验证数值 |
| RUTH (UniGrasp Data_DB) | 未找到 | 未找到 | 未找到 | 未找到 | contact_point_selection/UniGrasp/data/gripper_features/Data_DB/ruth/workspace.npy (1,256) float32 | 2021-09-10 23:33 / b840a651847bd9f0a4d23f63eb0b006b | **否**: 与 ruthArrays/mean.npy 不相等 (max abs diff 0.274); 早于本仓库所有 RUTH 工作空间数组 (2022-02-07) 和 9999model.ckpt |
| Robotiq 3F | 未找到 (rl_inverse_kinematics/robotiq_3f/gym/envs/kelin/workspace_generation.py 是 RUTH 环境的拷贝, __main__ 只做随机动作测试, 不保存) | 未找到 | 未找到 | 未找到 (rl_inverse_kinematics/robotiq_3f/ws1.npy 与 RUTH ws1.npy 逐字节相同, md5 84e3ed31…, 不是 Robotiq 的) | Data_DB/robotiq_3f/workspace.npy (1,256) (data/grippers/robotiq_3f/ 下只有 mean/max/min 和 2048x3 点云, 无 workspace.npy) | 2021-08-30 21:07 / 78d88f0e9ea765ae292545d46244bebc | **否** |
| Barrett BH-282 | 未找到 (同上, barrett/.../workspace_generation.py 为 RUTH 拷贝) | 未找到 | 未找到 | 未找到 (rl_inverse_kinematics/barrett/ws1.npy 同样是 RUTH 的 ws1.npy) | Data_DB/bh_282/workspace.npy == data/grippers/bh_282/workspace.npy (1,256) | 2021-08-30 21:09 / 222f063d25597721b09cd13d1206dabb | **否** |
| Kinova KG-3 | 未找到 | 未找到 | 未找到 | 未找到 | Data_DB/kinova_kg3/workspace.npy (1,256); unigrasp.py 推理未引用 (gripper_id 13 -> 'ruth') | 2021-08-30 21:08 / 7729470dbfc415bc68c8f4826c02a394 | **否** |

说明
- 两个仓库内所有形状为 (N,9) 的 .npy 只有: RUTH 的 (8649,9) 及其 (576,9) 子采样 (6 份拷贝, 同 md5), (485376,9) 的 RL 目标集 contact_points.npy (6 份拷贝, md5 f3f4b364, 2022-01-17, 用于 move_env.py 的 reset 目标采样, 非编码输入), 以及若干 (10,9) 的接触点结果文件. 不存在任何 Robotiq / Barrett / Kinova 的 (N,9) 工作空间数组.
- Data_DB 中 4 个 workspace.npy 的范数 (0.38–0.94) 与 ruthArrays/mean.npy (0.349) 同量级, 明显小于 UniGrasp 原版 mean/max/min 点云特征 (3.4–7.5), 说明它们确实来自某个 "workspace" 自编码器, 但生成它们的输入数组与 checkpoint (2021-08/09) 均不在两个仓库中.
- 另注: train_pssn.py L645-659 使用 mean/max/min 拼接 (768 维, gripper_id 13 -> kinova_kg3), 而 unigrasp.py L548-566 使用 workspace.npy (256 维, gripper_id 13 -> ruth); 两者输入维度不一致, 需另行核对 PSSN checkpoint 的输入维度.
