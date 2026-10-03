#!/usr/bin/env python3
"""复现 workspace_visual.py 的抽样步骤.

原脚本: rl_inverse_kinematics/ruth/workspace_visual.py (第 15-20 行)
    b = np.zeros((576,9))
    vertices_index = np.linspace(0, len(a)-1, 576, dtype=int)
    for i in range(576): b[i,:] = a[vertices_index[i],:]
    np.save('ws1', b)
其中 a 为 RUTH 指尖工作空间 contact_points.npy (8649,9).
本脚本从 gripper_representation/ruth_workspace_sampling/contact_points.npy 重新生成
576x9 子采样, 写入 results/thesis_ch3/A2_ws1_reproduced.npy, 并与
gripper_representation/feature_extraction/ruthArrays/ws1.npy 逐元素比对.
用法: python3 evaluation/thesis_ch3/subsample_576.py
"""
import hashlib
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'gripper_representation/ruth_workspace_sampling/contact_points.npy')
REF = os.path.join(ROOT, 'gripper_representation/feature_extraction/ruthArrays/ws1.npy')
OUT_DIR = os.path.join(ROOT, 'results/thesis_ch3')
OUT = os.path.join(OUT_DIR, 'A2_ws1_reproduced.npy')

a = np.load(SRC)
print('contact_points.npy shape:', a.shape)

# --- 与 workspace_visual.py 完全相同的抽样 ---
b = np.zeros((576, 9))
vertices_index = np.linspace(0, len(a) - 1, 576, dtype=int)
for i in range(576):
    b[i, :] = a[vertices_index[i], :]

os.makedirs(OUT_DIR, exist_ok=True)
np.save(OUT, b)

# --- 抽样间隔统计 ---
d = np.diff(vertices_index)
print('index range: %d..%d, n=%d' % (vertices_index[0], vertices_index[-1], len(vertices_index)))
print('nominal step (8648/575) = %.4f' % (8648 / 575))
print('actual integer steps: unique=%s, counts=%s' % (np.unique(d), np.bincount(d)[np.unique(d)]))

# --- 与 ruthArrays/ws1.npy 比对 ---
ref = np.load(REF)
print('ref shape:', ref.shape, 'dtype:', ref.dtype)
print('array_equal(reproduced, ref):', np.array_equal(b, ref))
print('allclose(reproduced, ref):', np.allclose(b, ref))
print('max abs diff:', np.abs(b - ref).max())

# 每一行 ws1 是否能在 contact_points 中找到 (容差 1e-6)
found = 0
for row in ref:
    if np.any(np.all(np.abs(a - row) < 1e-6, axis=1)):
        found += 1
print('rows of ws1 found in contact_points within 1e-6: %d/576 (%.4f)' % (found, found / 576))

md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()
print('md5 reproduced:', md5(OUT))
print('md5 ref       :', md5(REF))
with open(os.path.join(OUT_DIR, 'A2_subsample_576.txt'), 'w') as f:
    f.write('index range %d..%d n=%d\n' % (vertices_index[0], vertices_index[-1], len(vertices_index)))
    f.write('nominal step %.4f; integer steps %s counts %s\n' % (8648 / 575, np.unique(d).tolist(), np.bincount(d)[np.unique(d)].tolist()))
    f.write('array_equal=%s allclose=%s maxabsdiff=%g\n' % (np.array_equal(b, ref), np.allclose(b, ref), np.abs(b - ref).max()))
    f.write('rows found within 1e-6: %d/576\n' % found)
    f.write('md5 reproduced %s\nmd5 ref %s\n' % (md5(OUT), md5(REF)))
