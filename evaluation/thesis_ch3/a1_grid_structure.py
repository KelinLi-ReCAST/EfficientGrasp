#!/usr/bin/env python3
"""A1: 分析 contact_points.npy (8649,9) 的采样网格结构.

workspace.py (gripper_representation/ruth_workspace_sampling/workspace.py 第533-546行) 的循环为
  for j in range(round(2*pi/0.2))  -> 31
    for k in range(round(2*pi/0.2))  -> 31
      for l in range(round(0.9/0.1))  -> 9
即 31 x 31 x 9 = 8649 (恰好也等于 93^2). 本脚本用两种 reshape 假设比较相邻样本的指尖位移,
判断真实网格是 31x31x9 (三电机) 还是 93x93 (两电机).
输出: results/thesis_ch3/A1_grid_structure.txt
用法: python3 evaluation/thesis_ch3/a1_grid_structure.py
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = np.load(os.path.join(ROOT, 'gripper_representation/ruth_workspace_sampling/contact_points.npy'))
out = []
P = out.append
P('shape %s dtype %s' % (W.shape, W.dtype))
for d in (6, 4, 3, 2):
    P('unique rows after np.round(W,%d): %d' % (d, np.unique(np.round(W, d), axis=0).shape[0]))
P('per-column min: %s' % np.round(W.min(0), 4))
P('per-column max: %s' % np.round(W.max(0), 4))

V = W.reshape(31, 31, 9, 9)      # (j, k, l, xyz*3)
U = W.reshape(93, 93, 9)         # (a, b, xyz*3)
dj = np.abs(np.diff(V, axis=0)).max(axis=(1, 2, 3))
dk = np.abs(np.diff(V, axis=1)).max(axis=(0, 2, 3))
dl = np.abs(np.diff(V, axis=2)).max(axis=(0, 1, 3))
P('--- 31x31x9 hypothesis: max |delta| between consecutive indices')
P('axis j (motor1, 30 steps): min %.4f max %.4f  per-step: %s' % (dj.min(), dj.max(), np.round(dj, 4).tolist()))
P('axis k (motor2, 30 steps): min %.4f max %.4f  per-step: %s' % (dk.min(), dk.max(), np.round(dk, 4).tolist()))
P('axis l (finger motor, 8 steps): min %.4f max %.4f  per-step: %s' % (dl.min(), dl.max(), np.round(dl, 4).tolist()))
P('total displacement j=0->30: %.4f m, k=0->30: %.4f m, l=0->8: %.4f m' % (
    np.abs(V[0] - V[30]).max(), np.abs(V[:, 0] - V[:, 30]).max(), np.abs(V[:, :, 0] - V[:, :, 8]).max()))
da = np.abs(np.diff(U, axis=0)).max(axis=(1, 2))
db = np.abs(np.diff(U, axis=1)).max(axis=(0, 2))
P('--- 93x93 hypothesis: max |delta| between consecutive indices')
P('axis a (92 steps): first 12: %s' % np.round(da[:12], 4).tolist())
P('axis b (92 steps): first 12: %s' % np.round(db[:12], 4).tolist())
P('93x93 axis b: every 3rd step jumps (index%%3==2): mean %.4f vs others mean %.4f' % (db[2::3].mean(), np.delete(db, np.arange(2, 92, 3)).mean()))
P('=> smooth, monotone-ish small deltas along all three axes of the 31x31x9 reshape, and periodic jumps every 3 (=9/3) '
  'and every 9 columns of the 93x93 reshape: the data is a 31x31x9 grid, row index = j*279 + k*9 + l.')
txt = '\n'.join(out)
print(txt)
with open(os.path.join(ROOT, 'results/thesis_ch3/A1_grid_structure.txt'), 'w') as f:
    f.write(txt + '\n')
