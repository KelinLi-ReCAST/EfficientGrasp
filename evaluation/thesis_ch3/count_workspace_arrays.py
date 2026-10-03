#!/usr/bin/env python3
"""A3(a): 统计参数化夹爪工作空间数组 (ws*.npy) 的行数分布.

对每个目录, 以 mmap_mode='r' 打开所有 ws*.npy, 只读取 .shape, 统计:
  文件总数 / 行数分布 / 空数组 (0,9) 数量 / 非空数量 / 文件名编号范围 / mtime 范围.
输出: results/thesis_ch3/A3_workspace_arrays_stats.md 与 .csv
用法: python3 evaluation/thesis_ch3/count_workspace_arrays.py
"""
import csv
import glob
import os
import re
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = {
    'WorkspaceArrays': os.path.join(ROOT, 'gripper_representation/data/WorkspaceArrays'),
    'coupRotWorkspaceArrays': os.path.join(ROOT, 'gripper_representation/feature_extraction/coupRotWorkspaceArrays'),
    'fourbarWorkspaceArrays': os.path.join(ROOT, 'gripper_representation/feature_extraction/fourbarWorkspaceArrays'),
    'singRotWorkspaceArrays': os.path.join(ROOT, 'gripper_representation/feature_extraction/singRotWorkspaceArrays'),
    'ruthArrays': os.path.join(ROOT, 'gripper_representation/feature_extraction/ruthArrays'),
}
OUT_DIR = os.path.join(ROOT, 'results/thesis_ch3')
os.makedirs(OUT_DIR, exist_ok=True)

rows_md = ['| 目录 | 路径 | 文件数(ws*.npy) | 编号范围 | 空数组(0,9) | 非空 | 行数分布 {rows: n_files} | 列数 | mtime范围 |',
           '|---|---|---|---|---|---|---|---|---|']
rows_csv = []
for name, d in DIRS.items():
    if not os.path.isdir(d):
        rows_md.append('| %s | %s | 未找到 | - | - | - | - | - | - |' % (name, d))
        rows_csv.append([name, d, 'NOT FOUND', '', '', '', '', '', ''])
        print(name, '未找到:', d)
        continue
    files = sorted(glob.glob(os.path.join(d, 'ws*.npy')))
    all_files = os.listdir(d)
    cnt = Counter()
    cols = Counter()
    mt = []
    ids = []
    for f in files:
        a = np.load(f, mmap_mode='r')
        cnt[a.shape[0]] += 1
        cols[a.shape[1:] if a.ndim > 1 else ('ndim%d' % a.ndim,)] += 1
        mt.append(os.path.getmtime(f))
        m = re.match(r'ws(\d+)\.npy', os.path.basename(f))
        if m:
            ids.append(int(m.group(1)))
    n_empty = cnt.get(0, 0)
    n_nonempty = len(files) - n_empty
    dist = ', '.join('%d: %d' % (k, v) for k, v in sorted(cnt.items()))
    tfmt = lambda t: time.strftime('%Y-%m-%d %H:%M', time.localtime(t))
    mtr = '%s ~ %s' % (tfmt(min(mt)), tfmt(max(mt))) if mt else '-'
    idr = '%d..%d (缺号 %d)' % (min(ids), max(ids), max(ids) - min(ids) + 1 - len(ids)) if ids else '-'
    print('%s: %d files (dir has %d entries), empty=%d, nonempty=%d, rows=%s, cols=%s, ids=%s, mtime=%s' %
          (name, len(files), len(all_files), n_empty, n_nonempty, dist, dict(cols), idr, mtr))
    rows_md.append('| %s | %s | %d | %s | %d | %d | %s | %s | %s |' %
                   (name, os.path.relpath(d, ROOT), len(files), idr, n_empty, n_nonempty, dist, dict(cols), mtr))
    rows_csv.append([name, os.path.relpath(d, ROOT), len(files), idr, n_empty, n_nonempty, dist, str(dict(cols)), mtr])

with open(os.path.join(OUT_DIR, 'A3_workspace_arrays_stats.md'), 'w') as f:
    f.write('# A3(a) 工作空间数组统计\n\n生成命令: `python3 evaluation/thesis_ch3/count_workspace_arrays.py`\n\n')
    f.write('\n'.join(rows_md) + '\n')
with open(os.path.join(OUT_DIR, 'A3_workspace_arrays_stats.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dir', 'path', 'n_files', 'id_range', 'n_empty', 'n_nonempty', 'row_count_distribution', 'cols', 'mtime_range'])
    w.writerows(rows_csv)
print('written', os.path.join(OUT_DIR, 'A3_workspace_arrays_stats.md'))
