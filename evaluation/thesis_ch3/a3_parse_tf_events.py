#!/usr/bin/env python3
"""A3(c): 不依赖 tensorflow/tensorboard, 用最小 TFRecord+protobuf 解码器读取
gripper_representation/feature_extraction/logs/ 下的 events.out.tfevents.* 文件.

Event proto 字段: 1 wall_time(double) 2 step(int64) 3 file_version(string)
                  4 graph_def(bytes) 5 summary(Summary) 6 log_message 7 session_log
                  8 tagged_run_metadata 9 meta_graph_def
Summary.Value:    1 tag(string) 2 simple_value(float) 3 obsolete_old_style_histogram
                  4 image 5 histo 6 audio 7 tensor 8 tensor(新) 9 metadata
输出: results/thesis_ch3/A3_tf_events.csv (每个文件一行) + 终端摘要.
用法: python3 evaluation/thesis_ch3/a3_parse_tf_events.py
"""
import csv
import glob
import os
import struct
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, 'gripper_representation/feature_extraction/logs')
OUT = os.path.join(ROOT, 'results/thesis_ch3/A3_tf_events.csv')


def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def parse_fields(buf):
    """Return list of (field_number, wire_type, value) for one protobuf message."""
    pos = 0
    out = []
    n = len(buf)
    while pos < n:
        key, pos = read_varint(buf, pos)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = read_varint(buf, pos)
        elif wt == 1:
            v = buf[pos:pos + 8]; pos += 8
        elif wt == 2:
            ln, pos = read_varint(buf, pos)
            v = buf[pos:pos + ln]; pos += ln
        elif wt == 5:
            v = buf[pos:pos + 4]; pos += 4
        else:
            raise ValueError('unsupported wire type %d' % wt)
        out.append((fn, wt, v))
    return out


def tfrecords(path):
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            (ln,) = struct.unpack('<Q', hdr)
            f.read(4)  # masked crc of length
            data = f.read(ln)
            f.read(4)  # masked crc of data
            yield data


def count_nodes(graph_def_bytes):
    # GraphDef: field 1 node (repeated NodeDef); NodeDef: 1 name, 2 op
    nodes = 0
    ops = Counter()
    names = []
    for fn, wt, v in parse_fields(graph_def_bytes):
        if fn == 1:
            nodes += 1
            name = op = ''
            for f2, w2, v2 in parse_fields(v):
                if f2 == 1: name = v2.decode('utf-8', 'replace')
                elif f2 == 2: op = v2.decode('utf-8', 'replace')
            ops[op] += 1
            names.append(name)
    return nodes, ops, names


rows = []
files = sorted(glob.glob(os.path.join(LOG_DIR, 'events.out.tfevents.*')))
for p in files:
    n_events = 0
    n_graph = 0
    scalars = Counter()
    steps = []
    wall = []
    file_version = ''
    graph_nodes = 0
    graph_ops = Counter()
    has_optimizer = False
    has_hist = 0
    for rec in tfrecords(p):
        n_events += 1
        for fn, wt, v in parse_fields(rec):
            if fn == 1:
                wall.append(struct.unpack('<d', v)[0])
            elif fn == 2:
                steps.append(v)
            elif fn == 3:
                file_version = v.decode()
            elif fn == 4:
                n_graph += 1
                graph_nodes, graph_ops, names = count_nodes(v)
                has_optimizer = any(('Adam' in nm or 'gradients' in nm) for nm in names)
            elif fn == 5:
                for f2, w2, v2 in parse_fields(v):
                    if f2 == 1:  # Value
                        tag = ''
                        kind = ''
                        for f3, w3, v3 in parse_fields(v2):
                            if f3 == 1: tag = v3.decode()
                            elif f3 == 2: kind = 'simple_value'
                            elif f3 == 5: kind = 'histo'; has_hist += 1
                            elif f3 in (7, 8): kind = 'tensor'
                        scalars[(tag, kind)] += 1
    ts = int(os.path.basename(p).split('.')[3])
    row = dict(file=os.path.basename(p), size=os.path.getsize(p),
               name_ts=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts)),
               mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(p))),
               n_events=n_events, n_graph_events=n_graph, graph_nodes=graph_nodes,
               has_adam_or_gradients=has_optimizer, file_version=file_version,
               n_summary_values=sum(scalars.values()), summary_tags=';'.join('%s[%s]x%d' % (k[0], k[1], c) for k, c in scalars.items()),
               steps_min=min(steps) if steps else '', steps_max=max(steps) if steps else '',
               wall_first=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wall[0])) if wall else '',
               wall_last=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wall[-1])) if wall else '',
               n_placeholders=graph_ops.get('Placeholder', 0), n_variables=graph_ops.get('VariableV2', 0))
    rows.append(row)
    print('%s size=%d events=%d graph=%d nodes=%d adam=%s summaries=%d steps=%s..%s wall=%s..%s' % (
        row['file'][20:30], row['size'], n_events, n_graph, graph_nodes, has_optimizer, row['n_summary_values'],
        row['steps_min'], row['steps_max'], row['wall_first'], row['wall_last']))

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('written', OUT)
sizes = Counter((r['size'], r['graph_nodes'], r['has_adam_or_gradients']) for r in rows)
print('size/nodes/adam variants:', sizes)
