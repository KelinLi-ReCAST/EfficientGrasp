#!/usr/bin/env python3
"""B3: read the GraphDef embedded in each TF event file (no TensorFlow) and
report (a) all TopKV2 nodes with the constant k they consume, (b) all
Placeholder nodes with their shapes, (c) AdamOptimizer learning-rate constants.

Usage: python3 evaluation/thesis_ch3/b3_graphdef_topk.py
Appends a section to results/thesis_ch3/B3_tf_event_runs.md
"""
import os, sys, glob, struct, collections
sys.path.insert(0, os.path.dirname(__file__))
from b3_parse_tf_events import read_records, parse_message, read_varint, ROOT, DEFAULT_LOGDIR, OUTDIR  # noqa


def parse_tensor_shape(buf):
    dims = []
    for fn, wt, v in parse_message(buf):
        if fn == 2:  # repeated Dim
            size = None
            for dfn, dwt, dv in parse_message(v):
                if dfn == 1:
                    size = dv if dv < (1 << 63) else dv - (1 << 64)
            dims.append(size)
    return dims


def parse_tensor_proto(buf):
    dtype = None; shape = []; ints = []; floats = []; content = b''
    for fn, wt, v in parse_message(buf):
        if fn == 1: dtype = v
        elif fn == 2: shape = parse_tensor_shape(v)
        elif fn == 4: content = v
        elif fn == 5:  # float_val packed or single
            if wt == 2:
                floats += list(struct.unpack('<%df' % (len(v) // 4), v))
            else:
                floats.append(struct.unpack('<f', v)[0])
        elif fn == 7:  # int_val
            if wt == 2:
                pos = 0
                while pos < len(v):
                    x, pos = read_varint(v, pos); ints.append(x)
            else:
                ints.append(v)
    if content and not ints and not floats:
        if dtype == 3 and len(content) % 4 == 0:  # DT_INT32
            ints = list(struct.unpack('<%di' % (len(content) // 4), content))
        elif dtype == 1 and len(content) % 4 == 0:  # DT_FLOAT
            floats = list(struct.unpack('<%df' % (len(content) // 4), content))
    return dtype, shape, ints, floats


def parse_graph(buf):
    nodes = {}
    for fn, wt, v in parse_message(buf):
        if fn != 1:
            continue
        name = op = None; inputs = []; attrs = {}
        for nfn, nwt, nv in parse_message(v):
            if nfn == 1: name = nv.decode()
            elif nfn == 2: op = nv.decode()
            elif nfn == 3: inputs.append(nv.decode())
            elif nfn == 5:
                k = None; val = None
                for afn, awt, av in parse_message(nv):
                    if afn == 1: k = av.decode()
                    elif afn == 2: val = av
                attrs[k] = val
        nodes[name] = (op, inputs, attrs)
    return nodes


def attr_value(av, kind):
    """kind: 'tensor' | 'shape' | 'f' """
    for fn, wt, v in parse_message(av):
        if kind == 'tensor' and fn == 8: return parse_tensor_proto(v)
        if kind == 'shape' and fn == 7: return parse_tensor_shape(v)
    return None


def main():
    files = sorted(glob.glob(os.path.join(DEFAULT_LOGDIR, 'events.out.tfevents.*')))
    lines = ['', '## GraphDef constants embedded in each event file (b3_graphdef_topk.py)', '']
    for f in files:
        graph = None
        for rec in read_records(f):
            for fn, wt, v in parse_message(rec):
                if fn == 4:
                    graph = v; break
            if graph is not None:
                break
        if graph is None:
            lines.append('- %s: no graph_def' % os.path.basename(f)); continue
        nodes = parse_graph(graph)
        lines.append('### %s (%d nodes)' % (os.path.basename(f), len(nodes)))
        lines.append('')
        lines.append('TopKV2 nodes and their k:')
        for name, (op, inputs, attrs) in nodes.items():
            if op == 'TopKV2':
                kin = inputs[1].split(':')[0] if len(inputs) > 1 else None
                kval = None
                if kin in nodes and nodes[kin][0] == 'Const' and 'value' in nodes[kin][2]:
                    kval = attr_value(nodes[kin][2]['value'], 'tensor')[2]
                lines.append('- %s  input=%s  k=%s' % (name, inputs[0], kval))
        lines.append('')
        lines.append('Placeholder shapes:')
        for name, (op, inputs, attrs) in nodes.items():
            if op == 'Placeholder' and 'shape' in attrs:
                lines.append('- %s %s' % (name, attr_value(attrs['shape'], 'shape')))
        lines.append('')
        lines.append('Adam learning-rate constants (nodes named */learning_rate):')
        for name, (op, inputs, attrs) in nodes.items():
            if op == 'Const' and name.endswith('learning_rate') and 'value' in attrs:
                lines.append('- %s = %s' % (name, attr_value(attrs['value'], 'tensor')[3]))
        lines.append('')
        # extra_feat first conv weight shape (gripper feature width)
        for name, (op, inputs, attrs) in nodes.items():
            if op in ('VariableV2', 'Variable') and name.endswith('pointnet2/Conv1D/W') and 'shape' in attrs:
                lines.append('- variable %s shape %s' % (name, attr_value(attrs['shape'], 'shape')))
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(OUTDIR, 'B3_tf_event_runs.md'), 'a') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
