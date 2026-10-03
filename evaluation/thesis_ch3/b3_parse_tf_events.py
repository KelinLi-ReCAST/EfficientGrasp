#!/usr/bin/env python3
"""B3: parse TensorFlow event files WITHOUT tensorflow/tensorboard.

Minimal TFRecord reader + protobuf wire-format decoder for
  Event{1: wall_time(double), 2: step(int64), 3: file_version(string),
        4: graph_def(bytes), 5: summary(Summary), 6: log_message, 7: session_log,
        8: tagged_run_metadata, 9: meta_graph_def}
  Summary{1: repeated Value}
  Value{1: tag(string), 2: simple_value(float), 3: obsolete_old_style_histogram,
        4: image, 5: histo, 6: audio, 8: tensor}

Writes results/thesis_ch3/B3_tf_event_runs.md and per-run CSV of scalars.
Usage: python3 evaluation/thesis_ch3/b3_parse_tf_events.py [logdir ...]
"""
import os, sys, struct, glob, datetime, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_LOGDIR = os.path.join(ROOT, 'contact_point_selection', 'UniGrasp', 'point_set_selection', 'logs')
OUTDIR = os.path.join(ROOT, 'results', 'thesis_ch3')


def read_records(path):
    """Yield raw record bytes from a TFRecord file (skips CRC checks)."""
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return
            (length,) = struct.unpack('<Q', hdr)
            f.read(4)  # masked crc of length
            data = f.read(length)
            if len(data) < length:
                return
            f.read(4)  # masked crc of data
            yield data


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


def parse_message(buf):
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


def parse_event(rec):
    ev = {'wall_time': None, 'step': None, 'values': [], 'kinds': set()}
    for fn, wt, v in parse_message(rec):
        if fn == 1 and wt == 1:
            ev['wall_time'] = struct.unpack('<d', v)[0]
        elif fn == 2 and wt == 0:
            ev['step'] = v
        elif fn == 3:
            ev['kinds'].add('file_version'); ev['file_version'] = v.decode('utf-8', 'replace')
        elif fn == 4:
            ev['kinds'].add('graph_def'); ev['graph_bytes'] = len(v)
        elif fn == 5:
            ev['kinds'].add('summary')
            for sfn, swt, sv in parse_message(v):
                if sfn == 1 and swt == 2:
                    tag, simple, other = None, None, []
                    for vfn, vwt, vv in parse_message(sv):
                        if vfn == 1:
                            tag = vv.decode('utf-8', 'replace')
                        elif vfn == 2 and vwt == 5:
                            simple = struct.unpack('<f', vv)[0]
                        else:
                            other.append(vfn)
                    ev['values'].append((tag, simple, other))
        elif fn == 6:
            ev['kinds'].add('log_message')
        elif fn == 7:
            ev['kinds'].add('session_log')
        elif fn == 9:
            ev['kinds'].add('meta_graph_def'); ev['meta_graph_bytes'] = len(v)
    return ev


def summarize(path):
    info = collections.OrderedDict()
    info['file'] = os.path.basename(path)
    info['size_bytes'] = os.path.getsize(path)
    n = 0
    kinds = collections.Counter()
    tags = collections.OrderedDict()
    steps = []
    wall = []
    fv = None
    graph_bytes = []
    for rec in read_records(path):
        n += 1
        ev = parse_event(rec)
        for k in ev['kinds']:
            kinds[k] += 1
        if 'file_version' in ev:
            fv = ev['file_version']
        if 'graph_bytes' in ev:
            graph_bytes.append(ev['graph_bytes'])
        if ev['step'] is not None:
            steps.append(ev['step'])
        if ev['wall_time'] is not None:
            wall.append(ev['wall_time'])
        for tag, simple, other in ev['values']:
            tags.setdefault(tag, []).append((ev['step'], ev['wall_time'], simple, other))
    info['n_records'] = n
    info['record_kinds'] = dict(kinds)
    info['file_version'] = fv
    info['graph_def_bytes'] = graph_bytes
    info['steps_min_max'] = (min(steps), max(steps)) if steps else None
    info['n_distinct_steps'] = len(set(steps))
    if wall:
        info['wall_first'] = datetime.datetime.fromtimestamp(min(wall)).isoformat()
        info['wall_last'] = datetime.datetime.fromtimestamp(max(wall)).isoformat()
    info['tags'] = tags
    return info


def main():
    logdirs = sys.argv[1:] or [DEFAULT_LOGDIR]
    os.makedirs(OUTDIR, exist_ok=True)
    lines = ['# B3 TF event-file inventory', '',
             'Generated by `python3 evaluation/thesis_ch3/b3_parse_tf_events.py` (pure-python TFRecord/protobuf decoder, no TensorFlow).', '']
    for ld in logdirs:
        lines.append('## %s' % os.path.relpath(ld, ROOT) if ld.startswith(ROOT) else '## %s' % ld)
        lines.append('')
        files = sorted(glob.glob(os.path.join(ld, '**', 'events.out.tfevents.*'), recursive=True))
        if not files:
            lines.append('(no event files)')
            continue
        lines.append('| file | size (MB) | records | record kinds | file_version | graph_def bytes | steps (min..max, #distinct) | wall-clock first..last | scalar tags (#points) |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for f in files:
            info = summarize(f)
            tagdesc = '; '.join('%s (%d)' % (t, len(v)) for t, v in info['tags'].items()) or 'NONE'
            lines.append('| %s | %.1f | %d | %s | %s | %s | %s / %d | %s .. %s | %s |' % (
                info['file'], info['size_bytes'] / 1e6, info['n_records'], info['record_kinds'],
                info['file_version'], info['graph_def_bytes'], info['steps_min_max'], info['n_distinct_steps'],
                info.get('wall_first'), info.get('wall_last'), tagdesc))
            # dump scalars
            for t, v in info['tags'].items():
                safe = t.replace('/', '_')
                csv = os.path.join(OUTDIR, 'B3_scalars_%s_%s.csv' % (info['file'].split('.')[-2], safe))
                with open(csv, 'w') as fh:
                    fh.write('step,wall_time,simple_value,other_fields\n')
                    for s, w, sv, o in v:
                        fh.write('%s,%s,%s,%s\n' % (s, w, sv, '|'.join(map(str, o))))
        lines.append('')
    txt = '\n'.join(lines)
    print(txt)
    with open(os.path.join(OUTDIR, 'B3_tf_event_runs.md'), 'w') as fh:
        fh.write(txt + '\n')


if __name__ == '__main__':
    main()
