#!/usr/bin/env python3
"""B5: try to reproduce the "9.85 %" aggregate improvement from thesis Table 3.2.

Hard-codes Table 3.2 (Top-1 / Top-10 in %, stages S1..S3) and prints many
candidate aggregates of E (EfficientGrasp) vs U (UniGrasp) / I (insufficient
inputs).  Also loads results/pssn_accuracy/*.mat|*.npy and checks whether any
Table 3.2 cell is literally contained there (already done in b2).

Usage: python3 evaluation/thesis_ch3/b5_table32_aggregates.py
Writes results/thesis_ch3/B5_table32_aggregates.md
"""
import os, itertools, collections
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'results', 'thesis_ch3', 'B5_table32_aggregates.md')
TARGET = 9.85
TOL = 0.05

# rows: gripper -> {'E': [(t1,t10) S1..S3], 'U': ..., 'I': ...}
T = {
    'Robotiq-3F': {'E': [(99.3, 100), (97.5, 98.9), (88.7, 96.1)],
                   'U': [(91.9, 99.6), (83.4, 90.8), (76.0, 86.2)]},
    'Kinova-3F': {'E': [(97.9, 100), (87.6, 97.2), (75.6, 89.8)],
                  'U': [(86.9, 98.2), (66.4, 80.2), (54.8, 70.7)]},
    'BarrettHand': {'E': [(100, 100), (96.5, 99.7), (91.9, 96.5)],
                    'U': [(94.3, 98.9), (86.9, 91.5), (77.7, 87.6)],
                    'I': [(92.2, 98.6), (85.9, 90.8), (77.4, 86.9)]},
}
G = list(T)


def arr(kind, grippers=G, stages=(0, 1, 2), metrics=(0, 1)):
    out = []
    for g in grippers:
        if kind not in T[g]:
            continue
        for s in stages:
            for m in metrics:
                out.append(T[g][kind][s][m])
    return np.array(out, dtype=float)


cands = collections.OrderedDict()


def add(name, val):
    cands[name] = float(val)


# --- absolute differences E - U
for label, kw in [('all 18 cells', {}),
                  ('Top-1 only (9)', {'metrics': (0,)}),
                  ('Top-10 only (9)', {'metrics': (1,)}),
                  ('Stage-1 only (6)', {'stages': (0,)}),
                  ('Stage-2 only (6)', {'stages': (1,)}),
                  ('Stage-3 only (6)', {'stages': (2,)}),
                  ('Stage-3 Top-1 (3)', {'stages': (2,), 'metrics': (0,)}),
                  ('Stage-3 Top-10 (3)', {'stages': (2,), 'metrics': (1,)}),
                  ('Stage-2+3 (12)', {'stages': (1, 2)}),
                  ('Stage-2+3 Top-1 (6)', {'stages': (1, 2), 'metrics': (0,)}),
                  ('Stage-2+3 Top-10 (6)', {'stages': (1, 2), 'metrics': (1,)})]:
    e, u = arr('E', **kw), arr('U', **kw)
    add('mean(E-U) %s' % label, np.mean(e - u))
    add('mean((E-U)/U) %s [relative %%]' % label, 100 * np.mean((e - u) / u))
    add('(mean E - mean U)/mean U %s [relative %%]' % label, 100 * (e.mean() - u.mean()) / u.mean())
    add('median(E-U) %s' % label, np.median(e - u))

for g in G:
    for label, kw in [('all 6', {}), ('Top-1 (3)', {'metrics': (0,)}), ('Top-10 (3)', {'metrics': (1,)}),
                      ('Stage-3 (2)', {'stages': (2,)})]:
        e, u = arr('E', [g], **kw), arr('U', [g], **kw)
        add('%s mean(E-U) %s' % (g, label), np.mean(e - u))
        add('%s mean((E-U)/U) %s [rel %%]' % (g, label), 100 * np.mean((e - u) / u))

# --- using I (insufficient input) instead of U for BarrettHand
e_b, i_b, u_b = arr('E', ['BarrettHand']), arr('I', ['BarrettHand']), arr('U', ['BarrettHand'])
add('BarrettHand mean(E-I) all 6', np.mean(e_b - i_b))
add('BarrettHand mean(U-I) all 6', np.mean(u_b - i_b))
add('BarrettHand mean(E-I) Top-1', np.mean(arr('E', ['BarrettHand'], metrics=(0,)) - arr('I', ['BarrettHand'], metrics=(0,))))
add('BarrettHand mean((E-I)/I) all 6 [rel %]', 100 * np.mean((e_b - i_b) / i_b))
# all 18 with I substituted for Barrett U
e_all = arr('E'); u_sub = np.concatenate([arr('U', ['Robotiq-3F']), arr('U', ['Kinova-3F']), i_b])
add('mean(E-U*) all 18, U*=I for BarrettHand', np.mean(e_all - u_sub))
add('mean((E-U*)/U*) all 18, U*=I for BarrettHand [rel %]', 100 * np.mean((e_all - u_sub) / u_sub))
# 24 cells: E vs U (18) plus E vs I (6)
add('mean over 24 diffs (E-U 18 + E-I 6)', np.mean(np.concatenate([e_all - arr('U'), e_b - i_b])))

# --- stage-wise means first, then average (3 numbers)
for m, mn in [(0, 'Top-1'), (1, 'Top-10'), ((0, 1), 'both')]:
    ms = (m,) if isinstance(m, int) else m
    stage_means = [np.mean(arr('E', stages=(s,), metrics=ms) - arr('U', stages=(s,), metrics=ms)) for s in range(3)]
    add('mean of per-stage mean diffs %s' % mn, np.mean(stage_means))
    add('per-stage mean diffs %s = %s' % (mn, np.round(stage_means, 3)), np.nan)

# --- the same candidates with the EXACT fractions k/283 (B2 shows every cell is k/283)
def exact(x):
    return 100.0 * np.round(np.asarray(x) / 100.0 * 283) / 283
for label, kw in [('all 18', {}), ('Top-1 (9)', {'metrics': (0,)}), ('Top-10 (9)', {'metrics': (1,)}), ('Stage-3 (6)', {'stages': (2,)})]:
    e, u = exact(arr('E', **kw)), exact(arr('U', **kw))
    add('[exact k/283] mean(E-U) %s' % label, np.mean(e - u))
    add('[exact k/283] mean((E-U)/U) %s [rel %%]' % label, 100 * np.mean((e - u) / u))
for g in G:
    for label, kw in [('all 6', {}), ('Top-1 (3)', {'metrics': (0,)})]:
        e, u = exact(arr('E', [g], **kw)), exact(arr('U', [g], **kw))
        add('[exact k/283] %s mean(E-U) %s' % (g, label), np.mean(e - u))

# --- geometric / other
add('mean(E)-mean(U) all 18 (same as mean diff)', arr('E').mean() - arr('U').mean())
add('mean E all 18', arr('E').mean()); add('mean U all 18', arr('U').mean())
add('mean E Top-1', arr('E', metrics=(0,)).mean()); add('mean U Top-1', arr('U', metrics=(0,)).mean())
add('mean E Stage-3', arr('E', stages=(2,)).mean()); add('mean U Stage-3', arr('U', stages=(2,)).mean())
# error-rate reductions
e, u = arr('E'), arr('U')
add('mean((100-U)-(100-E)) = mean diff (identity)', np.mean((100 - u) - (100 - e)))
add('relative error reduction mean(1-(100-E)/(100-U)) [%%] (cells with U<100)', 100 * np.mean([1 - (100 - a) / (100 - b) for a, b in zip(e, u) if b < 100]))
# product-of-stage style: final stage-3 only vs cumulative
# exhaustive search over subsets of the 18 differences is too large (2^18=262k is fine actually)
d = e - u
subset_hits = []
for r in range(1, 19):
    for comb in itertools.combinations(range(18), r):
        v = d[list(comb)].mean()
        if abs(v - TARGET) <= TOL:
            subset_hits.append((r, comb, v))
    if len(subset_hits) > 30:
        break

lines = ['# B5: candidate aggregates for the "9.85 %" figure (Table 3.2)', '',
         'Generated by `python3 evaluation/thesis_ch3/b5_table32_aggregates.py`', '',
         'Target: %.2f +- %.2f' % (TARGET, TOL), '',
         '| candidate | value | hit |', '|---|---|---|']
hits = []
for k, v in cands.items():
    hit = (not np.isnan(v)) and abs(v - TARGET) <= TOL
    if hit:
        hits.append(k)
    lines.append('| %s | %s | %s |' % (k, ('%.3f' % v) if not np.isnan(v) else '', 'HIT' if hit else ''))
lines.append('')
lines.append('## Result')
lines.append('')
if hits:
    lines.append('Candidates within +-%.2f of %.2f: %s' % (TOL, TARGET, hits))
else:
    lines.append('无法复现: no named candidate aggregate equals %.2f +- %.2f.' % (TARGET, TOL))
vals = [(k, v) for k, v in cands.items() if not np.isnan(v)]
vals.sort(key=lambda kv: abs(kv[1] - TARGET))
lines.append('')
lines.append('Closest named candidates:')
for k, v in vals[:8]:
    lines.append('- %s = %.3f (delta %.3f)' % (k, v, v - TARGET))
lines.append('')
lines.append('Brute-force: subsets of the 18 E-U differences whose mean is within tolerance (first %d shown, ordered by subset size):' % min(len(subset_hits), 30))
cell_names = ['%s S%d %s' % (g, s + 1, m) for g in G for s in range(3) for m in ('T1', 'T10')]
for r, comb, v in subset_hits[:30]:
    lines.append('- n=%d mean=%.3f: %s' % (r, v, ', '.join(cell_names[i] for i in comb)))
if not subset_hits:
    lines.append('- none')
lines.append('')
lines.append('The 18 differences E-U (gripper, stage, metric):')
for n, dv in zip(cell_names, d):
    lines.append('- %s: %.1f' % (n, dv))
txt = '\n'.join(lines)
print(txt)
with open(OUT, 'w') as fh:
    fh.write(txt + '\n')
print('written', OUT)
