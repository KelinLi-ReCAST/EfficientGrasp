#!/usr/bin/env python3
"""Re-plot thesis Fig 3.7 (RL inverse-kinematics fingertip error vs. training epoch)
directly from the raw evaluation logs written by rl_inverse_kinematics/<gripper>/val.py.

Usage (from the repo root):
    python3 evaluation/thesis_ch3/plot_rl_error.py

Inputs : results/rl_ik_error/<gripper>/epoch*.npy   (one float per evaluation episode)
Outputs: results/thesis_ch3/C3_rl_error_table.md / .csv  (n, mean, median, std, p5, p95, min, max per file)
         results/thesis_ch3/fig_rl_error.pdf / .png

What the saved quantity is (val.py lines 85-95, same in all three gripper copies):
    distance += r                # r = env.step() reward = sum_j ||p_j - goal_j|| * 1000  (mm, 3 fingers)
    d = (pit >= env._max_episode_steps)   # episode always runs the full 100 steps
    avg_dist = distance / pit / 3        # -> time-average over the 100 steps of the per-finger mean distance [mm]
    contact_list.append(avg_dist)
so each entry is "mean per-finger Euclidean fingertip error in mm, averaged over all 100 steps of one episode"
(it includes the early steps before the policy has converged, so it is an upper bound on the final-step error).

No sorting, no smoothing, no band rescaling (unlike evaluation/simulation/RL_eval.py).
"""
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IN_DIR = os.path.join(ROOT, "results", "rl_ik_error")
OUT_DIR = os.path.join(ROOT, "results", "thesis_ch3")
os.makedirs(OUT_DIR, exist_ok=True)

GRIPPERS = [("ruth", "RUTH"), ("robotiq_3f", "Robotiq 3-Finger"), ("barrett", "BarrettHand")]
# validated default categorical palette, slots 1-3 (dataviz skill references/palette.md, light mode)
COLORS = {"ruth": "#2a78d6", "robotiq_3f": "#eb6834", "barrett": "#1baf7a"}

# the series the published figure used (evaluation/simulation/RL_eval.py lines 14-46):
PAPER_SERIES = {
    "ruth": ["epoch_0", "epoch_10", "epoch_100", "epoch_200", "epoch_500", "epoch_1000",
             "epoch_1500", "epoch_2000", "epoch_2500", "epoch_2885"],
    "robotiq_3f": ["epoch_00", "epoch_10", "epoch_100", "epoch_200", "epoch_500", "epoch_1000",
                   "epoch_1500", "epoch_2000", "epoch_2500", "epoch_2885"],
    "barrett": ["epoch_0", "epoch_10", "epoch_100", "epoch_200", "epoch_500", "epoch_1000",
                "epoch_1500", "epoch_2000", "epoch_2500", "epoch_2885"],
}


def epoch_of(name):
    """'epoch_2885' -> 2885, 'epoch1000' -> 1000, 'epoch_00' -> 0"""
    m = re.match(r"epoch_?(\d+)$", name)
    return int(m.group(1)) if m else None


def stats(a):
    return dict(n=a.size, mean=a.mean(), median=np.median(a), std=a.std(ddof=0),
                p5=np.percentile(a, 5), p95=np.percentile(a, 95), min=a.min(), max=a.max())


def main():
    rows = []
    data = {}
    for g, _ in GRIPPERS:
        data[g] = {}
        for f in sorted(glob.glob(os.path.join(IN_DIR, g, "epoch*.npy"))):
            name = os.path.splitext(os.path.basename(f))[0]
            a = np.load(f).astype(float)
            s = stats(a)
            s.update(gripper=g, file=name, epoch=epoch_of(name),
                     underscore=("_" in name), in_paper_series=(name in PAPER_SERIES[g]),
                     mtime=os.path.getmtime(f))
            data[g][name] = (a, s)
            rows.append(s)

    # ---- tables -------------------------------------------------------------
    cols = ["gripper", "file", "epoch", "in_paper_series", "n", "mean", "median", "std", "p5", "p95", "min", "max"]
    with open(os.path.join(OUT_DIR, "C3_rl_error_table.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["gripper"], r["epoch"], r["file"])):
            w.writerow([r[c] if not isinstance(r[c], float) else f"{r[c]:.3f}" for c in cols])
    with open(os.path.join(OUT_DIR, "C3_rl_error_table.md"), "w") as fh:
        fh.write("# C3 · RL-IK evaluation logs, per file (unit: mm, per-finger mean distance averaged over the 100 steps of an episode)\n\n")
        fh.write("Source: `results/rl_ik_error/<gripper>/*.npy`, written by `rl_inverse_kinematics/<gripper>/val.py` "
                 "(`avg_dist = distance/pit/3`, `np.save(...)`). Computed by `evaluation/thesis_ch3/plot_rl_error.py`.\n\n")
        fh.write("| gripper | file | epoch | used in paper fig | n | mean | median | std | p5 | p95 | min | max |\n")
        fh.write("|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(rows, key=lambda r: (r["gripper"], r["epoch"], r["file"])):
            fh.write(f"| {r['gripper']} | {r['file']} | {r['epoch']} | {'yes' if r['in_paper_series'] else 'no'} | {r['n']} | "
                     f"{r['mean']:.2f} | {r['median']:.2f} | {r['std']:.2f} | {r['p5']:.2f} | {r['p95']:.2f} | {r['min']:.2f} | {r['max']:.2f} |\n")
        # reductions
        fh.write("\n## Reduction of the mean error, first -> last epoch of the paper series\n\n")
        fh.write("| gripper | first file (mean) | last file (mean) | reduction of mean | reduction of median |\n|---|---|---|---:|---:|\n")
        for g, _ in GRIPPERS:
            first = PAPER_SERIES[g][0]
            last = PAPER_SERIES[g][-1]
            a0, s0 = data[g][first]
            a1, s1 = data[g][last]
            fh.write(f"| {g} | {first} ({s0['mean']:.2f}) | {last} ({s1['mean']:.2f}) | "
                     f"{100*(1-s1['mean']/s0['mean']):.1f} % | {100*(1-s1['median']/s0['median']):.1f} % |\n")
        fh.write("\n## What the old script (evaluation/simulation/RL_eval.py) would have plotted as the per-epoch mean\n\n")
        fh.write("It computes `sum(values)/100` (wrong for ruth/epoch_200 which has 20 values) and then sorts the 10 means "
                 "in descending order before plotting (`pppp.sort(reverse=True)`), so the x position no longer corresponds to the epoch.\n\n")
        for g, _ in GRIPPERS:
            means = [float(data[g][n][0].sum() / 100) for n in PAPER_SERIES[g]]
            fh.write(f"- {g}: sum/100 per file = {[round(m, 2) for m in means]}; after descending sort = "
                     f"{[round(m, 2) for m in sorted(means, reverse=True)]}  (last plotted point = {sorted(means)[0]:.2f} mm)\n")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    for g, label in GRIPPERS:
        names = [n for n in PAPER_SERIES[g] if n in data[g]]
        ep = np.array([max(epoch_of(n), 1) for n in names], float)  # epoch 0 -> 1 for log axis
        med = np.array([data[g][n][1]["median"] for n in names])
        p5 = np.array([data[g][n][1]["p5"] for n in names])
        p95 = np.array([data[g][n][1]["p95"] for n in names])
        ns = [data[g][n][1]["n"] for n in names]
        c = COLORS[g]
        ax.fill_between(ep, p5, p95, color=c, alpha=0.15, linewidth=0)
        ax.plot(ep, med, color=c, lw=2, marker="o", ms=5, label=label)
        for x, y, n in zip(ep, med, ns):
            if n != 100:
                ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, color=c)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000, 2885])
    ax.set_xticklabels(["0/1", "10", "100", "1000", "2885"])
    ax.set_xlabel("training epoch (checkpoint label of the log file)")
    ax.set_ylabel("per-finger fingertip error, episode average (mm)")
    ax.set_title("median with 5–95 % band over 100 evaluation episodes", fontsize=10)
    ax.axhline(5, color="0.5", lw=0.8, ls="--")
    ax.text(1.05, 5.4, "5 mm", fontsize=7, color="0.4")
    ax.grid(True, which="both", color="0.9", lw=0.6)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # right panel: categorical x — barrett underscore vs no-underscore, robotiq epoch_0/epoch_00/epoch_1
    ax = axes[1]
    items = [("robotiq_3f", "epoch_0"), ("robotiq_3f", "epoch_00"), ("robotiq_3f", "epoch_1"),
             ("barrett", "epoch_1000"), ("barrett", "epoch1000"), ("barrett", "epoch_1500"), ("barrett", "epoch1500")]
    for i, (g, name) in enumerate(items):
        a, s = data[g][name]
        c = COLORS[g]
        filled = s["in_paper_series"]
        ax.errorbar([i], [s["median"]], yerr=[[s["median"] - s["p5"]], [s["p95"] - s["median"]]],
                    fmt="o", color=c, mfc=(c if filled else "white"), mec=c, capsize=3, ms=7)
        ax.text(i, s["p95"] + 1.5, f"{s['mean']:.1f}", ha="center", fontsize=7, color="0.3")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([f"{'robotiq' if g == 'robotiq_3f' else 'barrett'}\n{n}" for g, n in items], fontsize=7)
    ax.set_ylim(0, None)
    ax.set_title("variant / duplicate log files\n(filled = used in the paper figure; number = mean)", fontsize=9)
    ax.grid(True, axis="y", color="0.9", lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_rl_error.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_rl_error.png"), dpi=200)

    # console summary
    for g, _ in GRIPPERS:
        last = PAPER_SERIES[g][-1]
        first = PAPER_SERIES[g][0]
        a0, s0 = data[g][first]
        a1, s1 = data[g][last]
        print(f"{g:11s} {first}: mean {s0['mean']:.2f} med {s0['median']:.2f} | {last}: mean {s1['mean']:.2f} "
              f"med {s1['median']:.2f} p95 {s1['p95']:.2f} max {s1['max']:.2f} | reduction(mean) {100*(1-s1['mean']/s0['mean']):.1f} % "
              f"| frac episodes <5mm: {np.mean(a1 < 5):.2f}, <15mm: {np.mean(a1 < 15):.2f}")


if __name__ == "__main__":
    sys.exit(main())
