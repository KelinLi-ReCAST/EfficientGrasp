#!/usr/bin/env python3
"""Plot the RUTH fingertip workspace used as the EfficientGrasp gripper representation.

Usage:
    python3 make_fig3_workspace.py /path/to/contact_points.npy [out_prefix]

Input is an (L, 9) array: each row is one motor configuration (RUTH: 31 x 31 palm x 9 finger), columns are the
world-frame xyz of fingertip 1, 2, 3. Prints "L rows, N distinct rows" and writes
<out_prefix>.pdf and <out_prefix>.png (default prefix: fig_workspace_ruth).

Left panel: top-down view (x-y, the palm plane), one colour per finger.
Right panel: 3-D view of the same points. Units are converted to millimetres and
re-centred on the centroid of all fingertip samples so the axes read as hand size.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # finger 1, 2, 3
LABELS = ["Finger 1", "Finger 2", "Finger 3"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "fig_workspace_ruth"

    ws = np.load(path)
    assert ws.ndim == 2 and ws.shape[1] % 3 == 0, f"expected (L, 3N), got {ws.shape}"
    n_fingers = ws.shape[1] // 3
    distinct = np.unique(ws, axis=0).shape[0]
    print(f"{ws.shape[0]} rows, {distinct} distinct rows")

    # mm, centred on the overall centroid
    pts = ws.reshape(-1, 3)
    centre = pts.mean(axis=0)
    fingers = [(ws[:, 3 * i:3 * i + 3] - centre) * 1000.0 for i in range(n_fingers)]

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 3.2))

    # --- top-down view (palm plane) -----------------------------------------
    ax = fig.add_subplot(1, 2, 1)
    for f, c, lab in zip(fingers, COLORS, LABELS):
        ax.scatter(f[:, 0], f[:, 1], s=1.5, c=c, alpha=0.5, linewidths=0, label=lab, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Top view (palm plane)")
    ax.grid(True, lw=0.3, alpha=0.5)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, markerscale=6, frameon=False, handletextpad=0.2, columnspacing=1.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    # --- 3-D view -----------------------------------------------------------
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for f, c in zip(fingers, COLORS):
        ax3.scatter(f[:, 0], f[:, 1], f[:, 2], s=1.0, c=c, alpha=0.4, linewidths=0, rasterized=True)
    allp = np.vstack(fingers)
    span = (allp.max(0) - allp.min(0))
    mid = (allp.max(0) + allp.min(0)) / 2
    r = span.max() / 2
    ax3.set_xlim(mid[0] - r, mid[0] + r)
    ax3.set_ylim(mid[1] - r, mid[1] + r)
    ax3.set_zlim(mid[2] - r, mid[2] + r)
    ax3.set_xlabel("x (mm)", labelpad=2)
    ax3.set_ylabel("y (mm)", labelpad=2)
    ax3.set_zlabel("z (mm)", labelpad=2)
    ax3.set_title("3-D view")
    ax3.view_init(elev=28, azim=-55)
    ax3.tick_params(pad=1)

    fig.suptitle(f"RUTH fingertip workspace: {ws.shape[0]} motor configurations "
                 f"(31 x 31 palm motor steps x 9 finger-tendon steps), {n_fingers} fingers", y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{prefix}.{ext}", dpi=300, bbox_inches="tight")
        print("wrote", f"{prefix}.{ext}")


if __name__ == "__main__":
    main()
