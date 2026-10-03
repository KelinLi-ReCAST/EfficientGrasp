#!/usr/bin/env python3
"""C4 · Run the RUTH numeric/analytic IK (ruth_grasping_kinematics.ruthModel.compute_ruth_pose)
on reachable fingertip triplets and tabulate convergence, next to the RL-IK errors from C3.

Usage (repo root):
    python3 evaluation/thesis_ch3/c4_numeric_ik_vs_rl.py

Inputs
  gripper_representation/feature_extraction/ruth_grasping_kinematics.py   (imported, NOT modified)
  gripper_representation/ruth_workspace_sampling/contact_points.npy      (8649 x 9, m, world frame)
  rl_inverse_kinematics/ruth/contact_points.npy                          (485376 x 9, m, world frame; RL training targets)
  rl_inverse_kinematics/ruth/contact_list.npy                            (21 row indices actually used by the RL env)
  results/rl_ik_error/ruth/epoch_2885.npy                                (RL final-epoch per-episode error, mm)
Output
  results/thesis_ch3/C4_rl_vs_numeric_ik.md  (+ C4_numeric_ik_raw.csv)

The solver expects the three contact points in millimetres (its link lengths are l1=28.5, l2=70,
fingerLength=100; the historical callers do `rM.compute_ruth_pose(cp1*1000, cp2*1000, cp3*1000)`,
e.g. rl_inverse_kinematics/ruth/gym/envs/kelin/grasping_with_RUTH_2.py:196). It builds its own
frame from the contact-point plane, so no hand-frame transform is needed.

Instrumentation without editing the solver: the module-level names `fsolve` and `max` used inside
compute_ruth_pose are replaced in the imported module's namespace by recording wrappers, so that we
can read fsolve's `ier`/`fvec` and the three finger-base-to-contact distances that feed `hori_bend`.
"""
import csv
import importlib.util
import os
import sys
import warnings

import numpy as np
import scipy.optimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "results", "thesis_ch3")
os.makedirs(OUT_DIR, exist_ok=True)
SOLVER = os.path.join(ROOT, "gripper_representation", "feature_extraction", "ruth_grasping_kinematics.py")
N_SAMPLE = 1000
SEED = 0

# ---------------------------------------------------------------- import + instrument
spec = importlib.util.spec_from_file_location("ruth_grasping_kinematics", SOLVER)
rgk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rgk)

_rec = {}


def _fsolve_rec(func, x0, args=()):
    if not isinstance(args, tuple):
        args = (args,)
    x, info, ier, mesg = scipy.optimize.fsolve(func, x0, args=args, full_output=True)
    _rec["ier"] = ier
    _rec["fvec"] = float(np.abs(info["fvec"]).max())
    _rec["nfev"] = info["nfev"]
    _rec["mesg"] = mesg
    return x


def _max_rec(*args, **k):
    # compute_ruth_pose calls max(a, b) (line 94) and max([d1, d2, d3]) (line 190, the hori_bend reaches)
    if len(args) == 1 and isinstance(args[0], (list, tuple)) and len(args[0]) == 3:
        _rec["reaches"] = [float(v) for v in args[0]]
    return max(*args, **k)


rgk.fsolve = _fsolve_rec
rgk.max = _max_rec  # module globals shadow the builtin inside compute_ruth_pose
rM = rgk.ruthModel()


def solve_one(row_m):
    """row_m: 9 floats in metres. Returns dict."""
    cps = np.asarray(row_m, float).reshape(3, 3) * 1000.0  # -> mm
    _rec.clear()
    out = dict(exception="", nan_out=False, ier=-1, fvec=np.nan, theta_rad=np.nan,
               reach_max=np.nan, reach_min=np.nan, reach_spread=np.nan,
               tcp_x=np.nan, tcp_y=np.nan, tcp_z=np.nan, warned=False)
    with warnings.catch_warnings(record=True) as w, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        try:
            tcp, theta, R, RT = rM.compute_ruth_pose(cps[0], cps[1], cps[2])
            theta = float(np.ravel(theta)[0])
            tcp = np.asarray(tcp, float).ravel()
            out.update(theta_rad=theta, tcp_x=tcp[0], tcp_y=tcp[1], tcp_z=tcp[2],
                       nan_out=bool(np.isnan(tcp).any() or np.isnan(theta) or np.isnan(R).any()))
        except Exception as ex:  # noqa: BLE001
            out["exception"] = f"{type(ex).__name__}: {ex}"[:120]
        out["warned"] = len(w) > 0
    if "ier" in _rec:
        out["ier"] = int(_rec["ier"])
        out["fvec"] = _rec["fvec"]
    if "reaches" in _rec:
        r = np.array(_rec["reaches"])
        out.update(reach_max=r.max(), reach_min=r.min(), reach_spread=r.max() - r.min())
    return out


def run_set(name, rows):
    res = [solve_one(r) for r in rows]
    n = len(res)
    exc = sum(1 for r in res if r["exception"])
    nan_out = sum(1 for r in res if r["nan_out"])
    ier1 = np.array([r["ier"] == 1 for r in res])
    ok = ier1 & ~np.array([bool(r["exception"]) or r["nan_out"] for r in res])
    theta = np.array([r["theta_rad"] for r in res])
    spread = np.array([r["reach_spread"] for r in res])
    reach_max = np.array([r["reach_max"] for r in res])
    fvec = np.array([r["fvec"] for r in res])
    summ = dict(
        set=name, n=n, exceptions=exc, nan_outputs=nan_out,
        fsolve_ier1=int(ier1.sum()), fully_ok=int(ok.sum()),
        theta_in_0_pi2_of_ok=int(np.sum((theta[ok] >= 0) & (theta[ok] <= np.pi / 2))) if ok.any() else 0,
        theta_med_deg=float(np.degrees(np.nanmedian(theta[ok]))) if ok.any() else np.nan,
        reach_max_med=float(np.nanmedian(reach_max)), reach_max_p95=float(np.nanpercentile(reach_max[~np.isnan(reach_max)], 95)),
        spread_mean=float(np.nanmean(spread)), spread_med=float(np.nanmedian(spread)),
        spread_p95=float(np.nanpercentile(spread[~np.isnan(spread)], 95)),
        fvec_max_ok=float(np.nanmax(fvec[ok])) if ok.any() else np.nan,
        theta_neg=int(np.sum(theta[ok] < 0)) if ok.any() else 0,
        theta_min_deg=float(np.degrees(np.nanmin(theta[ok]))) if ok.any() else np.nan,
        theta_max_deg=float(np.degrees(np.nanmax(theta[ok]))) if ok.any() else np.nan,
        nonconv=[(int(i), float(r["reach_max"]), float(r["fvec"])) for i, r in enumerate(res) if r["ier"] != 1],
    )
    return res, summ


def main():
    rng = np.random.default_rng(SEED)
    sets = []
    ws = np.load(os.path.join(ROOT, "gripper_representation", "ruth_workspace_sampling", "contact_points.npy"))
    idx = rng.choice(ws.shape[0], size=min(N_SAMPLE, ws.shape[0]), replace=False)
    sets.append(("workspace_sampling/contact_points.npy, 1000 rows, seed 0", ws[idx], idx))
    rl = np.load(os.path.join(ROOT, "rl_inverse_kinematics", "ruth", "contact_points.npy"), mmap_mode="r")
    idx2 = np.sort(rng.choice(rl.shape[0], size=N_SAMPLE, replace=False))
    sets.append(("rl_inverse_kinematics/ruth/contact_points.npy, 1000 rows, seed 0", np.array(rl[idx2]), idx2))
    cl = np.load(os.path.join(ROOT, "rl_inverse_kinematics", "ruth", "contact_list.npy"))
    sets.append(("rl_inverse_kinematics/ruth/contact_points.npy, the 21 rows of contact_list.npy (RL training targets)", np.array(rl[cl]), cl))

    rl_err = np.load(os.path.join(ROOT, "results", "rl_ik_error", "ruth", "epoch_2885.npy"))

    all_rows, summaries = [], []
    for name, rows, ids in sets:
        res, summ = run_set(name, rows)
        summaries.append(summ)
        for i, r in zip(ids, res):
            r2 = dict(set=name, row=int(i))
            r2.update(r)
            all_rows.append(r2)

    cols = list(all_rows[0].keys())
    with open(os.path.join(OUT_DIR, "C4_numeric_ik_raw.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    with open(os.path.join(OUT_DIR, "C4_rl_vs_numeric_ik.md"), "w") as fh:
        fh.write("# C4 · RUTH numeric IK (fsolve) vs. RL IK — what can and cannot be compared\n\n")
        fh.write(f"Script: `evaluation/thesis_ch3/c4_numeric_ik_vs_rl.py` (seed {SEED}, {N_SAMPLE} rows per set). "
                 "Raw per-target results: `results/thesis_ch3/C4_numeric_ik_raw.csv`.\n\n")
        fh.write("## Numeric IK runs\n\n")
        fh.write("| target set | n | exceptions | NaN output | fsolve ier==1 | fully OK | θ∈[0,90°] of OK | median θ (deg) | "
                 "median / p95 required reach (mm) | finger-reach spread mean / median / p95 (mm) | max residual of OK |\n")
        fh.write("|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|\n")
        for s in summaries:
            fh.write(f"| {s['set']} | {s['n']} | {s['exceptions']} | {s['nan_outputs']} | {s['fsolve_ier1']} | {s['fully_ok']} | "
                     f"{s['theta_in_0_pi2_of_ok']} | {s['theta_med_deg']:.1f} | {s['reach_max_med']:.1f} / {s['reach_max_p95']:.1f} | "
                     f"{s['spread_mean']:.1f} / {s['spread_med']:.1f} / {s['spread_p95']:.1f} | {s['fvec_max_ok']:.1e} |\n")
        fh.write("\nθ range of the OK solutions (deg) and count of θ<0 (finger bent backwards, i.e. contact closer to the finger base than the straight-finger reach): ")
        fh.write("; ".join(f"{s_['set'].split(',')[0]}: [{s_['theta_min_deg']:.1f}, {s_['theta_max_deg']:.1f}], θ<0 in {s_['theta_neg']}/{s_['fully_ok']}" for s_ in summaries) + ".\n")
        for s_, (name, rows, ids) in zip(summaries, sets):
            for k, reach, fv in s_["nonconv"]:
                fh.write(f"Non-converged case: set '{name}', row {int(ids[k])}: required reach {reach:.1f} mm exceeds the planar finger model's maximum "
                         f"(|P9s_x| ≤ ≈67 mm for the 44.7 + 35 mm two-segment finger of `sym_fing_fip`, line 181-188); fsolve residual {fv:.1f} mm.\n")
        fh.write("\n*fully OK* = no exception, no NaN in (TCP, θ, R) and fsolve ier==1. "
                 "*required reach* = `hori_bend` = max over the three fingers of the in-plane distance from the finger base "
                 "(five-bar joints P2/P3/P4) to its contact point; *spread* = max−min of those three distances. "
                 "Because RUTH's three fingers are closed by one tendon motor, the solver bends all fingers by the same θ to the "
                 "largest reach, so the spread is the in-plane reach mismatch the model itself cannot remove for the other fingers "
                 "(a proxy only — it is **not** a fingertip position error from a forward model).\n\n")
        fh.write("## RL IK, RUTH, final checkpoint (results/rl_ik_error/ruth/epoch_2885.npy, from C3)\n\n")
        fh.write(f"n = {rl_err.size} episodes; per-finger error averaged over the 100 steps of each episode: "
                 f"mean {rl_err.mean():.2f} mm, median {np.median(rl_err):.2f} mm, std {rl_err.std():.2f}, "
                 f"p5 {np.percentile(rl_err,5):.2f}, p95 {np.percentile(rl_err,95):.2f}, min {rl_err.min():.2f}, max {rl_err.max():.2f} mm.\n\n")
        fh.write("## Comparison status: 未完成 (partial)\n\n")
        fh.write("A like-for-like fingertip-error comparison is **not possible** from the repository contents:\n\n")
        fh.write("1. `ruth_grasping_kinematics.py` contains no forward kinematics of the RUTH hand (five-bar palm + three "
                 "under-actuated fingers in 3-D). It returns only a TCP point for the arm, one finger-bend angle θ and the "
                 "contact-plane rotation (lines 210-213); the five-bar motor angle `opt_t` it searches for is not even returned. "
                 "Fingertip positions for a given (motor, θ) can only be obtained in PyBullet "
                 "(`move_env.py: get_pos()` reads link states), and PyBullet is not available here.\n")
        fh.write("2. The RL numbers are evaluated in PyBullet on the full arm+hand model with the random wrist perturbation of "
                 "`point_transform()` and are time-averages over 100 control steps; the numeric solver is a static planar model.\n")
        fh.write("3. The numeric solver's own metrics (convergence of the 1-D fsolve, reach spread) are reported above; they say "
                 "whether a planar solution exists, not how far the real fingertips would land.\n\n")
        fh.write("Therefore the table reports **solver convergence / feasibility** for the numeric IK and **fingertip error** for the RL IK, "
                 "and the thesis should not present them as the same quantity.\n")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    sys.exit(main())
