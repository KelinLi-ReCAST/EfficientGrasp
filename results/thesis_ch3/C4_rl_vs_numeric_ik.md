# C4 · RUTH numeric IK (fsolve) vs. RL IK — what can and cannot be compared

Script: `evaluation/thesis_ch3/c4_numeric_ik_vs_rl.py` (seed 0, 1000 rows per set). Raw per-target results: `results/thesis_ch3/C4_numeric_ik_raw.csv`.

## Numeric IK runs

| target set | n | exceptions | NaN output | fsolve ier==1 | fully OK | θ∈[0,90°] of OK | median θ (deg) | median / p95 required reach (mm) | finger-reach spread mean / median / p95 (mm) | max residual of OK |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| workspace_sampling/contact_points.npy, 1000 rows, seed 0 | 1000 | 0 | 0 | 1000 | 1000 | 800 | 8.0 | 22.6 / 60.7 | 3.9 / 0.3 / 38.9 | 6.3e-13 |
| rl_inverse_kinematics/ruth/contact_points.npy, 1000 rows, seed 0 | 1000 | 0 | 0 | 999 | 999 | 714 | 3.6 | 14.1 / 31.4 | 0.3 / 0.3 / 0.7 | 3.4e-13 |
| rl_inverse_kinematics/ruth/contact_points.npy, the 21 rows of contact_list.npy (RL training targets) | 21 | 0 | 0 | 21 | 21 | 13 | 3.2 | 13.3 / 17.0 | 0.2 / 0.2 / 0.5 | 4.1e-14 |

θ range of the OK solutions (deg) and count of θ<0 (finger bent backwards, i.e. contact closer to the finger base than the straight-finger reach): workspace_sampling/contact_points.npy: [-1.8, 36.9], θ<0 in 200/1000; rl_inverse_kinematics/ruth/contact_points.npy: [-2.2, 37.6], θ<0 in 285/999; rl_inverse_kinematics/ruth/contact_points.npy: [-1.4, 8.4], θ<0 in 8/21.
Non-converged case: set 'rl_inverse_kinematics/ruth/contact_points.npy, 1000 rows, seed 0', row 332345: required reach 147.0 mm exceeds the planar finger model's maximum (|P9s_x| ≤ ≈67 mm for the 44.7 + 35 mm two-segment finger of `sym_fing_fip`, line 181-188); fsolve residual 79.5 mm.

*fully OK* = no exception, no NaN in (TCP, θ, R) and fsolve ier==1. *required reach* = `hori_bend` = max over the three fingers of the in-plane distance from the finger base (five-bar joints P2/P3/P4) to its contact point; *spread* = max−min of those three distances. Because RUTH's three fingers are closed by one tendon motor, the solver bends all fingers by the same θ to the largest reach, so the spread is the in-plane reach mismatch the model itself cannot remove for the other fingers (a proxy only — it is **not** a fingertip position error from a forward model).

## RL IK, RUTH, final checkpoint (results/rl_ik_error/ruth/epoch_2885.npy, from C3)

n = 100 episodes; per-finger error averaged over the 100 steps of each episode: mean 15.54 mm, median 15.23 mm, std 4.68, p5 8.82, p95 24.14, min 8.30, max 29.43 mm.

## Comparison status: 未完成 (partial)

A like-for-like fingertip-error comparison is **not possible** from the repository contents:

1. `ruth_grasping_kinematics.py` contains no forward kinematics of the RUTH hand (five-bar palm + three under-actuated fingers in 3-D). It returns only a TCP point for the arm, one finger-bend angle θ and the contact-plane rotation (lines 210-213); the five-bar motor angle `opt_t` it searches for is not even returned. Fingertip positions for a given (motor, θ) can only be obtained in PyBullet (`move_env.py: get_pos()` reads link states), and PyBullet is not available here.
2. The RL numbers are evaluated in PyBullet on the full arm+hand model with the random wrist perturbation of `point_transform()` and are time-averages over 100 control steps; the numeric solver is a static planar model.
3. The numeric solver's own metrics (convergence of the 1-D fsolve, reach spread) are reported above; they say whether a planar solution exists, not how far the real fingertips would land.

Therefore the table reports **solver convergence / feasibility** for the numeric IK and **fingertip error** for the RL IK, and the thesis should not present them as the same quantity.
