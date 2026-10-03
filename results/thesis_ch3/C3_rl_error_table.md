# C3 · RL-IK evaluation logs, per file (unit: mm, per-finger mean distance averaged over the 100 steps of an episode)

Source: `results/rl_ik_error/<gripper>/*.npy`, written by `rl_inverse_kinematics/<gripper>/val.py` (`avg_dist = distance/pit/3`, `np.save(...)`). Computed by `evaluation/thesis_ch3/plot_rl_error.py`.

| gripper | file | epoch | used in paper fig | n | mean | median | std | p5 | p95 | min | max |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| barrett | epoch_0 | 0 | yes | 100 | 75.42 | 76.09 | 21.68 | 43.07 | 113.07 | 40.00 | 115.08 |
| barrett | epoch_1 | 1 | no | 100 | 45.61 | 41.54 | 13.58 | 28.46 | 70.46 | 25.98 | 85.62 |
| barrett | epoch_10 | 10 | yes | 100 | 38.77 | 33.05 | 16.97 | 22.38 | 68.55 | 21.12 | 83.08 |
| barrett | epoch_100 | 100 | yes | 100 | 24.98 | 22.10 | 8.57 | 14.17 | 44.66 | 12.42 | 46.59 |
| barrett | epoch_200 | 200 | yes | 100 | 18.05 | 18.61 | 4.79 | 11.17 | 25.98 | 10.92 | 27.72 |
| barrett | epoch_500 | 500 | yes | 100 | 13.77 | 12.16 | 3.95 | 10.04 | 22.43 | 9.99 | 23.92 |
| barrett | epoch1000 | 1000 | no | 100 | 12.03 | 11.19 | 2.79 | 9.29 | 17.16 | 8.79 | 24.97 |
| barrett | epoch_1000 | 1000 | yes | 100 | 35.48 | 36.52 | 7.91 | 19.89 | 42.82 | 14.93 | 59.67 |
| barrett | epoch1500 | 1500 | no | 100 | 12.41 | 11.06 | 4.31 | 9.09 | 24.30 | 8.68 | 27.57 |
| barrett | epoch_1500 | 1500 | yes | 100 | 36.33 | 36.67 | 5.90 | 24.80 | 43.88 | 21.50 | 47.16 |
| barrett | epoch_2000 | 2000 | yes | 100 | 11.44 | 11.09 | 2.00 | 9.80 | 18.02 | 9.49 | 18.30 |
| barrett | epoch_2500 | 2500 | yes | 100 | 11.50 | 11.27 | 1.75 | 9.41 | 16.51 | 8.99 | 16.99 |
| barrett | epoch_2885 | 2885 | yes | 100 | 10.88 | 10.72 | 1.64 | 9.08 | 12.74 | 8.91 | 17.34 |
| robotiq_3f | epoch_0 | 0 | no | 100 | 42.93 | 39.09 | 12.09 | 27.20 | 64.78 | 25.33 | 72.68 |
| robotiq_3f | epoch_00 | 0 | yes | 100 | 48.18 | 45.87 | 11.65 | 29.96 | 70.38 | 27.78 | 82.32 |
| robotiq_3f | epoch_1 | 1 | no | 100 | 45.61 | 41.54 | 13.58 | 28.46 | 70.46 | 25.98 | 85.62 |
| robotiq_3f | epoch_10 | 10 | yes | 100 | 19.85 | 16.84 | 9.45 | 9.96 | 42.05 | 9.35 | 43.37 |
| robotiq_3f | epoch_100 | 100 | yes | 100 | 17.48 | 16.36 | 6.82 | 9.29 | 31.94 | 7.32 | 36.84 |
| robotiq_3f | epoch_200 | 200 | yes | 100 | 36.26 | 36.59 | 14.91 | 13.81 | 62.56 | 10.68 | 76.13 |
| robotiq_3f | epoch_500 | 500 | yes | 100 | 33.29 | 30.83 | 12.44 | 20.51 | 49.64 | 18.02 | 85.60 |
| robotiq_3f | epoch_1000 | 1000 | yes | 100 | 28.55 | 23.81 | 11.53 | 15.59 | 55.96 | 10.87 | 58.95 |
| robotiq_3f | epoch_1500 | 1500 | yes | 100 | 17.93 | 17.46 | 7.49 | 8.26 | 29.20 | 8.11 | 39.07 |
| robotiq_3f | epoch_2000 | 2000 | yes | 100 | 18.62 | 18.77 | 6.35 | 8.39 | 28.04 | 8.06 | 38.80 |
| robotiq_3f | epoch_2500 | 2500 | yes | 100 | 17.54 | 16.52 | 7.02 | 8.33 | 36.53 | 7.64 | 38.40 |
| robotiq_3f | epoch_2885 | 2885 | yes | 100 | 18.93 | 18.99 | 7.17 | 8.72 | 37.44 | 8.17 | 38.57 |
| ruth | epoch_0 | 0 | yes | 100 | 84.25 | 85.17 | 24.15 | 45.84 | 129.48 | 41.34 | 135.41 |
| ruth | epoch_10 | 10 | yes | 100 | 41.57 | 29.76 | 19.97 | 19.28 | 79.74 | 14.17 | 82.51 |
| ruth | epoch_100 | 100 | yes | 100 | 36.94 | 28.68 | 18.33 | 17.81 | 71.71 | 14.10 | 80.32 |
| ruth | epoch_200 | 200 | yes | 20 | 25.31 | 20.04 | 13.09 | 10.60 | 49.18 | 10.27 | 50.21 |
| ruth | epoch_500 | 500 | yes | 100 | 43.06 | 43.35 | 14.90 | 20.45 | 80.24 | 9.88 | 84.23 |
| ruth | epoch_1000 | 1000 | yes | 100 | 26.06 | 23.02 | 11.91 | 9.17 | 47.09 | 8.60 | 57.67 |
| ruth | epoch_1500 | 1500 | yes | 100 | 39.86 | 42.27 | 8.98 | 23.31 | 56.71 | 13.49 | 59.70 |
| ruth | epoch_2000 | 2000 | yes | 100 | 18.28 | 16.65 | 6.25 | 11.48 | 34.31 | 10.53 | 35.77 |
| ruth | epoch_2500 | 2500 | yes | 100 | 17.90 | 16.87 | 5.99 | 12.06 | 33.72 | 9.46 | 34.61 |
| ruth | epoch_2885 | 2885 | yes | 100 | 15.54 | 15.23 | 4.68 | 8.82 | 24.14 | 8.30 | 29.43 |

## Reduction of the mean error, first -> last epoch of the paper series

| gripper | first file (mean) | last file (mean) | reduction of mean | reduction of median |
|---|---|---|---:|---:|
| ruth | epoch_0 (84.25) | epoch_2885 (15.54) | 81.6 % | 82.1 % |
| robotiq_3f | epoch_00 (48.18) | epoch_2885 (18.93) | 60.7 % | 58.6 % |
| barrett | epoch_0 (75.42) | epoch_2885 (10.88) | 85.6 % | 85.9 % |

## What the old script (evaluation/simulation/RL_eval.py) would have plotted as the per-epoch mean

It computes `sum(values)/100` (wrong for ruth/epoch_200 which has 20 values) and then sorts the 10 means in descending order before plotting (`pppp.sort(reverse=True)`), so the x position no longer corresponds to the epoch.

- ruth: sum/100 per file = [84.25, 41.57, 36.94, 5.06, 43.06, 26.06, 39.86, 18.28, 17.9, 15.54]; after descending sort = [84.25, 43.06, 41.57, 39.86, 36.94, 26.06, 18.28, 17.9, 15.54, 5.06]  (last plotted point = 5.06 mm)
- robotiq_3f: sum/100 per file = [48.18, 19.85, 17.48, 36.26, 33.29, 28.55, 17.93, 18.62, 17.54, 18.93]; after descending sort = [48.18, 36.26, 33.29, 28.55, 19.85, 18.93, 18.62, 17.93, 17.54, 17.48]  (last plotted point = 17.48 mm)
- barrett: sum/100 per file = [75.42, 38.77, 24.98, 18.05, 13.77, 35.48, 36.33, 11.44, 11.5, 10.88]; after descending sort = [75.42, 38.77, 36.33, 35.48, 24.98, 18.05, 13.77, 11.5, 11.44, 10.88]  (last plotted point = 10.88 mm)
