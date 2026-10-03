# File reference and reorganisation notes

Detailed per-file notes for this repository, moved here from the first version of the
README. The top-level [README](../README.md) covers installation and usage.

The repository is a reorganised copy of the original working directory
`~/RAL-IROS2022`, split by functional module. Every file was copied from it
(provenance table below). Duplicates, caches and post-paper scratch were dropped,
bringing the tree from about 15 GB down to about 12 GB (code plus the separately
distributed data).

Section, figure and table numbers of the form `3.x` refer to Chapter 3 of the thesis:
`gripper_representation` = Sec. 3.3.1, `contact_point_selection` + `grasp_quality` =
Sec. 3.3.2, `rl_inverse_kinematics` = Sec. 3.3.3, `evaluation/simulation` + `results` =
Sec. 3.4, `evaluation/real_world` = Sec. 3.5.1. Sec. 3.5.2 (soft reconfigurable gripper)
has no code in the original repo; the collaborators' work was not stored here. In the
paper, thesis Table 3.2 and Fig. 3.6 correspond to Table II and Fig. 6, and thesis
Fig. 3.7 to Fig. 7.

The thesis text still has a placeholder in Sec. 3.2.2 for how the RUTH five-bar loop
was closed in PyBullet; the answer is in
`rl_inverse_kinematics/ruth/gym/envs/kelin/move_env.py` and the `ur5_plus_RUTH.urdf`
it loads (look for the constraint that closes the loop).

---

## gripper_representation

| Path | What it is |
|---|---|
| `feature_extraction/workspace_generator.py` | Analytic fingertip-workspace generator for parametric 3-finger grippers. Only the 5-bar palm is implemented (the 4-bar / 3-RRR / single- and coupled-rotation branches are commented-out stubs, L153-160). Rows are L x 9 (3 fingertips x xyz). Produced `data/WorkspaceArrays`. The script that produced `coupRotWorkspaceArrays` was not found. |
| `feature_extraction/workspace.py` | PyBullet sampling of the RUTH fingertip workspace (uses `fo_t1o_t2o.urdf`). |
| `feature_extraction/encoder.py` | TF1/tflearn PointNet autoencoder, input `[None,576,9]`, 6 conv1d layers + max-pool -> 256-d feature, 3-FC decoder; lr 1e-3, batch 1, 10000 epochs, one checkpoint per epoch. **Contains the coupled Chamfer distance** (Eq. 3.2): the 9 columns are split into 3 fingers and per-finger distances are summed over a shared row index before the min. **Training set as committed:** the loop over `WorkspaceArrays` / `coupRot…` / `fourbar…` / `singRot…` (L26-45) is commented out and only `ruthArrays/ws*` is read (L46-47); there is no subsampling code, so only 576-row arrays can be fed. See `results/thesis_ch3/REPORT.md` A3. |
| `feature_extraction/encoder_test.py` | Restores `saved_models/workspace/9999model.ckpt`, writes `mean/max/min.npy` features for `ruthArrays`. |
| `feature_extraction/ruth_grasping_kinematics.py` | Analytic RUTH pose from three contact points (fsolve). Used for comparison only. |
| `feature_extraction/show3d_balls.py`, `render_balls_so.cpp`, `compile_render_balls_so.sh` | Point-cloud viewer (compile the `.so` locally; it was not copied). |
| `feature_extraction/saved_models/workspace/` | Trained autoencoder checkpoint (`9999model.ckpt`, epoch 9999, written 2022-02-07 22:15). From the committed code, the run timeline (42 min for 10000 epochs) and the empty TF logs, this checkpoint was trained on **one array, `ruthArrays/ws1.npy`** (`REPORT.md` A3). Earlier Jan-2022 runs and the 2021 encoder behind `Data_DB/*/workspace.npy` are not preserved. |
| `feature_extraction/logs/` | 43 TF event files from autoencoder runs (Jan–Feb 2022). They contain only the graph (no scalars, no steps: `add_summary` is commented out). |
| `feature_extraction/coupRotWorkspaceArrays/` | 3005 workspace arrays, all 576 x 9, numbered ws39..ws96759 (a selection from a larger set), Aug 2021 – Jan 2022. Generating script not found. |
| `feature_extraction/ruthArrays/` | `ws1.npy` = RUTH workspace (576 x 9) and its 256-d feature (`mean.npy` = `max.npy` = `min.npy`, since the directory holds a single array). `ws1.npy` is an **equal-interval subsample** of `ruth_workspace_sampling/contact_points.npy`: `contact_points[np.linspace(0, 8648, 576, dtype=int)]` (from `rl_inverse_kinematics/*/workspace_visual.py` L17; reproduced byte-for-byte by `evaluation/thesis_ch3/subsample_576.py`). Note: this feature is **not** equal to `Data_DB/ruth/workspace.npy`, which predates it. |
| `ruth_workspace_sampling/workspace.py` | Sweeps the RUTH motors through `gym.make('kelin-v0')` and saves the fingertip workspace. Needs the gym env in `rl_inverse_kinematics/early_prototype/gym` on `sys.path`. |
| `ruth_workspace_sampling/contact_points.npy` | The sampled RUTH fingertip workspace, 8649 x 9 (committed to git, 623 KB). The grid is **31 x 31 x 9**: palm motors `0.08*j`, `0.08*k` for j,k in 0..30 and finger-tendon motor `0.1*l` for l in 0..8 (`workspace.py` L533-546), row index `j*279 + k*9 + l`. `evaluation/thesis_ch3/make_fig3_workspace.py` plots it. |
| `data/WorkspaceArrays/` | 44 751 `ws<N>.npy` files (3.8 GB) from `workspace_generator.py`: 14 112 empty `(0,9)`, 30 639 non-empty with 432/648/864/1458/2187/2916 rows (`results/thesis_ch3/A3_workspace_arrays_stats.md`). **None has 576 rows**, so the committed `encoder.py` cannot consume them directly; whether an earlier, unsaved pipeline used them is unknown. |

## contact_point_selection

### `UniGrasp/` — baseline and the EfficientGrasp-modified PSSN
Kept as one tree because the modified scripts use relative paths into
`saved_models/`, `data/` and `tf_models/`.

**EfficientGrasp-modified / new (Kelin, Aug 2021 – Feb 2022)**

| Path | What it is |
|---|---|
| `point_set_selection/unigrasp.py` | **EfficientGrasp PSSN inference.** Placeholder `gripper_feat_tf [None,256]`, loads `data/gripper_features/Data_DB/<gripper>/workspace.npy` (the 256-d workspace feature), restores `saved_models/point_set_selection/220model.ckpt`, writes `top1_f{1,2,3}_index_*.txt`. Gripper ID 13 (Kinova-3F label set) is mapped to `'ruth'`, i.e. RUTH reuses the Kinova-3F training labels. `__main__` is commented out. |
| `point_set_selection/train_pssn.py`, `point_set_selection.py`, `unigrasp_train.py` | Staged PSSN training (stage 1 -> 2 -> 3, earlier stages frozen) restricted to gripper IDs 11/12/13 (robotiq_3f, bh_282, kinova). `unigrasp_train.py` differs from `point_set_selection.py` only in which training ops are enabled. |
| `point_set_selection/point_set_selection_test.py` | Test variant, restores `189model`. |
| `point_set_selection/data_preparing*.py`, `simulation.py`, `pointnet4/train.py` | Smaller edits. `simulation.py` still hardcodes `/home/robin-lab/Kelin/UniGrasp-master/`. |
| `gripper_representation/gripper_feature_extraction.py` | UniGrasp autoencoder with 2048 x 3 inputs (Feb 2022). |
| `data/gripper_features/Data_DB/{ruth,robotiq_3f,kinova_kg3,bh_282}/workspace.npy` | The (1,256) workspace features fed to the PSSN. |
| `data/ObjectPointClouds/` | 16 YCB point clouds and `Contact_Points/{Barret,Robotiq,Ruth}/*.npy`, the PSSN contact-point outputs per object. |
| `saved_models/point_set_selection/220model.ckpt` (282 MB) | **Most likely the trained EfficientGrasp PSSN** (size matches the 256-d input). `189model` and `221model` (316 MB) match the 768-d UniGrasp input and are retrained baselines. |
| `point_set_selection/logs/` | TF event files (129 MB) from PSSN training. |

**Unmodified upstream UniGrasp** (stanford-iprl-lab/UniGrasp, 2021-01-18): `README.md`,
`LICENSE`, `simulation/`, `vis_3d/`, `gripper_urdf/`, `tf_models/`,
`point_set_selection_raw_point_cloud.py`, `point_set_selection_test_with_gt.py`,
`pgm_loader.py`, `Train_Val_Test.py`, `pointnet4/`, `saved_models/220model.ckpt`
(303 MB, released PSSN), `saved_models/gripper_representation/2248model.ckpt`
(released gripper autoencoder), `data/objects/1812` (448 MB, UniGrasp labels for
one object), `data/real_world/d*.npy` (UniGrasp's own 2021 data).

## grasp_quality
Vendored *diverse-and-stable-grasp* (Liu et al., RA-L 2021, differentiable
force-closure estimator; originally `FC/`). **The GQS computation is the `__main__` block of
`utils/Losses.py`**: it evaluates the force-closure term for the top-10 contact
triplets of each of the 16 YCB objects with centroid-directed normals. The
`FCLoss` class itself is upstream. `data/` holds DeepSDF weights and
`mano/MANO_RIGHT.pkl` (licence-restricted); `utils/manopth/` is 257 MB of MANO
models. The upstream `synthesis/` output (525 MB of plotly HTML) was not copied.

## rl_inverse_kinematics

One directory per gripper, each self-contained and runnable from inside it:

| Directory | URDF | Action / obs dims | Trained seeds (`log/kelin-v0/seed--N`) |
|---|---|---|---|
| `ruth/` | `ur5_plus_RUTH.urdf` | 4 | 0–3 (last epochs 2885, 2475, 5865, 3365), 3.1 GB |
| `robotiq_3f/` | `ur5_plus_robotiq_3f.urdf` | 6 | 0 (to 2300), 1 (to 145), 63 MB |
| `barrett/` | `ur5_plus_barrett.urdf` | 9 | 0 (to 3050), 1 (to 2905), 1.3 GB |

Files in each gripper directory:

| File | What it is |
|---|---|
| `train.py` | SAC training on `gym.make('kelin-v0')`; logs to `log/kelin-v0/seed--N`. |
| `sac.py`, `model.py`, `replay_memory.py`, `utils_.py`, `utils11.py`, `utils/` | SAC implementation (Gaussian policy, twin Q). Identical in all three directories. |
| `val.py` | Runs a trained policy for 100 episodes and saves per-episode mean fingertip error as `epochNNNN.npy` in the cwd. Source of the logs in `results/rl_ik_error/`. |
| `RL_gripper.py` | Loads the trained agent for the demo (`log/kelin-v0/seed--0` in `ruth/` and `robotiq_3f/`, `seed--1` in `barrett/`; absolute path). |
| `demo.py` | Full simulation pipeline: loads UR5+gripper and the YCB object, reads PSSN contact points, positions the arm (`move_ur.calc_target_pos`), runs the RL policy, records MP4. |
| `move_ur.py` (+ `move_ur5_robotiq.py`) | UR5 positioning and the two-step rest-pose computation (Sec. 3.3.3 step one). |
| `gym/` | Vendored gym 0.14.0 with `envs/kelin/move_env.py` registered as `kelin-v0` (class name `MoveUr5RuthEnv` for all grippers). Holds the URDFs and meshes. |
| `obj_pc/` | Object point clouds (robotiq: 16; barrett: 16 x 4 orientations; ruth: 16 YCB + 17 real-world captures). |
| `contact_points.npy` (485376 x 9), `contact_list.npy`, `ws1.npy` | RL target pool (485376 = 79 x 32 x 32 x 6 UR5-wrist + RUTH motor sweep), the 21 row indices actually used as training targets, and the RUTH 576 x 9 workspace. Identical in all three directories: the `ws1.npy` under `robotiq_3f/` and `barrett/` is the **RUTH** array (a stray copy), not a Robotiq/Barrett workspace; no Robotiq/Barrett/Kinova workspace array exists in the repo. |
| `pybullet_object_models` | Symlink into `third_party/`. `ruth/`, `robotiq_3f/` -> `pybullet_object_models_demo_ruth_robotiq`; `barrett/` -> `pybullet_object_models_demo_barrett`; every `gym/envs/kelin/pybullet_object_models` -> `pybullet_object_models`. See *third_party* below. |
| `barrett/camera.py` | Demo variant with camera capture (May 2022). |
| `barrett/gym/envs/kelin/Barrett/` | Separate Barrett URDF set with `move_ur5_barrett.py`; not a duplicate of `urdf/barrett_model`. |

Other RL directories:

| Directory | What it is |
|---|---|
| `early_prototype/` | Nov 2021 – Feb 2022 first SAC version (deterministic policy) with its own gym env; `log/kelin-v0/seed--0` trained to epoch 1670. The RUTH workspace sampler that lived here is now `gripper_representation/ruth_workspace_sampling/`. |
| `revision_lr_ablation/` | June 2022 SAC learning-rate / step-LR study for the RA-L revision (`log/kelin-v0/LR001, LR003, stepLR, ...`, `plot_error.ipynb`, result PNGs). Originally `workspace_xian/SAC_train`. |
| `her_baseline_attempt/` | Unmodified clone of hemilpanchiwala/Hindsight-Experience-Replay with a `gym/envs/kelin` env added; DDPG/HER was never wired to it. Originally `workspace_xian/Hindsight-Experience-Replay`. |

## evaluation/simulation

| Path | What it is |
|---|---|
| `test.py` | Simulation trial driver: 16 YCB objects x 4 orientations (`000`, `pi00`, `pi0pi`, `pipipi`), calls `train_<gripper>/demo.main`. `--gripper {ruth,robotiq,barrett}`. |
| `RL_eval.py` | Produces the RL-IK error-vs-epoch figure (Fig. 3.7) from `results/rl_ik_error/*/epoch_*.npy` (paths inside still point at the old location, see Known issues). |

## results

| Path | What it is |
|---|---|
| `rl_ik_error/<gripper>/epoch_*.npy` | Per-episode fingertip error (100 episodes) at epochs 0/1/10/100/200/500/1000/1500/2000/2500/2885, written by `val.py`. `.mat` files are MATLAB exports of the same. |
| `pssn_accuracy/stage{1,2,3}_{acc.npy,ours.mat,unigrasp.mat}` | Per-stage Top-1/Top-10 accuracy of EfficientGrasp vs UniGrasp (Table 3.2, Fig. 3.6). |
| `videos/unigrasp_baseline/` | 128 MP4s: UniGrasp-baseline contact points, barrett and robotiq x 16 objects x 4 orientations. |

## evaluation/real_world

There is no robot-control or RealSense capture code in the repository; only the
captured data and the pre/post-processing scripts.

| Path | What it is |
|---|---|
| `pointcloud_process.py` | Downsamples a captured RealSense cloud to 2048 points (PSSN input size). |
| `target_position.py` | Computes the UR5 target pose from a PSSN contact-point set for a real object. Imports `move_ur` from `rl_inverse_kinematics/ruth/`. |
| `show_contact_points.py` | Visualises contact points on a real point cloud. Imports `show3d_balls` from `rl_inverse_kinematics/ruth/`. |
| `object_pointclouds/` | 17 real captures (2048 x 3): banana, bowl, bowl-bottom, clip-L, clip-M, drill-lay, drill-stand, football, mug, mug-bottom, screwdriver, soup-can (x4 poses), spam-can (x3 poses). A copy also remains in `rl_inverse_kinematics/ruth/obj_pc/` because `demo.py` loads from there. |
| `object_photos/` | Renders of the real point clouds with the three selected contact points marked in red, green and blue (May 2022). Despite the folder name these are not camera photos. |
| `occlusion/` | Material for the occlusion failure analysis: RGB captures, occluded point-cloud renders, and four RUTH MustardBottle sim clips. |

---

## third_party

`pybullet_object_models` (eleramp/pybullet-object-models, YCB meshes and URDFs)
existed in 12 identical-looking copies in the original repo. They differ only in
`ycb_objects/YcbMustardBottle/model.urdf`, in three variants, each kept once:

| Directory | MustardBottle `model.urdf` | Used by |
|---|---|---|
| `pybullet_object_models/` | friction 0.8, mass 0.603 kg, inertia 1e-3 (upstream values) | all `gym/envs/kelin` environments, i.e. RL training |
| `pybullet_object_models_demo_ruth_robotiq/` | mass changed to 0.01 kg (11 May 2022) | `ruth/demo.py`, `robotiq_3f/demo.py` (simulation trials) |
| `pybullet_object_models_demo_barrett/` | friction 1.0, inertia 0 (15 Aug 2022) | `barrett/demo.py` |

All other objects are byte-identical across the three.

---

## Provenance: original path -> new path

| Original (`~/RAL-IROS2022/`) | New |
|---|---|
| `environment.yml` | `environment.yml` |
| `feature_extraction/` | `gripper_representation/feature_extraction/` |
| `workspace_generation/workspace.py`, `contact_points.npy` | `gripper_representation/ruth_workspace_sampling/` |
| `WorkspaceArrays/` | `gripper_representation/data/WorkspaceArrays/` |
| `UniGrasp/` | `contact_point_selection/UniGrasp/` |
| `UniGrasp/point_set_selection/stage*` | `results/pssn_accuracy/` |
| `FC/` | `grasp_quality/` |
| `train_ruth/`, `train_robotiq/`, `train_barrett/` | `rl_inverse_kinematics/{ruth,robotiq_3f,barrett}/` |
| `train_*/epoch*.npy|.mat` | `results/rl_ik_error/<gripper>/` |
| `train_ruth/pybullet_object_models/` | `third_party/pybullet_object_models_demo_ruth_robotiq/` |
| `train_barrett/pybullet_object_models/` | `third_party/pybullet_object_models_demo_barrett/` |
| `train_*/gym/envs/kelin/pybullet_object_models/` | `third_party/pybullet_object_models/` |
| `train_ruth/{target_position,show_contact_points}.py`, `train_ruth/*.png`, `train_ruth/obj_pc/Ycb<lowercase>*.npy` | `evaluation/real_world/` |
| `workspace_generation/` (rest) | `rl_inverse_kinematics/early_prototype/` |
| `workspace_xian/SAC_train/` | `rl_inverse_kinematics/revision_lr_ablation/` |
| `workspace_xian/Hindsight-Experience-Replay/` | `rl_inverse_kinematics/her_baseline_attempt/` |
| `test.py`, `RL_eval.py` | `evaluation/simulation/` |
| `unigrasp_videos/` | `results/videos/unigrasp_baseline/` |
| `pointcloud_process.py` | `evaluation/real_world/` |
| `Occlusion/` | `evaluation/real_world/occlusion/` |

## Not copied (still available in `~/RAL-IROS2022`)

- All `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, compiled `render_balls_so.so` (rebuild with `compile_render_balls_so.sh`).
- `kelin.tar.xz` (5 identical copies, 99 MB each) and `kelin/` (4 identical copies, 149 MB each): a Nov 2021 snapshot of `gym/envs/kelin`, RUTH-only, superseded by the per-gripper `gym/` trees.
- `pybullet_object_models` duplicates (12 x 145 MB): three variants kept once each in `third_party/`, all other locations are symlinks.
- `FC/synthesis/` (now `grasp_quality/`; 525 MB of upstream plotly demo output) and `FC/grasp_quality.py` (Feb 2022 scratch, points at `~/workspace_kelin/RL_baselines`).
- `train_*/test.py` (4-line checkpoint loader pointing at `RL_IK`) and `train_*/RL_eval.py` (stale boxplot pointing at `RL_IK_2`); the maintained `RL_eval.py` is in `evaluation/simulation/`.
- `train_*/results_imgs.png`.
- `train_barrett/1111/`, `1111.py`, `apple*.png`, `banana*.png`, `show3d_screenshot_08.01.2023*.png`: Jan 2023 fruit/vegetable point clouds, a later project.
- `UniGrasp/saved_models/RECOVERED_FILES/` (empty), `exp_test.txt` (empty), `scene1.txt` (point dump), `ObjectPointClouds.zip` (duplicate of `data/ObjectPointClouds`), `gripper_urdf/2` (stray), `*.urdf.old`, empty `logging/` dirs.
- `feature_extraction/top1_f1_index_1.txt` (dump), `workspace_generation/log/kelin-v0/seed--{125,999}` (config only, no checkpoints).
- `27.txt` (empty).
- `videos/` (65 EfficientGrasp simulation MP4s): copied at first, then removed on request; the originals remain in `~/RAL-IROS2022/videos/`.
