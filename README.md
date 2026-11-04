# EfficientGrasp: A Unified Data-Efficient Learning to Grasp Method for Multi-fingered Robot Hands

<p align="center">
  <a href="https://arxiv.org/abs/2206.15159"><img src="https://img.shields.io/badge/Paper-333333?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://github.com/KelinLi-ReCAST/EfficientGrasp"><img src="https://img.shields.io/badge/Code-333333?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/datasets/KelinLiIC/EfficientGrasp"><img src="https://img.shields.io/badge/Dataset-333333?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"></a>
</p>

Code and data for the paper [*EfficientGrasp: A Unified Data-Efficient Learning to Grasp Method for Multi-fingered Robot Hands*](https://arxiv.org/abs/2206.15159) (Kelin Li, Nicholas Baron, Xian Zhang and Nicolas Rojas, IEEE Robotics and Automation Letters 2022, presented at IROS 2022).

The repository contains the three phases of the method: the fingertip-workspace autoencoder that produces the gripper feature, the point set selection network (PSSN) that selects contact points on an object point cloud, and the reinforcement-learning policies that solve the gripper inverse kinematics. It also contains the PyBullet grasping trials, the grasp quality computation and the processing scripts of the real-world experiments.

A gripper is described by the workspace of its fingertips instead of a URDF model, so the method also applies to grippers with closed kinematic loops such as the [RUTH hand](https://doi.org/10.1177/02783649211048929), which a URDF cannot represent. The workspace is encoded into a 256-d feature, concatenated with the [PointNet++](https://github.com/charlesq34/pointnet2) feature of the object point cloud, and fed to a PSSN derived from [UniGrasp](https://github.com/stanford-iprl-lab/UniGrasp). The arm then moves to a rest pose facing the selected contact points and a soft actor-critic policy drives the fingertips onto them. Three grippers are supported: the RUTH hand, the Robotiq 3-Finger gripper and the BarrettHand BH8-280, each mounted on a UR5.

<p align="center">
  <img src="media/readme/pipeline.png" width="860" alt="Flowchart of EfficientGrasp: feature extraction, contact points generation, inverse kinematics computation">
</p>

<p align="center">
  <img src="media/readme/sim_ruth_pi00.gif" width="32%" alt="UR5 with the RUTH hand grasping the mustard bottle in simulation">
  <img src="media/readme/sim_ruth_pi0pi.gif" width="32%" alt="UR5 with the RUTH hand grasping the mustard bottle in simulation">
  <img src="media/readme/sim_ruth_pipipi.gif" width="32%" alt="UR5 with the RUTH hand grasping the mustard bottle in simulation">
</p>

*UR5 with the RUTH hand grasping the YCB mustard bottle in PyBullet (3× speed).*

---

## 1. Hardware

Training and the simulation trials need only a PC. The robot, the hand and the camera are needed for the real-world experiments (§6).

| Component | Role | Notes |
|---|---|---|
| Linux PC with an NVIDIA GPU | Runs PyBullet and trains the autoencoder, the PSSN and the RL policies | The paper used an Intel Core i7-9700K with an RTX 2080 Ti for the PSSN and an RTX 3070 for the RL policies. `grasp_quality/utils/Losses.py` requires CUDA |
| UR5 robot arm | Carries the gripper | Simulated in PyBullet; the real arm is used in §6 |
| RUTH hand | Three-finger underactuated gripper with a five-bar reconfigurable palm (closed loop) | Simulated and real |
| Robotiq 3-Finger gripper, BarrettHand BH8-280 | Grippers without closed loops, used for the comparison with UniGrasp | Simulation only |
| Intel RealSense D435i | Depth image of the object, converted to a point cloud | Real-world experiments only |

<p align="center">
  <img src="media/readme/ruth_modes.png" width="620" alt="RUTH hand in different operation modes">
</p>

*The RUTH hand in three operation modes. The finger bases move with the five-bar palm, so the "most open" and "most closed" configurations that UniGrasp needs are ambiguous.*

<p align="center">
  <img src="media/readme/real_objects.jpg" width="38%" alt="YCB objects used in the real-world experiments">
  &nbsp;
  <img src="media/readme/ruth_real_grasp.jpg" width="59%" alt="RUTH hand configuring from its initial position to the contact points">
</p>

*Left: the YCB objects of the real-world experiments. Right: the RUTH hand on the UR5 moving from its initial configuration to the selected contact points (red, green, blue).*

---

## 2. Repository layout

Code, data and media live in separate top-level folders. Everything marked *(Hugging Face)* below is not in git; it is downloaded from [Hugging Face](https://huggingface.co/datasets/KelinLiIC/EfficientGrasp) (§3.4). The code is split by functional module. `contact_point_selection/UniGrasp/` is derived from [UniGrasp](https://github.com/stanford-iprl-lab/UniGrasp) and `grasp_quality/` from [diverse-and-stable-grasp](https://github.com/tengyu-liu/diverse-and-stable-grasp) (which vendors DeepSDF); both keep their own licences. Each gripper directory in `rl_inverse_kinematics/` ships its own copy of gym 0.14.0 with the custom environment.

```
environment.yml                        conda environment "RAL2023"
gripper_representation/
  feature_extraction/
    workspace_generator.py             fingertip workspaces of parametric three-finger grippers
    workspace.py                       RUTH fingertip workspace sampled in PyBullet
    encoder.py                         PointNet autoencoder with the coupled Chamfer distance
    encoder_test.py                    extracts the feature of a gripper with the trained encoder
  ruth_workspace_sampling/             RUTH workspace sampled through the gym environment
contact_point_selection/
  UniGrasp/
    point_set_selection/
      train_pssn.py                    staged PSSN training on the workspace feature
      unigrasp.py                      PSSN inference: point cloud in, contact points out
      pointnet4/                       PointNet++ and its TF operators (to be compiled)
    gripper_urdf/  simulation/  vis_3d/    unmodified UniGrasp
grasp_quality/
  utils/Losses.py                      force-closure estimator; its __main__ computes the GQS
rl_inverse_kinematics/
  ruth/  robotiq_3f/  barrett/         one self-contained directory per gripper
    train.py  val.py                   SAC training and validation
    sac.py  model.py  replay_memory.py
    demo.py  RL_gripper.py  move_ur.py full grasp pipeline in PyBullet
    gym/envs/kelin/                    environment kelin-v0, URDFs of UR5 + gripper
  early_prototype/                     first SAC version
  revision_lr_ablation/                learning-rate study (plots on Hugging Face)
  her_baseline_attempt/                HER baseline, not connected to the environment
evaluation/
  simulation/   test.py  RL_eval.py    trial driver (16 YCB objects), RL-IK error plot
  real_world/                          point cloud processing and target pose scripts
    object_pointclouds/                RealSense captures, 2048 × 3 (Hugging Face)
    object_photos/  occlusion/         contact-point renders, occlusion analysis (Hugging Face)
results/                               (Hugging Face)
  rl_ik_error/  pssn_accuracy/         evaluation logs behind the figures and tables of the paper
  videos/                              PyBullet recordings of the grasp trials
third_party/                           pybullet-object-models, YCB meshes (Hugging Face)
media/readme/                          images and GIFs used in this README
docs/file_reference.md                 per-file notes and provenance of every folder
```

The training data (`gripper_representation/data/`, `contact_point_selection/UniGrasp/data/`), the trained models (`saved_models/`, `rl_inverse_kinematics/<gripper>/log/`) and the meshes of the URDFs are also on Hugging Face, at the same paths.

---

## 3. Installation

### 3.1 Conda environment

```bash
conda env create -f environment.yml    # Python 3.6, PyTorch 1.10.1 (CUDA 11.3), PyBullet 3.2.1, Open3D 0.14.1
conda activate RAL2023
```

This environment covers `rl_inverse_kinematics/`, `evaluation/` and `grasp_quality/`. `train.py` logs to [visdom](https://github.com/fossasia/visdom) and uses `mpi4py`, both included.

### 3.2 TensorFlow (autoencoder and PSSN only)

`environment.yml` does not contain TensorFlow. The autoencoder (`encoder.py`) and the PSSN use the TensorFlow 1 API with `tflearn`, and the PSSN also needs `tensorflow.contrib.slim`, so a TensorFlow 1.x installation is required, preferably in a separate environment.

The PointNet++ operators have to be compiled once against that TensorFlow:

```bash
cd contact_point_selection/UniGrasp/point_set_selection/pointnet4/tf_ops
(cd sampling         && sh tf_sampling_compile.sh)
(cd grouping         && sh tf_grouping_compile.sh)
(cd 3d_interpolation && sh tf_interpolate_compile.sh)
```

### 3.3 gym

Do not rely on the `gym==0.21.0` listed in `environment.yml`. The environment `kelin-v0` is registered in the gym 0.14.0 copy inside each gripper directory, which is picked up when the scripts are started from that directory:

```bash
cd rl_inverse_kinematics/ruth     # or robotiq_3f / barrett
python -c "import gym; print(gym.__file__)"   # must point into this directory
```

### 3.4 Data and models

GitHub holds the code, the URDF/xacro files and the media of this README. The datasets, trained models, meshes, evaluation logs, figures and videos are on Hugging Face: [KelinLiIC/EfficientGrasp](https://huggingface.co/datasets/KelinLiIC/EfficientGrasp). The dataset has the same layout as this repository, so download it into the root of your clone:

```bash
pip install -U huggingface_hub
hf download KelinLiIC/EfficientGrasp --repo-type dataset --local-dir . --exclude README.md --exclude .gitattributes
tar -xf gripper_representation/data/WorkspaceArrays.tar -C gripper_representation/data   # autoencoder training set, 44 751 files
```

The downloaded folders are kept out of git by `.gitignore`.

The MANO models (`grasp_quality/utils/manopth/mano/models/` and `grasp_quality/data/mano/MANO_RIGHT.pkl`) are licence-restricted and are not redistributed: obtain them from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de).

The PSSN is trained on the UniGrasp dataset, which is not redistributed here either: download `train_data.tar.gz` and `test_data.tar.gz` from the links in `contact_point_selection/UniGrasp/README.md` and extract them to `contact_point_selection/UniGrasp/data/train_data/` and `test_data/`.

`rl_inverse_kinematics/*/pybullet_object_models` are symbolic links into `third_party/`. They are broken until `third_party/` is in place. The three variants differ only in the mass, friction and inertia of the mustard bottle (`docs/file_reference.md`).

### 3.5 Paths

The absolute paths of the machines the code was written on have been replaced by paths relative to the repository root, so the scripts run from any location. Each script derives the root from its own location. The changes, from the original path to the new one:

- `/home/kelin/workspace_kelin/RAL-IROS2022/train_ruth` (`train_robotiq`, `train_barrett`) → `rl_inverse_kinematics/ruth` (`robotiq_3f`, `barrett`): URDFs, trained policies in `log/`, `contact_list.npy` and `obj_pc/`.
- `import train_ruth.demo` (`train_robotiq`, `train_barrett`) in `evaluation/simulation/test.py` → the gripper directory is put on `sys.path` and `demo` is imported from it.
- `/home/kelin/workspace_xian/urdf/ur5_plus_RUTH.urdf` (`ruth/demo.py`) → `rl_inverse_kinematics/ruth/gym/envs/kelin/urdf/ur5_plus_RUTH.urdf`.
- `/home/kelin/workspace_kelin/RL_IK`, `RL_IK_2` and `/home/kelin/workspace_xian/RUTH_RL` → the gripper directory of the script (`contact_points.npy`, `contact_list.npy`, `log/`).
- `/home/kelin/workspace_kelin/previous_work/feature_extraction` and `/home/robin-lab/Kelin/saved_models` → `gripper_representation/feature_extraction` (`saved_models/`, `ruthArrays/`, `fo_t1o_t2o.urdf`).
- `/home/kelin/workspace_kelin/previous_work/WorkspaceArrays` → `gripper_representation/data/WorkspaceArrays`.
- `/home/kelin/workspace_xian/meshes/RUTH` (`fo_t1o_t2o.urdf`) → `rl_inverse_kinematics/ruth/gym/envs/kelin/meshes/RUTH`.
- `/home/robin-lab/Kelin/UniGrasp-master` and `/home/kelin/workspace_kelin/previous_work/UniGrasp` → `contact_point_selection/UniGrasp`.
- `/media/robin-lab/Jim/train_data`, `test_data` → `contact_point_selection/UniGrasp/data/train_data`, `test_data`.
- `/home/kelin/Downloads/contact_points` → `contact_point_selection/UniGrasp/data/ObjectPointClouds/Contact_Points/`, read as `Ruth_Ycb<Object>.npy` (`ruth/demo.py`, `grasp_quality`). These files are not on Hugging Face and are written with Step 2; the PSSN outputs on Hugging Face are `Contact_Points/{Barret,Robotiq,Ruth}/<Gripper>_<Object>.npy`.
- `/home/kelin/Downloads/contact_points_U` → `contact_point_selection/UniGrasp/data/ObjectPointClouds/Contact_Points_UniGrasp/`, read as `Robotiq_Ycb<Object>.npy` and `BH_Ycb<Object>.npy` (`robotiq_3f/demo.py`, `barrett/demo.py`). These files are not on Hugging Face either.
- `/home/kelin/Downloads/contact_points_revision` → `evaluation/real_world/contact_points/`.
- `/home/kelin/Downloads/revision_real_pc` → `evaluation/real_world/raw_pointclouds/`.
- `/home/kelin/Downloads/revision_sparse_real_pc` → `evaluation/real_world/object_pointclouds/`.
- `/home/kelin/Downloads/target_pos` → `evaluation/real_world/target_poses/`.
- `/home/kelin/workspace_kelin/RAL-IROS2022/train_<gripper>/epoch_*.npy` (`RL_eval.py`) → `results/rl_ik_error/<gripper>/`.
- `/home/kelin/workspace_kelin/RAL-IROS2022/videos`, `unigrasp_videos` → `results/videos/efficientgrasp/`, `results/videos/unigrasp_baseline/`, created when the first trial is recorded.
- `/home/robin-lab/anaconda3/envs/kelin/lib/python3.6/site-packages/tensorflow` (`pointnet4/tf_ops/*/tf_*_compile.sh`) → `TF_ROOT`, taken from the TensorFlow of the active Python environment unless set beforehand.
- `prefix: /home/kelin/anaconda3/envs/RAL2022` in `environment.yml` → removed; conda uses the environment name.

The UniGrasp files under `contact_point_selection/UniGrasp/gripper_urdf/` and `simulation/` still contain the paths of the UniGrasp authors. They are upstream code and are not called by the steps below.

---

## 4. Usage

### Step 1: Gripper workspace feature

The fingertip workspace of an *N*-finger gripper is an *L* × 3*N* array; each row is one set of fingertip positions. Three-finger grippers give *L* × 9.

```bash
cd gripper_representation/feature_extraction
python workspace_generator.py     # training set: ws<N>.npy for parametric grippers (5-bar, 4-bar, 3-RRR palms)
python encoder.py                 # train the autoencoder
python encoder_test.py            # restore 9999model.ckpt and extract the feature of one gripper
```

- `encoder.py` takes inputs of 576 × 9, uses 6 convolutional layers and a max-pooling layer to reach the 256-d feature, and a decoder with 3 fully connected layers. The loss is the coupled Chamfer distance (Eq. 2 of the paper): the 9 columns are split into the 3 fingers, and the per-finger distances are summed over a shared row index before the minimum is taken.
- As committed, `encoder.py` loads `ruthArrays/`. The block that loads the full training set is commented out at the top of the file.
- `encoder_test.py` writes `mean.npy`, `max.npy` and `min.npy` next to the workspace arrays in `gripper_dir`.
- The PSSN reads the feature of each gripper from `contact_point_selection/UniGrasp/data/gripper_features/Data_DB/<gripper>/workspace.npy` (1 × 256), with `<gripper>` one of `ruth`, `robotiq_3f`, `bh_282`, `kinova_kg3`.

For a new gripper, sample its fingertip workspace in any way (for RUTH this is done in PyBullet with `workspace.py` or `ruth_workspace_sampling/workspace.py`), extract the feature, and store it under a new `Data_DB/<gripper>/`.

### Step 2: Contact points (PSSN)

**Training.** The three stages are trained one after the other, with the earlier stages frozen. Training is restricted to the gripper IDs 11 (Robotiq-3F), 12 (BarrettHand) and 13 (Kinova-3F).

```bash
cd contact_point_selection/UniGrasp/point_set_selection
python train_pssn.py
```

The stage is selected in the code, not on the command line: the `__main__` block restores a checkpoint with `restore_stage2_v2` / `restore_stage3_v2` and the matching training operation is enabled in `train()`. As committed it continues stage three from `221model.ckpt`.

**Inference.** `unigrasp.py` takes an object point cloud of 2048 × 3 and returns the 10 best-ranked contact points for each of the three fingers. Its `__main__` block is commented out, so call it from Python:

```python
# run from contact_point_selection/UniGrasp/point_set_selection
import numpy as np
import unigrasp

unigrasp.restore_stage3(220)                       # saved_models/point_set_selection/220model.ckpt
p1, p2, p3 = unigrasp.test(0, np.load('../data/ObjectPointClouds/<object>.npy'))
```

| Setting | Where | Meaning |
|---|---|---|
| `gripper_index` | first lines of `test()` | `11`: Robotiq-3F, `12`: BarrettHand, `13`: RUTH. Set to `11` as committed |
| checkpoint | argument of `restore_stage3` | `220` is the EfficientGrasp PSSN |

RUTH has no labels of its own in the UniGrasp dataset. It uses ID 13, i.e. the network is trained with the Kinova-3F labels and queried with the RUTH workspace feature.

`test()` opens an Open3D window showing the point cloud with the top-1 contact points in red, green and blue.

<p align="center">
  <img src="media/readme/real_contact_points.png" width="720" alt="Contact points selected on real point clouds">
</p>

*Contact points selected for the RUTH hand on point clouds captured with the RealSense camera. Renders of all captures are in `evaluation/real_world/object_photos/` on Hugging Face.*

### Step 3: Train the inverse-kinematics policy

One policy is trained per gripper. The end-effector orientation is fixed during training and the targets are expressed in the end-effector frame, so the policy is independent of the arm pose.

```bash
python -m visdom.server &          # train.py connects to visdom
cd rl_inverse_kinematics/ruth      # or robotiq_3f / barrett
python train.py --seed 0 --cuda
```

| Gripper | URDF | Action | Observation |
|---|---|---|---|
| `ruth` | `ur5_plus_RUTH.urdf` | 4: last UR5 joint, 2 palm motors, finger flexion | 4 joint values + 9 target coordinates |
| `robotiq_3f` | `ur5_plus_robotiq_3f.urdf` | 6 | 6 + 9 |
| `barrett` | `ur5_plus_barrett.urdf` | 9 | 9 + 9 |

Defaults of `train.py` are the values used in the paper: Gaussian policy, `--alpha 0.8`, `--lr 0.003`, `--tau 0.005`, `--gamma 0.99`, `--batch_size 256`, 10 episodes per epoch, at most 100 steps per episode. Targets are drawn from `contact_points.npy`, the sampled fingertip workspace of the gripper. Checkpoints go to `log/kelin-v0/seed--<seed>/`.

The RUTH palm is a closed five-bar linkage. Its URDF is an open chain, and the loop is closed at run time with a PyBullet point-to-point constraint between `Link_2` and `Link_4` (`gym/envs/kelin/move_env.py`, `demo.py`).

**Validation.** `val.py` runs a policy for 100 episodes without rendering and stores the mean fingertip error of each episode in millimetres:

```bash
python val.py
```

The command-line arguments of `val.py` are overwritten in its `__main__` block. Set the run (`args.fpath`), the epoch (`args.itr`) and the name of the output file (`np.save` in `run_policy`) there.

### Step 4: Grasp trials in simulation

```bash
python evaluation/simulation/test.py --gripper ruth     # or robotiq / barrett
```

This runs 16 YCB objects in 4 orientations (`000`, `pi00`, `pi0pi`, `pipipi`) in the PyBullet GUI, and records each one. The contact points are not computed on the fly; they are read from the files written in Step 2, saved under the names and folders listed in §3.5. Every trial calls `demo.main` of the chosen gripper:

1. **Scene.** The UR5 with the gripper is loaded and moved to its start configuration.
2. **Contact points.** The best-ranked set (`point_nos = 0`) is read for the object.
3. **Rest pose.** The end effector is placed 0.3 m from the centroid of the contact points, perpendicular to their plane (`move_ur.calc_target_pos`).
4. **Gripper configuration.** The object is spawned at `[0.5, 0, 0.01]` and the trained policy is applied for `rl_step = 3` steps.
5. **Grasp.** The arm approaches along the normal by `depth = 0.24` m, the fingers close, and the object is lifted.

The policy is loaded by `RL_gripper.py` from `log/kelin-v0/seed--0` (`ruth`, `robotiq_3f`) or `seed--1` (`barrett`).

### Step 5: Grasp Quality Score

```bash
python grasp_quality/utils/Losses.py
```

The `__main__` block evaluates the force-closure term of [Liu et al.](https://arxiv.org/abs/2104.09194) for the 10 best contact-point sets of each of the 16 YCB objects, with the force directions taken along the line between each contact point and the centroid of the three points, and saves a 16 × 10 array. The GQS of the paper follows from this value through Eq. 5; sets with GQS above 0.75 are considered force-closure grasps.

---

## 5. Output

| File | Content |
|---|---|
| `gripper_representation/feature_extraction/saved_models/workspace/<epoch>model.ckpt` | Autoencoder checkpoints |
| `<gripper_dir>/{mean,max,min}.npy` | Workspace feature written by `encoder_test.py` |
| `Ruth_Ycb<Object>[_<ori>].npy`, `Robotiq_Ycb<Object>_<ori>.npy`, `BH_Ycb<Object>_<ori>.npy` | Contact points, 10 × 9: one row per ranked set, three points × xyz. Metres for the simulated objects, millimetres for the real captures |
| `top1_f{1,2,3}_index_*.txt`, `scene1.txt` | Top-1 contact point of each finger and the rest of the point cloud, written by `unigrasp.test` into the working directory |
| `rl_inverse_kinematics/<gripper>/log/kelin-v0/seed--<N>/` | `agent.pkl`, `GaussianPolicy.*.pt`, `QNetwork.*.pt`, `config.json`, `progress.txt` |
| `epoch_<N>.npy` | Output of `val.py`: mean fingertip error per episode (mm), 100 episodes. The published logs are in `results/rl_ik_error/<gripper>/` |
| `results/videos/{efficientgrasp,unigrasp_baseline}/<gripper>_<Object>_<ori>.mp4` | PyBullet recording of one trial |

Results reported in the paper (5 trials per object; a grasp is successful if the object is held for 5 s above the table):

| Gripper | UniGrasp success (%) | EfficientGrasp success (%) | EfficientGrasp mean GQS |
|---|:---:|:---:|:---:|
| Robotiq-3F (simulation) | 81.3 | 85.0 | 0.8387 |
| BarrettHand (simulation) | 85.0 | 87.5 | 0.8518 |
| RUTH (simulation) | not applicable | 83.8 | 0.8587 |
| RUTH (real world) | not applicable | 83.3 | 0.8868 |

Using the workspace instead of gripper point clouds reduced the memory needed for feature extraction from 4777 MB to 873 MB (81.7 %).

---

## 6. Optional: real-world experiments

The real-world experiments used the RUTH hand on a UR5 with a RealSense D435i placed as in simulation. **The code that captures the point cloud and the code that commands the UR5 and the hand are not part of this repository.** What is included are the scripts between capture and execution, in `evaluation/real_world/`, and the captured point clouds, on Hugging Face (§3.4):

| Script | Input | Output |
|---|---|---|
| `pointcloud_process.py` | Captured point cloud | The same cloud downsampled to 2048 points, the input size of the PSSN |
| Step 2 (`unigrasp.py`) | 2048 × 3 point cloud | Contact points `Ruth_Ycb<object>.npy` (mm) |
| `target_position.py` | Contact points | UR5 target: position (3), orientation quaternion (4), centroid of the contact points (3) |
| `show_contact_points.py` | Point cloud and contact points | Viewer window with the contact points in red, green and blue |

Set `obj_name` at the top of each script. `target_position.py` and `show_contact_points.py` import `move_ur` and `show3d_balls` from `rl_inverse_kinematics/ruth/`, so run them with that directory on `PYTHONPATH`. `show3d_balls` needs its C++ renderer, built with `compile_render_balls_so.sh`.

`occlusion/` on Hugging Face holds the material of the failure analysis: RGB captures, point clouds of self-occluded objects, and four simulation clips of the mustard bottle.

---

## 7. Known limitations

- The raw RealSense point clouds before downsampling and the saved UR5 target poses were not preserved; `evaluation/real_world/object_pointclouds/` holds the downsampled clouds.
- `demo.py` cannot be started on its own (`__main__` calls `main()` without arguments); use `evaluation/simulation/test.py`.
- The object mass is assumed to be uniformly distributed, and success in the real world depends on friction and weight that the simulation does not model.

---

## License

The code is released under the [MIT License](LICENSE). Code derived from other projects keeps its own licence: `contact_point_selection/UniGrasp/` (UniGrasp), `grasp_quality/` (diverse-and-stable-grasp, including DeepSDF and manopth; the MANO models are licence-restricted), and the copies of gym in `rl_inverse_kinematics/` (see the `LICENSE` files in those folders).

---

## Citation

```bibtex
@article{li2022efficientgrasp,
  title   = {EfficientGrasp: A Unified Data-Efficient Learning to Grasp Method for Multi-Fingered Robot Hands},
  author  = {Li, Kelin and Baron, Nicholas and Zhang, Xian and Rojas, Nicolas},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {7},
  number  = {4},
  pages   = {8619--8626},
  year    = {2022},
  doi     = {10.1109/LRA.2022.3187875},
  note    = {arXiv:2206.15159}
}
```
