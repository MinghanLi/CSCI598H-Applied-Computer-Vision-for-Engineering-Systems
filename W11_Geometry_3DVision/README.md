# 3D Scene Visualization Project

This project helps students inspect **Mip-NeRF 360** scenes through COLMAP cameras and sparse 3D points.

## Dataset

Reference: <https://arxiv.org/abs/2111.12077>

Place the dataset under:

```text
data/360_v2/
```

Available scenes in this project:

`bicycle`, `bonsai`, `counter`, `garden`, `kitchen`, `room`, `stump`

Note: local notes in this repo say `flowers` and `treehill` are not publicly redistributable.

## Data Structure

Each scene follows this layout:

```text
data/360_v2/bonsai/
├── images/
├── images_2/
├── images_4/
├── images_8/
├── poses_bounds.npy
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

- `images*`: original and downsampled input images
- `poses_bounds.npy`: pose and bound metadata used by NeRF pipelines
- `cameras.bin`: camera intrinsics
- `images.bin`: registered images and camera poses
- `points3D.bin`: sparse COLMAP point cloud

## Step 1: Visualize the Sparse 3D Point Cloud

This step visualizes the existing **COLMAP sparse reconstruction**: camera poses, camera frustums, and sparse 3D points.

Install dependencies:

```bash
python -m pip install numpy plotly
conda install conda-forge::colmap
```

Run:

```bash
python visualize_mipnerf360.py --scene bonsai
```

You can pass either a scene name or a full path such as `data/360_v2/room`.

Output:

```text
output/<scene_name>_colmap_visualization.html
```

Useful optional flags:

```bash
python visualize_mipnerf360.py --scene bonsai --max_points 20000 --every_nth_camera 5
```

- `--max_points`: reduce displayed sparse points
- `--every_nth_camera`: show fewer cameras
- `--camera_scale`: control frustum size
- `--output`: set a custom HTML path
- `--data_root`: change the base dataset folder

## Step 2: Run COLMAP Dense Reconstruction

The dataset already contains **COLMAP poses** and **sparse points**, so you can continue with **dense reconstruction**.

The standard COLMAP workflow is:

1. `SfM` recovers the sparse scene and camera poses.
2. `Multi-View Stereo (MVS)` recovers the dense scene representation.

In this project, Step 1 helps you inspect the sparse result first, and Step 2 moves on to dense reconstruction from that existing COLMAP output.

Important: save the **depth maps** produced during dense reconstruction, since they will be needed for later analysis and downstream tasks.
