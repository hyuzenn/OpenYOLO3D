# nuScenes Setup

This document covers how to wire the nuScenes dataset into OpenYOLO3D's
new `dataloaders/nuscenes_loader.py`. Only the dataloader is implemented
right now — pipeline integration (lifter, mask predictor, detector) is a
separate task.

## 1. Download nuScenes mini

Get the mini split (~4 GB) from <https://www.nuscenes.org/nuscenes>.
Sign in, accept the terms, and download:

- `v1.0-mini.tgz` (annotations + sample + sweep data)

## 2. Expected layout

The dataloader expects:

```
data/nuscenes/
├── v1.0-mini/         # JSON annotations
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── sample_annotation.json
│   └── ...
├── samples/           # Keyframe sensor data
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── ... (6 cameras)
│   └── LIDAR_TOP/
└── sweeps/            # Non-keyframe sensor data (used only when multi_sweep=true)
    ├── CAM_FRONT/
    ├── ...
    └── LIDAR_TOP/
```

Note: `data/` in this repo is a symlink. The dataloader reads via
`config["nuscenes"]["dataroot"]`, which defaults to `data/nuscenes`.

## 3. Environment

Use a clone of the existing env so the in-progress evaluation in
`openyolo3d` is not disturbed:

```bash
conda create -n openyolo3d-dev --clone openyolo3d
conda activate openyolo3d-dev
pip install nuscenes-devkit
```

`environment.yml` is intentionally NOT updated yet — only after the
dataloader is verified end-to-end.

## 4. Frame conventions (returned per sample)

| key | shape | frame |
|---|---|---|
| `point_cloud` | (N, 4) — x, y, z, intensity | EGO at LIDAR_TOP timestamp |
| `images[cam]` | (H, W, 3) uint8 RGB | image plane |
| `intrinsics[cam]` | (3, 3) | pixel coords from camera coords |
| `cam_to_ego[cam]` | (4, 4) | **cam → ego** (multiply a homogeneous point in camera coords by this to get its position in ego coords). To project ego-frame points into the image, invert this first. |
| `ego_pose` | (4, 4) | **ego → global** at the LIDAR_TOP timestamp |
| `timestamp` | int (microseconds) | LIDAR_TOP sample timestamp |
| `sample_token` | str | nuScenes sample token |
| `gt_boxes` | list of dicts | translation/size/rotation in **global** frame |

## 5. Smoke test (one sample)

From the repo root, with `openyolo3d-dev` active:

```bash
python run_nuscenes.py --config configs/nuscenes_baseline.yaml
```

Expected output: types and shapes for each key on a single sample. Does
not invoke OpenYOLO3D inference.

## 6. Projection sanity check

Verify the LiDAR-to-camera transform chain:

```bash
python -m dataloaders.sanity_check --config configs/nuscenes_baseline.yaml --cam CAM_FRONT --n 100
```

Picks 100 random LiDAR points (in ego frame), inverts the cam-to-ego
extrinsic to bring them into the camera frame, and projects with the
intrinsic. Reports the fraction of points (that are actually in front of
the camera) which land inside the image bounds. CAM_FRONT typically
captures roughly 1/6 of a 360° sweep — judge by the
`in_bounds / in_front` ratio, not the absolute count.
