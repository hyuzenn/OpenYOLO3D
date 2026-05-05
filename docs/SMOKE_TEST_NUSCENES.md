# nuScenes ↔ OpenYOLO3D Smoke Test

End-to-end crash test: feed one nuScenes sample through OpenYOLO3D and
verify the pipeline runs to completion. **Output quality is not
evaluated** — this only checks that the data plumbing works.

## What it does

1. Loads ONE sample from nuScenes mini via `dataloaders/nuscenes_loader.py`.
2. Materializes it as an OpenYOLO3D scene directory using
   `adapters/nuscenes_to_openyolo3d.py`:
   - `color/0.jpg` ← CAM_FRONT image
   - `depth/0.png` ← LiDAR projected to CAM_FRONT (uint16 mm, sparse, 0 = invalid)
   - `poses/0.txt` ← `cam_to_ego` (treating ego as OpenYOLO3D's "world")
   - `intrinsics.txt` ← 4×4 padded K
   - `lidar.ply` ← LiDAR point cloud in ego frame, RGB-colored from CAM_FRONT
3. Runs `OpenYolo3D.predict()` with the outdoor text prompt set in
   `configs/openyolo3d_nuscenes.yaml`.
4. Saves the predicted instance .ply.

## Coordinate convention

The ego frame at the LIDAR_TOP timestamp is OpenYOLO3D's "world". Pose is
the cam-to-world matrix, which equals our dataloader's `cam_to_ego[CAM_FRONT]`
directly (no inversion). Point cloud is also in ego frame. `ego_pose` (ego
→ global) is unused at this stage.

## Single camera

Only `CAM_FRONT` is used. Multi-view fusion is a future task. The other
five cameras' data is loaded by `NuScenesLoader` but ignored here.

## Sparse depth (no completion)

`adapters/nuscenes_to_openyolo3d.project_lidar_to_depth` projects every
LiDAR point through `K · inv(cam_to_ego)` and writes the cam-frame z-value
to the corresponding pixel (with a per-pixel z-buffer). No interpolation,
no inpainting. Empty pixels stay 0; OpenYOLO3D's depth-loading code treats
0 as invalid.

`depth_scale = 1000` (uint16 mm) → max representable depth ≈ 65.5 m.
Points beyond that are dropped, not clipped, so OpenYOLO3D never sees a
phantom flat wall at 65.5 m.

## How to run

GPU is required (Mask3D + YOLO-World). On the cluster, submit the PBS
job from the worktree root:

```bash
cd /home/rintern16/OpenYOLO3D-nuscenes
qsub scripts/run_nuscenes_smoke_test.pbs
```

This activates `openyolo3d-dev` (clone of `openyolo3d` with
`nuscenes-devkit` installed) and writes
`results/smoke_nuscenes/run.log`.

Local equivalent (if you already have a GPU on the current node):

```bash
conda activate openyolo3d-dev
python run_nuscenes_smoke_test.py
```

## What success looks like

The driver prints a 4-step progress log and at the end:

```
✓ End-to-end smoke test passed
```

Acceptance criteria (all must hold):

| # | check |
|---|---|
| 1 | `python run_nuscenes_smoke_test.py` exits 0 |
| 2 | `results/smoke_nuscenes/prediction.ply` exists with non-zero size |
| 3 | OpenYOLO3D produced ≥1 mask proposal AND ≥1 label |
| 4 | No file modified outside `dataloaders/`, `adapters/`, `configs/`, `docs/`, `scripts/`, `results/`, top-level `run_nuscenes*.py` |

## Known limitations / non-goals

- Mask3D weights are `scannet200_val.ckpt` (indoor-trained). Running them
  on outdoor LiDAR is **not expected to produce semantically meaningful
  segments**. Crash-free completion + ≥1 mask is the bar at this step,
  not segmentation quality.
- nuScenes outdoor scenes are 100 m+ scale vs. ScanNet's ~10 m rooms. If
  Mask3D hits OOM or shape mismatches under that scale, the failure
  itself is the diagnostic — fallback strategies are intentionally not
  attempted in this task.
- Single-frame, single-camera. No multi-view aggregation, no temporal
  fusion.
- `ego_pose` (ego → global) is loaded by the dataloader but unused here.

## File map (this task only)

```
adapters/__init__.py
adapters/nuscenes_to_openyolo3d.py
configs/openyolo3d_nuscenes.yaml
run_nuscenes_smoke_test.py
scripts/run_nuscenes_smoke_test.pbs
docs/SMOKE_TEST_NUSCENES.md       (this file)
```
