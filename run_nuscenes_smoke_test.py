"""End-to-end smoke test: nuScenes sample → OpenYOLO3D pipeline.

Loads ONE nuScenes mini sample, materializes it as an OpenYOLO3D scene
directory via the adapter, runs OpenYOLO3D inference, and saves the
predicted .ply. Crash-free completion + ≥1 mask + ≥1 label is the only
success criterion. Output quality is NOT evaluated here.

Stops and reports the failure point on any error — no fallbacks.
"""

import argparse
import os
import os.path as osp
import shutil
import time

import yaml

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.nuscenes_to_openyolo3d import adapt_sample, CAMERA


def _count_outputs(prediction):
    """prediction is dict[scene_name -> tuple(masks, labels, scores)] per
    OpenYOLO3D's existing convention (see run_evaluation.py).

    masks has shape (N_points, N_instances), so we read the instance count
    off `labels` (one entry per instance). Using masks.shape[0] would
    accidentally report the point count instead.
    """
    n_masks, n_labels = 0, 0
    for _, val in prediction.items():
        try:
            labels = val[1]
            n_instances = int(labels.shape[0])
            n_masks += n_instances
            n_labels += n_instances
        except Exception:
            pass
    return n_masks, n_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--openyolo-config", default="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--out-dir", default="results/smoke_nuscenes")
    parser.add_argument("--sample-idx", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"Step 1/4: load nuScenes sample {args.sample_idx}")
    print("=" * 60)
    loader = NuScenesLoader(config_path=args.data_config)
    if args.sample_idx >= len(loader):
        raise SystemExit(f"sample_idx {args.sample_idx} ≥ {len(loader)} samples")
    item = loader[args.sample_idx]
    print(f"  sample_token = {item['sample_token']}")
    print(f"  point_cloud  = {item['point_cloud'].shape}")
    print(f"  image        = {item['images'][CAMERA].shape}")

    print()
    print("=" * 60)
    print("Step 2/4: adapt to OpenYOLO3D scene dir")
    print("=" * 60)
    os.makedirs(args.out_dir, exist_ok=True)
    scene_dir = osp.join(args.out_dir, "scene")
    if osp.exists(scene_dir):
        shutil.rmtree(scene_dir)

    stats = adapt_sample(item, scene_dir, camera=CAMERA)
    print(f"  → {scene_dir}")
    print(f"  image_hw            = {stats['image_hw']}")
    print(f"  lidar points        = {stats['n_lidar_points']}")
    print(f"  depth pixels filled = {stats['n_depth_pixels_filled']} ({stats['depth_pixel_coverage']:.3%})")
    print(f"  depth_scale (mm)    = {stats['depth_scale_mm']}")

    if stats["n_depth_pixels_filled"] == 0:
        print("\n✗ STOP: zero depth pixels filled — adapter produced an empty depth map.")
        return 1

    print()
    print("=" * 60)
    print("Step 3/4: load OpenYOLO3D and run predict()")
    print("=" * 60)
    print(f"  config = {args.openyolo_config}")
    with open(args.openyolo_config) as f:
        oy3d_cfg = yaml.safe_load(f)
    depth_scale = oy3d_cfg["openyolo3d"]["depth_scale"]
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]
    print(f"  depth_scale = {depth_scale}")
    print(f"  text prompts ({len(text_prompts)}) = {text_prompts}")

    from utils import OpenYolo3D
    print(f"\n  initializing OpenYOLO3D ... (cold start can take ~30s)")
    t_init = time.time()
    openyolo3d = OpenYolo3D(args.openyolo_config)
    print(f"  initialized in {time.time() - t_init:.1f}s")

    print(f"\n  running predict() ...")
    t_pred = time.time()
    prediction = openyolo3d.predict(
        path_2_scene_data=scene_dir,
        depth_scale=depth_scale,
        datatype="point cloud",
        text=text_prompts,
    )
    print(f"  predict() finished in {time.time() - t_pred:.1f}s")

    print()
    print("=" * 60)
    print("Step 4/4: save output and verify")
    print("=" * 60)
    out_ply = osp.join(args.out_dir, "prediction.ply")
    openyolo3d.save_output_as_ply(out_ply)
    ply_exists = osp.exists(out_ply)
    ply_size = osp.getsize(out_ply) if ply_exists else 0
    print(f"  output .ply = {out_ply}  (exists={ply_exists}, size={ply_size}B)")

    n_masks, n_labels = _count_outputs(prediction)
    print(f"  mask proposals = {n_masks}")
    print(f"  labels         = {n_labels}")

    elapsed = time.time() - t0
    print(f"\n  total wall time = {elapsed:.1f}s")

    print()
    if ply_exists and ply_size > 0 and n_masks >= 1 and n_labels >= 1:
        print("✓ End-to-end smoke test passed")
        return 0
    print("✗ Smoke test failed — see numbers above for the failure point")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
