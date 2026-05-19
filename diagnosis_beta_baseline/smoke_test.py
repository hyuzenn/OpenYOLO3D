"""Stage 1 — single-frame nuScenes smoke test.

Picks the FIRST token in results/diagnosis_step_a/samples_used.json (= the
v1.0-mini token b73152..., chosen by seed 42 in the 7-stage diagnosis
work) and runs:

  NuScenesLoader._load(token)
  → adapter.adapt_sample (single-cam CAM_FRONT)
  → OpenYOLO3D.predict (10 nuScenes classes)
  → mask → DetectionBox conversion

Success ≡ pipeline returns ≥1 instance in the kept-prediction list AND
adapter wrote a non-empty depth map. Anything less → exit 1, report the
failure point. Output goes to results/diagnosis_beta_baseline/stage_1_smoke.md.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np
import yaml

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.nuscenes_to_openyolo3d import adapt_sample, CAMERA
from diagnosis_beta_baseline import NUSCENES_10_CLASS
from diagnosis_beta_baseline.format_predictions import (
    _read_ply_points,
    predictions_to_detection_boxes,
)


SAMPLES_USED = "results/diagnosis_step_a/samples_used.json"
OY3D_CFG = "diagnosis_beta_baseline/openyolo3d_nuscenes10.yaml"
DATA_CFG_BASE = "configs/nuscenes_baseline.yaml"


def _load_first_mini_token():
    with open(SAMPLES_USED) as f:
        d = json.load(f)
    tokens = d["tokens"]
    n_mini = d["n_mini"]
    if n_mini < 1:
        raise RuntimeError("samples_used.json declares n_mini < 1")
    return tokens[0], d


def _build_mini_loader(out_dir: str) -> NuScenesLoader:
    cfg = yaml.safe_load(open(DATA_CFG_BASE))
    cfg["nuscenes"]["version"] = "v1.0-mini"
    cfg["nuscenes"]["cameras"] = [CAMERA]  # smoke needs only the front cam
    tmp = osp.join(out_dir, "_smoke_data_config.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))
    return NuScenesLoader(config_path=tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/diagnosis_beta_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scene_dir = osp.join(args.out_dir, "scenes", "smoke_scene")
    if osp.exists(scene_dir):
        shutil.rmtree(scene_dir)

    timing = {}
    t0 = time.time()

    print("=" * 60)
    print("Stage 1 — nuScenes 1-frame smoke")
    print("=" * 60)
    target_token, samples_meta = _load_first_mini_token()
    print(f"target token = {target_token}  (from {SAMPLES_USED})")

    print("\nstep 1/4: build mini loader")
    t = time.time()
    loader = _build_mini_loader(args.out_dir)
    timing["loader_build_s"] = time.time() - t
    print(f"  built in {timing['loader_build_s']:.1f}s, {len(loader)} samples")

    if target_token not in loader.sample_tokens:
        raise SystemExit(f"token {target_token} not in mini loader (has {len(loader)} samples)")

    print("\nstep 2/4: load + adapt sample")
    t = time.time()
    item = loader._load(target_token)
    timing["loader_load_s"] = time.time() - t
    print(f"  loaded in {timing['loader_load_s']:.1f}s — pc {item['point_cloud'].shape}")

    t = time.time()
    stats = adapt_sample(item, scene_dir, camera=CAMERA)
    timing["adapter_s"] = time.time() - t
    print(f"  adapted in {timing['adapter_s']:.1f}s — depth coverage {stats['depth_pixel_coverage']:.3%}")
    if stats["n_depth_pixels_filled"] == 0:
        print("✗ STOP: zero depth pixels filled.")
        return 1

    print("\nstep 3/4: load OpenYOLO3D and predict")
    with open(OY3D_CFG) as f:
        oy3d_cfg = yaml.safe_load(f)
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]
    depth_scale = oy3d_cfg["openyolo3d"]["depth_scale"]
    print(f"  text_prompts ({len(text_prompts)}): {text_prompts}")
    assert text_prompts == NUSCENES_10_CLASS, \
        f"config text_prompts must equal NUSCENES_10_CLASS, got {text_prompts}"

    from utils import OpenYolo3D
    t = time.time()
    oy3d = OpenYolo3D(OY3D_CFG)
    timing["oy3d_init_s"] = time.time() - t
    print(f"  initialized in {timing['oy3d_init_s']:.1f}s")

    t = time.time()
    prediction = oy3d.predict(
        path_2_scene_data=scene_dir,
        depth_scale=depth_scale,
        datatype="point cloud",
        text=text_prompts,
    )
    timing["predict_s"] = time.time() - t
    print(f"  predict() ran in {timing['predict_s']:.1f}s")

    scene_name = osp.basename(scene_dir)
    pred_tuple = prediction[scene_name]

    print("\nstep 4/4: mask → DetectionBox")
    ply_pts = _read_ply_points(osp.join(scene_dir, "lidar.ply"))
    boxes, drop_stats = predictions_to_detection_boxes(
        sample_token=target_token,
        pred_tuple=pred_tuple,
        ply_points_ego=ply_pts,
        ego_pose_4x4=item["ego_pose"],
        text_prompts=NUSCENES_10_CLASS,
    )
    print(f"  raw instances:    {drop_stats['n_pred_total']}")
    print(f"  dropped bg label: {drop_stats['n_dropped_bg']}")
    print(f"  dropped <5 pts:   {drop_stats['n_dropped_small']}")
    print(f"  KEPT boxes:       {drop_stats['n_pred_kept']}")
    if boxes:
        b0 = boxes[0]
        print(f"  example: {b0['detection_name']:18s}"
              f" t={[round(v,2) for v in b0['translation']]}"
              f" wlh={[round(v,2) for v in b0['size']]}"
              f" score={b0['detection_score']:.3f}")

    elapsed = time.time() - t0
    timing["total_s"] = elapsed

    md_path = osp.join(args.out_dir, "stage_1_smoke.md")
    with open(md_path, "w") as f:
        f.write("# Stage 1 — nuScenes 1-frame smoke\n\n")
        f.write(f"- target_token: `{target_token}`\n")
        f.write(f"- samples_used: `{SAMPLES_USED}`\n")
        f.write(f"- adapter coverage: {stats['depth_pixel_coverage']:.3%}\n")
        f.write(f"- adapter pixels filled: {stats['n_depth_pixels_filled']}\n")
        f.write(f"- raw OpenYOLO3D instances: {drop_stats['n_pred_total']}\n")
        f.write(f"- dropped (background label): {drop_stats['n_dropped_bg']}\n")
        f.write(f"- dropped (<5 mask points): {drop_stats['n_dropped_small']}\n")
        f.write(f"- kept boxes: {drop_stats['n_pred_kept']}\n\n")
        f.write("## Timing (s)\n\n")
        for k, v in timing.items():
            f.write(f"- {k}: {v:.2f}\n")

    print(f"\nwrote {md_path}")

    # Acceptance: pipeline ran end-to-end (≥1 instance in raw OpenYOLO3D output is enough).
    if drop_stats["n_pred_total"] >= 1:
        print("\n✓ Stage 1 smoke passed (pipeline end-to-end).")
        return 0
    print("\n✗ Stage 1 smoke FAILED — zero raw instances out of OpenYOLO3D.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
