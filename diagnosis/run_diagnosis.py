"""Tier-1 diagnosis: lifting reliability + Mask3D output distribution on nuScenes.

Runs OpenYOLO3D on a fixed-seed random subset of nuScenes mini samples
(CAM_FRONT only) and records per-detection / per-GT measurements.

Usage:
    python -m diagnosis.run_diagnosis --num-samples 20 --seed 42
"""

import argparse
import json
import os
import os.path as osp
import shutil
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import yaml

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.nuscenes_to_openyolo3d import adapt_sample, CAMERA
from diagnosis.measurements import (
    DISTANCE_BIN_LABELS,
    LIFTABLE_K_VALUES,
    OVERSEG_THRESHOLDS,
    distance_bin,
    project_points_to_camera,
    points_inside_2d_box,
    valid_projection_ratio,
    gt_box_to_ego,
    points_inside_3d_box,
    gt_box_visible,
    project_3d_box_corners_to_image,
    iou_2d_xyxy,
    is_oversegmented_at,
)
from diagnosis.aggregate import aggregate, render_all


PER_SAMPLE_TIMEOUT_S = 90


class SampleTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SampleTimeout()


def _select_samples(loader, num_samples, seed):
    rng = np.random.default_rng(seed)
    n_total = len(loader)
    if num_samples >= n_total:
        chosen = list(range(n_total))
    else:
        chosen = sorted(rng.choice(n_total, size=num_samples, replace=False).tolist())
    return chosen


def _measure_2d_detections(preds_2d_frame, uv_inbounds, depth_inbounds, image_hw, text_prompts):
    """preds_2d_frame: dict with bbox/labels/scores tensors (image-original resolution)."""
    detections = []
    bboxes = preds_2d_frame["bbox"].cpu().numpy() if hasattr(preds_2d_frame["bbox"], "cpu") else np.asarray(preds_2d_frame["bbox"])
    labels = preds_2d_frame["labels"].cpu().numpy() if hasattr(preds_2d_frame["labels"], "cpu") else np.asarray(preds_2d_frame["labels"])
    scores = preds_2d_frame["scores"].cpu().numpy() if hasattr(preds_2d_frame["scores"], "cpu") else np.asarray(preds_2d_frame["scores"])

    H, W = image_hw

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        # clip to image for area / projection
        x1c = max(0.0, x1)
        y1c = max(0.0, y1)
        x2c = min(float(W), x2)
        y2c = min(float(H), y2)
        box_clip = [x1c, y1c, x2c, y2c]
        box_area = max(0.0, (x2c - x1c)) * max(0.0, (y2c - y1c))

        inside = points_inside_2d_box(uv_inbounds, box_clip) if uv_inbounds.shape[0] > 0 else np.zeros((0,), dtype=bool)
        n_pts = int(inside.sum())
        if n_pts == 0:
            d_med = None
            density = None
        else:
            d_med = float(np.median(depth_inbounds[inside]))
            density = float(n_pts / box_area) if box_area > 0 else None

        vpr = valid_projection_ratio(uv_inbounds[inside], box_clip) if box_area > 0 else 0.0

        label_idx = int(labels[i])
        label_text = text_prompts[label_idx] if 0 <= label_idx < len(text_prompts) else f"class_{label_idx}"

        detections.append({
            "box_id": i,
            "label_text": label_text,
            "label_idx": label_idx,
            "confidence": float(scores[i]),
            "bbox_xyxy": [x1, y1, x2, y2],
            "box_area_pixels": float(box_area),
            "distance_to_ego": d_med,
            "distance_bin": distance_bin(d_med),
            "num_lidar_points": n_pts,
            "point_density": density,
            "valid_projection_ratio": float(vpr),
            "is_empty": n_pts == 0,
        })

    return detections


def _measure_gt_boxes(gt_boxes, ego_pose_4x4, pc_ego_xyz, K, T_cam_to_ego, image_hw, detections_xyxy):
    """One row per visible GT box."""
    H, W = image_hw
    rows = []
    for i, gt in enumerate(gt_boxes):
        box_ego = gt_box_to_ego(gt, ego_pose_4x4)
        if not gt_box_visible(box_ego, K, T_cam_to_ego, image_hw):
            continue
        center = box_ego.center
        d = float(np.linalg.norm(center))
        # 3D-box point containment in ego frame
        in3d = points_inside_3d_box(box_ego, pc_ego_xyz)
        n_pts = int(in3d.sum())

        gt_xyxy = project_3d_box_corners_to_image(box_ego, K, T_cam_to_ego, image_hw)
        match_idx = None
        match_iou = 0.0
        if gt_xyxy is not None and len(detections_xyxy) > 0:
            ious = [iou_2d_xyxy(gt_xyxy, det_box) for det_box in detections_xyxy]
            j = int(np.argmax(ious))
            if ious[j] >= 0.3:
                match_idx = j
                match_iou = float(ious[j])

        rows.append({
            "gt_id": i,
            "category": gt["category"],
            "distance_to_ego": d,
            "distance_bin": distance_bin(d),
            "center_ego": center.tolist(),
            "num_lidar_points": n_pts,
            "matched_to_detection": match_idx,
            "match_iou": match_iou if match_idx is not None else None,
            "projected_2d_bbox": gt_xyxy,
        })
    return rows


def _measure_mask3d(preds_3d, scene_dir):
    """preds_3d: tuple(masks_tensor, scores_tensor). masks shape (N_points, N_instances)."""
    masks, scores = preds_3d
    masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)

    if masks_np.ndim == 2:
        n_instances = int(masks_np.shape[1])
        instance_sizes = [int(masks_np[:, j].sum()) for j in range(n_instances)]
    else:
        n_instances = 0
        instance_sizes = []

    return {
        "num_instances": n_instances,
        "instance_sizes": instance_sizes,
        "instance_scores": [float(s) for s in scores_np.tolist()] if scores_np.size > 0 else [],
        "is_empty_frame": n_instances == 0,
        "is_oversegmented_at": {str(t): is_oversegmented_at(n_instances, t) for t in OVERSEG_THRESHOLDS},
    }


def _detection_loss_summary(detections, gt_rows):
    n_visible = len(gt_rows)
    n_with_pts = sum(1 for r in gt_rows if r["num_lidar_points"] >= 1)
    lost = 0
    for r in gt_rows:
        if r["num_lidar_points"] < 1:
            continue
        m = r["matched_to_detection"]
        if m is None:
            lost += 1
        else:
            if detections[m]["num_lidar_points"] == 0:
                lost += 1
    ratio = (lost / n_with_pts) if n_with_pts > 0 else None
    return {
        "num_gt_boxes_visible": n_visible,
        "num_gt_with_points": n_with_pts,
        "num_gt_lost_by_detection": lost,
        "detection_loss_ratio": ratio,
    }


def _process_one_sample(item, openyolo3d, oy3d_cfg, work_root):
    """Run pipeline + measurements for one nuScenes sample. Returns the per-sample dict."""
    sample_token = item["sample_token"]
    scene_dir = osp.join(work_root, sample_token)
    if osp.exists(scene_dir):
        shutil.rmtree(scene_dir)
    os.makedirs(scene_dir, exist_ok=True)

    adapt_stats = adapt_sample(item, scene_dir, camera=CAMERA)

    depth_scale = oy3d_cfg["openyolo3d"]["depth_scale"]
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]
    t_pred = time.time()
    prediction = openyolo3d.predict(
        path_2_scene_data=scene_dir,
        depth_scale=depth_scale,
        datatype="point cloud",
        text=text_prompts,
    )
    pred_seconds = float(time.time() - t_pred)

    image = item["images"][CAMERA]
    H, W = image.shape[:2]
    K = item["intrinsics"][CAMERA]
    T_cam_to_ego = item["cam_to_ego"][CAMERA]
    pc_ego_xyz = item["point_cloud"][:, :3]
    ego_pose = item["ego_pose"]

    proj = project_points_to_camera(pc_ego_xyz, K, T_cam_to_ego, (H, W))
    uv = proj["uv"]
    depth = proj["depth"]

    preds_2d = openyolo3d.preds_2d
    frame_keys = list(preds_2d.keys())
    frame_pred = preds_2d[frame_keys[0]] if frame_keys else {"bbox": np.zeros((0, 4)), "labels": np.zeros((0,)), "scores": np.zeros((0,))}

    detections = _measure_2d_detections(frame_pred, uv, depth, (H, W), text_prompts)
    detection_xyxys = [d["bbox_xyxy"] for d in detections]
    gt_rows = _measure_gt_boxes(item["gt_boxes"], ego_pose, pc_ego_xyz, K, T_cam_to_ego, (H, W), detection_xyxys)
    mask3d_stats = _measure_mask3d(openyolo3d.preds_3d, scene_dir)
    loss_summary = _detection_loss_summary(detections, gt_rows)

    # cleanup
    shutil.rmtree(scene_dir, ignore_errors=True)

    return {
        "sample_token": sample_token,
        "image_hw": [H, W],
        "n_lidar_points": int(pc_ego_xyz.shape[0]),
        "n_lidar_in_image": int(uv.shape[0]),
        "depth_pixel_coverage": adapt_stats["depth_pixel_coverage"],
        "predict_seconds": pred_seconds,
        "detections_2d": detections,
        "gt_boxes": gt_rows,
        "detection_loss": loss_summary,
        "mask3d": mask3d_stats,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--openyolo-config", default="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--output-dir", default="results/diagnosis")
    args = parser.parse_args()

    out_dir = args.output_dir
    per_sample_dir = osp.join(out_dir, "per_sample")
    figures_dir = osp.join(out_dir, "figures")
    os.makedirs(per_sample_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Force loader to CAM_FRONT only — we override via config dict, not editing the loader.
    with open(args.data_config) as f:
        data_cfg_text = f.read()
    data_cfg = yaml.safe_load(data_cfg_text)
    data_cfg["nuscenes"]["cameras"] = [CAMERA]
    tmp_cfg_path = osp.join(out_dir, "_data_config_camfront_only.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(data_cfg, f)

    loader = NuScenesLoader(config_path=tmp_cfg_path)
    chosen = _select_samples(loader, args.num_samples, args.seed)
    chosen_tokens = [loader.sample_tokens[i] for i in chosen]

    with open(osp.join(out_dir, "samples_used.json"), "w") as f:
        json.dump({
            "seed": args.seed,
            "num_requested": args.num_samples,
            "num_total_in_loader": len(loader),
            "indices": chosen,
            "sample_tokens": chosen_tokens,
        }, f, indent=2)

    print("=" * 60)
    print(f"Diagnosis: {len(chosen)} samples, seed={args.seed}, camera={CAMERA}")
    print("=" * 60)

    with open(args.openyolo_config) as f:
        oy3d_cfg = yaml.safe_load(f)

    print("Initializing OpenYOLO3D ...")
    t_init = time.time()
    from utils import OpenYolo3D
    openyolo3d = OpenYolo3D(args.openyolo_config)
    print(f"  initialized in {time.time() - t_init:.1f}s")

    work_root = tempfile.mkdtemp(prefix="diag_scene_")
    print(f"  scratch dir: {work_root}")

    succeeded = []
    failed = []

    signal.signal(signal.SIGALRM, _alarm_handler)

    for idx, sample_idx in enumerate(chosen):
        token = loader.sample_tokens[sample_idx]
        print(f"\n[{idx + 1}/{len(chosen)}] sample {token}")
        t_s = time.time()
        signal.alarm(PER_SAMPLE_TIMEOUT_S)
        try:
            item = loader[sample_idx]
            result = _process_one_sample(item, openyolo3d, oy3d_cfg, work_root)
            signal.alarm(0)
            elapsed = time.time() - t_s
            result["wall_seconds"] = elapsed
            with open(osp.join(per_sample_dir, f"{token}.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(f"  ✓ {elapsed:.1f}s — "
                  f"{len(result['detections_2d'])} dets, "
                  f"{len(result['gt_boxes'])} GT, "
                  f"{result['mask3d']['num_instances']} masks")
            succeeded.append(result)
        except SampleTimeout:
            signal.alarm(0)
            elapsed = time.time() - t_s
            failed.append({"sample_token": token, "reason": "timeout", "wall_seconds": elapsed})
            print(f"  ✗ timeout after {elapsed:.1f}s")
        except Exception as e:
            signal.alarm(0)
            elapsed = time.time() - t_s
            tb = traceback.format_exc()
            print(f"  ✗ exception: {e}\n{tb}")
            failed.append({"sample_token": token, "reason": str(e), "traceback": tb, "wall_seconds": elapsed})

    shutil.rmtree(work_root, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed")
    print("=" * 60)

    if len(succeeded) >= 10:
        agg = aggregate(succeeded)
        with open(osp.join(out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        render_all(succeeded, agg, failed, figures_dir, osp.join(out_dir, "report.md"))
        print(f"  → {out_dir}/aggregate.json")
        print(f"  → {out_dir}/report.md")
    else:
        msg = f"Only {len(succeeded)} succeeded (< 10). Skipping aggregate/report per spec."
        print("  " + msg)
        with open(osp.join(out_dir, "report.md"), "w") as f:
            f.write(f"# nuScenes diagnosis — INCOMPLETE\n\n{msg}\n\n## Failed samples\n\n")
            for fr in failed:
                f.write(f"- `{fr['sample_token']}` — {fr['reason']}\n")

    with open(osp.join(out_dir, "_failed.json"), "w") as f:
        json.dump(failed, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
