"""Tier-2 diagnosis: multi-view + uniformity on the same 20 nuScenes mini samples
as Tier 1.

This pipeline does NOT call Mask3D / OpenYOLO3D. Only YOLO-World per cam
(via Network_2D) and LiDAR projection per cam.

Usage:
    python -m diagnosis_tier2.run_diagnosis_tier2 \
        --use-tier1-samples results/diagnosis/samples_used.json
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
from PIL import Image

from dataloaders.nuscenes_loader import NuScenesLoader
from diagnosis.measurements import (
    DISTANCE_BIN_LABELS,
    distance_bin,
    project_points_to_camera,
    points_inside_2d_box,
    gt_box_to_ego,
    points_inside_3d_box,
    project_3d_box_corners_to_image,
)
from diagnosis_tier2.measurements_tier2 import (
    SIX_CAMERAS,
    per_cam_geom_visibility,
    per_cam_det_visibility,
    per_cam_inbox_centroid,
    pair_distances,
    quadrant_entropy,
)
from diagnosis_tier2.aggregate_tier2 import aggregate_tier2, render_all_tier2, _load_mini_baseline


PER_SAMPLE_TIMEOUT_S = 90


class SampleTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SampleTimeout()


def _save_six_images(item, work_root, sample_token):
    """Materialise the 6 cam images to a flat tmp dir as 0.jpg..5.jpg.

    We bypass the OpenYOLO3D scene_dir layout entirely — Network_2D only
    needs file paths.
    """
    sample_dir = osp.join(work_root, sample_token)
    if osp.exists(sample_dir):
        shutil.rmtree(sample_dir)
    os.makedirs(sample_dir, exist_ok=True)
    paths = []
    for i, cam in enumerate(SIX_CAMERAS):
        img = item["images"][cam]
        p = osp.join(sample_dir, f"{i}.jpg")
        Image.fromarray(img).save(p, quality=95)
        paths.append(p)
    return sample_dir, paths


def _detections_per_cam(yolo_preds, text_prompts):
    """Convert Network_2D output (keyed by frame_id "0"…"5") into per-cam
    detection dicts. Each cam dict has lists: bbox xyxy float, label_text, score, idx.
    """
    per_cam = {cam: {"xyxy": [], "labels": [], "scores": [], "indices": []} for cam in SIX_CAMERAS}
    for i, cam in enumerate(SIX_CAMERAS):
        pred = yolo_preds.get(str(i))
        if pred is None:
            continue
        bboxes = pred["bbox"].cpu().numpy() if hasattr(pred["bbox"], "cpu") else np.asarray(pred["bbox"])
        labels = pred["labels"].cpu().numpy() if hasattr(pred["labels"], "cpu") else np.asarray(pred["labels"])
        scores = pred["scores"].cpu().numpy() if hasattr(pred["scores"], "cpu") else np.asarray(pred["scores"])
        for j in range(bboxes.shape[0]):
            x1, y1, x2, y2 = [float(v) for v in bboxes[j]]
            label_idx = int(labels[j])
            label_text = text_prompts[label_idx] if 0 <= label_idx < len(text_prompts) else f"class_{label_idx}"
            per_cam[cam]["xyxy"].append([x1, y1, x2, y2])
            per_cam[cam]["labels"].append(label_text)
            per_cam[cam]["scores"].append(float(scores[j]))
            per_cam[cam]["indices"].append(j)
    return per_cam


def _process_one_sample(item, network_2d, oy3d_cfg, work_root):
    sample_token = item["sample_token"]
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]

    # --- 6-cam YOLO-World ---
    sample_dir, paths = _save_six_images(item, work_root, sample_token)
    t0 = time.time()
    yolo_preds = network_2d.get_bounding_boxes(paths, text=text_prompts)
    t_yolo = float(time.time() - t0)
    per_cam_dets = _detections_per_cam(yolo_preds, text_prompts)
    n_dets_per_cam = {cam: len(per_cam_dets[cam]["xyxy"]) for cam in SIX_CAMERAS}

    # --- LiDAR projections per cam (cached for reuse in M1/M2/M3) ---
    pc_ego = item["point_cloud"][:, :3]
    ego_pose = item["ego_pose"]
    cam_proj = {}
    image_hw = {}
    K_per_cam = {}
    T_per_cam = {}
    t1 = time.time()
    for cam in SIX_CAMERAS:
        H, W = item["images"][cam].shape[:2]
        K = item["intrinsics"][cam]
        T = item["cam_to_ego"][cam]
        image_hw[cam] = (H, W)
        K_per_cam[cam] = K
        T_per_cam[cam] = T
        cam_proj[cam] = project_points_to_camera(pc_ego, K, T, (H, W))
    t_proj = float(time.time() - t1)

    # --- Identify LiDAR-supported GT (Tier 1 def: ≥1 pt in 3D box, ego frame) ---
    t2 = time.time()
    gt_lidar_supported = []
    for gt_idx, gt in enumerate(item["gt_boxes"]):
        box_ego = gt_box_to_ego(gt, ego_pose)
        in3d = points_inside_3d_box(box_ego, pc_ego)
        n = int(in3d.sum())
        if n < 1:
            continue
        d = float(np.linalg.norm(box_ego.center))
        gt_lidar_supported.append({
            "gt_idx": gt_idx,
            "class": gt["category"],
            "distance_m": d,
            "n_lidar_pts_in_3d_box": n,
            "box_ego": box_ego,
        })

    # --- M1 view diversity per GT (per cam geom + det) ---
    view_diversity_per_gt = []
    histogram_geom = [0] * 7
    histogram_det = [0] * 7
    for g in gt_lidar_supported:
        per_cam = {}
        n_geom = 0
        n_det = 0
        for cam in SIX_CAMERAS:
            geom_vis, bbox2d, n_in = per_cam_geom_visibility(
                g["box_ego"], pc_ego, K_per_cam[cam], T_per_cam[cam], image_hw[cam]
            )
            det_vis, det_idx, det_iou = per_cam_det_visibility(bbox2d, per_cam_dets[cam]["xyxy"])
            per_cam[cam] = {
                "geom_visible": bool(geom_vis),
                "det_visible": bool(det_vis),
                "n_inbox_lidar_pts": n_in,
                "matched_det_idx": det_idx if det_vis else None,
                "matched_det_iou": float(det_iou) if det_vis else None,
                "projected_2d_bbox": bbox2d,
            }
            n_geom += int(geom_vis)
            n_det += int(det_vis)
        histogram_geom[n_geom] += 1
        histogram_det[n_det] += 1
        view_diversity_per_gt.append({
            "gt_idx": g["gt_idx"],
            "class": g["class"],
            "distance_m": g["distance_m"],
            "distance_bin": distance_bin(g["distance_m"]),
            "n_lidar_pts_in_3d_box": g["n_lidar_pts_in_3d_box"],
            "n_geom_visible": n_geom,
            "n_det_visible": n_det,
            "per_cam": per_cam,
        })

    # --- M2 multi-view consistency: GTs with n_geom_visible ≥ 2 ---
    multi_view = []
    for vd in view_diversity_per_gt:
        if vd["n_geom_visible"] < 2:
            continue
        gt = next(x for x in gt_lidar_supported if x["gt_idx"] == vd["gt_idx"])
        cams_geom = [c for c in SIX_CAMERAS if vd["per_cam"][c]["geom_visible"]]
        centroids = []
        cam_used = []
        for cam in cams_geom:
            r = per_cam_inbox_centroid(gt["box_ego"], pc_ego, K_per_cam[cam], T_per_cam[cam], image_hw[cam])
            if r["centroid"] is not None:
                centroids.append(r["centroid"])
                cam_used.append(cam)
        if len(centroids) < 2:
            continue
        ds = pair_distances(centroids)
        multi_view.append({
            "gt_idx": vd["gt_idx"],
            "class": vd["class"],
            "distance_m": vd["distance_m"],
            "distance_bin": vd["distance_bin"],
            "cams_used": cam_used,
            "n_views": len(centroids),
            "centroids_ego": centroids,
            "pair_distances_m": ds,
            "max_pair_distance_m": float(max(ds)) if ds else 0.0,
            "mean_pair_distance_m": float(np.mean(ds)) if ds else 0.0,
        })

    # --- M3 uniformity per detection (n ≥ 4 in-box LiDAR points) ---
    uniformity_per_det = []
    for cam in SIX_CAMERAS:
        uv = cam_proj[cam]["uv"]
        depth = cam_proj[cam]["depth"]
        for det_idx, bbox in enumerate(per_cam_dets[cam]["xyxy"]):
            inside = points_inside_2d_box(uv, bbox) if uv.shape[0] > 0 else np.zeros((0,), dtype=bool)
            n_total = int(inside.sum())
            if n_total < 4:
                continue
            uv_inbox = uv[inside]
            H_norm, counts = quadrant_entropy(uv_inbox, bbox)
            d_med = float(np.median(depth[inside])) if n_total > 0 else None
            uniformity_per_det.append({
                "cam": cam,
                "det_idx": det_idx,
                "class": per_cam_dets[cam]["labels"][det_idx],
                "score": per_cam_dets[cam]["scores"][det_idx],
                "bbox_xyxy": bbox,
                "n_total": n_total,
                "quadrant_counts": counts,
                "entropy_norm": float(H_norm),
                "distance_to_ego": d_med,
                "distance_bin": distance_bin(d_med),
            })
    t_meas = float(time.time() - t2)

    # cleanup tmp images
    shutil.rmtree(sample_dir, ignore_errors=True)

    return {
        "sample_token": sample_token,
        "n_gt_lidar_supported": len(gt_lidar_supported),
        "n_detections_per_cam": n_dets_per_cam,
        "view_diversity": {
            "per_gt": [
                {k: v for k, v in vd.items() if k != "_internal"} for vd in view_diversity_per_gt
            ],
            "histogram_geom": histogram_geom,
            "histogram_det": histogram_det,
        },
        "multi_view_consistency": {
            "per_gt": multi_view,
            "n_multi_view_gts": len(multi_view),
        },
        "uniformity": {
            "per_detection": uniformity_per_det,
            "n_eligible": len(uniformity_per_det),
        },
        "timings_s": {
            "yolo_6cams": t_yolo,
            "lidar_projection_6cams": t_proj,
            "measurements": t_meas,
            "total": t_yolo + t_proj + t_meas,
        },
        "status": "ok",
    }


def _all_files_exist_for_sample(nusc, dataroot, sample_token, cams=SIX_CAMERAS):
    """Check that LIDAR_TOP and all 6 camera sample_data files exist on disk."""
    sample = nusc.get("sample", sample_token)
    needed_channels = ["LIDAR_TOP", *cams]
    for ch in needed_channels:
        sd_token = sample["data"].get(ch)
        if sd_token is None:
            return False
        sd = nusc.get("sample_data", sd_token)
        if not osp.exists(osp.join(dataroot, sd["filename"])):
            return False
    return True


def _select_random_with_filter(loader, num_samples, seed):
    """Pre-filter loader.sample_tokens by file existence, then random-select.

    Returns (chosen_tokens_in_loader_order, n_pool_after_filter, n_pool_before_filter).
    """
    rng = np.random.default_rng(seed)
    pool = loader.sample_tokens
    valid = [t for t in pool if _all_files_exist_for_sample(loader.nusc, loader.dataroot, t)]
    n_total = len(pool)
    n_valid = len(valid)
    if num_samples >= n_valid:
        chosen = valid
    else:
        idx = rng.choice(n_valid, size=num_samples, replace=False)
        chosen = [valid[i] for i in sorted(idx)]
    return chosen, n_valid, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="seed for random sample selection (only used when --use-tier1-samples is omitted)")
    parser.add_argument("--version", default=None, help="override loader version (e.g. v1.0-trainval). When set, switches to random-with-file-filter selection unless --use-tier1-samples is also given.")
    parser.add_argument("--use-tier1-samples", default=None, help="reuse a samples_used.json (e.g. Tier 1's). When set, --num-samples / --seed are ignored.")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--openyolo-config", default="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir", default="results/diagnosis_tier2")
    args = parser.parse_args()

    out_dir = args.output_dir
    per_sample_dir = osp.join(out_dir, "per_sample")
    figures_dir = osp.join(out_dir, "figures")
    os.makedirs(per_sample_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Force loader to all 6 cams + apply --version override if provided.
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)
    data_cfg["nuscenes"]["cameras"] = list(SIX_CAMERAS)
    if args.version is not None:
        data_cfg["nuscenes"]["version"] = args.version
    tmp_cfg_path = osp.join(out_dir, "_data_config_six_cams.yaml")
    with open(tmp_cfg_path, "w") as f:
        yaml.safe_dump(data_cfg, f)

    print(f"Loading NuScenes (version={data_cfg['nuscenes']['version']})...")
    t_load = time.time()
    loader = NuScenesLoader(config_path=tmp_cfg_path)
    print(f"  loader ready in {time.time() - t_load:.1f}s — {len(loader)} samples in pool")

    # Sample selection: prefer Tier 1 list if given; else random with file filter.
    if args.use_tier1_samples is not None:
        with open(args.use_tier1_samples) as f:
            tier1 = json.load(f)
        sample_tokens_req = tier1["sample_tokens"][: args.num_samples]
        token_to_idx = {tok: i for i, tok in enumerate(loader.sample_tokens)}
        indices, missing = [], []
        for tok in sample_tokens_req:
            if tok in token_to_idx:
                indices.append(token_to_idx[tok])
            else:
                missing.append(tok)
        if missing:
            print(f"WARNING: {len(missing)} tier1 tokens not found in current loader: {missing[:3]}...")
        provenance = {
            "source": "tier1",
            "tier1_path": args.use_tier1_samples,
            "version": data_cfg["nuscenes"]["version"],
            "num_used": len(indices),
            "sample_tokens": [loader.sample_tokens[i] for i in indices],
            "indices_in_loader": indices,
            "tier1_seed": tier1.get("seed"),
        }
    else:
        print(f"Random selection (seed={args.seed}) with file-existence pre-filter...")
        t_filter = time.time()
        chosen_tokens, n_valid, n_total = _select_random_with_filter(loader, args.num_samples, args.seed)
        print(f"  pool: {n_total} total, {n_valid} valid after file-existence filter ({time.time() - t_filter:.1f}s)")
        token_to_idx = {tok: i for i, tok in enumerate(loader.sample_tokens)}
        indices = [token_to_idx[t] for t in chosen_tokens]
        provenance = {
            "source": "random",
            "version": data_cfg["nuscenes"]["version"],
            "seed": args.seed,
            "n_requested": args.num_samples,
            "n_pool_total": n_total,
            "n_pool_after_file_check": n_valid,
            "tokens": chosen_tokens,
            "sample_tokens": chosen_tokens,  # alias for downstream compat
            "indices_in_loader": indices,
        }

    with open(osp.join(out_dir, "samples_used.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print("=" * 60)
    print(f"Tier 2 diagnosis: {len(indices)} samples (source={provenance['source']}, "
          f"version={data_cfg['nuscenes']['version']}), 6 cams")
    print("=" * 60)

    # --- only init YOLO-World; NO Mask3D ---
    print("Initializing YOLO-World only (no Mask3D)...")
    with open(args.openyolo_config) as f:
        oy3d_cfg = yaml.safe_load(f)
    t_init = time.time()
    from utils.utils_2d import Network_2D
    network_2d = Network_2D(oy3d_cfg)
    print(f"  initialized in {time.time() - t_init:.1f}s")

    work_root = tempfile.mkdtemp(prefix="diag_t2_")
    print(f"  scratch dir: {work_root}")

    succeeded, failed = [], []
    signal.signal(signal.SIGALRM, _alarm_handler)

    for i, sample_idx in enumerate(indices):
        token = loader.sample_tokens[sample_idx]
        print(f"\n[{i + 1}/{len(indices)}] sample {token}")
        t_s = time.time()
        signal.alarm(PER_SAMPLE_TIMEOUT_S)
        try:
            item = loader[sample_idx]
            result = _process_one_sample(item, network_2d, oy3d_cfg, work_root)
            signal.alarm(0)
            elapsed = time.time() - t_s
            result["wall_seconds"] = elapsed
            with open(osp.join(per_sample_dir, f"{token}.json"), "w") as f:
                json.dump(result, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            print(f"  ✓ {elapsed:.1f}s — "
                  f"{result['n_gt_lidar_supported']} LiDAR-supp GT, "
                  f"{result['multi_view_consistency']['n_multi_view_gts']} multi-view, "
                  f"{result['uniformity']['n_eligible']} M3-eligible dets")
            succeeded.append(result)
        except SampleTimeout:
            signal.alarm(0)
            failed.append({"sample_token": token, "reason": "timeout", "wall_seconds": time.time() - t_s})
            print(f"  ✗ timeout")
        except Exception as e:
            signal.alarm(0)
            tb = traceback.format_exc()
            failed.append({"sample_token": token, "reason": str(e), "traceback": tb,
                          "wall_seconds": time.time() - t_s})
            print(f"  ✗ {e}\n{tb}")

    shutil.rmtree(work_root, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed")
    print("=" * 60)

    if len(succeeded) >= 10:
        mini_baseline = _load_mini_baseline(out_dir)
        if mini_baseline is not None:
            print(f"  Loaded mini baseline for comparison: cond_C={mini_baseline['cond_C']:.4f} "
                  f"(n={mini_baseline['n_samples']})")
        agg = aggregate_tier2(succeeded, provenance=provenance, mini_baseline=mini_baseline)
        with open(osp.join(out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        render_all_tier2(succeeded, agg, failed, figures_dir, osp.join(out_dir, "report.md"),
                         provenance=provenance, mini_baseline=mini_baseline)
        print(f"  → {out_dir}/aggregate.json")
        print(f"  → {out_dir}/report.md")
    else:
        msg = f"Only {len(succeeded)} succeeded (< 10). Skipping aggregate/report per spec."
        print("  " + msg)
        with open(osp.join(out_dir, "report.md"), "w") as f:
            f.write(f"# nuScenes Tier 2 diagnosis — INCOMPLETE\n\n{msg}\n")

    with open(osp.join(out_dir, "_failed.json"), "w") as f:
        json.dump(failed, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
