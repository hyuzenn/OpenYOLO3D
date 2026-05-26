"""Task 2.5 — native CenterPoint mAP sanity (single-sweep vs multi-sweep).

Decisive diagnostic for the γ baseline anomaly (Stage C mAP 0.0526 vs the
standard nuScenes CenterPoint ~0.58). The Stage C pipeline scores boxes with
*YOLO-World* labels, not CenterPoint's own head, so its mAP conflates detector
quality with the 3D→2D relabeling step. This script removes that confound:

  - Run CenterPoint and feed its NATIVE 10-class label + score straight to the
    nuScenes-devkit detection eval (no YOLO, no temporal gate, no association).
  - Do it twice on the SAME scenes: single-sweep input, then 10-sweep input.
    Sweep count is the only changed variable.

Coordinate transforms are byte-identical to nuscenes_evaluator.step_sample
(Fix 2 orientation), so the only difference from the Stage C pipeline is the
label/score source. Interpretation:
  - multi-sweep native mAP ~0.5+  → checkpoint is healthy; single-sweep input
    was the root cause; remaining Stage C gap is the YOLO relabeling.
  - multi-sweep native mAP still low → deeper config/coordinate problem.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from dataloaders.nuscenes_loader import NuScenesLoader
from method_scannet.streaming.nuscenes_evaluator import (
    NUSC_10_SET, _detection_box_dict, _list_val_scenes, _set_mm_scope,
)


def _scene_sample_tokens(nusc, scene_token):
    scene = nusc.get("scene", scene_token)
    toks, cur = [], scene["first_sample_token"]
    while cur:
        toks.append(cur)
        cur = nusc.get("sample", cur)["next"]
    return toks


def _scene_sweeps_present(nusc, dataroot, scene_token, num_sweeps):
    """True iff every sample's prev-sweep chain (up to num_sweeps-1) is on disk
    OR terminates at scene start. Guarantees from_file_multisweep won't raise
    FileNotFoundError on this scene. The 146/150 keyframe-only val scenes fail
    this; only the ~4 with downloaded sweeps pass.
    """
    for tok in _scene_sample_tokens(nusc, scene_token):
        cur = nusc.get("sample", tok)["data"]["LIDAR_TOP"]
        for _ in range(num_sweeps - 1):
            prev = nusc.get("sample_data", cur)["prev"]
            if not prev:
                break  # scene start: fewer sweeps is fine, not a missing file
            sd_prev = nusc.get("sample_data", prev)
            if not os.path.exists(os.path.join(dataroot, sd_prev["filename"])):
                return False
            cur = prev
    return True


def _covered_val_scenes(loader, num_sweeps):
    val = _list_val_scenes(loader)
    return [s for s in val
            if _scene_sweeps_present(loader.nusc, loader.dataroot, s, num_sweeps)]


def _native_boxes_for_sample(loader, cp, sample_token, gid):
    """Run CenterPoint on one sample; return (pred_dicts, gt_dicts, n_points)."""
    item = loader._load(sample_token)
    pc = item["point_cloud"]
    ego_pose = item["ego_pose"]
    ego_translation = ego_pose[:3, 3]
    ego_quat = Quaternion(matrix=ego_pose[:3, :3])

    nusc = loader.nusc
    lidar_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    T_lidar_to_ego = transform_matrix(
        translation=lidar_cs["translation"],
        rotation=Quaternion(lidar_cs["rotation"]),
    )
    lidar_to_ego_q = Quaternion(matrix=T_lidar_to_ego[:3, :3])

    _set_mm_scope("mmdet3d")
    out = cp.generate(pc, T_lidar_to_ego, tmp_bin_path="/tmp/_native_pc.bin")

    preds = []
    for p in out["proposals"]:
        cls_name = p.get("cls_name")
        if cls_name is None or cls_name not in NUSC_10_SET:
            continue
        centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
        bbox_lidar = p["bbox_lidar"]
        yaw_lidar = float(bbox_lidar[6]) if len(bbox_lidar) >= 7 else 0.0
        box_q_ego = lidar_to_ego_q * Quaternion(axis=(0.0, 0.0, 1.0), angle=yaw_lidar)
        centroid_global = (ego_pose[:3, :3] @ centroid_ego[:3]) + ego_translation
        global_q = ego_quat * box_q_ego
        preds.append(_detection_box_dict(
            global_id=gid,
            sample_token=sample_token,
            bbox_lidar=bbox_lidar,
            centroid_global=centroid_global,
            ego_translation=ego_translation,
            rotation_global_wxyz=[float(global_q.w), float(global_q.x),
                                  float(global_q.y), float(global_q.z)],
            detection_name=cls_name,          # NATIVE CenterPoint class
            score=float(p.get("score", 0.0)),  # NATIVE CenterPoint score
        ))
        gid += 1

    from nuscenes.eval.detection.utils import category_to_detection_name
    gts = []
    for gt in item["gt_boxes"]:
        det = category_to_detection_name(gt["category"])
        if det is None or det not in NUSC_10_SET:
            continue
        gts.append({
            "sample_token": sample_token,
            "translation": [float(x) for x in gt["translation"]],
            "size": [float(x) for x in gt["size"]],
            "rotation": [float(x) for x in gt["rotation"]],
            "velocity": [0.0, 0.0],
            "ego_translation": [float(x) for x in ego_translation],
            "num_pts": int(gt.get("num_lidar_pts", 0)),
            "detection_name": det,
            "detection_score": -1.0,
            "attribute_name": "",
        })
    return preds, gts, int(pc.shape[0]), gid


def _eval_pass(loader, cp, scenes, out_dir, label):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate

    pred_eb, gt_eb = EvalBoxes(), EvalBoxes()
    gid = 0
    n_samples = 0
    n_errors = 0
    n_points_acc = []
    t0 = time.time()
    for si, sc in enumerate(scenes):
        for tok in _scene_sample_tokens(loader.nusc, sc):
            try:
                preds, gts, npts, gid = _native_boxes_for_sample(loader, cp, tok, gid)
            except Exception as exc:
                n_errors += 1
                print(f"    [{label}] sample {tok[:8]} FAILED: {exc!r}", flush=True)
                pred_eb.add_boxes(tok, [])
                gt_eb.add_boxes(tok, [])
                continue
            pred_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in preds])
            gt_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in gts])
            n_points_acc.append(npts)
            n_samples += 1
        print(f"  [{label}] scene {si+1}/{len(scenes)} done "
              f"({n_samples} samples, {n_errors} err, "
              f"mean_pts={np.mean(n_points_acc) if n_points_acc else 0:.0f})", flush=True)
    wall = time.time() - t0

    eval_dir = Path(out_dir) / f"native_{label}"
    summary = nu_evaluate(pred_boxes=pred_eb, gt_boxes=gt_eb,
                          output_dir=str(eval_dir), config_name="detection_cvpr_2019")
    res = {
        "label": label,
        "n_scenes": len(scenes),
        "n_samples": n_samples,
        "n_errors": n_errors,
        "mean_points_per_sample": float(np.mean(n_points_acc)) if n_points_acc else 0.0,
        "mAP": summary.get("mean_ap"),
        "NDS": summary.get("nd_score"),
        "n_pred_boxes": summary["counts"]["n_pred_boxes"],
        "n_gt_boxes": summary["counts"]["n_gt_boxes"],
        "wall_s": wall,
    }
    print(f"  [{label}] mAP={res['mAP']:.4f} NDS={res['NDS']:.4f} "
          f"mean_pts={res['mean_points_per_sample']:.0f} wall={wall:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scene-limit", type=int, default=0,
                    help="cap on number of scenes (0 = use all covered scenes)")
    ap.add_argument("--score-threshold", type=float, default=0.0,
                    help="0.0 keeps all head outputs (best for AP recall)")
    ap.add_argument("--num-sweeps", type=int, default=10)
    ap.add_argument("--scenes-mode", choices=["covered", "all"], default="covered",
                    help="'covered' = only val scenes with LiDAR sweeps on disk (for "
                         "multi-sweep); 'all' = full 150-scene val (single-sweep safe).")
    ap.add_argument("--sweep-mode", choices=["single", "multi", "both"], default="both",
                    help="which passes to run. 'multi'/'both' require sweep-covered scenes.")
    args = ap.parse_args()

    if args.sweep_mode in ("multi", "both") and args.scenes_mode == "all":
        raise SystemExit("multi-sweep over 'all' val scenes would crash on the 146 "
                         "keyframe-only scenes; use --scenes-mode covered for multi.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    val = _list_val_scenes(loader)
    if args.scenes_mode == "covered":
        scenes = _covered_val_scenes(loader, args.num_sweeps)
        names = [loader.nusc.get("scene", s)["name"] for s in scenes]
        print(f"  val scenes total {len(val)}; sweep-covered: {len(scenes)} {names}",
              flush=True)
    else:
        scenes = val
        print(f"  val scenes total {len(val)}; using ALL (single-sweep)", flush=True)
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    if not scenes:
        raise SystemExit("No scenes selected — check --scenes-mode / data.")
    print(f"  using {len(scenes)} scenes", flush=True)

    print("Loading CenterPoint ...", flush=True)
    from adapters.centerpoint_proposals import CenterPointProposalGenerator
    CKPT = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
            "centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_"
            "20220810_011659-04cb3a3b.pth")
    CFG = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
           "centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")
    cp = CenterPointProposalGenerator(config_path=CFG, checkpoint_path=CKPT,
                                      score_threshold=args.score_threshold,
                                      device="cuda:0")

    results = {}
    if args.sweep_mode in ("single", "both"):
        loader.multi_sweep = False
        loader.num_sweeps = 1
        results["single_sweep"] = _eval_pass(loader, cp, scenes, out, "single_sweep")
    if args.sweep_mode in ("multi", "both"):
        loader.multi_sweep = True
        loader.num_sweeps = args.num_sweeps
        results["multi_sweep"] = _eval_pass(loader, cp, scenes, out, "multi_sweep")

    (out / "native_map_sanity.json").write_text(json.dumps(results, indent=2))
    print("\n=== NATIVE CenterPoint mAP ===")
    for k in ("single_sweep", "multi_sweep"):
        if k in results:
            r = results[k]
            print(f"{k:13s}: mAP={r['mAP']:.4f} NDS={r['NDS']:.4f} "
                  f"pts={r['mean_points_per_sample']:.0f} "
                  f"n_samples={r['n_samples']} n_scenes={r['n_scenes']}")
    print(f"wrote {out / 'native_map_sanity.json'}")


if __name__ == "__main__":
    main()
