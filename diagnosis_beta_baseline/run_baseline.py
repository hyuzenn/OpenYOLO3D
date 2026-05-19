"""β baseline — 50-sample OpenYOLO3D inference + nuScenes-devkit mAP/NDS.

Pipeline per sample (single-cam CAM_FRONT, single OY3D init reused):
  1. NuScenesLoader._load(token) — picks v1.0-mini vs v1.0-trainval based
     on which loader's sample_tokens contains the token.
  2. adapter.adapt_sample → scene_dir (color/0.jpg, depth/0.png, poses/0.txt,
     intrinsics.txt, lidar.ply).
  3. OpenYOLO3D.predict (10 nuScenes detection classes as text prompts).
  4. mask → DetectionBox, ego→global via ego_pose.
  5. instance-level M/L/D/miss bookkeeping (W1-style).

After the loop:
  6. nuScenes-devkit-equivalent mAP/NDS via diagnosis_beta_baseline.evaluate_nuscenes
  7. dump aggregate.json + per_sample/<token>.json + samples_used.json copy.

Crash policy
------------
A per-sample exception is *recorded* (n_skipped++) and we continue, so a
single bad token cannot kill the run. The AC requires ≥ 45/50 samples
evaluated.

The OpenYOLO3D core is NEVER patched here — we only call its public
`predict()` and `OpenYolo3D.__init__`.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import shutil
import sys
import time
import traceback
from typing import Dict, List

import numpy as np
import torch
import yaml

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.nuscenes_to_openyolo3d import adapt_sample, CAMERA
from diagnosis_beta_baseline import NUSCENES_10_CLASS, DETECTION_CLASS_RANGE
from diagnosis_beta_baseline.format_predictions import (
    _read_ply_points,
    gt_to_detection_boxes,
    predictions_to_detection_boxes,
)
from diagnosis_beta_baseline.evaluate_nuscenes import (
    build_eval_boxes,
    evaluate as evaluate_nuscenes,
)
from diagnosis_beta_baseline.instance_metrics import (
    aggregate_distance_strata,
    per_sample_case_breakdown,
)


SAMPLES_USED = "results/diagnosis_step_a/samples_used.json"
OY3D_CFG = "diagnosis_beta_baseline/openyolo3d_nuscenes10.yaml"
DATA_CFG_BASE = "configs/nuscenes_baseline.yaml"


def _build_loader(out_dir: str, version: str, label: str) -> NuScenesLoader:
    cfg = yaml.safe_load(open(DATA_CFG_BASE))
    cfg["nuscenes"]["version"] = version
    cfg["nuscenes"]["cameras"] = [CAMERA]  # single-cam baseline (OY3D constraint)
    tmp = osp.join(out_dir, f"_data_config_{label}.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))
    print(f"  loading {version} (cameras={[CAMERA]}) ...")
    t0 = time.time()
    loader = NuScenesLoader(config_path=tmp)
    print(f"    {version} ready ({len(loader)} samples) in {time.time() - t0:.1f}s")
    return loader


def _route_token(loaders: Dict[str, NuScenesLoader], token: str) -> NuScenesLoader:
    for L in loaders.values():
        if token in L.sample_tokens:
            return L
    raise KeyError(f"token {token} not in any loader")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/diagnosis_beta_baseline")
    parser.add_argument("--samples-used", default=SAMPLES_USED)
    parser.add_argument("--limit", type=int, default=None,
                        help="for debugging — process only the first N tokens")
    parser.add_argument("--keep-scene-dirs", action="store_true",
                        help="don't delete per-sample scene dirs after inference")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "per_sample"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "scenes"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "nuscenes_eval"), exist_ok=True)

    print("=" * 70)
    print("β baseline — OpenYOLO3D nuScenes 50-sample eval")
    print("=" * 70)

    with open(args.samples_used) as f:
        samples_meta = json.load(f)
    tokens = list(samples_meta["tokens"])
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"loaded {len(tokens)} sample tokens from {args.samples_used}")
    shutil.copyfile(args.samples_used, osp.join(args.out_dir, "samples_used.json"))

    loaders = {
        "mini": _build_loader(args.out_dir, "v1.0-mini", "mini"),
        "trainval": _build_loader(args.out_dir, "v1.0-trainval", "trainval"),
    }

    with open(OY3D_CFG) as f:
        oy3d_cfg = yaml.safe_load(f)
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]
    depth_scale = oy3d_cfg["openyolo3d"]["depth_scale"]
    assert text_prompts == NUSCENES_10_CLASS, \
        f"config text_prompts must equal NUSCENES_10_CLASS, got {text_prompts}"

    print("\ninitializing OpenYOLO3D ...")
    t = time.time()
    from utils import OpenYolo3D
    oy3d = OpenYolo3D(OY3D_CFG)
    init_s = time.time() - t
    print(f"  initialized in {init_s:.1f}s")

    pred_per_sample: Dict[str, List[dict]] = {}
    gt_per_sample: Dict[str, List[dict]] = {}
    per_sample_records: List[dict] = []
    n_skipped = 0
    timing_buckets = {"adapter_s": [], "predict_s": [], "format_s": [], "metrics_s": [], "total_s": []}

    for i, token in enumerate(tokens):
        t_start = time.time()
        scene_dir = osp.join(args.out_dir, "scenes", token)
        if osp.exists(scene_dir):
            shutil.rmtree(scene_dir)

        try:
            L = _route_token(loaders, token)
            source = "mini" if L is loaders["mini"] else "trainval"
            t = time.time()
            item = L._load(token)
            t_load_s = time.time() - t

            t = time.time()
            stats = adapt_sample(item, scene_dir, camera=CAMERA)
            t_adapt_s = time.time() - t
            if stats["n_depth_pixels_filled"] == 0:
                raise RuntimeError(
                    f"adapter produced empty depth for {token} — pc {item['point_cloud'].shape}"
                )

            t = time.time()
            prediction = oy3d.predict(
                path_2_scene_data=scene_dir,
                depth_scale=depth_scale,
                datatype="point cloud",
                text=text_prompts,
            )
            t_pred_s = time.time() - t
            scene_name = osp.basename(scene_dir)
            pred_tuple = prediction[scene_name]

            t = time.time()
            ply_pts = _read_ply_points(osp.join(scene_dir, "lidar.ply"))
            pred_dicts, drop_stats = predictions_to_detection_boxes(
                sample_token=token,
                pred_tuple=pred_tuple,
                ply_points_ego=ply_pts,
                ego_pose_4x4=item["ego_pose"],
                text_prompts=NUSCENES_10_CLASS,
            )
            gt_dicts = gt_to_detection_boxes(token, item["gt_boxes"], ego_pose_4x4=item["ego_pose"])
            t_format_s = time.time() - t

            # Cap predictions per sample to nuScenes-devkit's max_boxes_per_sample (500).
            pred_dicts.sort(key=lambda d: -d["detection_score"])
            pred_dicts = pred_dicts[:500]

            pred_per_sample[token] = pred_dicts
            gt_per_sample[token] = gt_dicts

            t = time.time()
            # instance-level M/L/D/miss
            masks = pred_tuple[0]
            classes = pred_tuple[1]
            if hasattr(masks, "detach"):
                masks = masks.detach().cpu().numpy()
            else:
                masks = np.asarray(masks)
            masks = masks.astype(bool)
            if hasattr(classes, "detach"):
                classes = classes.detach().cpu().numpy().astype(np.int64)
            else:
                classes = np.asarray(classes).astype(np.int64)
            inst_metrics = per_sample_case_breakdown(
                gt_boxes_global=item["gt_boxes"],
                ego_pose_4x4=item["ego_pose"],
                pc_ego_xyz=item["point_cloud"][:, :3],
                instance_masks=masks,
                instance_classes=classes,
                text_prompts=NUSCENES_10_CLASS,
            )
            t_metrics_s = time.time() - t

            sample_total_s = time.time() - t_start
            timing_buckets["adapter_s"].append(t_adapt_s)
            timing_buckets["predict_s"].append(t_pred_s)
            timing_buckets["format_s"].append(t_format_s)
            timing_buckets["metrics_s"].append(t_metrics_s)
            timing_buckets["total_s"].append(sample_total_s)

            rec = {
                "sample_token": token,
                "source": source,
                "n_lidar_points": int(item["point_cloud"].shape[0]),
                "depth_coverage": float(stats["depth_pixel_coverage"]),
                "drop_stats": drop_stats,
                "n_pred_kept": len(pred_dicts),
                "n_gt_eval_classes": len(gt_dicts),
                "instance_metrics": {k: v for k, v in inst_metrics.items() if k != "per_gt"} | {
                    "per_gt": inst_metrics["per_gt"]
                },
                "timing_s": {
                    "loader_load":   t_load_s,
                    "adapter":       t_adapt_s,
                    "predict":       t_pred_s,
                    "format":        t_format_s,
                    "metrics":       t_metrics_s,
                    "total":         sample_total_s,
                },
            }
            per_sample_records.append(rec)
            with open(osp.join(args.out_dir, "per_sample", f"{token}.json"), "w") as f:
                json.dump(rec, f, indent=2)

            print(f"[{i+1}/{len(tokens)}] {source[:4]} {token[:8]} "
                  f"adapt {t_adapt_s:4.1f}s "
                  f"predict {t_pred_s:5.1f}s "
                  f"raw={drop_stats['n_pred_total']:3d} "
                  f"kept={drop_stats['n_pred_kept']:3d} "
                  f"GT={len(gt_dicts):2d} "
                  f"M_rate={inst_metrics['M_rate']:.2f}")

        except Exception as e:
            n_skipped += 1
            err = traceback.format_exc()
            with open(osp.join(args.out_dir, "per_sample", f"{token}.error.txt"), "w") as f:
                f.write(err)
            print(f"[{i+1}/{len(tokens)}] {token[:8]}  SKIPPED: {e!r}")

        finally:
            if not args.keep_scene_dirs and osp.exists(scene_dir):
                try:
                    shutil.rmtree(scene_dir)
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nfinished inference: {len(per_sample_records)} ok, {n_skipped} skipped")

    # Cap per-sample preds for safety (already done above).
    pred_eb = build_eval_boxes(pred_per_sample)
    gt_eb = build_eval_boxes(gt_per_sample)

    print("\nrunning nuScenes-devkit DetectionEval-equivalent ...")
    eval_dir = osp.join(args.out_dir, "nuscenes_eval")
    summary = evaluate_nuscenes(pred_eb, gt_eb, eval_dir)

    # also dump raw pred/gt boxes for reproducibility
    with open(osp.join(eval_dir, "pred_boxes.json"), "w") as f:
        json.dump(pred_per_sample, f, indent=2)
    with open(osp.join(eval_dir, "gt_boxes.json"), "w") as f:
        json.dump(gt_per_sample, f, indent=2)

    # aggregate.json — top-level summary suitable for the report builder.
    distance_strata = aggregate_distance_strata(per_sample_records)

    # global instance-level rates
    cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    n_gt_total = 0
    for r in per_sample_records:
        c = r["instance_metrics"]["case_counts"]
        for k in cases:
            cases[k] += int(c.get(k, 0))
        n_gt_total += int(r["instance_metrics"]["n_gt"])

    def _stats(arr):
        if not arr:
            return {"median": None, "p95": None, "mean": None, "n": 0}
        a = np.asarray(arr, dtype=np.float64)
        return {
            "median": float(np.median(a)),
            "p95":    float(np.percentile(a, 95)),
            "mean":   float(a.mean()),
            "n":      int(a.size),
        }

    aggregate = {
        "n_samples_total": len(tokens),
        "n_samples_evaluated": len(per_sample_records),
        "n_samples_skipped": n_skipped,
        "openyolo3d_init_s": init_s,
        "samples_used_path": args.samples_used,
        "samples_used_seed": samples_meta.get("seed"),
        "text_prompts": text_prompts,
        "instance_level": {
            "n_gt_total": int(n_gt_total),
            "case_counts": cases,
            "M_rate": (cases["M"] / n_gt_total) if n_gt_total else 0.0,
            "L_rate": (cases["L"] / n_gt_total) if n_gt_total else 0.0,
            "D_rate": (cases["D"] / n_gt_total) if n_gt_total else 0.0,
            "miss_rate": (cases["miss"] / n_gt_total) if n_gt_total else 0.0,
            "distance_strata": distance_strata,
        },
        "timing_s": {k: _stats(v) for k, v in timing_buckets.items()},
        "nuscenes_eval": summary,
    }
    with open(osp.join(args.out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nwrote {osp.join(args.out_dir, 'aggregate.json')}")
    print(f"wrote {osp.join(eval_dir, 'eval_summary.json')}")
    print(f"mAP={summary['mean_ap']:.4f}  NDS={summary['nd_score']:.4f}  "
          f"n_pred={summary['counts']['n_pred_boxes']}  n_gt={summary['counts']['n_gt_boxes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
