"""Diagnose β baseline mAP=0 GT-loading bug — minimal-import version.

Compares GT/pred JSON files written by run_baseline.py and confirms the
root cause of nuScenes-devkit `accumulate` returning 0 GT matches:

    `gt_to_detection_boxes()` in format_predictions.py omits the
    `ego_translation` field. `DetectionBox.deserialize` defaults the
    field to (0, 0, 0). evaluate_nuscenes._filter_by_range uses the
    formula `||box.translation[:2] - box.ego_translation[:2]||` to
    measure ego-distance, which for GT becomes `||box.translation[:2]||`
    in the global frame (~hundreds of meters in any nuScenes city) and
    so every GT box exceeds class_range and gets filtered out.

Pred boxes have `ego_translation = ego_pose_translation_in_global`, so
their formula correctly yields ~box-to-ego distance — they pass.

This version avoids the slow nuscenes-devkit import (NFS-bound on this
host); it inlines the cvpr_2019 class_range and replicates the filter.

Run:
    python diagnosis_beta_baseline/diagnose_gt_loading.py
"""

from __future__ import annotations

import json
import os.path as osp

import numpy as np

EVAL_DIR = "results/diagnosis_beta_baseline/nuscenes_eval"

# detection_cvpr_2019.json class_range, inlined to avoid the slow nuScenes
# import.  Verified to match nuscenes-devkit configs/detection_cvpr_2019.json.
CLASS_RANGE_CVPR_2019 = {
    "car": 50.0,
    "truck": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "construction_vehicle": 50.0,
    "pedestrian": 40.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
    "traffic_cone": 30.0,
    "barrier": 30.0,
}


def main() -> int:
    with open(osp.join(EVAL_DIR, "gt_boxes.json")) as f:
        gt_data = json.load(f)
    with open(osp.join(EVAL_DIR, "pred_boxes.json")) as f:
        pred_data = json.load(f)

    print("=== file-level counts ===")
    n_gt_file = sum(len(v) for v in gt_data.values())
    n_pred_file = sum(len(v) for v in pred_data.values())
    print(f"GT samples:   {len(gt_data)},  total boxes: {n_gt_file}")
    print(f"Pred samples: {len(pred_data)}, total boxes: {n_pred_file}")
    n_pred_nonempty = sum(1 for v in pred_data.values() if v)
    print(f"Pred non-empty samples: {n_pred_nonempty}")

    print("\n=== class-name match ===")
    gt_classes = {b["detection_name"] for boxes in gt_data.values() for b in boxes}
    pred_classes = {b["detection_name"] for boxes in pred_data.values() for b in boxes}
    print(f"GT classes:   {sorted(gt_classes)}")
    print(f"Pred classes: {sorted(pred_classes)}")
    print(f"Intersection: {sorted(gt_classes & pred_classes)}")
    print(f"GT-only:   {sorted(gt_classes - pred_classes)}")
    print(f"Pred-only: {sorted(pred_classes - gt_classes)}")

    print("\n=== ego_translation field presence ===")
    gt_with_ego = sum(1 for boxes in gt_data.values() for b in boxes if "ego_translation" in b)
    pred_with_ego = sum(1 for boxes in pred_data.values() for b in boxes if "ego_translation" in b)
    print(f"GT boxes with ego_translation:   {gt_with_ego} / {n_gt_file}")
    print(f"Pred boxes with ego_translation: {pred_with_ego} / {n_pred_file}")

    print("\n=== sample first GT vs first pred (same sample) ===")
    first_token = list(gt_data.keys())[0]
    print(f"sample_token: {first_token}")
    print(f"GT[0]:   trans={gt_data[first_token][0]['translation']}, ego={gt_data[first_token][0].get('ego_translation', 'MISSING')}")
    same_pred = pred_data.get(first_token, [])
    if same_pred:
        print(f"Pred[0]: trans={same_pred[0]['translation']}, ego={same_pred[0]['ego_translation']}")

    print("\n=== _filter_by_range simulation ===")
    print(f"class_range (cvpr_2019): {CLASS_RANGE_CVPR_2019}")

    def filter_count(data: dict, label: str) -> int:
        kept = 0
        total = 0
        survives_per_class: dict = {}
        for token, dicts in data.items():
            for d in dicts:
                total += 1
                cls = d["detection_name"]
                if cls not in CLASS_RANGE_CVPR_2019:
                    continue
                r = CLASS_RANGE_CVPR_2019[cls]
                ego = np.asarray(d.get("ego_translation", [0.0, 0.0, 0.0]), dtype=np.float64)
                xyz = np.asarray(d["translation"], dtype=np.float64)
                dist = float(np.linalg.norm(xyz[:2] - ego[:2]))
                if dist <= r:
                    kept += 1
                    survives_per_class[cls] = survives_per_class.get(cls, 0) + 1
        print(f"{label}: kept {kept}/{total} after _filter_by_range")
        for c, n in sorted(survives_per_class.items()):
            print(f"  {c}: {n}")
        return kept

    n_pred_kept = filter_count(pred_data, "Pred")
    n_gt_kept = filter_count(gt_data, "GT")

    print("\n=== distance distribution (||xyz - ego||) ===")

    def dist_stats(data: dict, label: str) -> None:
        ds = []
        for token, dicts in data.items():
            for d in dicts:
                ego = np.asarray(d.get("ego_translation", [0.0, 0.0, 0.0]), dtype=np.float64)
                xyz = np.asarray(d["translation"], dtype=np.float64)
                ds.append(float(np.linalg.norm(xyz[:2] - ego[:2])))
        if not ds:
            print(f"{label}: empty")
            return
        ds_arr = np.asarray(ds)
        print(f"{label}: n={len(ds_arr)}, min={ds_arr.min():.2f}, median={np.median(ds_arr):.2f}, max={ds_arr.max():.2f}, p99={np.percentile(ds_arr, 99):.2f}")

    dist_stats(gt_data, "GT (with ego defaulted to [0,0,0])")
    dist_stats(pred_data, "Pred")

    print("\n=== ROOT CAUSE ===")
    if gt_with_ego == 0 and pred_with_ego > 0 and n_gt_kept == 0:
        print("CONFIRMED: gt_to_detection_boxes() in format_predictions.py omits")
        print("ego_translation. DetectionBox.deserialize defaults to (0,0,0). The")
        print("formula ||xyz - ego||  becomes ||xyz|| (global frame, ~hundreds of m),")
        print("which exceeds class_range (max 50m). All GT filtered → mAP = 0.")
        print("FIX: write ego_translation = ego_pose_translation_in_global into each")
        print("GT box, matching the convention used by predictions_to_detection_boxes.")
        return 0
    print(f"GT kept = {n_gt_kept}; pattern does not exactly match the hypothesis.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
