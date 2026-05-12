# β baseline GT-loading bug — root-cause diagnosis

Source: `diagnosis_beta_baseline/diagnose_gt_loading.py` (run 2026-05-09).

## Symptom

`results/diagnosis_beta_baseline/aggregate.json` reports
`n_pred_boxes=42`, `n_gt_boxes=0`, `mAP = 0.0` despite
`results/diagnosis_beta_baseline/nuscenes_eval/gt_boxes.json` containing
**1735 GT boxes across 43 samples**.

## Diagnosis

### File-level

| | samples | total boxes | non-empty samples |
|---|---:|---:|---:|
| GT (file) | 43 | **1735** | 43 |
| Pred (file) | 43 | 42 | 29 |

### Class-name match — OK

`{barrier, bicycle, bus, car, construction_vehicle, pedestrian, traffic_cone, truck}` are present in both. Pred is missing `motorcycle`, `trailer` — that is a model-level absence, not a class-name mismatch (pred is a strict subset of GT, no underscore/space discrepancy).

### `ego_translation` field — root cause

| | with `ego_translation` | total |
|---|---:|---:|
| GT  | **0** | 1735 |
| Pred | 42 | 42 |

Pred:  `gt_to_detection_boxes()` in `diagnosis_beta_baseline/format_predictions.py` (line 126-150) **omits the `ego_translation` field** when serializing GT dicts.
Pred (line 105-116, `predictions_to_detection_boxes`) writes it as `ego_pose_4x4[:3, 3]` (ego pose in global frame).

`DetectionBox.deserialize` (nuscenes-devkit) defaults missing `ego_translation` to `(0, 0, 0)`.

### `_filter_by_range` simulation

`evaluate_nuscenes._filter_by_range` (line 54-58) measures ego-distance with the formula `||box.translation[:2] - box.ego_translation[:2]||`. Distance distribution under that formula on the on-disk JSON:

| | n | min | median | max | p99 |
|---|---:|---:|---:|---:|---:|
| GT (ego defaulted to (0,0,0)) | 1735 | 716.83 m | **1631.68 m** | 3163.17 m | 3131.23 m |
| Pred (ego properly populated) | 42 | 0.57 m | 1.32 m | 7.71 m | 7.49 m |

Filter result vs cvpr_2019 `class_range` (max 50 m, min 30 m):

- **Pred kept: 42/42** (every prediction within ~10 m of ego — passes)
- **GT kept: 0/1735** (every GT distance >716 m — exceeds every class_range)

→ `accumulate(gt_boxes={}, pred_boxes=...)` returns `tp=0, fp>0, n_gt=0` for every (class, dist_th), so AP = 0 for every class. mAP = 0.

## Root cause

`gt_to_detection_boxes` in `format_predictions.py` does not propagate ego pose into each GT dict, so `_filter_by_range`'s ego-distance formula sees `xyz - 0 ≈ xyz` (global frame) and rejects every GT box as out-of-range. The pred path is correct because `predictions_to_detection_boxes` already takes `ego_pose_4x4` and writes `ego_translation = ego_pose_4x4[:3, 3]` into each pred dict.

This is a **GT-side serialization omission**, not a metric / model / class-name bug.

## Fix (Stage 2)

1. Extend `gt_to_detection_boxes(sample_token, gt_boxes_global, ego_pose_4x4)` to write `ego_translation = [ego_pose_4x4[i, 3] for i in range(3)]`, matching the pred convention.
2. Without re-running OpenYOLO3D inference, regenerate `gt_boxes.json` only — read sample → ego_pose mapping straight from `data/nuscenes/{v1.0-mini, v1.0-trainval}/{sample_data, ego_pose}.json` (avoiding the slow `NuScenes(...)` constructor on this NFS-bound host).
3. Re-run `evaluate_nuscenes.evaluate(pred_boxes, fixed_gt_boxes, ...)` on the existing predictions.

## Out of scope (per user constraints)

- Do not edit nuscenes-devkit (sys-installed).
- Do not edit `instance_metrics.py` (W1-style instance loader is independent and correct — its 1/1735 matched-rate result in v1 must be preserved unchanged in v2 as a regression check).
- Do not re-run OpenYOLO3D inference; reuse `pred_boxes.json` as-is.
