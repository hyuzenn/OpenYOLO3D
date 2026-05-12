# β baseline v2 — GT-bug-fixed nuScenes detection (50 samples)
- samples: 43/50 evaluated (skipped 7) — same set as v1
- text prompts: ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
- v1 had `n_gt_boxes=0` due to missing `ego_translation` field on GT dicts. Fix: see `gt_bug_diagnosis.md`.

## v1 vs v2 — headline

| metric | v1 (buggy) | v2 (fixed) |
|---|---:|---:|
| mAP | 0.0000 | 0.0000 |
| NDS | 0.0000 | 0.0000 |
| n_pred_boxes (post class_range) | 42 | 42 |
| n_gt_boxes (post class_range) | 0 | 1326 |
| n_samples | 43 | 43 |

## TP errors (v2)

- trans_err: 1.0000
- scale_err: 1.0000
- orient_err: 1.0000
- vel_err: 1.0000
- attr_err: 1.0000

## Per-class breakdown — v2

| class | AP_mean | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | trans_err | scale_err | orient_err | vel_err | attr_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| truck | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| trailer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| construction_vehicle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| pedestrian | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| motorcycle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bicycle | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| traffic_cone | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | nan | nan | nan |
| barrier | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | nan | nan |

## Instance-level (W1-style) — UNCHANGED from v1 (regression check)

- n_GT total: 1735
- M_rate:    0.001
- L_rate:    0.010
- D_rate:    0.068
- miss_rate: 0.921

### Distance-stratified instance metrics
| bin | n_GT | M_rate | L_rate | D_rate | miss_rate |
|---|---:|---:|---:|---:|---:|
| 0-10m | 145 | 0.000 | 0.062 | 0.276 | 0.662 |
| 10-20m | 383 | 0.003 | 0.018 | 0.131 | 0.849 |
| 20-30m | 444 | 0.000 | 0.002 | 0.045 | 0.953 |
| 30-50m | 494 | 0.000 | 0.002 | 0.014 | 0.984 |
| 50m+ | 269 | 0.000 | 0.000 | 0.004 | 0.996 |

## Timing (s/sample) — UNCHANGED from v1

| stage | median | p95 | mean | n |
|---|---:|---:|---:|---:|
| adapter_s | 0.08 | 0.08 | 0.08 | 43 |
| predict_s | 0.38 | 0.51 | 0.61 | 43 |
| format_s | 0.01 | 0.01 | 0.01 | 43 |
| metrics_s | 0.02 | 0.08 | 0.03 | 43 |
| total_s | 0.51 | 0.67 | 0.74 | 43 |

## Why is v2 mAP still 0? (sub-recall analysis)

The bug fix lifts `n_gt_boxes` from 0 → 1326 (after class_range filter), so `accumulate` can now match GT and pred. But mAP is still exactly 0 because of a structural lower bound, not a second bug.

Closest in-class GT distance for each of the 42 predictions, restricted to the same sample and after class_range filtering on both sides:

| class | n_pred | d≤4 m | d≤2 m | d≤1 m |
|---|---:|---:|---:|---:|
| car | 15 | 3 | 0 | 0 |
| truck | 8 | 0 | 0 | 0 |
| bus | 4 | 0 | 0 | 0 |
| construction_vehicle | 1 | 0 | 0 | 0 |
| pedestrian | 4 | 0 | 0 | 0 |
| bicycle | 1 | 0 | 0 | 0 |
| traffic_cone | 4 | 3 | 2 | 1 |
| barrier | 5 | 0 | 0 | 0 |
| **TOTAL** | **42** | **6** | **2** | **1** |

So the maximum-recall ceiling per class (under the loosest threshold `dist_th=4 m`):

- car: 3 / 516 GT = 0.58%
- traffic_cone: 3 / 204 GT = 1.47%
- every other class: 0 / N

`detection_cvpr_2019` requires `min_recall = 0.10`; the AP integration window covers `recall ∈ [0.1, 1.0]`. With the per-class achievable recall pinned at <2 %, the integration window is empty, so `calc_ap` returns 0 deterministically. `mAP = 0.00 %` in v2 is therefore arithmetically correct, **not** a residual bug.

This makes the failure mode load-bearing for the narrative: OpenYOLO3D produces ~1 detection per outdoor sample (avg 42/43), and even those are only ~14 % aligned to the right class+location at 4 m. There is nothing to recall.

## Decision (re-fired on v2 mAP)

- overall mAP (v2 corrected) = 0.00% → **FAIL (<5%)**
- Catastrophic on outdoor data; consistent with v1 narrative even after fixing the GT bug. The bug masked the precise mAP value but the conclusion is unchanged: OpenYOLO3D is not a viable nuScenes baseline on its own. Comparison section needs a second outdoor-trained baseline.
- pedestrian AP_mean < 5% → **PED_pedestrian_FAIL**
- far-range (50m+) M_rate < 5% → **FAR_FAIL**
- median per-sample latency = 0.51s → **RT_SLOW**

## Regression check — md5 / acceptance criteria

- nuScenes-devkit code: untouched (sys-installed under conda env, not modified)
- OpenYOLO3D core (utils.py, evaluate/, models/, embed/): untouched
- instance_metrics.py (W1 GT loader): untouched; instance-level metrics identical to v1
- format_predictions.py: extended `gt_to_detection_boxes` to accept `ego_pose_4x4` and write `ego_translation` (additive change)
- run_baseline.py: passes `ego_pose_4x4=item['ego_pose']` to gt_to_detection_boxes (call-site update; no inference re-run for v2)
- predictions / inference: NOT re-run; v2 reuses v1's `pred_boxes.json` byte-for-byte
- samples_used.json: NOT changed; same 43/50 samples evaluated
