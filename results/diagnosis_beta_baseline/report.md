# β baseline — OpenYOLO3D × nuScenes detection (50 samples)

- samples: 43/50 evaluated (skipped 7)
- samples_used: `results/diagnosis_step_a/samples_used.json` (seed=42)
- text prompts: `['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']`
- OpenYOLO3D init: 17.7s

## Standard nuScenes detection metrics

- **mAP**: 0.00%  (0.0000)
- **NDS**: 0.0000
- trans_err: 1.0000
- scale_err: 1.0000
- orient_err: 1.0000
- vel_err: 1.0000
- attr_err: 1.0000
- n_pred_boxes (after class_range filter): 42
- n_gt_boxes (after class_range filter):   0

### Per-class breakdown (10 detection classes)
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

## Instance-level (W1-style)

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

## Timing (s/sample)

| stage | median | p95 | mean | n |
|---|---:|---:|---:|---:|
| adapter_s | 0.08 | 0.08 | 0.08 | 43 |
| predict_s | 0.38 | 0.51 | 0.61 | 43 |
| format_s | 0.01 | 0.01 | 0.01 | 43 |
| metrics_s | 0.02 | 0.08 | 0.03 | 43 |
| total_s | 0.51 | 0.67 | 0.74 | 43 |

## Decision

- overall mAP = 0.00% → **β-D**
  - FAIL < 5 — catastrophic; OpenYOLO3D is not a viable nuScenes baseline. Narrative is very strong; ALSO consider adding a second baseline so the comparison is meaningful.
- pedestrian AP_mean = 0.0000 → **PED_pedestrian_FAIL**
- far-range (50m+) M_rate = 0.000 → **FAR_FAIL**
- median per-sample latency = 0.51s → **RT_SLOW**

**Trace**: mAP < 5 → β-D. Baseline is meaningless on its own; we still report it, but additionally need a second nuScenes baseline (e.g. 3D detector trained on outdoor data) so the comparison section is not vacuous.

## Figures

- ![mAP_per_class.png](figures/mAP_per_class.png)
- ![mAP_per_distance.png](figures/mAP_per_distance.png)
- ![M_rate_breakdown.png](figures/M_rate_breakdown.png)
- ![timing.png](figures/timing.png)
