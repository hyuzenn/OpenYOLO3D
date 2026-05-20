# Task 2.4 — Outdoor Stage C (v1.0-trainval val anchor) — setup notes

Date: 2026-05-20
Branch: main @ 420eb77 (post Task 2.3)
Env: openyolo3d-dev

## Stage 1 — data + scope verification (PASS)

| Check | Result |
|---|---|
| v1.0-trainval metadata | ✓ present (`data/nuscenes/v1.0-trainval/*.json`) |
| keyframe blobs | ✓ all 10 (`.v1.0-trainval01..10_keyframes.txt`) |
| LIDAR_TOP sample files | 34,149 (≈ full trainval keyframes) |
| official val scenes resolved | **150 / 150** in DB |
| val samples total | **6,019** |
| file spot-check (LiDAR+CAM_FRONT) | 0 missing of 40 checked |

Full val data is present. Stage C is viable.

## Scope

- Split: official nuScenes v1.0-trainval **val** (150 scenes, 6,019 samples) via
  `nuscenes.utils.splits.val`. New evaluator flag `--scene-split val`.
- Proposal source: **γ (CenterPoint) only** (Stage B showed hybrid −12% mAP; γ is the
  outdoor production stack).
- Axes: baseline, M11 (N=3), M12 (threshold 0.85 = confirm@2, the Stage B outdoor
  operating point).
- Metrics: mAP + NDS + temporal (lsc, ttc) + per-class AP.

## Walltime estimate

Stage A v03: 404 mini samples × 3 axes ≈ 14 min ⇒ ≈ 0.7 s/sample/axis.
Val = 6,019 samples ≈ 15× mini ⇒ ≈ 70 min/axis.
- PBS A (baseline + M11): ~2.3 h. Walltime 10 h (generous margin for cluster + the
  full-trainval NuScenes object load, which is heavier than mini).
- PBS B (M12 thr=0.85): ~1.2 h. Walltime 5 h.

## PBS split + shared output dir

| PBS | axes | output |
|---|---|---|
| A (`run_task_2_4_stage_c_pbs_a.pbs`) | baseline, M11 | `results/<date>_outdoor_stage_c_v01/` (fixed name) |
| B (`run_task_2_4_stage_c_pbs_b.pbs`) | M12_thr085 | same dir (PBS B globs PBS A's dir) |

Shared dir: PBS A creates `results/${DATE}_outdoor_stage_c_v01`; PBS B resolves it via
`ls -d results/*_outdoor_stage_c_v01`. Each invocation overwrites `all_summaries.json`
(only its own axes), so Stage 5 aggregates from the persistent per-axis
`axis_*/metrics.json` instead. Sequential per 1-GPU/user policy; PBS B auto-enters only
on PBS A scenario P.

## Scenario gates (PBS A)

- **P**: Exit 0 + baseline mAP > 0.04 (mini was 0.0620) → auto PBS B.
- **Q**: Exit 0 + baseline mAP < 0.02 (< ⅓ of mini) → stop, user review.
- **R**: crash / mAP = 0 → stop, user review.

## Expected outputs

- `results/<date>_outdoor_stage_c_v01/axis_{baseline,M11,M12_thr085}/{metrics,temporal_metrics}.json`
  + `nuscenes_eval/{eval_summary,per_class}.json`
- `report.md` (Stage 5): smoke-vs-anchor consistency, 3-axis anchor table,
  Indoor-vs-Outdoor comparison, advisor §6 material.

## Code change

Only `nuscenes_evaluator.py`: `+ _list_val_scenes()`, `+ --scene-split {all,val}` flag,
scene-selection branch. Indoor / adapters / dataloaders / diagnosis_alpha unchanged.
