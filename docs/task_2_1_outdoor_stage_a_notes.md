# Task 2.1 — Outdoor Stage A — design + diagnostic notes

Date: 2026-05-20
Branch: `main` @ 793bad5
Environment: `openyolo3d-dev` (mmdet3d 1.4.0)

## Stage 1 diagnostic — environment + paths (PASS)

| Component | Status | Path / Version |
|---|:--:|---|
| `openyolo3d-dev` conda env | ✓ | `/home/rintern16/miniconda3/envs/openyolo3d-dev/` |
| mmdet3d | ✓ | 1.4.0 |
| nuscenes-devkit | ✓ | importable |
| hdbscan + open3d | ✓ | both importable |
| CenterPoint adapter | ✓ | `adapters/centerpoint_proposals.py` |
| NuScenesLoader | ✓ | `dataloaders/nuscenes_loader.py` |
| Indoor metrics reuse | ✓ | `method_scannet/streaming/metrics.py` |
| Indoor `running_labeler` | ✓ (importable but **not used directly**; see §design choice 3 below) |
| YOLO-World checkpoint | ✓ | `pretrained/checkpoints/yolo_world_v2_x_*.pth` (520 MB) |
| CenterPoint checkpoint | ✓ | `/home/rintern16/pretrained/centerpoint_nuscenes/*.pth` (34 MB) |
| CenterPoint config | ✓ | same dir, `*.py` config |
| nuScenes v1.0-mini data | ✓ | `data/nuscenes/v1.0-mini/` (symlink → `/home/rintern16/SemWorld-3D/data/nuscenes/`) |
| nuScenes detection config | ✓ | `configs/nuscenes_baseline.yaml` |
| OpenYolo3D YOLO-World config | ✓ | `configs/openyolo3d_nuscenes.yaml` (10 nuScenes classes + 5 open-vocab extensions) |

All Stage 1 conditions met. Auto-proceed to Stage 2.

## Stage 2 — StreamingNuScenesEvaluator design choices

The Indoor `StreamingScanNetEvaluator` doesn't transfer line-by-line because four
assumptions break on nuScenes:

1. **Proposal source has built-in instance IDs.** Indoor: Mask3D outputs N
   per-scene instances, IDs `[0, N-1]` stable across all frames. Outdoor: CenterPoint
   produces fresh per-sample proposals with no temporal continuity.
2. **One camera per frame.** Indoor: one RGB+depth per frame; YOLO-World runs once.
   Outdoor: 6 surround cameras per sample; YOLO-World must run 6 times and the
   per-proposal label is fused across the cameras where the proposal projects.
3. **Vertex-mesh-based labeling.** Indoor's `RunningInstanceLabeler.update_frame`
   takes `instance_vertex_masks`, `projection`, `inside_mask`, `label_map`. These
   are vertex-of-mesh abstractions tied to Mask3D's output shape. Outdoor proposals
   are 3D bounding boxes, not vertex sets.
4. **AP metric.** Indoor: ScanNet200 per-instance mask AP via
   `evaluate_scannet200`. Outdoor: nuScenes detection mAP / NDS via
   nuScenes-devkit `DetectionEval` (per-class, per-distance-threshold).

### Decisions

#### 1. Cross-sample association — `CentroidAssociator`

A greedy spatial tracker keyed on `(class, centroid_xy)` runs each sample:
- order new proposals by descending CenterPoint score
- match each to the closest active id of the same class within
  `association_threshold_m` (default 2.0 m, ego frame xy)
- unmatched proposals allocate fresh global ids; matched ids carry over
- active ids age by +1 each sample; dropped after `max_age=5` consecutive misses

Without this, M11/M12 ("confirmed after K appearances") are no-ops because
every proposal would be a "first sighting" each sample. **Documented as a
research decision** — the 2 m threshold and class-aware constraint were chosen
to give the 6.2.3 α-style hybrid an honest baseline. Sensitivity to these
hyperparameters is itself a future ablation.

#### 2. YOLO-World multi-camera fusion — `_yolo_label_for_proposal`

For each proposal centroid:
- project into each of the 6 cameras (extrinsic inverse + intrinsic)
- in-frame cameras: find YOLO-World bbox centers within
  `pixel_match_radius=80` px of the projection
- keep the highest YOLO-score match across cameras
- only nuScenes-10 class names survive (`tree, pole, building, traffic_sign,
  traffic_light` extensions in the YAML are dropped because there's no GT
  category for them in nuScenes detection)

#### 3. NuScenesRunningLabeler — parallel to Indoor's, simpler

We do **not** modify or reuse Indoor's `RunningInstanceLabeler`. It depends on
vertex masks. Instead a tiny dataset-agnostic labeler holds per-`global_id`
class histograms, accepts `add_vote(gid, cls_idx, weight)`, and exposes
`snapshot(ids)` returning `{gid → argmax_class}`. Both classes implement the
same public surface used downstream by `metrics.label_switch_count` and
`metrics.time_to_confirm`.

#### 4. Pred-history format is identical to Indoor

`pred_history: list[dict[global_id, class_idx]]` — one snapshot per sample.
Indoor's `metrics.label_switch_count` and `metrics.time_to_confirm` are
dataset-agnostic and consume this format unchanged. **Zero Indoor file
modified.**

#### 5. M11 / M12 hooks reuse Indoor's `hooks_streaming.install_method_streaming`

`StreamingNuScenesEvaluator` implements the attribute contract
(`self.method_11`, `self.method_12`, etc.) so the Indoor installer just sets
these attributes on it. No fork of `hooks_streaming.py`, no per-method
re-implementation. The gate's contract — `gate(visible_ids) → confirmed_ids` —
is purely id-based and works without modification.

#### 6. nuScenes-devkit evaluation via β-baseline's existing `evaluate()`

`diagnosis_beta_baseline/evaluate_nuscenes.py:evaluate(pred_eb, gt_eb, out)`
is reused as-is (it was written for the May β-baseline run). Our predictions
are formatted as DetectionBox dicts per sample token (size + global
translation + global rotation built from CenterPoint's lidar-frame yaw +
ego rotation).

#### 7. What is *not* in Stage A

- Open-vocab classes beyond nuScenes-10 — deferred (no GT to score against).
- β1 (HDBSCAN) proposal source — Stage B target.
- α distance-aware hybrid — Stage B target.
- Mask3D — not used outdoor (β baseline showed mAP=0 with it).
- M21 / M22 / M31 / M32 — out of Stage A scope per spec.
- Multi-sweep LiDAR aggregation — single keyframe only.

## Stage 2 import test result

```
syntax OK, lines = 739
All classes/functions importable.
associator step 1 → [0]  step 2 (same car @ 0.5 m offset) → [0]  matched=True
labeler snapshot ({0:1 w=0.9} + {0:1 w=0.5} + {0:2 w=0.2}) → {0: 1}  argmax=1 ✓
```

All Stage 2 acceptance conditions met. Auto-proceed to Stage 3.

## Stage 3 — PBS smoke (10 mini scenes × 3 axes)

`scripts/run_task_2_1_outdoor_smoke.pbs`:
- coss_agpu A100 1 GPU, walltime 6 h (smoke expected ≤ 3 h based on β-baseline
  per-sample timings: γ ≈ 0.5–2 s/sample, OY3D init ~17 s, 6× YOLO inference
  ≈ 5–10 s/sample × ~40 samples/scene × 10 scenes ≈ 30 min/axis)
- per-axis output: `metrics.json`, `temporal_metrics.json`,
  `nuscenes_eval/eval_summary.json`, `nuscenes_eval/per_class.json`
- top-level: `all_summaries.json`

## Expected scenarios

- **(P)** mAP > 0 in baseline, M11/M12 produce non-trivial lsc/ttc deltas →
  proceed to Stage B.
- **(Q)** mAP > 0 in baseline but temporal metrics degenerate (all-zero lsc
  or empty ttc) → debug `pred_history` snapshot; not a Stage-B blocker.
- **(R)** mAP = 0 across all axes (β-baseline-like catastrophic) → STOP per
  the absolute-prohibition rule; investigate γ proposal quality, YOLO label
  matching, association threshold.
