# Task 3.0 — Architecture inventory (Indoor + Outdoor unified)

Date: 2026-05-21 · Scope: **read-only**, zero code changes · CPU-only.
Purpose: paper §3 architecture material + an honest separation of upstream
vs. user contribution. Every claim below is grounded in a file:line; design
intent is cited from docstrings / prior task docs, **not** inferred.

> Note on the "A–F" step labels: the user's task framing names a 6-step
> A–F pipeline. OpenYOLO3D's source does not literally name steps A–F; the
> mapping below ties each conceptual step to the actual function that
> implements it (authoritative trace: `docs/task_1_1_pipeline_analysis.md`).

---

## 1.1 Upstream OpenYOLO3D (base system — NOT user contribution)

Entry: `run_evaluation.py:__main__` → `test_pipeline_full` →
`OpenYolo3D.predict` (`utils/__init__.py:103`). Per-scene one call;
312-scene loop then `evaluate_scannet200`.

| Step | Role | Module / function | In | Out |
|------|------|-------------------|----|----|
| **A** | 3D proposal generator | Mask3D via `network_3d.get_class_agnostic_masks` (`utils/__init__.py:115`); NMS+threshold `:116-118` | scene point cloud | `masks[V,K]`, `scores[K]` (K≈50–150) |
| **B** | 2D open-vocab labeler | YOLO-World via `network_2d.get_bounding_boxes(color_paths, text)` (`utils/__init__.py:127`); per-frame `inference_detector` | RGB frames + text prompts | `dict[frame → {bbox,labels,scores}]` |
| **C** | projection + visibility | `WORLD_2_CAM.get_mesh_projections` (`utils/__init__.py:391`) | mesh verts, poses, depth | `projected_points[F,V,2]`, `inside_mask[F,V]` (frustum + depth 0.05 m) |
| **D** | label assignment (MVPDist) | `label_3d_masks_from_label_maps` (`utils/__init__.py:158`): `construct_label_maps` → `get_visibility_mat` (top-K frames) → per-instance **mode** vote | A,B,C outputs | per-instance class + score |
| **E** | aggregation | top-K representative frames (`topk=40`, GT=25) + `topk_per_image=600` final filter (inside D) | vote distribution | `K_final` instance-label pairs |
| **F** | spatial merge / output | upstream: none (predictions emitted directly); user's M31/M32 plug in here | D/E output | `(masks[V,K_final], classes, scores)` |

Upstream is used **unmodified** (md5 stability verified across Task 1.4a/c).
Steps C/D/E/F all live inside `label_3d_masks_from_label_maps` in the
offline pipeline — the streaming layer (1.2) is what splits them per-frame.

---

## 1.2 Indoor temporal layer (USER contribution — `method_scannet/`)

### Method classes (6), by axis and pipeline step

| ID | Class · file | Axis | Step | One-line mechanism (per docstring) |
|----|--------------|------|------|------------------------------------|
| **M11** | `FrameCountingGate` · `method_11_frame_counting.py` | registration | **C** | confirm instance after N (cumulative) visible frames |
| **M12** | `BayesianGate` · `method_12_bayesian.py` | registration | **C** | posterior P(real\|history) ≥ threshold confirms |
| **M21** | `WeightedVoting` · `method_21_weighted_voting.py` | label | **D** | replace mode-vote with `α·exp(-d3/D)+(1-α)·exp(-d2/C)`-weighted vote |
| **M22** | `FeatureFusionEMA` · `method_22_feature_fusion.py` | label | **D** | per-instance CLIP-feature EMA, cosine-classify vs prompt embeddings |
| **M31** | `IoUMerger` · `method_31_iou_merging.py` | spatial merge | **F** | class-aware 3D vertex-IoU NMS (keep higher score) |
| **M32** | `HungarianMerger` · `method_32_hungarian_merging.py` | spatial merge | **F** | Hungarian on `α·dist+(1-α)·(1-cos)`, union-find clusters |

Axis design (registration=C, label=D, merge=F) is documented in
`docs/task_1_1_pipeline_analysis.md §7` and `hooks_streaming.py:10-24`.
M11/M12 are **streaming-only** — meaningless in static offline ScanNet
(all instances already present); they activate online (pipeline_analysis §7).

### Streaming infrastructure (USER contribution — `method_scannet/streaming/`)

| Component | File | Role |
|-----------|------|------|
| `StreamingScanNetEvaluator` | `wrapper.py` | per-frame driver; `step_frame` + `compute_method_predictions` install all 6 axes |
| `RunningInstanceLabeler` | `running_labeler.py` | per-frame cumulative label histogram snapshot (baseline + M21 weighting; M22 via `snapshot_method22`) |
| temporal metrics | `metrics.py` | `label_switch_count` (lsc), `time_to_confirm` (ttc), incremental mAP, mask-IoU AP |
| axis install hooks | `hooks_streaming.py` | attribute-injection installers `install_method_{11..32}` + compounds (`phase1`,`phase2`, …) |
| May↔streaming adapters | `method_adapters.py` | packages per-frame ↔ batch calls; calls the **unmodified** May classes |
| visibility | `visibility.py` | D1/D2/D3 instance-visibility thresholds |

**Indoor wiring is complete.** `wrapper.compute_method_predictions`
(`wrapper.py:566-646`) calls every axis in order **label (M21/M22) →
registration filter (M11/M12) → spatial merge (M31/M32)**:
- label: `compute_predictions_method21` (`:580-594`) / `_method22` (`:605-613`)
- registration: `apply_registration_filter` (`:626-632`)
- merge: `apply_method31_merge` (`:636-638`) / `apply_method32_merge` (`:640-646`)
- M22 per-frame CLIP encode: `_method22_per_frame` (`:489-560`).

This is why the Indoor M21/M22/M31/M32 results are **real** (the Task 1.4a
redesign fixed the earlier no-op/crash; 10/10 axes verified non-no-op,
`docs/task_1_4a_redesign_notes.md`). Contrast with Outdoor in §1.4 / §2.5.

---

## 1.3 Outdoor extension (USER contribution)

| Component | File | Role | Notes |
|-----------|------|------|-------|
| γ proposals | `adapters/centerpoint_proposals.py` | CenterPoint (mmdet3d 1.4.0) → proposals; **replaces Step A** | Task 2.5 `NUSC_10` class-map fix (checkpoint head order) `:37-40`; Δt 5th-channel feed `:113` |
| β1 proposals | `adapters/lidar_proposals.py` | HDBSCAN class-agnostic clustering; alt Step A | ground filter + clustering only |
| α union | `diagnosis_alpha/union_strategies.py` | 4 β1∪γ union strategies | **GT-oracle simulation, not runtime** |
| nuScenes bridge | `adapters/nuscenes_to_openyolo3d.py` | one sample → OpenYOLO3D scene dir | ego frame = "world" |
| data loader | `dataloaders/nuscenes_loader.py` | production loader | Task 2.5 multi-sweep Δt 5th column `:75-100` |
| Stage C evaluator | `method_scannet/streaming/nuscenes_evaluator.py` | OpenYOLO3D-pipeline outdoor eval (YOLO-World relabel) | **only M11/M12 wired** (§2.5) |
| native γ dump | `scripts/centerpoint_native_map_sanity.py` | native CenterPoint mAP, YOLO bypassed | Step 2a (mAP 0.3407) |
| native temporal eval | `method_scannet/streaming/nuscenes_native_evaluator.py` | Step 2b: native γ + temporal layer | **in progress on separate PBS — not analyzed here, not modified** |

The Outdoor evaluator imports the Indoor temporal layer
(`nuscenes_evaluator.py:46-50`) but, in Stage C, honestly exercises **only
M11/M12** (§2.5).

Two `NUSC_10` tuples exist and are **not** the same — important:
- `centerpoint_proposals.py:37-40` — **checkpoint CenterHead emit order**
  (used as label index → name). This was the Task 2.5 permutation bug.
- `nuscenes_evaluator.py:54-57` — devkit order, used **name-based** for
  `text_prompts` / `NUSC_10_SET` membership; emission uses the stored name
  (`:788`), so this tuple's order does **not** cause a permutation bug.

---

## 1.4 Data flow (text)

### Indoor

```
ScanNet scene
  → A: Mask3D → 3D proposals (masks[V,K])              [one-shot per scene]
  → B: YOLO-World → 2D labels (single RGB-D stream, per frame)
  → C: WORLD_2_CAM projection/visibility ; M11 OR M12 registration gate
  → D: MVPDist label vote ; M21 (weighted vote) OR M22 (CLIP-EMA reclassify)
  → F: M31 (IoU NMS) OR M32 (Hungarian) spatial merge
  → confirmed instance map (per-frame snapshot, RunningInstanceLabeler)
  → temporal metrics (lsc, ttc) + Mean AP
```
All six axes are genuinely invoked (`wrapper.py:566-646`).

### Outdoor — two modes (same γ proposal source; differ in class assignment)

**Mode 1 — OpenYOLO3D pipeline (Stage C, `nuscenes_evaluator.py`, mAP 0.0526):**
```
nuScenes sample (6 cams + LiDAR)
  → A: γ CenterPoint (or β1 / hybrid) → 3D proposals     [Mask3D unusable outdoor]
  → cross-sample CentroidAssociator → stable global_id   (enables M11/M12)
  → B: YOLO-World on 6 cameras → 2D open-vocab labels
       (γ native class DISCARDED; YOLO assigns class by 3D-box↔2D-bbox IoU)
  → C: M11 OR M12 registration gate (step_sample:771-774)  ← FUNCTIONAL
  → D: NuScenesRunningLabeler.add_vote(YOLO class)         ← NOT M21/M22
  → F: (no spatial-merge step)                             ← NOT M31/M32
  → confirmed instance map → mAP 0.0526 (open-vocab capability demo)
```
M21/M22/M31/M32 install but are never read here → silent no-op (§2.5).

**Mode 2 — Native classifier (Step 2a done, Step 2b in progress, mAP 0.3407):**
```
nuScenes sample
  → A: γ CenterPoint → 3D proposals + native class labels  [Task 2.5 class-map fix]
  → B/C/D YOLO bypass (no 6-camera relabel)
  → native γ class label → mAP 0.3407 (Step 2a baseline, single-sweep)
  → native γ + temporal layer → Step 2b (separate PBS, not analyzed here)
```

**Framing:** Mode 1 = open-vocab capability (paper supplementary);
Mode 2 = detection metric (paper main). Both share the γ proposal source;
the only difference is *who assigns the class* (YOLO-World vs native head).

---

## 1.5 Contribution ledger (upstream vs. user-authored)

**Upstream (NOT user contribution):** Mask3D, YOLO-World, OpenYOLO3D base
(`run_evaluation.py`, `utils/`), CenterPoint checkpoint + mmdet3d, the A–F
inference core.

**User contribution = (2) Indoor temporal layer + (3) Outdoor extension:**
- `method_scannet/method_11..32.py` — 6 method classes
- `method_scannet/streaming/*` — wrapper, running_labeler, metrics, hooks_streaming, method_adapters
- `adapters/centerpoint_proposals.py`, `lidar_proposals.py`, `nuscenes_to_openyolo3d.py`
- `dataloaders/nuscenes_loader.py`
- `diagnosis_alpha/union_strategies.py` (GT-oracle, not runtime)
- `method_scannet/streaming/nuscenes_evaluator.py` + `nuscenes_native_evaluator.py`

**Results status (for paper §6):**
- Indoor: M11 sole stabilizer; M12 = M11 until silent-bug fix (Task 1.4c);
  M21/M22/M31/M32 negative (see audit doc).
- Outdoor Mode 1 (Stage C): mAP 0.0526 (open-vocab capability).
- Outdoor Mode 2 (Step 2a native γ): mAP 0.3407 / NDS 0.3145, single-sweep.
- Outdoor Step 2b: in progress (separate PBS).

---

## Acceptance (Stage 1)
- 1.1–1.3 inventory (upstream + Indoor + Outdoor): ✓
- 1.4 data flow, both domains, Outdoor two modes: ✓
- 1.5 this file + honest upstream/user split: ✓
