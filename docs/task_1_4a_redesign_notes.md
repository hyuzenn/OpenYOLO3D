# Task 1.4a redesign — failure analysis + smoke results (2026-05-14)

## TL;DR

First-attempt Task 1.4a (committed `fe7f9eb`, May 13) shipped hook plumbing
with mock-only unit tests. When Task 1.4b launched the 12-axis ablation
on top, the first 4 axes (`baseline`, `M11`, `M12`, `M21`) all produced
the exact same AP = **0.19560** as the baseline; the remaining 8 axes
were expected to crash. Diagnosis revealed three independent
no-op/crash root causes — all in the streaming wrapper, none in the May
method classes. After redesigning the wrapper to match the May class
signatures exactly, the 1-scene integration smoke on `scene0011_00`
now shows **10/10 methods PASS** (each method produces a fingerprint
that differs from baseline, i.e. is verifiably not a no-op).

## What went wrong in the first attempt

| Method axis | Failure mode | Root cause |
|---|---|---|
| M11 / M12 (registration) | Silent no-op (AP = baseline) | Gate updated `_confirmed` during step_frame, but `compute_baseline_predictions()` never consulted it — final preds came from all Mask3D proposals regardless. |
| M21 (WeightedVoting) | Silent no-op | Wrapper called `method_21.observe_frame(...)` and `finalize(...)` guarded by `hasattr(...)` — neither method exists on May's `WeightedVoting`. Both calls were silently skipped. |
| M22 (FeatureFusionEMA) | Silent no-op | Same as M21: `observe_frame`/`finalize` don't exist on `FeatureFusionEMA`. Also no CLIP image-feature encoding was wired up at all. |
| M31 (IoUMerger) | Crash (TypeError, masked as `NotImplementedError`-catch miss) | Wrapper called `merger.merge(pred_masks=…, scene_vertices=…)` — May signature uses `predicted_masks=…, vertex_coords=…`. Wrong kwargs ≠ no-op; the try/except caught only `NotImplementedError`, so TypeError propagated and crashed the axis. |
| M32 (HungarianMerger) | Crash | Same kwarg mismatch + the wrapper never built the `instance_list` / `instance_features` dicts that `HungarianMerger.merge` takes. |

The "47/47 unit tests PASS" before launch did not catch any of this
because every test fed mock data with sentinel labels through a stub
hook — `hasattr` guards make the stubs equivalent to baseline, and the
mock evaluator was never wired to call the real May methods with real
shapes.

## Redesign summary

Three localized changes; **no May method class was touched**.

### 1. `method_scannet/streaming/method_adapters.py` (new)

Owns the streaming↔May adaptation layer with five public helpers:

- `apply_registration_filter(preds, mask_idx, confirmed_set)` — used by
  M11/M12. Takes the gate's confirmed set and slices `pred_masks`
  columns whose underlying Mask3D proposal idx is *not* confirmed.
- `compute_predictions_method21(accumulator, voter, scene_vertices,
  camera_positions, image_width, image_height)` — replays the
  accumulator's stacked per-frame data and substitutes
  `WeightedVoting`'s distance-weighted vote for the baseline's
  mode-vote (matches offline `_patched_label_3d_masks_from_label_maps`
  in `hooks.py`).
- `compute_predictions_method22(accumulator, fusion, topk_per_image)` —
  consumes the per-instance feature dict that
  `wrapper._method22_per_frame` populated during streaming and runs
  the same cosine-distribution top-k expansion as the offline
  `_apply_method_22`.
- `apply_method31_merge(preds, merger, scene_vertices)` — single
  batched call with the correct kwargs (`predicted_masks=`,
  `vertex_coords=`).
- `apply_method32_merge(preds, merger, scene_vertices, mask_idx,
  instance_features)` — builds the `instance_list` of
  `{id, label, centroid}` dicts and the `instance_features` dict that
  `HungarianMerger.merge` requires, class-aware-grouped exactly like
  offline `_apply_method_32`.

### 2. `method_scannet/streaming/wrapper.py` — `step_frame` + `compute_method_predictions`

- New per-frame state: `_camera_positions[]` (M21 needs the world-frame
  camera translation per pose).
- Removed the dead `method_21.observe_frame` / `method_22.observe_frame`
  / `label_method.finalize(...)` hasattr-guarded calls.
- Added `_method22_per_frame(...)` — for each confirmed-visible
  proposal, project its vertex set to the depth frame, match to the
  best YOLO 2D bbox by IoU, crop, CLIP-encode, and call
  `method_22.update_instance_feature(...)`.
- Rewrote `compute_method_predictions()` to call the new adapter
  helpers in the documented order: label assignment → registration
  filter → spatial merge.

### 3. `method_scannet/streaming/baseline.py`

- `BaselineLabelAccumulator.compute_predictions()` now stashes the
  `mask_idx` mapping (output column k → Mask3D proposal idx) on
  `self._last_mask_idx` so the wrapper can post-filter by registration
  confirmed set.
- New public `stack_for_methods()` returns the per-frame buffers
  (numpy arrays + raw bbox dicts) the adapter helpers need at
  finalize.

### 4. `method_scannet/streaming/hooks_streaming.py`

- `install_method_22` now also constructs a `CLIPImageEncoder` and
  loads the prompt-embeddings subset (`openyolo3d_inference_classes`),
  matching offline `install_method_22_only` semantics. Encoder
  defaults to `device='cuda'` for the streaming path (CPU would push
  Task 1.4b walltime past the queue limit — see below).

### 5. M32-only secondary bug (caught during smoke, fixed before resubmit)

The first smoke pass (results/2026-05-14_task14a_smoke_v01) showed M32
matching the baseline fingerprint exactly. Root cause: `HungarianMerger`
defaults to `semantic_threshold=0.3`, which masks every pair when the
feature dict is empty (M32-only, no upstream M22) — because cosine sim
against zero-vector features is 0 < 0.3. The May offline
`_apply_method_32` handles this by instantiating a fresh merger with
`semantic_threshold=-1.0` when no features are passed; the streaming
adapter (`apply_method32_merge`) now does the same. Verified in
results/2026-05-14_task14a_smoke_m32fix_v02: M32 alone now drops 600
→ 316 predictions (strong spatial merge).

## 1-scene smoke results (scene0011_00)

Both runs use the cached Mask3D output (results/2026-05-13_mask3d_cache).
"PASS" means the output fingerprint differs from baseline beyond
numerical noise — verifies that the method is not a no-op. AP not
computed (single-scene mAP is too noisy to compare; the verification is
fingerprint-level).

| method   | n_pred | n_cls | scores_mean | scores_max | PASS? |
|----------|-------:|------:|------------:|-----------:|-------|
| baseline |    600 |   107 |      0.1432 |     1.0000 | ref   |
| M11      |    569 |    90 |      0.1493 |     1.0000 | PASS  |
| M12      |    569 |    90 |      0.1493 |     1.0000 | PASS  |
| M21      |    600 |   108 |      0.1340 |     1.0000 | PASS  |
| M22      |    600 |   140 |      0.2594 |     0.2887 | PASS  |
| M31      |    579 |   107 |      0.1442 |     1.0000 | PASS  |
| M32      |    316 |   107 |      0.1984 |     1.0000 | PASS (fix v02) |
| phase1   |    550 |    91 |      0.1400 |     1.0000 | PASS  |
| phase2   |    362 |   141 |      0.2612 |     0.2887 | PASS  |
| M21+M31  |    579 |   108 |      0.1347 |     1.0000 | PASS  |
| M22+M32  |    362 |   140 |      0.2612 |     0.2887 | PASS  |

What this confirms axis-by-axis:
- **M11/M12**: registration filter is active — drops 31 columns whose
  proposal idx was never confirmed.
- **M21**: WeightedVoting changes the per-instance class scores
  (mean and one additional unique class observed).
- **M22**: full cosine-distribution reclassification — scores live in
  [0, 0.29] cosine space, 33 more unique classes than baseline.
- **M31**: 21 column drops due to class-aware NMS.
- **M32 (fixed)**: 284 column drops with `semantic_threshold=-1.0` for
  spatial-only mode.
- **Compounds**: each compound's fingerprint is consistent with its
  axes combined (e.g. phase2 ≈ M22 reclassification followed by M32
  merge → 600→362 predictions).

Per-axis walltime on scene0011_00 (238 frames after frequency=10
sub-sampling): plain axes ≈ 35–40 s; M22 axes ≈ 155–162 s (CPU CLIP);
total smoke + retry ≈ 14 min on a single A100.

## md5 verification (May classes + OpenYOLO3D core, unchanged)

```
1efd60771aa9e3a868a4d23c1958bd80  method_scannet/method_11_frame_counting.py
01f0d59a812c5f8bf47d81f04e118f13  method_scannet/method_12_bayesian.py
ced26619420af839947c60ae683525ef  method_scannet/method_21_weighted_voting.py
3b7509f65b05177aa209ab1c010933ef  method_scannet/method_22_feature_fusion.py
80735839b7621b31230df155c9fa8162  method_scannet/method_31_iou_merging.py
b0423e5a403c02aa5633493208046989  method_scannet/method_32_hungarian_merging.py
078e70661242856c64511398bdf68b30  method_scannet/hooks.py
81905ea69feee4b2a92e30f9c971a1f2  method_scannet/clip_image_encoder.py
2e74b191796c435a5a63172be307da42  utils/__init__.py
```

`git diff HEAD` against these paths returns empty — Task 1.4a touches
the streaming wrapper / adapters / hooks_streaming only.

## Walltime headroom for Task 1.4b

CPU CLIP would push M22-flavored axes (M22, phase2, M22+M32) to
~14 h × 3 = 42 h on top of the ~17 h non-M22 work, exceeding the 6-PBS
sequential budget. The fix:
- `install_method_22` defaults the `CLIPImageEncoder` to `device='cuda'`
  (falls back to CPU if CUDA is not visible).
- `apply_method22_finalize` moves prompt embeddings to the feature's
  device before the cosine matmul and back to CPU for the
  distribution tensor.

Plus walltime bumps for the M22-heavy PBS scripts (pbs3, pbs5, pbs6:
6h → 12h). CUDA library path (`CUDA_HOME=/tools/cuda/cuda11.7`) is now
exported by every streaming PBS (one compute node missing it caused the
first M32-fix smoke to crash on import).

## Task 1.4b restart readiness

✅ All 10 method configs no longer no-op (1-scene smoke).
✅ All 5 May method classes unchanged (md5 stable).
✅ All 6 PBS scripts patched (CUDA env + 12h walltime for heavy ones).
✅ Mask3D cache (results/2026-05-13_mask3d_cache) reused — 312 scenes
present.

Pending: this user-facing review, then `qsub` 6 PBS scripts → v2 result
directories preserve the first attempt's data.
