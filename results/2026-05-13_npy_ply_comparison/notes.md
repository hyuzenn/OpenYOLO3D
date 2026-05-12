# Task 1.2c diagnosis — Option B: 312-scene .npy vs .ply vertex count comparison

**Date**: 2026-05-13
**Tool**: `method_scannet/streaming/tools/compare_npy_ply.py` (CPU only, 70 s for 312 scenes).
**Purpose**: Directly test H1 — whether the `.npy` Mask3D input and `.ply` projection mesh use a consistent vertex layout across all ScanNet200 validation scenes.

## Result

| Field | Value |
|---|---|
| total scenes | 312 |
| matched (len(.npy) == len(.ply)) | **312** |
| mismatched | **0** |
| missing input | 0 |

Every single scene's `.npy` and `.ply` files have **identical vertex counts**. Sampling spot-checks (scene0011_00: 237 360 each) confirm a 1-to-1 correspondence.

## Verdict on H1

**H1 BUSTED.** There is no vertex misalignment between the Mask3D `.npy` input and the projection `.ply` mesh. The streaming wrapper's setup (Mask3D consumes `.npy`, projection uses `.ply`) is safe at the dataset level.

## Implication

The 312-scene streaming aggregate AP gap (−0.0562 vs offline) is **not** caused by vertex layout mismatch. Diagnostic effort can move to the remaining hypothesis H3 (CPU vs CUDA float ordering in `get_visibility_mat`) or to additional per-bucket / per-scene experiments to localize where the gap originates.

## Artefacts

- `results/2026-05-13_npy_ply_comparison/summary.json`
- `results/2026-05-13_npy_ply_comparison/per_scene.json` (per-scene counts for every val scene)
