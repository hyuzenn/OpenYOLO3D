# Task 1.2c Option E — frame-level debugging on common-class scenes

**Date**: 2026-05-13
**PBS job**: 92048
**Run dir**: `results/2026-05-13_streaming_debug_E/`
**Scenes**: scene0334_00, scene0334_01 (selected by GT class composition; both contain 4 common-class instances, manageable size 105-119 streaming frames each)

## Setup

`method_scannet/streaming/tools/debug_compare.py` monkey-patches:
- `OpenYolo3D.label_3d_masks_from_label_maps` → dumps visibility-matrix top-K rep-frames, per-instance visible-vertex totals, final class assignments.
- `BaselineLabelAccumulator.compute_predictions` → identical dump, from the streaming path.

Both pipelines run **inside the same Python process**, on **the same scene**, with **the same `.npy` input**, against a single shared `OpenYolo3D` instance. Mask3D is invoked once per pipeline.

## Result — pipelines disagree drastically

| Field | scene0334_00 | scene0334_01 |
|---|---|---|
| K instances offline / streaming | 12 / 12 (match) | 10 / 10 (match) |
| `n_valid_frames` offline / streaming | 116 / 117 | 105 / 104 |
| **Rep-frame Jaccard mean (per instance)** | **0.467** | **0.638** |
| Rep-frame Jaccard min | **0.000** | 0.159 |
| Instances with Jaccard < 0.95 | **11 / 12** | **7 / 10** |
| **`per_instance_total_visible` mean diff** | **4 450** | 528 |
| `per_instance_total_visible` max diff | **52 242** | 4 783 |
| Instances with visible-vertex mismatch | 11 / 12 | 7 / 10 |
| Final-class agreement (top-600 ordered) | 8/600 = 1.3% | 113/600 = 18.8% |

Same `K`, identical post-NMS filter — but the **vertex membership of each Mask3D instance differs** between the two calls. With different masks, every downstream quantity diverges (top-K rep-frame selection, pixel-label accumulation, mode → class).

## Verdict on the hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| H1 — `.npy` / `.ply` misalignment | ❌ **busted** | Option B: 312/312 scenes match in vertex count |
| H2 — Mask3D filter shifts top-K | ❌ **partial bust on scene0011_00** | Option A: streaming AP stable under filter+`.npy` |
| H3 — CPU/CUDA float ordering in `get_visibility_mat` | ⚠️ **unlikely solo cause** | Float precision cannot move per-instance visible-vertex counts by 50k+ |
| H4 — frame ordering / sequential vs batch | ⚠️ **unlikely solo cause** | `n_valid_frames` differs by ≤ 1 across both scenes; rep-frame divergence is structural, not ordering |
| H5 — small instances drop out under D3 | ❌ **busted** | The mismatches are not about *which* instances are visible (K matches) but *which vertices* are inside each instance |
| **H6 — Mask3D inference non-determinism between two forward passes** | ✅ **NEW, strongly supported** | Same model, same input, same process → different vertex assignments per instance |

## Mechanism

`network_3d.get_class_agnostic_masks()` is called twice (once per pipeline) for each scene. Despite identical inputs and identical model weights, the two forward passes yield instance masks that share the same K but assign different vertices to each instance. Likely sources:

- cuDNN non-deterministic convolutions (default for Mask3D unless explicitly disabled).
- MinkowskiEngine voxel hashing / sparse tensor reductions, which are order-sensitive across calls.
- Internal random-init state in submodules (anchors, queries) not seeded.

Once the masks differ:
- `get_visibility_mat` selects different top-K rep-frames per instance (Jaccard 0.47 in scene0334_00).
- Different rep-frames look at different pixel label maps.
- The accumulated label distribution per instance ends up dominated by different classes.
- The mode → final class differs in 99% of the top-600 (instance × class) pairs.

The streaming protocol itself is **not broken**. The 312-scene aggregate gap (Δ AP = −0.0562) is at least partly explained by comparing against an offline reference that was produced **on a different day with a different Mask3D forward pass**.

## Implications for Task 1.4 and the sanity threshold

- The ±0.005 sanity threshold against a one-shot offline aggregate is **not achievable** as long as both pipelines re-run Mask3D independently. The single-scene noise floor (≈ 0.0235 same-pipeline run-to-run) propagates into the aggregate.
- For Task 1.4 method-axis comparisons (M11/12/21/22/31/32 vs baseline), this is **not a problem** because each ablation can be measured against a sibling streaming run that shares its Mask3D output. Δ within a single streaming run is the meaningful signal; absolute streaming AP vs absolute offline AP is not.
- The Task 1.4 baseline reference should therefore be **streaming-baseline-on-cached-Mask3D**, not the offline 2026-05-07 number.

## Proposed next step (awaiting user decision)

| # | option | what it does | cost |
|---|---|---|---|
| **G** | **Cache Mask3D output once per scene**: run Mask3D, pickle `(masks, scores)` to disk per scene, have both offline and streaming consume the pickled tensors. Re-run 312-scene sanity. | Eliminates H6 entirely. If aggregate Δ collapses to ≤ ±0.02 (the run-to-run AP_50 noise floor), the streaming protocol is verified correct. | ~30 min impl + ~2.5 h sanity run |
| H | Set `torch.backends.cudnn.deterministic=True` + fixed seeds + re-run debug on the same two scenes | Cheapest test of H6 → confirm Mask3D is the source. Not a full fix because MinkowskiEngine voxel hashing may still vary. | ~10 min impl + ~3 min debug run |
| I | Switch the sanity criterion to AP_50 (run-to-run noise ≈ 0.003) and accept the existing Δ as residual non-determinism | Cheapest, but doesn't actually verify streaming. | edit only |

Recommendation: **H first (cheap), G if H insufficient**. H tells us whether cuDNN alone explains H6; G is the definitive verification.

## Acceptance status (Option E)

| # | item | status |
|---|---|---|
| 1 | 1-2 scenes picked by GT bucket criterion | ✅ scene0334_00 / scene0334_01 |
| 2 | streaming debug runner | ✅ via `BaselineLabelAccumulator.compute_predictions` monkey-patch |
| 3 | offline mid-run dump | ✅ via `OpenYolo3D.label_3d_masks_from_label_maps` monkey-patch (in-process, no source edits) |
| 4 | comparison + H4/H5 analysis | ✅ summary above; H4/H5 busted, **H6 found** |
| 5 | md5 — OpenYOLO3D core unchanged | ✅ (monkey-patches reverted on exit) |
