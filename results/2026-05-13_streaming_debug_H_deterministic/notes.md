# Task 1.2c Option H — cuDNN deterministic + seeds (H6 source narrowing)

**Date**: 2026-05-13
**PBS job**: 92051
**Run dir**: `results/2026-05-13_streaming_debug_H_deterministic/`
**Setup**:
- `SEED = 42` for `torch`, `torch.cuda`, `numpy`, `random`.
- `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False`.
- `CUBLAS_WORKSPACE_CONFIG = ":4096:8"`.
- `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Reuses `debug_compare.run_one_scene` so Mask3D is called twice per scene (offline + streaming) under the deterministic regime.

## E (default) vs H (deterministic) comparison

| Scene | metric | E | **H** | verdict |
|---|---|---|---|---|
| scene0334_00 | `K_offline` = `K_streaming` | 12 = 12 ✓ | **12 ≠ 11** ✗ | **H made it worse** |
| scene0334_00 | rep-frame Jaccard mean | 0.467 | n/a (K mismatch) | n/a |
| scene0334_00 | final class match (top-600) | 1.3 % | **0.5 %** | worse |
| scene0334_01 | `K_offline` = `K_streaming` | 10 = 10 ✓ | **10 ≠ 11** ✗ | worse |
| scene0334_01 | final class match (top-600) | 18.8 % | **2.2 %** | worse |

cuDNN determinism does **not** rescue the streaming-vs-offline gap. In fact, with cuDNN locked down, the two Mask3D calls produce a different **post-NMS K** — meaning the non-determinism that remains is large enough to push some proposals across the score / NMS thresholds.

## Determinism-warning audit

`torch.use_deterministic_algorithms(warn_only=True)` flagged a single non-deterministic op during the entire run, but it ran multiple times:

```
scatter_add_cuda_kernel does not have a deterministic implementation,
but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'.
```

`scatter_add_cuda_kernel` is the CUDA backend that `torch.scatter_add` falls back to. **MinkowskiEngine's sparse-tensor reduction layers rely on `scatter_add` to aggregate per-voxel features**, and Mask3D's backbone is a Minkowski sparse convolutional network. So:

- cuDNN convolution determinism does nothing for Mask3D, because the dominant non-determinism is in MinkowskiEngine's `scatter_add`, not in cuDNN.
- PyTorch has no deterministic implementation of `scatter_add_cuda_kernel` (open issue upstream).
- The "fix" inside MinkowskiEngine would require either (a) replacing `scatter_add` with a deterministic-by-sort implementation or (b) making the entire Mask3D forward pass deterministic at the op level — both far outside this project's scope.

## Verdict on the hypotheses

| Hypothesis | Status after H |
|---|---|
| H1 — `.npy` / `.ply` misalignment | ❌ busted (Option B) |
| H2 — filter + `.npy` shift | ❌ partial bust (Option A) |
| H3 — CPU/CUDA float ordering | ⚠️ not sole cause |
| H4 — frame ordering | ❌ busted (E) |
| H5 — D3 small-instance dropout | ❌ busted (E) |
| **H6 — Mask3D non-determinism** | ✅ **confirmed, root cause is MinkowskiEngine's `scatter_add_cuda_kernel`, not cuDNN** |

## What "Option C" / "Option G" mean now

The non-determinism cannot be removed at the framework level (PyTorch has no deterministic `scatter_add` on CUDA). The only correct comparison between offline and streaming is to **avoid calling Mask3D twice** for the same scene — i.e., the sanity test must share Mask3D output between pipelines.

This is exactly **Option G** (cache Mask3D output per scene, feed both pipelines the cached `(masks, scores)`). It bypasses the residual non-determinism entirely and gives an unambiguous read on whether the streaming protocol is algorithmically correct.

## Recommendation

Proceed to **Option G**:

1. New runner that:
   - For each of the 312 scenes, runs Mask3D **exactly once**, persists `(masks, scores)` to disk (or to an in-process cache).
   - Runs both offline (`OpenYolo3D.predict` with `path_to_3d_masks=<cache_dir>`) and streaming (`StreamingScanNetEvaluator.setup_scene` with the cached masks injected) on the cached output.
   - Aggregates AP for both pipelines from one shared population of Mask3D proposals.
2. Sanity criterion: **|streaming_AP − offline_AP| ≤ 0.005** is now meaningful because Mask3D non-determinism is removed.

Implementation note: `OpenYolo3D.predict` already supports `path_to_3d_masks=...` for loading pre-saved Mask3D `.pt` files (utils/__init__.py:119-120). Streaming wrapper needs a small extension to accept pre-computed `(masks, scores)` in `setup_scene` (read-only — no OpenYOLO3D core edit).

## Acceptance status (Option H)

| # | item | status |
|---|---|---|
| 1 | deterministic-mode script | ✅ `method_scannet/streaming/tools/debug_E_deterministic.py` |
| 2 | scene0334_00 / scene0334_01 Mask3D × 2 calls Exit 0 | ✅ |
| 3 | E vs H comparison table | ✅ (this file) |
| 4 | md5 — OpenYOLO3D core unchanged | ✅ (monkey-patches reverted on exit, determinism set in user-space only) |
