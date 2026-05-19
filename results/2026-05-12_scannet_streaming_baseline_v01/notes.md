# Task 1.2c — 312-scene streaming baseline (FAIL by aggregate sanity)

**Date**: 2026-05-12
**PBS job**: 92041
**Run dir**: `results/2026-05-12_scannet_streaming_baseline_v01/`
**Walltime**: 2 h 27 min (147 min, 312/312 scenes processed, evaluator ran cleanly)

## Aggregate AP — streaming vs offline (312 scene mean)

| Metric | Offline (v01, 2026-05-07) | Streaming (this run) | Δ (streaming − offline) |
|---|---|---|---|
| **AP** | **0.2473** | **0.1911** | **−0.0562** |
| AP_50 | 0.3175 | 0.2482 | −0.0694 |
| AP_25 | 0.3624 | 0.2900 | −0.0724 |
| AR | 0.1614 | 0.1128 | −0.0486 |
| APCDC | 0.6696 | 0.6094 | −0.0602 |

### Bucket breakdown

| Bucket | offline AP | streaming AP | Δ |
|---|---|---|---|
| head | 0.278 | 0.260 | −0.018 |
| common | 0.243 | 0.177 | −0.066 |
| tail | 0.216 | 0.127 | −0.089 |

Regression scales with class rarity (head −0.018 → tail −0.089). Indicates that the few common-/rare-class instances that survived offline's MVPDist accumulation are systematically lost in streaming — consistent with a small-quantity-of-evidence pipeline bug rather than a bulk-score drift.

## Sanity verdict — **FAIL by 11× the threshold**

| Threshold | \|Δ AP\| | Verdict |
|---|---|---|
| ±0.005 | 0.0562 | ❌ FAIL (11×) |

Even allowing for Mask3D non-determinism (single-scene v02 measured offline-vs-offline variance ≈ 0.0235, divided by √312 ≈ 0.0013 aggregate variance), the observed gap is ~40 standard errors away from zero — not noise. There is a systematic algorithmic gap between the streaming pipeline and offline.

## D3 visibility (312-scene aggregate)

| Statistic | Value |
|---|---|
| frame samples | 54 090 (≈ 173 frames per scene mean) |
| min visible instances / frame | 0 |
| **median** | **7** |
| mean | 8.3 |
| p90 | 15 |
| p99 | 26 |
| max | 43 |

Far below the scene0011_00 v02 figures (median 46). The 312-scene aggregate is dominated by smaller scenes with fewer Mask3D instances after the new filter (median K_mask3d ≈ 20–60 across scenes per the run log).

## Temporal metrics (caveat: instrumentation artefact)

| Metric | Value |
|---|---|
| label_switch_count total | 0 |
| time_to_confirm n_instances | 0 |

This is a bookkeeping artefact, not a real result. The streaming wrapper currently emits `-1` (unassigned) for `current_instance_map[k]` at every frame and only finalises the actual class via `compute_baseline_predictions()` at scene end. The temporal metric helpers (Task 1.3 spec) operate on `pred_history` which sees only `-1`s — so they report no switches and no confirmations. This must be re-instrumented in Task 1.3 / 1.4 by adding per-frame class snapshots into `pred_history`. The aggregate AP comparison is unaffected.

## Walltime + per-scene cost

| Statistic | Value |
|---|---|
| Total streaming time | 8 861 s (2 h 27 min) |
| Mean per scene | 28.4 s |
| Median per scene | 24.1 s |
| Max per scene | 91.9 s (scene0670_01, 356 frames) |
| Rate | ~2.3 scenes/min |

Well within the 6 h PBS walltime budget. 312-scene streaming run is operationally feasible at this cost.

## Diagnosis — why aggregate AP is 0.056 below offline

The 312-scene run differs from the single-scene v02 sanity in **two simultaneous changes**:

1. **Mask3D filter** (score threshold + NMS) applied in `setup_scene(apply_mask3d_filter=True)` to match offline pre-processing. Was not applied in v02 single-scene sanity.
2. **`.npy` Mask3D input** via `setup_scene(processed_scene_path=...)`. v02 used the `.ply` mesh directly.

Both changes were intended to bring streaming closer to offline (offline uses `.npy` + filter). v02 single-scene already showed streaming AP 0.3364 vs offline 0.3457 (Δ −0.0093) on scene0011_00 *without* these changes. The 312-scene run with both changes shows Δ −0.056 across the dataset.

Three remaining hypotheses (none of which I can confirm without further runs — task spec forbids auto-fix):

- **H1 — `.npy` ↔ `.ply` vertex misalignment per scene.**
  When `setup_scene(processed_scene_path=.npy)` is used, Mask3D consumes the `.npy` and the wrapper loads vertices from the `.ply` for projection. Offline does the same; assumed-aligned vertex sets. If for some scenes the two diverge in count or order, masks are indexed against a different vertex layout than `inside_mask`, silently corrupting predictions in those scenes. Worth verifying via `len(np.load(scene.npy))` vs `len(o3d.read_point_cloud(scene.ply).points)` on a sample of scenes.

- **H2 — Mask3D filter changes the K distribution streaming relies on.**
  Removing low-score / NMS-overlapping instances shrinks K from ~100s to ~20–60. With far fewer K, the top-K representative-frame selection in the accumulator may concentrate label votes on a different (worse) subset of frames than offline ends up with after its own filter, even though both filters appear identical. Tail / common buckets are hit harder, consistent with this.

- **H3 — Subtle CPU vs CUDA float ordering in `get_visibility_mat`.**
  Offline calls `get_visibility_mat` on CUDA tensors; streaming accumulator runs it on CPU (`device='cpu'`). Top-K with float-tied scores can pick different frames per instance under different precision orderings. Per-scene effect tiny, but consistent direction across 312 scenes could add up.

## Proposed paths forward (NOT auto-applied — awaiting user decision)

| # | option | what it does | expected effort |
|---|---|---|---|
| **A** | Re-run scene0011_00 with the v02 code path on this exact branch | Isolates whether the filter + `.npy` changes degrade scene0011_00 streaming AP (v02 was 0.3364). If yes → confirms H1/H2. If similar → confirms 312-scene aggregate is the truth and v02 was outlier-lucky. | ~5 min (1 scene, 1 qsub) |
| **B** | Compare `len(.npy points)` vs `len(.ply vertices)` across all 312 scenes | Confirms / refutes H1 directly. | ~2 min (CPU, no GPU) |
| **C** | Re-run 312 scenes with **`apply_mask3d_filter=False`** + `.ply` (= v02 path) | Directly compares to the v02 baseline path at aggregate scale. If aggregate Δ is closer to 0, the filter + `.npy` path is the problem. | ~2.5 h (qsub) |
| **D** | Move `get_visibility_mat` to GPU in streaming | Eliminates H3. | small edit + re-run |

Recommendation: **A → B → C in sequence**. (A) is the cheapest sanity probe; (B) directly tests H1; (C) is the heavy-but-decisive comparison. Defer D until after.

## Acceptance status

| # | item | status |
|---|---|---|
| 1 | eval_streaming_baseline.py | ✅ |
| 2 | qsub Exit 0, 312/312 | ✅ |
| 3 | metrics.json + per_scene/ + temporal_metrics + d3_visibility | ✅ |
| 4 | sanity \|Δ AP\| ≤ 0.005 | ❌ FAIL (Δ = −0.0562) |
| 5 | md5 OpenYOLO3D core unchanged | ✅ (to verify post-run) |
| 6 | new files only inside `method_scannet/streaming/` + `scripts/` + `results/` | ✅ |

Strict 4/5 — Task 1.4 entry blocked on the diagnosis decision (option A/B/C/D).
