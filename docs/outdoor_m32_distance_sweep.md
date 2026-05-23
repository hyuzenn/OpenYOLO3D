# Outdoor M32 distance-threshold sweep — nuScenes v1.0-trainval val (150 scenes)

Native γ CenterPoint baseline + M32 Hungarian merge (spatial-only; YOLO bypassed, M22 deferred). Single-sweep, score_threshold=0.0, cache-replay of the Step-2b CenterPoint proposals (no GPU, no re-inference). Reference: γ-fixed baseline mAP **0.3407**. Values pulled directly from `axis_M32/metrics.json` (generated, not transcribed).


## Sweep

| distance_threshold (m) | mAP | NDS | M32 merges | emitted/proposals | Δ mAP vs baseline |
|---|---|---|---|---|---|
| 2.0 | 0.3198 | 0.3028 | 133,198 | 896,182/1,029,380 | -0.0209 |
| 1.5 | 0.3269 | 0.3067 | 115,633 | 913,747/1,029,380 | -0.0138 |
| 1.0 | 0.3332 | 0.3105 | 62,887 | 966,493/1,029,380 | -0.0075 |
| 0.5 | 0.3407 | 0.3146 | 20,041 | 1,009,339/1,029,380 | -0.0000 |

## Distance distribution (same-class candidate pairs, measured from the 6019-sample cache)

- All same-class pairwise centroid distances: min **0.42 m**, p10 7.8, p25 19.1, median **35.3 m** (objects are scene-spread).
- Within the 2.0 m gate: min 0.42, p25 0.78, median 1.18, p90 1.83 m.
- Pairs ≤1.0 m: 115,155 (0.84%) · ≤0.5 m: 23,272 (0.17%) · **≤0.3 m: 0**.
- Large classes (car/truck/bus, ~4.5 m long) cannot have centroids <2 m apart unless they are duplicate detections → the 2.0 m gate over-merges distinct objects.

## Conclusion

- **Monotonic recovery**: mAP 0.3198 (2.0 m) → 0.3269 (1.5) → 0.3332 (1.0) → **0.3407 (0.5)**.
- **0.5 m fully recovers the −0.0209 over-merge loss to the baseline 0.3407** (NDS 0.3146 ≈ baseline 0.3145), while still consolidating 20,041 duplicate merges at zero AP cost.
- `dist_2.0` reproduces the Step-2b M32 result (0.3198) exactly → cache-replay harness verified.
- M32 cannot exceed the γ-fixed baseline outdoors; best case is **parity at 0.5 m** (over-merge eliminated; remaining role = harmless duplicate consolidation).
- **Production default set**: `nuscenes_native_evaluator --m32-distance` default 2.0 → **0.5**.

## Source files

- `results/outdoor_m32_sweep/dist_{2.0,1.5,1.0,0.5}/axis_M32/metrics.json`
- run log: `results/outdoor_m32_sweep/run_*.log`
