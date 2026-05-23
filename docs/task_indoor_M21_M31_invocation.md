# Indoor (ScanNet200) M21 / M31 module-invocation — diagnosis, M21 refactor, M31 IoU sweep

Scope: 5 ScanNet200 val scenes (`scene0011_00, scene0011_01, scene0015_00,
scene0019_00, scene0019_01`), streaming evaluator (`frequency=10`, cached
Mask3D). Measurements only; the 5-scene AP below is a subset figure and is not
the 312-scene number.

## 1. Changes made

**M21 (refactor only — numerics preserved).** The per-frame voting weight was
reimplemented inline in two streaming-path sites; both now route through the
`WeightedVoting.frame_weight` class method:
- `method_scannet/streaming/running_labeler.py::_compute_m21_weight`
- `method_scannet/streaming/method_adapters.py::compute_predictions_method21`

Numeric-equivalence check (CPU, 20000 random inputs): max relative error vs the
old inline formulas = **8.0e-16** (running_labeler) and **8.8e-16**
(method_adapters). The float64 vote accumulation is untouched.
`vote_distribution` / `vote_label` are not routed: their float32 aggregation
differs from the pipeline's float64 accumulation, so routing them would change
AP; they remain unused wrappers.

**M31 (unchanged) + IoU sweep added.** The once-per-scene finalize merge is
kept (3D masks are scene-constant: computed once in `setup_scene`, so a
per-frame merge would recompute identical results). Added
`diagnosis/m31_iou_threshold_sweep.py` to measure merge count and AP across IoU
thresholds.

## 2. M21 invocation counters — before vs after refactor

| counter | before | after |
|---|---:|---:|
| `WeightedVoting.__init__` | 5 | 5 |
| `WeightedVoting.frame_weight` | **0** | **11270** |
| `WeightedVoting.vote_distribution` | 0 | 0 |
| `WeightedVoting.vote_label` | 0 | 0 |
| `RunningInstanceLabeler._compute_m21_weight` (inline site) | 7116 | 7116 |
| `method_adapters.compute_predictions_method21` (inline site) | 5 | 5 |
| value effect — relabels (weighted vs uniform vote) | 13 / 133 (0.097744) | 13 / 133 (0.097744) |

`frame_weight` 0 → 11270 (7116 per-frame via `_compute_m21_weight` + the
remainder from the 5 finalize calls of `compute_predictions_method21`); relabel
count identical. phase1 reproduces the same M21 numbers.

## 3. M31 invocation counters

| counter | value (M31 axis) |
|---|---:|
| `IoUMerger.__init__` | 5 |
| `IoUMerger.merge` | 5 (1 / scene, at finalize) |
| `method_adapters.apply_method31_merge` | 5 |
| value effect — merges (Σ K_in − K_out) | 92 (in 3000 → out 2908) |

phase1: `IoUMerger.merge` = 5, merges = 92 (in 2772 → out 2680; smaller input
after the M11 registration gate).

## 4. M31 IoU-threshold sweep (5 scenes, finalize merge)

| label | merges | proposals in → out | AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| no_merge | 0 | 3000 → 3000 | 0.26047 | 0.36423 | 0.42017 |
| iou_0.5 | 92 | 3000 → 2908 | 0.26029 | 0.36600 | 0.42268 |
| iou_0.4 | 106 | 3000 → 2894 | 0.24742 | 0.35313 | 0.42322 |
| iou_0.3 | 144 | 3000 → 2856 | 0.22702 | 0.32860 | 0.42454 |
| iou_0.25 | 164 | 3000 → 2836 | 0.22345 | 0.32494 | 0.42524 |

As the IoU threshold decreases, merge count rises (92 → 164) and AP / AP50
decrease (0.26047 → 0.22345 / 0.36423 → 0.32494) while AP25 rises slightly
(0.42017 → 0.42524). iou_0.5 AP equals no_merge within 2e-4 (AP50 +0.0018).

## 5. Files

Changed: `method_scannet/streaming/running_labeler.py`,
`method_scannet/streaming/method_adapters.py`.
Added: `diagnosis/verify_indoor_module_invocation.py`,
`diagnosis/m31_iou_threshold_sweep.py`,
`scripts/run_indoor_module_invocation.pbs`,
`scripts/run_indoor_m21_refactor_m31_sweep.pbs`.
Result JSONs: `results/indoor_module_invocation/{M21,M31,phase1}_invocation.json`,
`results/indoor_module_invocation/m31_iou_sweep.json`.
