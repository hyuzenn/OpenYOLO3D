# Indoor temporal-method ablation — ScanNet200 val (312 scenes)

Online streaming harness (`StreamingScanNetEvaluator`, `frequency=10`, cached Mask3D), metrics via `RunningInstanceLabeler` per-frame snapshots. AP = ScanNet200 instance-seg AP; `lsc` = label-switch count total. All values pulled directly from the result JSONs (generated, not transcribed).


## Before / After (6-axis)

| Axis | AP (After) | AP (Before) | ΔAP | lsc (After) | lsc (Before) | Baseline lsc | Reference base / mechanism |
|---|---|---|---|---|---|---|---|
| baseline | 0.19560 | — | — | — | — | 23,385 | floor reference |
| M11 (FrameCounting) | 0.19540 | 0.19540 | -0.00020 | 17,023 | 17,023 | 23,385 | Δ vs baseline; unchanged module — lsc −27.2% single-handed |
| M21 (WeightedVoting) | 0.19589 | 0.19589 | +0.00030 | 23,629 | 23,629 | 23,385 | Δ vs baseline; inline→class-method refactor (numerics identical to 8e-16); near-null |
| M31 (IoUMerger) | 0.19522 | 0.19522 | -0.00037 | 23,359 | 23,359 | 23,385 | Δ vs baseline; once-per-scene justified (IoU sweep: AP falls as thr lowers); null |
| M22 (CLIP EMA) | 0.14205 | 0.14245 | -0.00040 | 19,662 | n/a@312 | 23,385 | Δ vs before-fix; margin gate m=0.006 + per-frame L2-norm. lsc ≤ baseline (no inflation); AP not rescued (weak image↔text head) |
| M32 (Hungarian) | 0.18561 | 0.17287 | +0.01274 | 23,385 | 23,385 | 23,385 | Δ vs before-fix; distance 2.0→0.5 m. AP recovers toward baseline; lsc=baseline (finalize-only) |
| M22+M32 (integrated) | 0.13469 | 0.09979 | +0.03490 | 19,662 | 103,420 | 23,385 | Δ vs before-fix; open-vocab cascade decoupled — AP +0.035, lsc 103,420→19,662 (−81%) |

## Notes

- **Two reference bases**: M11/M21/M31 have no before/after (M11 unchanged; M21 = numerics-preserving refactor; M31 unchanged) → ΔAP is vs **baseline**. M22/M32/M22+M32 ΔAP is vs the recorded **before-fix** control.
- **M22 before-fix lsc @312 not measured** (the broken-M22 axis was excluded to save compute). The 5-scene before/after measured it directly: 1889 → 316 (baseline 360), i.e. the gate removes the inflation and lands below baseline; linear projection of broken-M22 to 312 ≈ 118k.
- **M22+M32 lsc before/after IS measured @312**: 103,420 → 19,662 (the open-vocab cascade lsc fix).
- M22 `lsc` reduction is vs broken-M22, not vs baseline; baseline itself is ~23k (its floor).

## Source files (per cell)

- baseline / M11: `results/2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01/axis_{baseline,M11}/`
- M21 / M31: `…/pbs_B_m21_m31_v01/axis_{M21,M31}/`
- M22+M32 before: `…/pbs_D_m22m32_v01/axis_M22+M32/`
- M22 before: `results/2026-05-14_streaming_ablation_pbs3_label_merge_c_v01/axis_M22/summary.json`
- M32 before: `results/2026-05-14_streaming_ablation_pbs4_merge_axis_d_v02/axis_M32/summary.json`
- M22/M32/M22+M32 after: `results/m22_m32_fix/{m22_after_312,m32_after_312,m22m32_after_312}.json`
