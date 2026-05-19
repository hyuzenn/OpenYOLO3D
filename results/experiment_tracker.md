# Experiment Tracker — ScanNet200 val

Average metrics across 312 scenes. Higher is better.

| Date       | Run dir                                          | AP     | AP50   | AP25   | Notes                                       |
|------------|--------------------------------------------------|--------|--------|--------|---------------------------------------------|
| 2026-05-05 | `2026-05-05_scannet_eval_v02`                    | 0.2470 | 0.3180 | 0.3630 | baseline                                    |
| 2026-05-07 | `2026-05-07_scannet_eval_v01`                    | 0.2473 | 0.3175 | 0.3624 | baseline (re-run)                           |
| 2026-05-07 | `2026-05-07_scannet_method_21_only_v01`          | 0.2465 | 0.3176 | 0.3636 | METHOD_21 single-axis ablation              |
| 2026-05-07 | `2026-05-07_scannet_method_31_only_v01`          | 0.2442 | 0.3152 | 0.3606 | METHOD_31 only                              |
| 2026-05-07 | `2026-05-07_scannet_method_31_iou07_v01`         | 0.2468 | 0.3172 | 0.3626 | METHOD_31 iou=0.7 (near-no-op)              |
| 2026-05-07 | `2026-05-07_scannet_phase1_v02`                  | 0.2443 | 0.3157 | 0.3629 | Phase 1 (METHOD_21 + METHOD_31)             |
| 2026-05-08 | `2026-05-08_scannet_method_22_only_v02`          | 0.2188 | 0.2797 | 0.3189 | METHOD_22 only (raw class-name prompts)     |
| 2026-05-08 | `2026-05-08_scannet_method_32_only_v01`          | 0.2141 | 0.2751 | 0.3157 | METHOD_32 only                              |
| 2026-05-08 | `2026-05-08_scannet_method_22_v2_v01`            | 0.2192 | 0.2821 | 0.3230 | METHOD_22 + "a photo of a {class}" template |
| 2026-05-09 | `2026-05-09_scannet_phase2_v02`                  | 0.1698 | 0.2156 | 0.2505 | Phase 2 (METHOD_22 + METHOD_32 combined)    |

## Status summary
- **Phase 1 (METHOD_21 + METHOD_31)**: neutral / near-no-op — no clear win.
- **Phase 2 axes (METHOD_22, METHOD_32 individually)**: each regresses ~−0.03 AP.
  Prompt template tweak in METHOD_22_v2 does not recover the gap (+0.0004 AP);
  regression lives in the CLIP-image-feature re-classification logic itself, not
  in the text side.
- **Phase 2 combined (METHOD_22 + METHOD_32)**: AP 0.1698 (Δ −0.0775 vs baseline).
  Super-additive negative — worse than either axis alone (M22 −0.0285, M32 −0.0332;
  additive estimate −0.0617). Indicates the two regressions compound rather than
  cancel.

---

## Streaming evaluation (frequency=10, cached Mask3D, RunningInstanceLabeler)

312-scene ScanNet200 val. Streaming AP is computed on per-instance final-frame predictions; not directly comparable to non-streaming offline AP above. `lsc` = label switch count total. `ttc` = time-to-confirm mean (frames).

| Date       | Run dir                                                                          | Axis        | AP      | lsc total | ttc mean | Notes                                                            |
|------------|----------------------------------------------------------------------------------|-------------|---------|-----------|----------|------------------------------------------------------------------|
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | baseline    | 0.19560 | 23,385    | 6.709    | Part 3 reference                                                 |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | M11         | 0.19540 | 17,023    | 6.296    | FrameCountingGate N=3 — lsc −27 %, AP cost −0.0002              |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | M12 (buggy) | 0.19540 | 17,023    | 6.296    | bit-identical to M11 (silent bug, see Task 1.4c)                |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_B_m21_m31_v01`                   | M21         | 0.19589 | 23,629    | 6.622    | WeightedVoting — lsc +1 %, near-null                            |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_B_m21_m31_v01`                   | M31         | 0.19522 | 23,359    | 6.712    | IoUMerger — null                                                |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_C_phase1_m21m31_v01`             | phase1      | 0.19507 | 17,525    | 6.345    | M11+M21+M31; benefit attributable to M11                        |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_C_phase1_m21m31_v01`             | M21+M31     | 0.19527 | 23,593    | 6.623    | phase1 vs this isolates M11 add-on                              |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_D_m22m32_v01`                    | M22+M32     | 0.09979 | 103,420   | 6.633    | Phase 2 cascade — AP halves, lsc 4.4×                           |
| 2026-05-18 | `2026-05-18_streaming_ablation_m12_fixed_v01`                                    | M12 (fixed) | 0.19498 | 16,518    | 6.179    | BayesianGate silent-bug fix; slight strict-dominance over M11   |

### Streaming status summary
- **Phase 1 temporal stabilizer**: only M11 (or fixed M12) moves temporal metrics; M21 / M31 nulls.
- **M11 ≡ M12 was a silent bug** (Task 1.4c): BayesianGate.gate() missed the decay branch. Post-fix M12 reaches lsc −29 % / ttc −8 % at AP cost −0.0006.
- **Phase 2 cascade (M22+M32)**: dual failure (AP −49 %, lsc +342 %), no rescue path.
- **§5.2 framing**: "any monotonic confirmation gate at ≈ 3 visible frames" — both M11 (counting) and fixed M12 (Bayesian) qualify; difference is hyperparameter tuning, not structural.

### M11 N sensitivity sweep — Task 1.5 (paper §7.5)

50-scene subset (`seed=42`, scene0011_00 excluded — Part 3 outlier). NOT directly comparable to 312-scene rows above; subset has slightly higher AP base.
Run dir: `2026-05-19_m11_n_sweep_v02/`.

| N | AP      | lsc total | lsc mean | ttc mean | ttc n_inst | Notes                                                     |
|--:|--------:|----------:|---------:|---------:|-----------:|-----------------------------------------------------------|
| 2 | 0.21533 | 3,201     | 64.02    | 6.347    | 1,627      | bit-identical AP to N=3                                  |
| 3 | 0.21533 | 2,776     | 55.52    | 6.141    | 1,618      | current production default                               |
| 4 | 0.21507 | 2,432     | 48.64    | 6.020    | 1,611      | weak strict-dominance over N=3 (every temporal metric ↓) |
| 5 | 0.21467 | 2,186     | 43.72    | 6.263    | 1,598      | ttc starts climbing                                      |
| 7 | 0.21317 | 1,814     | 36.28    | 9.200    | 1,557      | 41 instances dropped from confirmation; AP −1 %          |

Sweet spot N=4 on this subset (same AP within 1.2e-3 vs N=3, lsc −12 %, ttc −2 %). N=3 retained as production default for direct comparability with Part 3; N=4 reported as no-cost alternative in §7.5. 312-scene re-validation of N=4 strict-dominance is the natural follow-up.
