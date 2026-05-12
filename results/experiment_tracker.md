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
