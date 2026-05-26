"""CPU-only validation of the new id_switch gt_matching path (no GPU/model).

1. Synthetic unit checks of full_scene_iou / build_gt_matching / id_switch_count.
2. Real-data integration: scene0011_00 Mask3D cache masks + GT .txt, with a
   synthetic pred_history, exercises GT parsing + vertex alignment + matching.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method_scannet.streaming.gt_matching import (
    full_scene_iou, build_gt_matching, load_gt_instance_masks, gt_matching_for_scene,
)
from method_scannet.streaming.metrics import id_switch_count

ok = True
def check(cond, msg):
    global ok
    print(("PASS" if cond else "FAIL"), "-", msg)
    ok = ok and cond

# ---- 1. synthetic ----------------------------------------------------------
# 6 vertices. Proposal 0 = {0,1,2}; proposal 1 = {2,3,4}; proposal 2 = {0,1,2,3}
ivm = np.array([
    [1,1,1,0,0,0],
    [0,0,1,1,1,0],
    [1,1,1,1,0,0],
], dtype=bool)
gt = np.array([1,1,1,0,0,0], dtype=bool)  # GT == proposal 0 exactly
iou = full_scene_iou(gt, ivm)
check(abs(iou[0]-1.0) < 1e-9, f"full_scene_iou exact match -> 1.0 (got {iou[0]:.3f})")
check(abs(iou[2]-0.75) < 1e-9, f"full_scene_iou partial -> 0.75 (got {iou[2]:.3f})")

# pred_history: live sets change so the GT's best-live proposal re-routes.
# gt_masks: single GT 'g' equal to proposal 0 (iou: p0=1.0, p2=0.75, p1=0.2)
gt_masks = {5000: gt}
# frame0 live {0,2} -> best=0 ; frame1 live {2} -> best=2 (switch 0->2) ;
# frame2 live {} -> None ; frame3 live {0,2} -> best=0 (None->0, not a switch)
pred_history = [
    {0: 1, 2: 1},
    {2: 1},
    {},
    {0: 1, 2: 1},
]
gm = build_gt_matching(pred_history, ivm, gt_masks, iou_threshold=0.5)
check(gm[5000] == [0, 2, None, 0], f"per-frame matching sequence (got {gm[5000]})")
n_sw = id_switch_count(pred_history, gm)
check(n_sw == 1, f"id_switch_count counts only 0->2 re-route (got {n_sw})")

# threshold gating: raise threshold so p2 (0.75) still ok but p1 (0.2) excluded
gm2 = build_gt_matching([{1: 1}], ivm, gt_masks, iou_threshold=0.5)
check(gm2[5000] == [None], f"below-threshold match -> None (got {gm2[5000]})")

# ---- 2. real data integration ---------------------------------------------
scene = "scene0011_00"
gt_txt = ROOT / "data" / "scannet200" / "ground_truth" / f"{scene}.txt"
cache = ROOT / "results" / "2026-05-13_mask3d_cache" / f"{scene}.pt"
if gt_txt.exists() and cache.exists():
    import torch
    masks_raw, _ = torch.load(str(cache), map_location="cpu")
    masks_np = masks_raw.cpu().numpy()
    # normalise to (K, V) like the wrapper does (K << V)
    ivm_real = masks_np.T if masks_np.shape[0] >= masks_np.shape[1] else masks_np
    ivm_real = ivm_real.astype(bool)
    V = ivm_real.shape[1]
    gt_masks_real = load_gt_instance_masks(str(gt_txt), n_vertices=V)
    check(len(gt_masks_real) > 0, f"real GT instances parsed (n_gt={len(gt_masks_real)})")
    # synthetic 3-frame history alternating which proposals are 'live'
    K = ivm_real.shape[0]
    ph = [
        {i: 1 for i in range(0, K, 2)},     # even proposals
        {i: 1 for i in range(1, K, 2)},     # odd proposals
        {i: 1 for i in range(K)},           # all
    ]
    gm_real, n_gt = gt_matching_for_scene(ph, ivm_real, str(gt_txt), iou_threshold=0.25)
    nsw = id_switch_count(ph, gm_real)
    check(n_gt == len(gt_masks_real), f"gt_matching_for_scene n_gt consistent ({n_gt})")
    check(all(len(v) == 3 for v in gm_real.values()), "per-GT sequence length == n_frames")
    check(isinstance(nsw, int) and nsw >= 0, f"id_switch on real masks runs (switches={nsw})")
    print(f"  [real] scene={scene} K={K} V={V} n_gt={n_gt} synthetic_switches={nsw}")
else:
    print("SKIP - real data not found (cache/gt missing)")

print("\n=== RESULT:", "ALL PASS" if ok else "FAILURES PRESENT", "===")
sys.exit(0 if ok else 1)
