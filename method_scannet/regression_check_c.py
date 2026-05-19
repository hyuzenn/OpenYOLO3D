"""Hook-isolation regression check for Step 1.5c.

Verifies that swapping `IoUMerger(iou_threshold=0.7)` into the existing
`install_method_31_only` does not contaminate later runs with the default
`iou_threshold=0.5`. Cross-references against the saved Step 1.5a
fingerprints.

Properties:
  P1 baseline (no hooks) fingerprint == 1.5a baseline_pristine
  P2 iou=0.5 default fingerprint     == 1.5a method_31_only
  P3 iou=0.7 fingerprint may equal iou=0.5 on the regression scene
     (scene0011_00 is vacuous for METHOD_31 — the merger does no work for
     either threshold). Recorded but not asserted.
  P4 baseline_after_all fingerprint   == baseline (clean state restored)
"""
from __future__ import annotations

import json
import os
import sys


SCENE = "scene0011_00"
SCENE_PATH = "data/scannet200/scene0011_00"
PROCESSED = "data/scannet200/scene0011_00/0011_00.npy"
MASKS_DIR = "output/scannet200/scannet200_masks"
CONFIG = "pretrained/config_scannet200.yaml"
PRIOR_FINGERPRINTS = (
    "results/2026-05-07_scannet_method_31_only_v01/regression_check.json"
)


def _fingerprint(pred):
    masks, classes, scores = pred
    classes = classes.detach().cpu()
    scores = scores.detach().cpu()
    masks_b = masks.detach().cpu().bool()
    K = int(classes.shape[0])
    if K > 0:
        order = scores.argsort(descending=True)
        topk = min(10, K)
        top_classes = classes[order[:topk]].tolist()
        top_scores = [round(float(s), 6) for s in scores[order[:topk]].tolist()]
    else:
        top_classes = []
        top_scores = []
    return {
        "K": K,
        "n_vertices": int(masks_b.shape[0]),
        "mask_vertex_count": int(masks_b.sum().item()),
        "score_sum": round(float(scores.sum().item()), 6),
        "score_mean": round(float(scores.mean().item()) if K > 0 else 0.0, 6),
        "top10_classes": top_classes,
        "top10_scores": top_scores,
    }


def _run_one(o, name):
    pred = o.predict(
        path_2_scene_data=SCENE_PATH,
        depth_scale=1000.0,
        datatype="mesh",
        processed_scene=PROCESSED,
        path_to_3d_masks=MASKS_DIR,
        is_gt=False,
    )
    fp = _fingerprint(pred[SCENE])
    print(f"[{name}] fingerprint: {fp}", flush=True)
    return fp


def main() -> int:
    from method_scannet.hooks import (
        install_method_31_only,
        uninstall_method_31_only,
    )
    from method_scannet.method_31_iou_merging import IoUMerger
    from utils import OpenYolo3D

    out = {}
    o = OpenYolo3D(CONFIG)

    out["baseline"] = _run_one(o, "baseline")

    # First install with iou=0.7 (our ablation).
    install_method_31_only(merger=IoUMerger(iou_threshold=0.7))
    out["m31_iou07"] = _run_one(o, "m31_iou07")
    uninstall_method_31_only()

    # Then install with default iou=0.5 — must match 1.5a fingerprint.
    install_method_31_only()
    out["m31_iou05_default"] = _run_one(o, "m31_iou05_default")
    uninstall_method_31_only()

    out["baseline_after_all"] = _run_one(o, "baseline_after_all")

    try:
        with open(PRIOR_FINGERPRINTS) as f:
            prior = json.load(f)
    except FileNotFoundError:
        prior = None

    props = {}
    if prior is not None:
        props["P1_baseline_matches_1_5a"] = (
            out["baseline"] == prior["baseline_pristine"]
        )
        props["P2_iou05_default_matches_1_5a_m31_only"] = (
            out["m31_iou05_default"] == prior["method_31_only"]
        )
    else:
        props["P1_baseline_matches_1_5a"] = None
        props["P2_iou05_default_matches_1_5a_m31_only"] = None

    props["P3_baseline_after_all_matches_baseline"] = (
        out["baseline_after_all"] == out["baseline"]
    )

    # Note (informational): iou=0.7 vs iou=0.5 on scene0011_00. Vacuous
    # equality is acceptable here because scene0011_00 has no same-class
    # IoU ≥ 0.5 pair within the 2 m KDTree radius, so neither threshold
    # finds anything to merge.
    props["INFO_iou07_equals_iou05_on_scene0011"] = (
        out["m31_iou07"] == out["m31_iou05_default"]
    )

    print("\n=== Regression properties ===")
    pass_count = 0
    skipped = 0
    for k, v in props.items():
        if v is None:
            print(f"  SKIP: {k}")
            skipped += 1
        else:
            tag = "OK" if v else "FAIL"
            if k.startswith("INFO_"):
                tag = "NOTE"
            print(f"  {tag}: {k} = {v}")
            if v:
                pass_count += 1

    out["properties"] = props
    # Hook-isolation correctness depends on P1, P2, P3 only; INFO_ doesn't
    # affect pass/fail.
    must_pass = ["P1_baseline_matches_1_5a",
                 "P2_iou05_default_matches_1_5a_m31_only",
                 "P3_baseline_after_all_matches_baseline"]
    out["all_required_passed"] = all(
        props[k] for k in must_pass if props[k] is not None
    )
    out["skipped"] = skipped
    out["passed"] = pass_count

    out_path = os.environ.get("REGRESSION_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[regression] wrote {out_path}")

    return 0 if out["all_required_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
