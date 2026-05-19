"""Hook-isolation regression check.

Runs a single ScanNet200 scene through OpenYolo3D under three hook
configurations and verifies:

  1) baseline (no hooks) is reproduced after install→uninstall round-trips
     of phase1 and method_31_only.
  2) phase1 predictions ≠ baseline predictions (METHOD_21 + METHOD_31 do
     something).
  3) method_31_only predictions ≠ baseline predictions (METHOD_31 does
     something).
  4) phase1 predictions ≠ method_31_only predictions (METHOD_21 does
     something on top of METHOD_31).

These four properties together demonstrate that the install/uninstall
mechanism cleanly toggles each method without contamination across runs.

Writes a JSON report to $REGRESSION_OUT (or stdout if unset) with the per-
config (class_count, score_sum, top10_classes) fingerprints plus the four
boolean properties.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch


SCENE = "scene0011_00"
SCENE_PATH = "data/scannet200/scene0011_00"
PROCESSED = "data/scannet200/scene0011_00/0011_00.npy"
MASKS_DIR = "output/scannet200/scannet200_masks"
CONFIG = "pretrained/config_scannet200.yaml"


def _fingerprint(pred):
    """Return a small dict that uniquely identifies a prediction tuple
    (predicted_masks, pred_classes, pred_scores) up to the precision needed
    for hook-isolation regression."""
    masks, classes, scores = pred
    classes = classes.detach().cpu()
    scores = scores.detach().cpu()
    masks_b = masks.detach().cpu().bool()
    K = int(classes.shape[0])
    # Top-K-by-score class ordering is the most discriminative scalar
    # signature; mask vertex sums add a structural check.
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


def _equal(a, b) -> bool:
    return a == b


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


def main():
    from method_scannet.hooks import (
        install_phase1,
        uninstall_phase1,
        install_method_31_only,
        uninstall_method_31_only,
    )
    from utils import OpenYolo3D

    out = {}

    # 1) Baseline (pristine state).
    o = OpenYolo3D(CONFIG)
    out["baseline_pristine"] = _run_one(o, "baseline_pristine")

    # 2) Phase 1 (both hooks).
    install_phase1()
    out["phase1"] = _run_one(o, "phase1")
    uninstall_phase1()

    # 3) Baseline after uninstall_phase1 round-trip.
    out["baseline_after_phase1_uninstall"] = _run_one(o, "baseline_after_phase1_uninstall")

    # 4) METHOD_31 only.
    install_method_31_only()
    out["method_31_only"] = _run_one(o, "method_31_only")
    uninstall_method_31_only()

    # 5) Baseline after uninstall_method_31_only round-trip.
    out["baseline_after_m31only_uninstall"] = _run_one(o, "baseline_after_m31only_uninstall")

    # ---- Property checks ----
    props = {}
    b = out["baseline_pristine"]
    b_p1 = out["baseline_after_phase1_uninstall"]
    b_m31 = out["baseline_after_m31only_uninstall"]
    p1 = out["phase1"]
    m31 = out["method_31_only"]

    props["P1_uninstall_phase1_restores_baseline"] = _equal(b, b_p1)
    props["P2_uninstall_method_31_only_restores_baseline"] = _equal(b, b_m31)
    props["P3_phase1_differs_from_baseline"] = not _equal(p1, b)
    props["P4_method_31_only_differs_from_baseline"] = not _equal(m31, b)
    props["P5_phase1_differs_from_method_31_only"] = not _equal(p1, m31)

    print("\n=== Regression properties ===")
    for k, v in props.items():
        status = "OK" if v else "FAIL"
        print(f"  {status}: {k} = {v}")

    out["properties"] = props
    out["all_properties_passed"] = all(props.values())

    out_path = os.environ.get("REGRESSION_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[regression] wrote {out_path}")

    return 0 if out["all_properties_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
