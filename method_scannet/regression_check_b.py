"""Hook-isolation regression check for Step 1.5b.

Adds METHOD_21-only to the configurations covered by 1.5a's regression_check
and cross-references against the saved fingerprints from
results/2026-05-07_scannet_method_31_only_v01/regression_check.json.

Properties:
  P1 baseline (no hooks) fingerprint == 1.5a baseline_pristine
  P2 phase1 fingerprint               == 1.5a phase1
  P3 method_31_only fingerprint       == 1.5a method_31_only
  P4 method_21_only fingerprint       != baseline (METHOD_21 changes preds)
  P5 method_21_only fingerprint       != phase1   (METHOD_31 also changes them)
  P6 method_21_only fingerprint       != method_31_only

These together prove (a) determinism — running again under the same hook
configuration produces identical predictions; (b) hook isolation — toggling
which methods are installed changes the predictions accordingly.

Writes a JSON report to $REGRESSION_OUT (or stdout if unset).
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
        install_phase1,
        uninstall_phase1,
        install_method_31_only,
        uninstall_method_31_only,
        install_method_21_only,
        uninstall_method_21_only,
    )
    from utils import OpenYolo3D

    out = {}
    o = OpenYolo3D(CONFIG)

    # Hook configurations.
    out["baseline"] = _run_one(o, "baseline")

    install_phase1()
    out["phase1"] = _run_one(o, "phase1")
    uninstall_phase1()

    install_method_31_only()
    out["method_31_only"] = _run_one(o, "method_31_only")
    uninstall_method_31_only()

    install_method_21_only()
    out["method_21_only"] = _run_one(o, "method_21_only")
    uninstall_method_21_only()

    # Verify clean state after the last uninstall.
    out["baseline_after_all"] = _run_one(o, "baseline_after_all")

    # Cross-reference vs Step 1.5a fingerprints.
    prior_path = PRIOR_FINGERPRINTS
    try:
        with open(prior_path) as f:
            prior = json.load(f)
    except FileNotFoundError:
        prior = None
        print(f"[warn] prior fingerprint file not found: {prior_path}", flush=True)

    props = {}
    if prior is not None:
        props["P1_baseline_matches_1_5a"] = out["baseline"] == prior["baseline_pristine"]
        props["P2_phase1_matches_1_5a"] = out["phase1"] == prior["phase1"]
        props["P3_method_31_only_matches_1_5a"] = (
            out["method_31_only"] == prior["method_31_only"]
        )
    else:
        props["P1_baseline_matches_1_5a"] = None
        props["P2_phase1_matches_1_5a"] = None
        props["P3_method_31_only_matches_1_5a"] = None

    props["P4_method_21_only_differs_from_baseline"] = (
        out["method_21_only"] != out["baseline"]
    )
    props["P5_method_21_only_differs_from_phase1"] = (
        out["method_21_only"] != out["phase1"]
    )
    props["P6_method_21_only_differs_from_method_31_only"] = (
        out["method_21_only"] != out["method_31_only"]
    )

    # Round-trip property: returning to clean hooks reproduces baseline.
    props["P7_clean_state_after_all_uninstalls_matches_baseline"] = (
        out["baseline_after_all"] == out["baseline"]
    )

    print("\n=== Regression properties ===")
    pass_count = 0
    skipped = 0
    for k, v in props.items():
        if v is None:
            print(f"  SKIP: {k} (no prior fingerprints to compare)")
            skipped += 1
        else:
            status = "OK" if v else "FAIL"
            print(f"  {status}: {k} = {v}")
            if v:
                pass_count += 1

    out["properties"] = props
    out["all_properties_passed"] = all(v for v in props.values() if v is not None)
    out["skipped"] = skipped
    out["passed"] = pass_count

    out_path = os.environ.get("REGRESSION_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[regression] wrote {out_path}")

    return 0 if out["all_properties_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
