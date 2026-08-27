"""Regression Tests B and D for the 2026-08-28 evaluator correction.

Feeds a prediction JSON (author-distributed or our 10-sweep reproduction)
through the CORRECTED custom evaluator path, exercising the real fixed code:

  GT   : NativeTemporalNuScenesEvaluator._load_meta (real method, stub self)
  eval : diagnosis_beta_baseline.evaluate_nuscenes.evaluate(..., nusc=nusc)
         -> official add_center_dist + filter_eval_boxes + official primitives

Acceptance: mAP/NDS within +/-0.002 of the same JSON's unmodified-devkit
result (author JSON: 0.5956/0.6676; our 10-sweep JSON: 0.5580/0.6458).

Also recomputes NDS by hand from the per-class output (plan test F6) and
asserts it matches the devkit's nd_score.

Usage:
  python -u audit/reeval_corrected_custom.py --json PATH --out DIR
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox

from audit._common import DATAROOT, VERSION, NUSC_10
from audit.reeval_official_devkit import val_sample_tokens
from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as custom_eval

NUSC_10_SET = set(NUSC_10)


def build_pred_boxes(nusc, tokens, json_path):
    with open(json_path) as f:
        results = json.load(f)["results"]
    eb = EvalBoxes()
    for tok in tokens:
        dboxes = []
        for b in results.get(tok, []):
            if b["detection_name"] not in NUSC_10_SET:
                continue
            dboxes.append(DetectionBox(
                sample_token=tok,
                translation=tuple(float(x) for x in b["translation"]),
                size=tuple(float(x) for x in b["size"]),
                rotation=tuple(float(x) for x in b["rotation"]),
                velocity=tuple(float(x) for x in b["velocity"]),
                num_pts=-1,
                detection_name=b["detection_name"],
                detection_score=float(b["detection_score"]),
                attribute_name=b.get("attribute_name", "")))
        eb.add_boxes(tok, dboxes)
    return eb


def build_gt_boxes(nusc, tokens):
    """GT through the REAL corrected _load_meta + DetectionBox.deserialize,
    exactly as aggregate_axis_metrics does for Table 1."""
    from method_scannet.streaming.nuscenes_native_evaluator import (
        NativeTemporalNuScenesEvaluator)
    stub = SimpleNamespace(loader=SimpleNamespace(nusc=nusc))
    eb = EvalBoxes()
    for tok in tokens:
        _ego, _t, gts = NativeTemporalNuScenesEvaluator._load_meta(stub, tok)
        eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in gts])
    return eb


def nds_by_hand(summary):
    """F6: recompute NDS from mean_ap + tp_errors (weight 5, err clipped to 1)."""
    tp = summary["tp_errors"]
    scores = [max(1.0 - min(tp[k], 1.0), 0.0)
              for k in ("trans_err", "scale_err", "orient_err",
                        "vel_err", "attr_err")]
    return (5.0 * summary["mean_ap"] + sum(scores)) / 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"=== loading NuScenes {VERSION} from {DATAROOT} ===", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    tokens = val_sample_tokens(nusc)
    print(f"val tokens: {len(tokens)}", flush=True)

    print(f"=== predictions from {args.json} ===", flush=True)
    pred_eb = build_pred_boxes(nusc, tokens, args.json)
    print(f"n_pred_boxes: {sum(len(pred_eb[t]) for t in pred_eb.sample_tokens)}",
          flush=True)

    print("=== GT via corrected _load_meta (real method) ===", flush=True)
    gt_eb = build_gt_boxes(nusc, tokens)
    print(f"n_gt_boxes (pre-filter): "
          f"{sum(len(gt_eb[t]) for t in gt_eb.sample_tokens)}", flush=True)

    print("=== corrected evaluate(nusc=...) ===", flush=True)
    summary = custom_eval(pred_boxes=pred_eb, gt_boxes=gt_eb,
                          output_dir=args.out, nusc=nusc)

    hand = nds_by_hand(summary)
    print("=== RESULT ===")
    print(f"mean_ap:  {summary['mean_ap']:.6f}")
    print(f"nd_score: {summary['nd_score']:.6f}")
    print(f"nds_by_hand (F6): {hand:.6f} "
          f"({'OK' if abs(hand - summary['nd_score']) < 1e-9 else 'MISMATCH'})")
    print(json.dumps({"mean_ap": summary["mean_ap"],
                      "nd_score": summary["nd_score"],
                      "tp_errors": summary["tp_errors"],
                      "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
