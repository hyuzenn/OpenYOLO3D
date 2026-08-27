"""#8 decision analysis (READ-ONLY investigation; no source code changed).

Question: how much does the pipeline's hardcoded prediction-attribute
heuristic (_detection_box_dict: "vehicle.moving" for 7 vehicle classes,
"" otherwise) cost vs the official mmdet3d CenterPoint rule
(nuscenes_metric.py::_format_lidar_bbox, speed>0.2 gate + DefaultAttribute)?

Method: take our existing 10-sweep CenterPoint JSON (whose attributes were
produced by the official rule), and re-evaluate it through the corrected
custom evaluator with attributes REPLACED by our pipeline heuristic.
Everything else (boxes, scores, velocity, GT) identical, so the delta is
exactly the attribute-policy effect. Only mAAE/NDS can move; mAP cannot
(attributes enter only tp attr_err).

Also dumps per-class attribute distributions: official-rule vs heuristic,
and the fraction of boxes whose attribute changes.
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter
from types import SimpleNamespace

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox

from audit._common import DATAROOT, VERSION, NUSC_10
from audit.reeval_official_devkit import val_sample_tokens
from audit.reeval_corrected_custom import build_gt_boxes, nds_by_hand
from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as custom_eval

NUSC_10_SET = set(NUSC_10)

# EXACT copy of the pipeline heuristic in
# method_scannet/streaming/nuscenes_evaluator.py::_detection_box_dict L414-417.
HEURISTIC_VEHICLES = ("car", "truck", "bus", "trailer", "construction_vehicle",
                      "motorcycle", "bicycle")


def pipeline_attr(name: str) -> str:
    return "vehicle.moving" if name in HEURISTIC_VEHICLES else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"=== loading NuScenes {VERSION} from {DATAROOT} ===", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    tokens = val_sample_tokens(nusc)

    with open(args.json) as f:
        results = json.load(f)["results"]

    dist_official = {}   # cls -> Counter(attr)
    dist_heuristic = {}
    n_changed = Counter()
    n_total = Counter()

    eb = EvalBoxes()
    for tok in tokens:
        dboxes = []
        for b in results.get(tok, []):
            name = b["detection_name"]
            if name not in NUSC_10_SET:
                continue
            off_attr = b.get("attribute_name", "")
            heu_attr = pipeline_attr(name)
            dist_official.setdefault(name, Counter())[off_attr] += 1
            dist_heuristic.setdefault(name, Counter())[heu_attr] += 1
            n_total[name] += 1
            if off_attr != heu_attr:
                n_changed[name] += 1
            dboxes.append(DetectionBox(
                sample_token=tok,
                translation=tuple(float(x) for x in b["translation"]),
                size=tuple(float(x) for x in b["size"]),
                rotation=tuple(float(x) for x in b["rotation"]),
                velocity=tuple(float(x) for x in b["velocity"]),
                num_pts=-1,
                detection_name=name,
                detection_score=float(b["detection_score"]),
                attribute_name=heu_attr))
        eb.add_boxes(tok, dboxes)

    print("=== attribute distributions (official rule vs pipeline heuristic) ===")
    for cls in NUSC_10:
        if cls not in n_total:
            continue
        tot = n_total[cls]
        print(f"{cls}: n={tot} changed={n_changed[cls]} "
              f"({100.0 * n_changed[cls] / tot:.1f}%)")
        print(f"  official : {dict(dist_official[cls].most_common())}")
        print(f"  heuristic: {dict(dist_heuristic[cls].most_common())}")

    print("=== GT via corrected _load_meta ===", flush=True)
    gt_eb = build_gt_boxes(nusc, tokens)

    print("=== corrected evaluate(nusc=...) with HEURISTIC attributes ===",
          flush=True)
    summary = custom_eval(pred_boxes=eb, gt_boxes=gt_eb,
                          output_dir=args.out, nusc=nusc)
    hand = nds_by_hand(summary)
    print("=== RESULT (heuristic-attribute arm) ===")
    print(f"mean_ap:  {summary['mean_ap']:.6f}")
    print(f"nd_score: {summary['nd_score']:.6f}")
    print(f"nds_by_hand: {hand:.6f} "
          f"({'OK' if abs(hand - summary['nd_score']) < 1e-9 else 'MISMATCH'})")
    print(json.dumps({"mean_ap": summary["mean_ap"],
                      "nd_score": summary["nd_score"],
                      "tp_errors": summary["tp_errors"]}, indent=2))
    print("Reference (official attributes, Test D): "
          "mAP 0.558050 NDS 0.645809 attr_err 0.184449")


if __name__ == "__main__":
    main()
