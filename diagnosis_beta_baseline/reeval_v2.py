"""Re-run nuScenes-devkit eval on existing pred_boxes.json + fixed gt_boxes.json.

Inputs:
  - results/diagnosis_beta_baseline/nuscenes_eval/pred_boxes.json   (unchanged from v1)
  - results/diagnosis_beta_baseline_v2/nuscenes_eval/gt_boxes.json  (rebuilt with ego_translation)

Calls `evaluate_nuscenes.evaluate(...)` so it goes through the same
filter / accumulate / calc_ap / calc_tp pipeline as v1, only with the
GT bug fixed.

Output:
  - results/diagnosis_beta_baseline_v2/nuscenes_eval/eval_summary.json
  - results/diagnosis_beta_baseline_v2/nuscenes_eval/per_class.json
  - results/diagnosis_beta_baseline_v2/aggregate.json (combined: nuscenes_eval +
    instance_level (copied verbatim from v1, since predictions are unchanged) +
    timing (verbatim) + counts).
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

# Make the diagnosis_beta_baseline package importable when this script
# is run directly without `python -m`.
_this = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.dirname(_this))

from diagnosis_beta_baseline.evaluate_nuscenes import build_eval_boxes, evaluate

V1_DIR = "results/diagnosis_beta_baseline"
V2_DIR = "results/diagnosis_beta_baseline_v2"


def main() -> int:
    pred_path = osp.join(V1_DIR, "nuscenes_eval", "pred_boxes.json")
    gt_path = osp.join(V2_DIR, "nuscenes_eval", "gt_boxes.json")
    out_dir = osp.join(V2_DIR, "nuscenes_eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading pred: {pred_path}")
    with open(pred_path) as f:
        pred_per_sample = json.load(f)
    print(f"loading gt:   {gt_path}")
    with open(gt_path) as f:
        gt_per_sample = json.load(f)

    pred_eb = build_eval_boxes(pred_per_sample)
    gt_eb = build_eval_boxes(gt_per_sample)

    summary = evaluate(pred_eb, gt_eb, out_dir)

    print("\n=== summary ===")
    print(f"mAP: {summary.get('mean_ap', 0.0):.4f}")
    print(f"NDS: {summary.get('nd_score', 0.0):.4f}")
    print(f"counts: {summary.get('counts')}")
    print(f"per-class mean AP: {summary.get('mean_dist_aps')}")

    # Build aggregate.json combining v2 nuscenes_eval with v1's
    # instance_level + timing (predictions / instance GT are unchanged).
    with open(osp.join(V1_DIR, "aggregate.json")) as f:
        v1_agg = json.load(f)
    v2_agg = dict(v1_agg)
    v2_agg["nuscenes_eval"] = summary
    v2_agg["v2_meta"] = {
        "fix": "gt_to_detection_boxes now writes ego_translation; rebuilt gt_boxes.json without re-running inference",
        "src_v1": V1_DIR,
        "predictions_unchanged": True,
        "instance_level_unchanged": True,
    }
    with open(osp.join(V2_DIR, "aggregate.json"), "w") as f:
        json.dump(v2_agg, f, indent=2, default=lambda o: float(o))
    print(f"\nwrote {osp.join(V2_DIR, 'aggregate.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
