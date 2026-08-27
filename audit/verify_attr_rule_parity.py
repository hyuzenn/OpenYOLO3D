"""#8 fix validation: box-for-box attribute parity + eval regression.

Step 1 (parity): for every box in the official 10-sweep CenterPoint JSON,
recompute the attribute with OUR _official_attribute(name, vx, vy) using the
JSON's own (global-frame) velocity and compare with the JSON's
attribute_name, which mmdet3d's _format_lidar_bbox produced. Expect 0
mismatches; any mismatch is printed with full box context.

Step 2 (regression): evaluate the JSON through the corrected custom evaluator
with OUR generated attributes replacing the stored ones. Expect bit-identical
mAP/mATE/mASE/mAOE/mAVE to Test D and NDS back at 0.645809 (from the
heuristic's 0.577841).
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox

from audit._common import DATAROOT, VERSION, NUSC_10
from audit.reeval_official_devkit import val_sample_tokens
from audit.reeval_corrected_custom import build_gt_boxes, nds_by_hand
from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as custom_eval
from method_scannet.streaming.nuscenes_evaluator import _official_attribute

NUSC_10_SET = set(NUSC_10)


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

    n_total = 0
    n_mismatch = 0
    mismatch_by_cls = Counter()
    eb = EvalBoxes()
    for tok in tokens:
        dboxes = []
        for b in results.get(tok, []):
            name = b["detection_name"]
            if name not in NUSC_10_SET:
                continue
            vx, vy = float(b["velocity"][0]), float(b["velocity"][1])
            ours = _official_attribute(name, vx, vy)
            theirs = b.get("attribute_name", "")
            n_total += 1
            if ours != theirs:
                n_mismatch += 1
                mismatch_by_cls[name] += 1
                if n_mismatch <= 20:
                    print(f"MISMATCH {name} v=({vx:.4f},{vy:.4f}) "
                          f"ours={ours!r} official={theirs!r} tok={tok}")
            dboxes.append(DetectionBox(
                sample_token=tok,
                translation=tuple(float(x) for x in b["translation"]),
                size=tuple(float(x) for x in b["size"]),
                rotation=tuple(float(x) for x in b["rotation"]),
                velocity=(vx, vy),
                num_pts=-1,
                detection_name=name,
                detection_score=float(b["detection_score"]),
                attribute_name=ours))
        eb.add_boxes(tok, dboxes)

    print(f"=== STEP 1 parity: {n_mismatch}/{n_total} mismatches ===")
    if mismatch_by_cls:
        print(f"by class: {dict(mismatch_by_cls)}")
    parity_ok = n_mismatch == 0
    print("PARITY:", "EXACT" if parity_ok else "FAILED")

    print("=== STEP 2: eval with OUR generated attributes ===", flush=True)
    gt_eb = build_gt_boxes(nusc, tokens)
    summary = custom_eval(pred_boxes=eb, gt_boxes=gt_eb,
                          output_dir=args.out, nusc=nusc)
    hand = nds_by_hand(summary)
    print("=== RESULT ===")
    print(f"mean_ap:  {summary['mean_ap']:.6f}")
    print(f"nd_score: {summary['nd_score']:.6f}")
    print(f"nds_by_hand: {hand:.6f} "
          f"({'OK' if abs(hand - summary['nd_score']) < 1e-9 else 'MISMATCH'})")
    print(json.dumps({"mean_ap": summary["mean_ap"],
                      "nd_score": summary["nd_score"],
                      "tp_errors": summary["tp_errors"]}, indent=2))
    print("Reference Test D (official attrs): mAP 0.558050 NDS 0.645809")
    print("Old heuristic attrs:               mAP 0.558050 NDS 0.577841")
    sys.exit(0 if parity_ok else 1)


if __name__ == "__main__":
    main()
