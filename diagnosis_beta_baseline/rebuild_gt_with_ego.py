"""Rebuild gt_boxes.json with the missing ego_translation field.

Stage 2 of the β baseline GT-loading bug fix. Reads the existing
`gt_boxes.json` (43 samples, 1735 boxes) and adds an `ego_translation`
field to each box, sourced from `data/nuscenes/{v1.0-mini, v1.0-trainval}/
{sample_data, ego_pose}.json` (raw JSON lookup — avoids constructing a
NuScenes object, which on this NFS-bound host takes minutes per split).

For each sample_token in the GT file:
  sample_data[sample_token, channel='LIDAR_TOP'].ego_pose_token
    → ego_pose[ego_pose_token].translation (3-vec, global frame)

That translation is exactly what `predictions_to_detection_boxes` writes
into pred dicts as `ego_translation`, so after this rebuild the GT and
pred conventions match and `_filter_by_range` ego-distance is correct.

Output: writes a new gt_boxes.json next to the source file (default
`results/diagnosis_beta_baseline_v2/nuscenes_eval/gt_boxes.json`).
Also writes `ego_translation_lookup.json` for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict


DATAROOT = "data/nuscenes"
SPLITS = ("v1.0-mini", "v1.0-trainval")


def _build_token_to_egotrans(dataroot: str, splits=SPLITS) -> Dict[str, list]:
    """sample_token (LIDAR_TOP) → ego_pose translation (global, 3-vec)."""
    out: Dict[str, list] = {}
    for split in splits:
        sd_path = osp.join(dataroot, split, "sample_data.json")
        ep_path = osp.join(dataroot, split, "ego_pose.json")
        if not osp.exists(sd_path) or not osp.exists(ep_path):
            print(f"[skip] {split}: missing sample_data.json / ego_pose.json")
            continue

        with open(sd_path) as f:
            sample_data = json.load(f)
        with open(ep_path) as f:
            ego_pose = json.load(f)

        ep_by_token = {ep["token"]: ep for ep in ego_pose}

        n_added = 0
        for sd in sample_data:
            # `is_key_frame=True` and channel `LIDAR_TOP` is the key-frame lidar
            # for a sample. We want the ego_pose at that sample's lidar timestamp.
            if not sd.get("is_key_frame"):
                continue
            # we don't have channel directly; sample_data has calibrated_sensor_token,
            # but the cheap proxy: filename contains "LIDAR_TOP/" for the lidar sweeps.
            if "LIDAR_TOP/" not in sd.get("filename", ""):
                continue
            sample_token = sd["sample_token"]
            ep_token = sd["ego_pose_token"]
            ep_rec = ep_by_token.get(ep_token)
            if ep_rec is None:
                continue
            out[sample_token] = list(ep_rec["translation"])
            n_added += 1
        print(f"[{split}] indexed {n_added} sample_token → ego_translation entries")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_gt", default="results/diagnosis_beta_baseline/nuscenes_eval/gt_boxes.json")
    ap.add_argument("--dst_gt", default="results/diagnosis_beta_baseline_v2/nuscenes_eval/gt_boxes.json")
    ap.add_argument("--lookup_out", default="results/diagnosis_beta_baseline_v2/ego_translation_lookup.json")
    ap.add_argument("--dataroot", default=DATAROOT)
    args = ap.parse_args()

    print(f"loading source GT: {args.src_gt}")
    with open(args.src_gt) as f:
        gt_data = json.load(f)
    n_gt = sum(len(v) for v in gt_data.values())
    print(f"  samples: {len(gt_data)}, boxes: {n_gt}")

    print("indexing sample_token → ego_translation from raw JSON ...")
    lookup = _build_token_to_egotrans(args.dataroot)
    print(f"  total mapped tokens: {len(lookup)}")

    missing = [t for t in gt_data if t not in lookup]
    if missing:
        print(f"WARNING: {len(missing)} GT sample_tokens missing from lookup; first: {missing[:3]}")

    print("rewriting GT with ego_translation ...")
    new_gt: Dict[str, list] = {}
    n_total = 0
    n_with_ego = 0
    for token, dicts in gt_data.items():
        ego = lookup.get(token)
        new_list = []
        for d in dicts:
            new_d = dict(d)
            if ego is not None:
                new_d["ego_translation"] = [float(x) for x in ego]
                n_with_ego += 1
            new_list.append(new_d)
            n_total += 1
        new_gt[token] = new_list
    print(f"  GT boxes: {n_with_ego}/{n_total} now have ego_translation")

    os.makedirs(osp.dirname(args.dst_gt), exist_ok=True)
    with open(args.dst_gt, "w") as f:
        json.dump(new_gt, f)
    print(f"wrote {args.dst_gt}")

    os.makedirs(osp.dirname(args.lookup_out), exist_ok=True)
    samples_in_gt_lookup = {t: lookup[t] for t in gt_data if t in lookup}
    with open(args.lookup_out, "w") as f:
        json.dump(samples_in_gt_lookup, f, indent=2)
    print(f"wrote {args.lookup_out}")

    if n_with_ego < n_total:
        print(f"FAIL: {n_total - n_with_ego} GT boxes still missing ego_translation")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
