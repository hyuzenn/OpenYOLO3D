"""Instance-level diagnostic metrics (M / L / D / miss) for OpenYOLO3D
mask predictions, mirroring the W1 / β1 definition for clusters.

  M    — 1 GT ↔ 1 instance      (clean per-object proposal)
  L    — 1 GT ↔ multiple instances  (over-segmentation)
  D    — multiple GTs share 1 instance (under-segmentation)
  miss — GT has no instance covering ≥ MIN_INBOX_HITS of its in-3D-box pts
         OR has no in-box LiDAR pts to begin with.

Replaces the cluster_id-based pass with an instance-mask-based pass:
each instance owns the points where its mask is True, then those points
are intersected with each GT's 3D box-interior set.

Distance bin labels match diagnosis/measurements.DISTANCE_BIN_LABELS.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from diagnosis.measurements import (
    DISTANCE_BIN_LABELS,
    distance_bin,
    gt_box_to_ego,
    points_inside_3d_box,
)
from nuscenes.eval.detection.utils import category_to_detection_name


MIN_INBOX_HITS = 5  # ≥ this many in-box points → instance covers the GT


def per_sample_case_breakdown(
    gt_boxes_global: list,
    ego_pose_4x4: np.ndarray,
    pc_ego_xyz: np.ndarray,
    instance_masks: np.ndarray,  # (N_points, N_instances) bool
    instance_classes: np.ndarray,  # (N_instances,) int (text-prompt index)
    text_prompts: list,
) -> Dict:
    """Return per-GT classification + summary case counts + distance bins.

    GT is filtered to the 10 official nuScenes detection classes — others
    have no AP to score against, so case counts are computed only on those.
    """
    if instance_masks.size == 0 or instance_masks.ndim != 2:
        instance_masks = np.zeros((pc_ego_xyz.shape[0], 0), dtype=bool)
    n_instances = instance_masks.shape[1]

    n_classes_text = len(text_prompts)
    valid_inst_idx = [i for i in range(n_instances)
                      if 0 <= int(instance_classes[i]) < n_classes_text]

    # Pass 1: per GT, compute its in-box point-set + which instance ids cover it.
    gt_inbox_total = []
    gt_inbox_pred = []
    gt_inst_sets = []  # list[list[int]] — instance ids that hit this GT (≥ MIN_INBOX_HITS)
    gt_distance = []
    gt_kept_idx = []   # filtered GT indices
    gt_categories = []
    for gt_idx, gt in enumerate(gt_boxes_global):
        det_name = category_to_detection_name(gt.get("category", ""))
        if det_name is None:
            continue
        try:
            box_ego = gt_box_to_ego(gt, ego_pose_4x4)
        except Exception:
            continue
        in3d = points_inside_3d_box(box_ego, pc_ego_xyz)
        n_in = int(in3d.sum())
        gt_kept_idx.append(gt_idx)
        gt_categories.append(det_name)
        gt_inbox_total.append(n_in)
        gt_distance.append(float(np.linalg.norm(box_ego.center)))
        if n_in == 0 or n_instances == 0:
            gt_inbox_pred.append(0)
            gt_inst_sets.append([])
            continue

        # For each kept instance: how many of the GT's in-box points fall in its mask?
        hits = []
        any_pred = 0
        for inst_i in valid_inst_idx:
            mask_i = instance_masks[:, inst_i]
            n_hit = int(np.logical_and(in3d, mask_i).sum())
            if n_hit >= MIN_INBOX_HITS:
                hits.append(inst_i)
            any_pred += n_hit
        gt_inst_sets.append(hits)
        gt_inbox_pred.append(int(any_pred))

    # Pass 2: per instance, which GTs does it claim?
    inst_to_gts: Dict[int, set] = {}
    for k, hits in enumerate(gt_inst_sets):
        for inst_i in hits:
            inst_to_gts.setdefault(inst_i, set()).add(k)

    # Pass 3: classify each (kept) GT.
    cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    per_gt = []
    for k, kept_idx in enumerate(gt_kept_idx):
        hits = gt_inst_sets[k]
        if not hits:
            case = "miss"
        elif len(hits) > 1:
            case = "L"
        else:
            inst_i = hits[0]
            if len(inst_to_gts.get(inst_i, set())) > 1:
                case = "D"
            else:
                case = "M"
        cases[case] += 1
        d = gt_distance[k]
        per_gt.append({
            "gt_idx": int(kept_idx),
            "category": gt_categories[k],
            "distance_m": d,
            "distance_bin": distance_bin(d),
            "case": case,
            "n_inbox_total": int(gt_inbox_total[k]),
            "n_inbox_pred": int(gt_inbox_pred[k]),
            "matched_instance_ids": [int(x) for x in hits],
        })

    n_gt = len(per_gt)
    return {
        "n_gt": int(n_gt),
        "n_instances": int(n_instances),
        "n_instances_valid": int(len(valid_inst_idx)),
        "case_counts": cases,
        "M_rate": (cases["M"] / n_gt) if n_gt else 0.0,
        "L_rate": (cases["L"] / n_gt) if n_gt else 0.0,
        "D_rate": (cases["D"] / n_gt) if n_gt else 0.0,
        "miss_rate": (cases["miss"] / n_gt) if n_gt else 0.0,
        "per_gt": per_gt,
    }


def aggregate_distance_strata(per_sample_records: List[dict]) -> Dict:
    """Roll per-GT records up by distance bin (M/L/D/miss rate per bin).
    """
    bin_counts = {b: {"n": 0, "M": 0, "L": 0, "D": 0, "miss": 0} for b in DISTANCE_BIN_LABELS}
    for rec in per_sample_records:
        for g in rec["instance_metrics"].get("per_gt", []):
            b = g["distance_bin"]
            if b not in bin_counts:
                continue
            bin_counts[b]["n"] += 1
            bin_counts[b][g["case"]] += 1
    out = {}
    for b, c in bin_counts.items():
        n = c["n"]
        out[b] = {
            "n_gt": int(n),
            "M_rate": (c["M"] / n) if n else 0.0,
            "L_rate": (c["L"] / n) if n else 0.0,
            "D_rate": (c["D"] / n) if n else 0.0,
            "miss_rate": (c["miss"] / n) if n else 0.0,
        }
    return out
