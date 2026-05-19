"""Hybrid (Mask3D ∪ HDBSCAN) coverage simulator.

Given per-sample masks from each source, build the union and report:
  covered_by_mask3d_only / covered_by_hdbscan_only / covered_by_both / covered_by_neither

Per-GT covered = at least one instance from that source overlaps the 3D box.
This is identical to the matching primitive but reduced to a binary signal
per source. We avoid duplicating that logic by reusing match_gt_to_instances
on each source separately.
"""

from __future__ import annotations

import numpy as np

from diagnosis_step1.matching import match_gt_to_instances


def simulate_hybrid(gt_boxes, ego_pose_4x4, pc_ego_xyz,
                    mask3d_masks: np.ndarray, hdbscan_masks: np.ndarray):
    """Return per-GT coverage flags and aggregate counts.

    mask3d_masks: (N_pts, N_M) bool. hdbscan_masks: (N_pts, N_H) bool.
    """
    per_gt_m, _ = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_ego_xyz, mask3d_masks)
    per_gt_h, _ = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_ego_xyz, hdbscan_masks)

    per_gt = []
    counts = {"mask3d_only": 0, "hdbscan_only": 0, "both": 0, "neither": 0,
              "covered_by_either": 0}
    for gt_idx, (gm, gh) in enumerate(zip(per_gt_m, per_gt_h)):
        m_cov = gm["case"] != "miss"
        h_cov = gh["case"] != "miss"
        if m_cov and h_cov:
            tag = "both"
        elif m_cov and not h_cov:
            tag = "mask3d_only"
        elif h_cov and not m_cov:
            tag = "hdbscan_only"
        else:
            tag = "neither"
        counts[tag] += 1
        if m_cov or h_cov:
            counts["covered_by_either"] += 1
        per_gt.append({
            "gt_idx": gt_idx,
            "category": gm.get("category"),
            "distance_m": gm.get("distance_m"),
            "distance_bin": gm.get("distance_bin"),
            "mask3d_case": gm["case"],
            "hdbscan_case": gh["case"],
            "covered_tag": tag,
        })

    n_gt = len(gt_boxes)
    rates = {k: (v / n_gt) if n_gt else 0.0 for k, v in counts.items()}

    # Stack masks for unified instance-level matching ("Hybrid M_rate")
    if mask3d_masks.shape[0] != hdbscan_masks.shape[0]:
        raise ValueError("Mask shapes disagree on N_pts axis")
    n_total = mask3d_masks.shape[1] + hdbscan_masks.shape[1]
    if n_total > 0:
        union_masks = np.concatenate([mask3d_masks, hdbscan_masks], axis=1)
    else:
        union_masks = np.zeros((mask3d_masks.shape[0], 0), dtype=bool)
    union_per_gt, union_cases = match_gt_to_instances(gt_boxes, ego_pose_4x4,
                                                       pc_ego_xyz, union_masks)
    return {
        "per_gt": per_gt,
        "counts": counts,
        "rates": rates,
        "n_proposals_mask3d": int(mask3d_masks.shape[1]),
        "n_proposals_hdbscan": int(hdbscan_masks.shape[1]),
        "n_proposals_total": int(n_total),
        "union_case_counts": union_cases,
        "union_per_gt": union_per_gt,
    }
