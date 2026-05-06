"""GT ↔ instance matching for Step 1.

Mask3D produces (N_pts, N_inst) boolean masks (one column per instance);
HDBSCAN produces (N_pts,) integer cluster_ids. To compare apples-to-apples
we expand HDBSCAN ids into the same (N_pts, N_inst) shape and apply one
matching function to both.

Matching criterion (W1 convention preserved): an instance overlaps a GT box
if at least one of its points lies inside the 3D GT box. M/L/D/miss labels
are then derived from the (GT × instance) bipartite adjacency.
"""

from __future__ import annotations

import numpy as np

from diagnosis.measurements import distance_bin, gt_box_to_ego, points_inside_3d_box


def cluster_ids_to_masks(cluster_ids: np.ndarray, n_clusters: int) -> np.ndarray:
    """(N_pts,) int cluster ids → (N_pts, n_clusters) bool one-hot.

    cluster_ids ∈ {-1, 0, 1, ..., n_clusters-1}. -1 (noise) does not get a column.
    """
    N = cluster_ids.shape[0]
    out = np.zeros((N, n_clusters), dtype=bool)
    for cid in range(n_clusters):
        out[:, cid] = (cluster_ids == cid)
    return out


def match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_ego_xyz, masks):
    """Per-GT case classification + per-instance GT membership.

    Args:
        masks: (N_pts, N_inst) boolean, one column per instance proposal.

    Returns:
        per_gt: list of dicts with keys gt_idx, category, distance_m,
                distance_bin, case ∈ {"M", "L", "D", "miss"}, matched_instances.
        case_counts: {"M","L","D","miss" → int}.
    """
    N_pts, N_inst = masks.shape

    # Pre-compute per-GT in-box mask once.
    gt_in_box = []          # list of (N_pts,) bool arrays
    gt_distance = []
    for gt in gt_boxes:
        try:
            box_ego = gt_box_to_ego(gt, ego_pose_4x4)
            in3d = points_inside_3d_box(box_ego, pc_ego_xyz)
            gt_in_box.append(in3d)
            gt_distance.append(float(np.linalg.norm(box_ego.center)))
        except Exception:
            gt_in_box.append(np.zeros(N_pts, dtype=bool))
            gt_distance.append(None)

    # For each GT, find the set of instance ids whose points overlap the box.
    gt_instance_sets = []
    for in3d in gt_in_box:
        if N_inst == 0 or not in3d.any():
            gt_instance_sets.append([])
            continue
        # masks[in3d, :] is (n_in_box, N_inst); column-wise any() = does this instance touch the box?
        touches = masks[in3d, :].any(axis=0)
        gt_instance_sets.append(sorted(int(i) for i in np.where(touches)[0]))

    # Inverse map: instance → set of GTs
    inst_to_gts = {}
    for gt_idx, ids in enumerate(gt_instance_sets):
        for i in ids:
            inst_to_gts.setdefault(i, set()).add(gt_idx)

    # Classify each GT
    per_gt = []
    cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for gt_idx, gt in enumerate(gt_boxes):
        ids = gt_instance_sets[gt_idx]
        if not ids:
            case = "miss"
        elif len(ids) > 1:
            case = "L"
        else:
            i = ids[0]
            if len(inst_to_gts.get(i, set())) > 1:
                case = "D"
            else:
                case = "M"
        cases[case] += 1
        d = gt_distance[gt_idx]
        per_gt.append({
            "gt_idx": gt_idx,
            "category": gt.get("category"),
            "distance_m": d,
            "distance_bin": distance_bin(d) if d is not None else None,
            "case": case,
            "matched_instances": ids,
            "n_inbox_total": int(gt_in_box[gt_idx].sum()),
        })

    return per_gt, cases


def aggregate_cases(case_counts_list, n_gt_total):
    """Sum case counts across samples and convert to rates."""
    totals = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for c in case_counts_list:
        for k in totals:
            totals[k] += c.get(k, 0)
    return {
        "totals": totals,
        "rates": {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in totals.items()},
        "n_gt_total": n_gt_total,
    }
