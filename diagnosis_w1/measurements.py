"""W1 measurement primitives.

GT-cluster matching with explicit M/L/D/miss case classification:
  M  — 1 GT ↔ 1 cluster (ideal: clean per-object proposal)
  L  — 1 GT ↔ multiple clusters (over-segmentation: GT spans many proposals)
  D  — multiple GTs share 1 cluster (under-segmentation: proposal merges
       neighbouring objects)
  miss — GT has zero in-3D-box clustered LiDAR points (HDBSCAN classified
         all of its points as noise, or it had none to begin with)

Reuses Tier-1 ego-frame box transform + 3D box containment.
"""

from __future__ import annotations

import numpy as np

from diagnosis.measurements import (
    distance_bin,
    gt_box_to_ego,
    points_inside_3d_box,
)


def match_gt_to_clusters(gt_boxes, ego_pose_4x4, pc_ego_xyz, cluster_ids):
    """Per-GT case classification + per-cluster GT membership.

    Args:
        gt_boxes: list[dict] from NuScenesLoader (translation/size/rotation/category).
        ego_pose_4x4: 4x4 ego→global transform.
        pc_ego_xyz: (N, 3) point cloud in ego frame.
        cluster_ids: (N,) int array; -1 = noise.

    Returns:
        per_gt: list of dicts (one per GT) with case + matched cluster ids
                + diagnostic counts. NB: ``case`` == ``"miss"`` is reported
                whether the GT had zero LiDAR points in its 3D box OR all
                of its in-box points were classified as noise — distinguish
                by ``n_inbox_total`` vs ``n_inbox_clustered``.
        case_counts: {"M": ..., "L": ..., "D": ..., "miss": ...}
    """
    n_pts = pc_ego_xyz.shape[0]

    # Pass 1: per GT, find which cluster ids its 3D-box-interior points fall into.
    gt_cluster_sets = {}  # gt_idx -> sorted list of unique cluster ids (excluding -1)
    gt_inbox_total = {}   # gt_idx -> total points in 3D box
    gt_inbox_clust = {}   # gt_idx -> points in 3D box that are clustered (cid != -1)
    gt_distance = {}
    for gt_idx, gt in enumerate(gt_boxes):
        try:
            box_ego = gt_box_to_ego(gt, ego_pose_4x4)
        except Exception:
            gt_cluster_sets[gt_idx] = []
            gt_inbox_total[gt_idx] = 0
            gt_inbox_clust[gt_idx] = 0
            gt_distance[gt_idx] = None
            continue
        in3d = points_inside_3d_box(box_ego, pc_ego_xyz)
        n_in = int(in3d.sum())
        gt_inbox_total[gt_idx] = n_in
        if n_in == 0:
            gt_cluster_sets[gt_idx] = []
            gt_inbox_clust[gt_idx] = 0
        else:
            cids = cluster_ids[in3d]
            non_noise = cids[cids != -1]
            unique = sorted(int(c) for c in set(non_noise.tolist()))
            gt_cluster_sets[gt_idx] = unique
            gt_inbox_clust[gt_idx] = int(non_noise.size)
        gt_distance[gt_idx] = float(np.linalg.norm(box_ego.center))

    # Pass 2: per cluster, build the set of GTs whose 3D box overlaps it.
    cluster_to_gts = {}  # cid -> set(gt_idx)
    for gt_idx, cids in gt_cluster_sets.items():
        for c in cids:
            cluster_to_gts.setdefault(c, set()).add(gt_idx)

    # Pass 3: classify each GT.
    per_gt = []
    cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for gt_idx, gt in enumerate(gt_boxes):
        cids = gt_cluster_sets[gt_idx]
        if not cids:
            case = "miss"
        elif len(cids) > 1:
            case = "L"
        else:
            c = cids[0]
            if len(cluster_to_gts.get(c, set())) > 1:
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
            "matched_clusters": cids,
            "n_inbox_total": gt_inbox_total[gt_idx],
            "n_inbox_clustered": gt_inbox_clust[gt_idx],
        })

    return per_gt, cases


def cluster_extents(centroids, bboxes, sizes):
    """Cluster-level summary stats. Returns dict of arrays for aggregator use."""
    if centroids.shape[0] == 0:
        return {"extent_xy": np.zeros((0,)), "extent_z": np.zeros((0,)),
                "distance": np.zeros((0,)), "size": np.asarray(sizes)}
    extent_xy = np.maximum(bboxes[:, 3] - bboxes[:, 0], bboxes[:, 4] - bboxes[:, 1])
    extent_z = bboxes[:, 5] - bboxes[:, 2]
    distance = np.linalg.norm(centroids[:, :2], axis=1)  # xy distance from ego
    return {
        "extent_xy": extent_xy,
        "extent_z": extent_z,
        "distance": distance,
        "size": np.asarray(sizes),
    }
