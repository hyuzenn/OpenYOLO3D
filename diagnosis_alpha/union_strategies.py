"""α hybrid simulation — four union strategies over β1 + γ proposals.

All strategies operate on a *unified proposal record* per source: each
proposal is reduced to a (N_pts,) bool mask in the ego frame plus a few
scalars (score, BEV axis-aligned bbox, point count). The four strategies
differ only in which subset of the union they hand to the matching primitive
``match_gt_to_instances``:

  1. naive_union           — concatenate β1 ∪ γ; let the matcher decide.
  2. distance_aware_union  — per-GT, prefer γ near (d < threshold) or β1 far.
                              Fall back to the other source on miss.
  3. score_weighted_union  — global greedy score-NMS using mask IoU; both
                              sources scored on a normalized [0, 1] axis.
  4. spatial_nms_union     — pairwise BEV IoU between β1 and γ proposals
                              only; if IoU > threshold, keep the higher-score
                              one. Within-source proposals untouched.

The β1 path produces (N, n_clusters) one-hot masks via ``cluster_ids``.
The γ path produces (N, n_proposals) masks via per-proposal points-in-3D-box.
Both are precomputed once per sample by ``measurements.run_sources_once``
and re-used across the 10 sweep combos so we never re-run inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from diagnosis_step1.matching import match_gt_to_instances


# -- helpers -----------------------------------------------------------------

def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min–max normalize within sample to [0, 1]; constant arrays → all 0.5."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Point-set IoU between two (N,) bool masks."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _bev_aabb_iou(a: tuple, b: tuple) -> float:
    """Axis-aligned BEV IoU for ``(xmin, ymin, xmax, ymax)`` tuples."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _per_gt_with_meta(per_gt: list, source_choice: list, distance_threshold: Optional[float]):
    out = []
    for g, src in zip(per_gt, source_choice):
        rec = dict(g)
        rec["chosen_source"] = src
        if distance_threshold is not None:
            rec["distance_threshold"] = distance_threshold
        out.append(rec)
    return out


# -- strategy 1: naive union --------------------------------------------------

def strategy_naive(
    beta1_masks: np.ndarray,
    gamma_masks: np.ndarray,
    gt_boxes: list,
    ego_pose_4x4: np.ndarray,
    pc_xyz: np.ndarray,
) -> dict:
    """Concatenate masks; all proposals are candidates.

    Returns dict with per_gt, case_counts, n_proposals_total/β1/γ, source_of_match.
    """
    n_b = beta1_masks.shape[1]
    n_g = gamma_masks.shape[1]
    if n_b + n_g == 0:
        masks = np.zeros((pc_xyz.shape[0], 0), dtype=bool)
    else:
        masks = np.concatenate([beta1_masks, gamma_masks], axis=1)
    sources = ["beta1"] * n_b + ["gamma"] * n_g
    per_gt, cases = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_xyz, masks)

    # annotate per-gt with which source(s) the matched instances came from
    out = []
    for g in per_gt:
        srcs = sorted({sources[i] for i in g["matched_instances"]}) if sources else []
        out.append({**g, "match_sources": srcs,
                    "chosen_source": (srcs[0] if len(srcs) == 1 else
                                       ("both" if len(srcs) == 2 else None))})
    return {
        "per_gt": out,
        "case_counts": cases,
        "n_proposals_total": n_b + n_g,
        "n_proposals_beta1": n_b,
        "n_proposals_gamma": n_g,
    }


# -- strategy 2: distance-aware union ----------------------------------------

def strategy_distance_aware(
    beta1_masks: np.ndarray,
    gamma_masks: np.ndarray,
    gt_boxes: list,
    ego_pose_4x4: np.ndarray,
    pc_xyz: np.ndarray,
    distance_threshold_m: float,
) -> dict:
    """Per-GT: distance < threshold → γ first, else β1 first; fallback on miss.

    Each GT is independently classified using whichever single source matched
    it. n_proposals_total reports the ``β1 ∪ γ`` candidate pool size.
    """
    per_gt_b1, _cb = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_xyz, beta1_masks)
    per_gt_g,  _cg = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_xyz, gamma_masks)

    cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    chosen_per_gt = []
    src_chain = []
    for g_b1, g_g in zip(per_gt_b1, per_gt_g):
        d = g_b1.get("distance_m")
        if d is None:
            d = g_g.get("distance_m")
        if d is None or d < distance_threshold_m:
            primary, secondary = g_g, g_b1
            primary_src, secondary_src = "gamma", "beta1"
        else:
            primary, secondary = g_b1, g_g
            primary_src, secondary_src = "beta1", "gamma"

        if primary["case"] != "miss":
            chosen, chosen_src = primary, primary_src
        elif secondary["case"] != "miss":
            chosen, chosen_src = secondary, secondary_src
        else:
            chosen, chosen_src = primary, primary_src   # both miss

        cases[chosen["case"]] += 1
        chosen_per_gt.append({
            "gt_idx": chosen["gt_idx"],
            "category": chosen["category"],
            "distance_m": chosen.get("distance_m") or g_g.get("distance_m"),
            "distance_bin": chosen.get("distance_bin") or g_g.get("distance_bin"),
            "case": chosen["case"],
            "matched_instances": chosen["matched_instances"],
            "chosen_source": chosen_src,
            "primary_source": primary_src,
            "fell_back": (chosen_src != primary_src),
        })
        src_chain.append(chosen_src)

    return {
        "per_gt": chosen_per_gt,
        "case_counts": cases,
        "n_proposals_total": beta1_masks.shape[1] + gamma_masks.shape[1],
        "n_proposals_beta1": beta1_masks.shape[1],
        "n_proposals_gamma": gamma_masks.shape[1],
        "distance_threshold_m": distance_threshold_m,
        "fell_back_count": int(sum(1 for g in chosen_per_gt if g["fell_back"])),
    }


# -- strategy 3: score-weighted union (greedy mask-IoU NMS) ------------------

def strategy_score_weighted(
    beta1_masks: np.ndarray,
    beta1_sizes: np.ndarray,
    gamma_masks: np.ndarray,
    gamma_scores: np.ndarray,
    gt_boxes: list,
    ego_pose_4x4: np.ndarray,
    pc_xyz: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """All β1 + γ candidates; greedy NMS by normalized score, mask IoU as overlap.

    β1 score proxy: cluster size / max(cluster size). γ score: CenterPoint score.
    Within each source we min–max normalize then concatenate. The greedy keep
    walks the union in descending score and discards any later candidate whose
    mask IoU with an already-kept one exceeds ``iou_threshold``.
    """
    n_b = beta1_masks.shape[1]
    n_g = gamma_masks.shape[1]

    # Normalize within source
    s_b = _normalize_scores(beta1_sizes.astype(np.float64)) if n_b > 0 else np.zeros((0,))
    s_g = _normalize_scores(gamma_scores.astype(np.float64)) if n_g > 0 else np.zeros((0,))
    scores = np.concatenate([s_b, s_g]) if n_b + n_g else np.zeros((0,))
    if n_b + n_g == 0:
        masks_all = np.zeros((pc_xyz.shape[0], 0), dtype=bool)
    else:
        masks_all = np.concatenate([beta1_masks, gamma_masks], axis=1)
    sources = (["beta1"] * n_b) + (["gamma"] * n_g)

    # Greedy NMS
    order = np.argsort(-scores) if scores.size else np.zeros((0,), dtype=np.int64)
    kept_idx = []
    suppressed = 0
    for i in order:
        keep = True
        for j in kept_idx:
            iou = _mask_iou(masks_all[:, i], masks_all[:, j])
            if iou > iou_threshold:
                keep = False
                break
        if keep:
            kept_idx.append(int(i))
        else:
            suppressed += 1

    if not kept_idx:
        kept_masks = np.zeros((pc_xyz.shape[0], 0), dtype=bool)
        kept_sources = []
    else:
        kept_masks = masks_all[:, kept_idx]
        kept_sources = [sources[i] for i in kept_idx]

    per_gt, cases = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_xyz, kept_masks)

    # annotate
    out = []
    for g in per_gt:
        srcs = sorted({kept_sources[i] for i in g["matched_instances"]}) if kept_sources else []
        out.append({**g, "match_sources": srcs,
                    "chosen_source": (srcs[0] if len(srcs) == 1 else
                                       ("both" if len(srcs) == 2 else None))})
    return {
        "per_gt": out,
        "case_counts": cases,
        "n_proposals_total": len(kept_idx),
        "n_proposals_beta1": int(sum(1 for s in kept_sources if s == "beta1")),
        "n_proposals_gamma": int(sum(1 for s in kept_sources if s == "gamma")),
        "n_suppressed": int(suppressed),
        "iou_threshold": float(iou_threshold),
    }


# -- strategy 4: spatial NMS union (cross-source only) ------------------------

def strategy_spatial_nms(
    beta1_masks: np.ndarray,
    beta1_sizes: np.ndarray,
    beta1_aabbs: np.ndarray,        # (n_b, 4) BEV xmin,ymin,xmax,ymax
    gamma_masks: np.ndarray,
    gamma_scores: np.ndarray,
    gamma_aabbs: np.ndarray,        # (n_g, 4)
    gt_boxes: list,
    ego_pose_4x4: np.ndarray,
    pc_xyz: np.ndarray,
    iou_threshold: float = 0.3,
) -> dict:
    """Pairwise BEV IoU β1↔γ. If iou > threshold, drop lower-normalized-score.

    Within-source proposals are kept as-is — the strategy only resolves
    cross-source duplicates. β1 score proxy = normalized cluster size,
    γ score = normalized CenterPoint score.
    """
    n_b = beta1_masks.shape[1]
    n_g = gamma_masks.shape[1]
    drop_b = np.zeros(n_b, dtype=bool)
    drop_g = np.zeros(n_g, dtype=bool)

    if n_b > 0 and n_g > 0:
        s_b = _normalize_scores(beta1_sizes.astype(np.float64))
        s_g = _normalize_scores(gamma_scores.astype(np.float64))
        for i in range(n_b):
            for j in range(n_g):
                iou = _bev_aabb_iou(tuple(beta1_aabbs[i]), tuple(gamma_aabbs[j]))
                if iou > iou_threshold:
                    if s_g[j] >= s_b[i]:
                        drop_b[i] = True
                    else:
                        drop_g[j] = True

    keep_b = np.where(~drop_b)[0]
    keep_g = np.where(~drop_g)[0]
    if n_b + n_g == 0:
        masks_all = np.zeros((pc_xyz.shape[0], 0), dtype=bool)
        sources = []
    else:
        masks_all = np.concatenate(
            [beta1_masks[:, keep_b], gamma_masks[:, keep_g]], axis=1
        )
        sources = (["beta1"] * keep_b.size) + (["gamma"] * keep_g.size)

    per_gt, cases = match_gt_to_instances(gt_boxes, ego_pose_4x4, pc_xyz, masks_all)
    out = []
    for g in per_gt:
        srcs = sorted({sources[i] for i in g["matched_instances"]}) if sources else []
        out.append({**g, "match_sources": srcs,
                    "chosen_source": (srcs[0] if len(srcs) == 1 else
                                       ("both" if len(srcs) == 2 else None))})
    return {
        "per_gt": out,
        "case_counts": cases,
        "n_proposals_total": int(keep_b.size + keep_g.size),
        "n_proposals_beta1": int(keep_b.size),
        "n_proposals_gamma": int(keep_g.size),
        "n_dropped_beta1": int(drop_b.sum()),
        "n_dropped_gamma": int(drop_g.sum()),
        "iou_threshold": float(iou_threshold),
    }


# -- strategy registry --------------------------------------------------------

STRATEGY_GRID = [
    {"strategy": "naive",            "params": {}},
    {"strategy": "distance_aware",   "params": {"distance_threshold_m": 15.0}},
    {"strategy": "distance_aware",   "params": {"distance_threshold_m": 20.0}},
    {"strategy": "distance_aware",   "params": {"distance_threshold_m": 25.0}},
    {"strategy": "distance_aware",   "params": {"distance_threshold_m": 30.0}},
    {"strategy": "distance_aware",   "params": {"distance_threshold_m": 35.0}},
    {"strategy": "score_weighted",   "params": {"iou_threshold": 0.5}},
    {"strategy": "spatial_nms",      "params": {"iou_threshold": 0.1}},
    {"strategy": "spatial_nms",      "params": {"iou_threshold": 0.3}},
    {"strategy": "spatial_nms",      "params": {"iou_threshold": 0.5}},
]


def combo_id(combo: dict) -> str:
    s, p = combo["strategy"], combo["params"]
    if s == "naive":
        return "naive"
    if s == "distance_aware":
        return f"distance_aware_thr{p['distance_threshold_m']:g}"
    if s == "score_weighted":
        return f"score_weighted_iou{p['iou_threshold']:g}"
    if s == "spatial_nms":
        return f"spatial_nms_iou{p['iou_threshold']:g}"
    raise ValueError(f"unknown strategy {s}")


def apply_strategy(combo: dict, sample_artifacts: dict) -> dict:
    """Dispatch one strategy on a single sample's β1+γ artifacts.

    ``sample_artifacts`` keys: gt_boxes, ego_pose, pc_xyz, beta1_masks,
    beta1_sizes, beta1_aabbs, gamma_masks, gamma_scores, gamma_aabbs.
    """
    s = combo["strategy"]
    p = combo["params"]
    a = sample_artifacts
    if s == "naive":
        return strategy_naive(
            a["beta1_masks"], a["gamma_masks"],
            a["gt_boxes"], a["ego_pose"], a["pc_xyz"],
        )
    if s == "distance_aware":
        return strategy_distance_aware(
            a["beta1_masks"], a["gamma_masks"],
            a["gt_boxes"], a["ego_pose"], a["pc_xyz"],
            distance_threshold_m=p["distance_threshold_m"],
        )
    if s == "score_weighted":
        return strategy_score_weighted(
            a["beta1_masks"], a["beta1_sizes"],
            a["gamma_masks"], a["gamma_scores"],
            a["gt_boxes"], a["ego_pose"], a["pc_xyz"],
            iou_threshold=p["iou_threshold"],
        )
    if s == "spatial_nms":
        return strategy_spatial_nms(
            a["beta1_masks"], a["beta1_sizes"], a["beta1_aabbs"],
            a["gamma_masks"], a["gamma_scores"], a["gamma_aabbs"],
            a["gt_boxes"], a["ego_pose"], a["pc_xyz"],
            iou_threshold=p["iou_threshold"],
        )
    raise ValueError(f"unknown strategy {s}")
