"""Outdoor proposal-recall probe (Task 2, stage 1).

PURPOSE
-------
Decide — *before* building the Geometry-guided OV bridge — whether
CenterPoint's proposal recall is the actual ceiling on outdoor mAP. We
compare two existing proposal sources side-by-side on the nuScenes val
set, using nothing but the pickled proposal caches already produced by
``method_scannet/streaming/nuscenes_native_evaluator.py``:

    γ-native CenterPoint   results/outdoor_native_temporal_cpcache_thr000_single/
    detguided clustering   results/outdoor_detguided_cpcache_thr000_full150/

Three metrics are emitted per source, all per-class and per-distance-bin:

  1. Recall_box      — at least one proposal lies within BEV centre
                       distance ≤ τ_d for each GT (τ_d ∈ {0.5, 1.0, 2.0, 4.0}
                       m, matching the official nuScenes mAP buckets).
  2. Recall_iou3d    — at least one proposal has IoU ≥ τ_iou (τ_iou ∈
                       {0.25, 0.50, 0.70}). The IoU metric is selectable
                       via ``--iou-mode``: ``iou3d`` (default) uses the
                       full 3D volume IoU (BEV intersection × z overlap);
                       ``bev`` reproduces the original BEV-only metric.
                       Class-agnostic to isolate geometric ceiling from
                       labelling.
  3. Oracle_mAP      — each proposal is re-labelled with the class of
                       its best-matching GT. ``--oracle-mode``:
                       ``maxiou`` (default) takes the highest-IoU GT
                       and enforces GT-deduplication (one GT serves at
                       most one proposal, top-score first), giving a
                       tight per-proposal upper-bound. ``nearest``
                       restores the legacy BEV-centre argmin (4 m
                       cutoff, no dedup). The resulting box set is fed
                       through the unchanged nuScenes-devkit eval.

NOTHING in this script modifies the evaluator, the leaderboard, or the
official mAP pipeline. The only side effect is the JSON written to
``results/<DATE>_outdoor_proposal_recall_v<NN>/recall_probe.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import shapely.geometry as sg
from shapely.affinity import rotate as _sh_rotate, translate as _sh_translate

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import val as VAL_SCENES
from nuscenes.eval.detection.utils import category_to_detection_name

# Class index order MUST match the cached proposal contracts. Both caches
# emit cls_name strings from this list (see adapters/centerpoint_proposals.py
# and nuscenes_native_evaluator._detguided_to_proposals).
NUSC_10 = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
NUSC_10_SET = set(NUSC_10)


# Distance bins in *ego* range (m). Matches diagnosis_gamma's BIN_EDGES.
DIST_BINS = [(0.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 80.0)]

# Match thresholds.
BOX_DIST_THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
IOU3D_THRESHOLDS = (0.25, 0.50, 0.70)

# Geometry-diagnostic grids (additive; never feed the metrics above).
# Histogram bin edges span [0, 1] in 0.05 steps; recall curves sweep τ in
# [0.0, 0.90] in 0.05 steps.
GEO_IOU_BIN_EDGES = tuple(round(0.05 * i, 2) for i in range(21))
GEO_RECALL_TAUS = tuple(round(0.05 * i, 2) for i in range(19))

# Oracle-mAP class assignment: any GT within this BEV centre distance is
# considered to *cover* the proposal. 4.0 m matches the loosest nuScenes
# mAP bucket, so we're declaring "this proposal could only ever count
# against a GT in that radius — assign that GT's label".
ORACLE_MATCH_RADIUS_M = 4.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ego_to_global_centroid(centroid_ego: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    return (ego_pose[:3, :3] @ np.asarray(centroid_ego, dtype=np.float64)[:3]
            ) + ego_pose[:3, 3]


def _yaw_lidar_to_global(yaw_lidar: float, lidar_to_ego_q: Quaternion,
                         ego_quat: Quaternion) -> float:
    """Compose lidar→ego→global rotations and return the global yaw."""
    q_box_lidar = Quaternion(axis=(0.0, 0.0, 1.0), angle=float(yaw_lidar))
    q_global = ego_quat * lidar_to_ego_q * q_box_lidar
    # Yaw from the rotation matrix's first column (atan2(r10, r00)).
    R = q_global.rotation_matrix
    return float(math.atan2(R[1, 0], R[0, 0]))


def _bev_rect(cx: float, cy: float, dx: float, dy: float, yaw: float) -> sg.Polygon:
    """Axis-aligned rectangle of size (dx, dy) at origin, then rotated by yaw
    (radians, ccw) and translated to (cx, cy). Returns a shapely polygon in
    BEV. Floors a degenerate 0-sized box to a 1 cm square so IoU is 0 (not
    NaN) when comparing.
    """
    dx = max(float(dx), 1e-2)
    dy = max(float(dy), 1e-2)
    rect = sg.box(-dx / 2.0, -dy / 2.0, dx / 2.0, dy / 2.0)
    rect = _sh_rotate(rect, yaw, origin=(0.0, 0.0), use_radians=True)
    return _sh_translate(rect, xoff=float(cx), yoff=float(cy))


def _bev_iou(rect_a: sg.Polygon, rect_b: sg.Polygon) -> float:
    if not rect_a.intersects(rect_b):
        return 0.0
    inter = rect_a.intersection(rect_b).area
    if inter <= 0.0:
        return 0.0
    union = rect_a.area + rect_b.area - inter
    return float(inter / union) if union > 0 else 0.0


def _iou_3d(a: dict, b: dict) -> float:
    """True 3D volume IoU between two axis-aligned-Z (yaw-rotated BEV) boxes.

    Each operand is a dict carrying:
        - "rect"     : shapely BEV polygon (oriented rectangle in (x, y))
        - "z_bottom" : float, world-frame bottom z of the box
        - "z_top"    : float, world-frame top z of the box

    Formula (axis-aligned-Z assumption; yaw acts in BEV only):
        bev_inter_area = rect_a.intersection(rect_b).area
        z_inter        = max(0, min(za_top, zb_top) - max(za_bot, zb_bot))
        inter_vol      = bev_inter_area * z_inter
        vol_a          = rect_a.area * (za_top - za_bot)
        vol_b          = rect_b.area * (zb_top - zb_bot)
        union_vol      = vol_a + vol_b - inter_vol
        iou3d          = inter_vol / union_vol  if union_vol > 0 else 0
    """
    rect_a = a["rect"]
    rect_b = b["rect"]
    if not rect_a.intersects(rect_b):
        return 0.0
    bev_inter = rect_a.intersection(rect_b).area
    if bev_inter <= 0.0:
        return 0.0
    za_bot = float(a["z_bottom"]); za_top = float(a["z_top"])
    zb_bot = float(b["z_bottom"]); zb_top = float(b["z_top"])
    z_inter = max(0.0, min(za_top, zb_top) - max(za_bot, zb_bot))
    if z_inter <= 0.0:
        return 0.0
    inter_vol = float(bev_inter) * float(z_inter)
    vol_a = float(rect_a.area) * max(za_top - za_bot, 0.0)
    vol_b = float(rect_b.area) * max(zb_top - zb_bot, 0.0)
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0.0:
        return 0.0
    return float(inter_vol / union_vol)


# ---------------------------------------------------------------------------
# Per-sample probe core
# ---------------------------------------------------------------------------

def _load_cached_proposals(cache_dir: Path, sample_token: str,
                           suffix: str = "") -> Optional[list[dict]]:
    """Returns ``None`` (not []) when the cache file is missing so the caller
    can distinguish a true zero-proposal sample from a missing cache entry.
    """
    fp = cache_dir / f"{sample_token}{suffix}.pkl"
    if not fp.exists():
        return None
    with open(fp, "rb") as f:
        return pickle.load(f)


def _proposals_to_global(proposals: list[dict], ego_pose: np.ndarray,
                         lidar_to_ego_q: Quaternion, ego_quat: Quaternion) -> list[dict]:
    """Augment each proposal with global-frame (centroid, yaw_global) +
    BEV rect for IoU + z-extent for true 3D IoU. Mutates a shallow copy;
    the input list is untouched.

    Both proposal sources publish ``bbox_lidar = [cx, cy, cz, dx, dy, dz,
    yaw, (vx, vy)]`` (cf. adapters/centerpoint_proposals.py:161 and
    method_scannet/streaming/nuscenes_native_evaluator.py:467). The z-extent
    is preserved via ``z_bottom``/``z_top`` in the GLOBAL frame so the
    `_iou_3d` formula above can apply uniformly to proposals and GT.
    """
    out: list[dict] = []
    for p in proposals:
        cls_name = p.get("cls_name")
        # Both caches gate to NUSC_10 on emit, but the detguided source can
        # carry "class_<idx>" / "object" when the YOLO label isn't in
        # NUSC_10 — drop those (recall on non-NUSC_10 isn't comparable).
        if cls_name not in NUSC_10_SET:
            continue
        centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
        bbox_lidar = list(p["bbox_lidar"])
        if len(bbox_lidar) < 7:
            continue
        dx, dy = float(bbox_lidar[3]), float(bbox_lidar[4])
        dz = float(bbox_lidar[5])
        yaw_lidar = float(bbox_lidar[6])
        centroid_global = _ego_to_global_centroid(centroid_ego, ego_pose)
        yaw_global = _yaw_lidar_to_global(yaw_lidar, lidar_to_ego_q, ego_quat)
        rect = _bev_rect(centroid_global[0], centroid_global[1], dx, dy, yaw_global)
        cz_global = float(centroid_global[2])
        # nuScenes convention: translation/centroid is the *centre* of the
        # box, so z_bottom = cz - dz/2 and z_top = cz + dz/2. This is
        # symmetric for both γ-native (CenterPoint) and detguided clusters
        # (centroid of pts, dims = max-min). Floor dz to 1 cm so a 0-height
        # box has a positive volume (and a 1 cm-thick slab, not NaN).
        dz_clamped = max(dz, 1e-2)
        z_bottom = cz_global - dz_clamped / 2.0
        z_top = cz_global + dz_clamped / 2.0
        out.append({
            "cls_name": cls_name,
            "score": float(p.get("score", 0.0)),
            "centroid_ego": centroid_ego,
            "centroid_global": centroid_global,
            "yaw_global": yaw_global,
            "size_xy": (dx, dy),
            "size_z": dz,
            "z_bottom": z_bottom,
            "z_top": z_top,
            "rect": rect,
        })
    return out


def _gt_records(nusc: NuScenes, sample_token: str,
                ego_pose: np.ndarray) -> list[dict]:
    """Loop sample.anns; filter to NUSC_10 via category_to_detection_name; tag
    with ego distance so the per-distance-bin counters work."""
    sample = nusc.get("sample", sample_token)
    out: list[dict] = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det = category_to_detection_name(ann["category_name"])
        if det is None or det not in NUSC_10_SET:
            continue
        centroid_g = np.asarray(ann["translation"], dtype=np.float64)
        size = ann["size"]  # nuScenes order: (w, l, h)
        rot = ann["rotation"]
        q_global = Quaternion(*rot)
        R = q_global.rotation_matrix
        yaw_global = float(math.atan2(R[1, 0], R[0, 0]))
        # nuScenes "size" is (w, l, h) but the BEV footprint uses (l, w) along
        # the box's local (x, y). We approximate as (size[1], size[0]) so
        # length runs along the heading direction — matches DetectionBox.
        rect = _bev_rect(centroid_g[0], centroid_g[1],
                         float(size[1]), float(size[0]), yaw_global)
        ego_dist = float(np.linalg.norm(centroid_g[:2] - ego_pose[:3, 3][:2]))
        # nuScenes annotation ``size = (w, l, h)``; the global ``translation``
        # is the box centre. We preserve the height ``size[2]`` as the GT's
        # z-extent so `_iou_3d` can compute the true 3D IoU without ever
        # needing the original ann dict again.
        h = float(size[2])
        h_clamped = max(h, 1e-2)
        cz_global = float(centroid_g[2])
        z_bottom = cz_global - h_clamped / 2.0
        z_top = cz_global + h_clamped / 2.0
        out.append({
            "cls_name": det,
            "centroid_global": centroid_g,
            "yaw_global": yaw_global,
            "size_xy": (float(size[1]), float(size[0])),
            "size_z": h,
            "z_bottom": z_bottom,
            "z_top": z_top,
            "rect": rect,
            "ego_distance_m": ego_dist,
            "translation": [float(x) for x in ann["translation"]],
            "size": [float(x) for x in ann["size"]],
            "rotation": [float(x) for x in ann["rotation"]],
        })
    return out


def _distance_bin(d: float) -> Optional[str]:
    for lo, hi in DIST_BINS:
        if lo <= d < hi:
            return f"{int(lo)}-{int(hi)}m"
    return f">{int(DIST_BINS[-1][1])}m" if d >= DIST_BINS[-1][1] else None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class RecallAccumulator:
    """Stores per-class × per-distance-bin GT/match counters for one source.

    iou_mode controls the IoU metric in BOTH the recall_iou3d counters and
    the maxiou oracle relabel step:
        - "iou3d" : true 3D volume IoU via _iou_3d (default)
        - "bev"   : legacy BEV-only IoU via _bev_iou (back-compat with the
                    earlier probe outputs)

    oracle_mode controls the oracle relabel step:
        - "maxiou"  : highest-IoU GT (per iou_mode), GT-deduplicated, score-
                      first ordering (default — true per-proposal upper bound)
        - "nearest" : legacy BEV-centre argmin within ORACLE_MATCH_RADIUS_M
                      with no GT deduplication
    """

    def __init__(self, iou_mode: str = "iou3d", oracle_mode: str = "maxiou",
                 oracle_iou_min: float = 0.0, geometry_diag: bool = True):
        if iou_mode not in ("iou3d", "bev"):
            raise ValueError(
                f"iou_mode must be 'iou3d' or 'bev', got {iou_mode!r}"
            )
        if oracle_mode not in ("maxiou", "nearest"):
            raise ValueError(
                f"oracle_mode must be 'maxiou' or 'nearest', got {oracle_mode!r}"
            )
        self.iou_mode = iou_mode
        self.oracle_mode = oracle_mode
        self.oracle_iou_min = float(oracle_iou_min)
        # match_counts[metric][threshold][cls_name][dbin] -> int
        self.gt_counts: dict[str, dict[str, int]] = {}
        self.box_hits: dict[float, dict[str, dict[str, int]]] = {
            t: {} for t in BOX_DIST_THRESHOLDS_M
        }
        self.iou_hits: dict[float, dict[str, dict[str, int]]] = {
            t: {} for t in IOU3D_THRESHOLDS
        }
        # For oracle_map: per-sample (token -> list of relabelled-proposal dicts).
        self.oracle_pred: dict[str, list[dict]] = {}
        self.oracle_gt: dict[str, list[dict]] = {}
        # Bookkeeping
        self.n_samples_seen = 0
        self.n_samples_missing_cache = 0
        self.n_proposals_seen = 0
        self.n_proposals_unlabeled = 0   # cls_name not in NUSC_10
        self.n_gt_total = 0
        # Maxiou oracle diagnostics — how many proposals failed to find an
        # unassigned GT or had IoU below oracle_iou_min.
        self.n_oracle_assigned = 0
        self.n_oracle_dropped_no_gt = 0
        self.n_oracle_dropped_low_iou = 0
        self.n_oracle_dropped_gt_consumed = 0

        # ---- Geometry diagnostic (additive; never affects the metrics above) --
        # For each GT we pair the BEV-IoU-argmax proposal and log signed
        # center/size errors plus three IoU variants (BEV, true 3D, and a
        # z-corrected 3D that reinterprets the cached proposal z as the box
        # BOTTOM). geo_recall arrays span ALL valid-dbin GTs (no-overlap GT ->
        # 0.0) so recall curves share the denominator with recall_iou3d;
        # geo_pairs holds errors only for GTs that have a BEV-overlapping
        # proposal (a meaningful pairing).
        self.geometry_diag_enabled = bool(geometry_diag)
        self.geo_recall: dict[str, list] = {
            "bev": [], "iou3d": [], "iou3d_zcorr": [], "cls": [], "dbin": [],
        }
        self.geo_pairs: dict[str, list] = {k: [] for k in (
            "cls", "dbin", "bev_iou", "iou3d", "iou3d_zcorr",
            "ex", "ey", "ez", "ez_zcorr", "el", "ew", "eh", "z_overlap_frac",
        )}
        self.n_geo_gt = 0
        self.n_geo_gt_no_overlap = 0

    def _pairwise_iou(self, a: dict, b: dict) -> float:
        if self.iou_mode == "iou3d":
            return _iou_3d(a, b)
        return _bev_iou(a["rect"], b["rect"])

    def _proposal_to_detbox(self, p: dict, gt_match: dict,
                            ego_pose: np.ndarray, sample_token: str) -> dict:
        """Build a DetectionBox-shaped dict from a (proposal, matched GT) pair.

        We use the *proposal's* geometry — translation, BEV size, rotation,
        and now the proposal's true ``size_z`` (dz from bbox_lidar[5]). The
        prior 1.7 m hard-coded height contaminated DetectionMetricData.scale_err
        (and therefore NDS) even though mAP itself is centre-distance based.
        """
        return {
            "sample_token": sample_token,
            "translation": [float(p["centroid_global"][0]),
                            float(p["centroid_global"][1]),
                            float(p["centroid_global"][2])],
            "size": [float(p["size_xy"][0]), float(p["size_xy"][1]),
                     float(p.get("size_z", 1.7))],
            "rotation": list(Quaternion(axis=(0.0, 0.0, 1.0),
                                        angle=float(p["yaw_global"])).q),
            "velocity": [0.0, 0.0],
            "ego_translation": [float(ego_pose[0, 3]), float(ego_pose[1, 3]),
                                float(ego_pose[2, 3])],
            "num_pts": 1,
            "detection_name": gt_match["cls_name"],
            "detection_score": float(p["score"]),
            "attribute_name": "",
        }

    def _ensure(self, store: dict, cls: str, dbin: str):
        store.setdefault(cls, {}).setdefault(dbin, 0)

    def record_sample(
        self,
        cls_proposals: list[dict],
        cls_gts: list[dict],
        ego_pose: np.ndarray,
        sample_token: str,
    ):
        self.n_samples_seen += 1
        self.n_proposals_seen += len(cls_proposals)

        for gt in cls_gts:
            cls = gt["cls_name"]
            dbin = _distance_bin(gt["ego_distance_m"])
            if dbin is None:
                continue
            self._ensure(self.gt_counts, cls, dbin)
            self.gt_counts[cls][dbin] += 1
            self.n_gt_total += 1

            # Box-distance recall: ANY proposal within τ_d (class-agnostic).
            gt_xy = gt["centroid_global"][:2]
            best_d = float("inf")
            for p in cls_proposals:
                d = float(np.linalg.norm(p["centroid_global"][:2] - gt_xy))
                if d < best_d:
                    best_d = d
            for th in BOX_DIST_THRESHOLDS_M:
                if best_d <= th:
                    self._ensure(self.box_hits[th], cls, dbin)
                    self.box_hits[th][cls][dbin] += 1

            # IoU recall: ANY proposal with IoU ≥ τ_iou. Metric selected by
            # `iou_mode` — true 3D-volume IoU by default; BEV-only kept as
            # a back-compat option.
            best_iou = 0.0
            best_bev = 0.0
            best_bev_p: Optional[dict] = None
            for p in cls_proposals:
                iou = self._pairwise_iou(gt, p)
                if iou > best_iou:
                    best_iou = iou
                # Additive: track the BEV-IoU-argmax proposal for the geometry
                # diagnostic. Computed independently of `iou_mode` so the diag
                # is identical regardless of the recall metric being run.
                if self.geometry_diag_enabled:
                    bev = _bev_iou(gt["rect"], p["rect"])
                    if bev > best_bev:
                        best_bev = bev
                        best_bev_p = p
            for th in IOU3D_THRESHOLDS:
                if best_iou >= th:
                    self._ensure(self.iou_hits[th], cls, dbin)
                    self.iou_hits[th][cls][dbin] += 1

            # ---- Geometry diagnostic capture (additive) ----
            if self.geometry_diag_enabled:
                self._record_geometry(gt, best_bev, best_bev_p, cls, dbin)

        # ---- Oracle relabelling for the mAP ceiling --------------
        # Dispatched by self.oracle_mode:
        #   "maxiou"  → score-ordered, GT-deduplicated, highest-IoU assignment.
        #               No 4 m radius cutoff (a positive IoU is already a much
        #               tighter constraint than 4 m centre distance).
        #   "nearest" → legacy BEV-centre argmin with 4 m cutoff, NO dedup
        #               (preserved for back-compat).
        oracle_pred: list[dict] = []
        if self.oracle_mode == "maxiou":
            # Score-descending order so a high-confidence proposal claims the
            # best GT first; subsequent proposals see that GT as consumed.
            ordered = sorted(cls_proposals, key=lambda q: float(q.get("score", 0.0)),
                             reverse=True)
            assigned: set[int] = set()
            n_gt = len(cls_gts)
            for p in ordered:
                if n_gt == 0:
                    self.n_oracle_dropped_no_gt += 1
                    continue
                best_iou = -1.0
                best_j = -1
                for j in range(n_gt):
                    if j in assigned:
                        continue
                    iou = self._pairwise_iou(cls_gts[j], p)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j < 0:
                    self.n_oracle_dropped_gt_consumed += 1
                    continue
                if best_iou <= self.oracle_iou_min:
                    # Strict per-proposal upper bound: a proposal whose best
                    # IoU is exactly 0 cannot serve any GT under official
                    # nuScenes matching either — drop it instead of inflating
                    # the ceiling with a free TP at far-off centre distance.
                    self.n_oracle_dropped_low_iou += 1
                    continue
                assigned.add(best_j)
                self.n_oracle_assigned += 1
                gt_match = cls_gts[best_j]
                oracle_pred.append(self._proposal_to_detbox(
                    p, gt_match, ego_pose, sample_token,
                ))
        else:  # "nearest" (legacy)
            gt_xy = np.asarray([g["centroid_global"][:2] for g in cls_gts],
                               dtype=np.float64) if cls_gts else None
            for p in cls_proposals:
                if gt_xy is None or gt_xy.size == 0:
                    continue
                dists = np.linalg.norm(gt_xy - p["centroid_global"][:2], axis=1)
                j = int(np.argmin(dists))
                if dists[j] > ORACLE_MATCH_RADIUS_M:
                    continue
                gt_match = cls_gts[j]
                oracle_pred.append(self._proposal_to_detbox(
                    p, gt_match, ego_pose, sample_token,
                ))
        self.oracle_pred[sample_token] = oracle_pred
        # Store GT in nuScenes-devkit DetectionBox format too.
        self.oracle_gt[sample_token] = [{
            "sample_token": sample_token,
            "translation": gt["translation"],
            "size": gt["size"],
            "rotation": gt["rotation"],
            "velocity": [0.0, 0.0],
            "ego_translation": [float(ego_pose[0, 3]), float(ego_pose[1, 3]),
                                float(ego_pose[2, 3])],
            "num_pts": 1,
            "detection_name": gt["cls_name"],
            "detection_score": -1.0,
            "attribute_name": "",
        } for gt in cls_gts]

    def _record_geometry(self, gt: dict, best_bev: float,
                         best_bev_p: Optional[dict], cls: str, dbin: str) -> None:
        """Log the BEV-IoU-argmax proposal for one GT: signed center/size
        errors, z-overlap fraction, and three IoU variants (BEV, true 3D, and
        a z-corrected 3D that reinterprets the cached proposal z as the box
        BOTTOM and lifts it by +dz/2). Purely additive — touches no counter
        consumed by `summary()`.
        """
        self.n_geo_gt += 1
        # Every valid-dbin GT contributes to the recall denominator; a GT with
        # no BEV-overlapping proposal scores 0 on all three IoU variants.
        if best_bev_p is None or best_bev <= 0.0:
            self.n_geo_gt_no_overlap += 1
            self.geo_recall["bev"].append(0.0)
            self.geo_recall["iou3d"].append(0.0)
            self.geo_recall["iou3d_zcorr"].append(0.0)
            self.geo_recall["cls"].append(cls)
            self.geo_recall["dbin"].append(dbin)
            return

        p = best_bev_p
        iou3d = _iou_3d(gt, p)
        # z-corrected counterfactual: the cached center cz is reinterpreted as
        # the box BOTTOM, so the corrected box spans [cz, cz + dz].
        dz = max(float(p["size_z"]), 1e-2)
        cz = 0.5 * (float(p["z_bottom"]) + float(p["z_top"]))
        p_zc = {"rect": p["rect"], "z_bottom": cz, "z_top": cz + dz}
        iou3d_zc = _iou_3d(gt, p_zc)

        pc = p["centroid_global"]
        gc = gt["centroid_global"]
        ex = float(pc[0] - gc[0])
        ey = float(pc[1] - gc[1])
        ez = float(pc[2] - gc[2])
        # Center error after the z-correction (corrected center = cz + dz/2).
        ez_zcorr = ez + dz / 2.0
        el = float(p["size_xy"][0] - gt["size_xy"][0])
        ew = float(p["size_xy"][1] - gt["size_xy"][1])
        eh = float(p["size_z"] - gt["size_z"])
        z_inter = max(0.0, min(float(p["z_top"]), float(gt["z_top"]))
                      - max(float(p["z_bottom"]), float(gt["z_bottom"])))
        gt_h = max(float(gt["z_top"]) - float(gt["z_bottom"]), 1e-9)
        z_overlap_frac = float(z_inter / gt_h)

        self.geo_recall["bev"].append(float(best_bev))
        self.geo_recall["iou3d"].append(float(iou3d))
        self.geo_recall["iou3d_zcorr"].append(float(iou3d_zc))
        self.geo_recall["cls"].append(cls)
        self.geo_recall["dbin"].append(dbin)

        gp = self.geo_pairs
        gp["cls"].append(cls)
        gp["dbin"].append(dbin)
        gp["bev_iou"].append(float(best_bev))
        gp["iou3d"].append(float(iou3d))
        gp["iou3d_zcorr"].append(float(iou3d_zc))
        gp["ex"].append(ex)
        gp["ey"].append(ey)
        gp["ez"].append(ez)
        gp["ez_zcorr"].append(ez_zcorr)
        gp["el"].append(el)
        gp["ew"].append(ew)
        gp["eh"].append(eh)
        gp["z_overlap_frac"].append(z_overlap_frac)

    def geometry_summary(self) -> dict:
        """Aggregate the additive geometry capture into error statistics, IoU
        histograms, and recall-vs-threshold curves (BEV, true 3D, z-corrected
        3D). Returns ``{}`` when geometry capture was disabled.

        error_stats are computed over the BEV-overlapping paired set; the
        histograms and recall curves span ALL valid-dbin GTs (no-overlap GTs
        sit in the 0-bin), so the recall denominator matches recall_iou3d.
        """
        if not self.geometry_diag_enabled:
            return {}

        def _stats(vals) -> dict:
            a = np.asarray(vals, dtype=np.float64)
            if a.size == 0:
                return {"n": 0}
            return {
                "n": int(a.size),
                "mean": float(a.mean()), "std": float(a.std()),
                "p10": float(np.percentile(a, 10)),
                "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max()),
            }

        edges = np.asarray(GEO_IOU_BIN_EDGES, dtype=np.float64)

        def _hist(vals) -> dict:
            counts, _ = np.histogram(np.asarray(vals, dtype=np.float64), bins=edges)
            return {"edges": [float(e) for e in edges],
                    "counts": [int(c) for c in counts]}

        den = self.n_geo_gt

        def _recall_curve(vals) -> dict:
            a = np.asarray(vals, dtype=np.float64)
            return {f"{t:.2f}": (float((a >= t).sum()) / den) if den else None
                    for t in GEO_RECALL_TAUS}

        gp = self.geo_pairs
        err_keys = ("ex", "ey", "ez", "ez_zcorr", "el", "ew", "eh",
                    "z_overlap_frac", "bev_iou", "iou3d", "iou3d_zcorr")
        overall_errors = {k: _stats(gp[k]) for k in err_keys}

        cls_arr = np.asarray(gp["cls"]) if gp["cls"] else np.asarray([], dtype=object)
        per_class_errors: dict[str, dict] = {}
        for cls in sorted(set(gp["cls"])):
            mask = cls_arr == cls
            per_class_errors[cls] = {
                k: _stats(np.asarray(gp[k], dtype=np.float64)[mask])
                for k in err_keys
            }

        return {
            "enabled": True,
            "pairing": "bev_iou_argmax",
            "n_gt_total": int(self.n_geo_gt),
            "n_gt_with_bev_overlap": int(self.n_geo_gt - self.n_geo_gt_no_overlap),
            "n_gt_no_bev_overlap": int(self.n_geo_gt_no_overlap),
            "recall_denominator": int(den),
            "error_stats": {
                "overall": overall_errors,
                "per_class": per_class_errors,
            },
            "histograms": {
                "bev_iou": _hist(self.geo_recall["bev"]),
                "iou3d": _hist(self.geo_recall["iou3d"]),
                "iou3d_zcorr": _hist(self.geo_recall["iou3d_zcorr"]),
            },
            "recall_curves": {
                "bev": _recall_curve(self.geo_recall["bev"]),
                "iou3d": _recall_curve(self.geo_recall["iou3d"]),
                "iou3d_zcorr": _recall_curve(self.geo_recall["iou3d_zcorr"]),
            },
        }

    def summary(self) -> dict:
        """Aggregate to per-class/per-dbin recall tables. Top-level
        ``recall_macro`` averages over classes that have at least one GT in
        the corresponding bin (avoids inflating recall with empty classes).
        """
        per_class: dict[str, dict] = {}
        all_dbins = sorted({db for c in self.gt_counts.values() for db in c})

        def _rates(hit_store):
            class_rates: dict[str, dict[str, float]] = {}
            macro_per_bin: dict[str, list[float]] = {}
            overall_num = 0
            overall_den = 0
            for cls, by_bin in self.gt_counts.items():
                cls_d = {}
                for dbin, n_gt in by_bin.items():
                    n_hit = hit_store.get(cls, {}).get(dbin, 0)
                    if n_gt > 0:
                        rate = n_hit / n_gt
                        cls_d[dbin] = {"recall": float(rate), "n_gt": int(n_gt),
                                       "n_hit": int(n_hit)}
                        macro_per_bin.setdefault(dbin, []).append(rate)
                        overall_num += n_hit
                        overall_den += n_gt
                class_rates[cls] = cls_d
            macro = {db: float(np.mean(v)) if v else None
                     for db, v in macro_per_bin.items()}
            overall = float(overall_num / overall_den) if overall_den else None
            return {
                "per_class": class_rates,
                "macro_per_dbin": macro,
                "micro_overall": overall,
            }

        return {
            "n_samples_seen": int(self.n_samples_seen),
            "n_samples_missing_cache": int(self.n_samples_missing_cache),
            "n_proposals_seen": int(self.n_proposals_seen),
            "n_proposals_unlabeled": int(self.n_proposals_unlabeled),
            "n_gt_total": int(self.n_gt_total),
            "dbins_present": all_dbins,
            "iou_mode": self.iou_mode,
            "oracle_mode": self.oracle_mode,
            "oracle_iou_min": self.oracle_iou_min,
            "oracle_diag": {
                "n_assigned": int(self.n_oracle_assigned),
                "n_dropped_no_gt": int(self.n_oracle_dropped_no_gt),
                "n_dropped_low_iou": int(self.n_oracle_dropped_low_iou),
                "n_dropped_gt_consumed": int(self.n_oracle_dropped_gt_consumed),
            },
            "recall_box": {
                f"thr_{t}m": _rates(self.box_hits[t])
                for t in BOX_DIST_THRESHOLDS_M
            },
            "recall_iou3d": {
                f"thr_{t:.2f}": _rates(self.iou_hits[t])
                for t in IOU3D_THRESHOLDS
            },
        }


# ---------------------------------------------------------------------------
# Oracle mAP driver
# ---------------------------------------------------------------------------

def _eval_oracle_map(acc: RecallAccumulator, out_dir: Path, label: str) -> dict:
    """Run the nuScenes-devkit eval (mirrors `centerpoint_native_map_sanity`).

    This calls the EXISTING evaluator entry — does not modify it.
    """
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate

    pred_eb, gt_eb = EvalBoxes(), EvalBoxes()
    n_pred = n_gt = 0
    for tok, preds in acc.oracle_pred.items():
        pred_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in preds])
        gt_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in acc.oracle_gt.get(tok, [])])
        n_pred += len(preds)
        n_gt += len(acc.oracle_gt.get(tok, []))

    eval_dir = out_dir / f"oracle_map_{label}"
    summary = nu_evaluate(pred_boxes=pred_eb, gt_boxes=gt_eb,
                          output_dir=str(eval_dir),
                          config_name="detection_cvpr_2019")
    return {
        "label": label,
        "n_pred_boxes": int(n_pred),
        "n_gt_boxes": int(n_gt),
        "mAP": summary.get("mean_ap"),
        "NDS": summary.get("nd_score"),
        "per_class_AP": summary.get("label_aps"),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _list_val_scenes(nusc: NuScenes) -> list[str]:
    name2tok = {s["name"]: s["token"] for s in nusc.scene}
    return [name2tok[n] for n in VAL_SCENES if n in name2tok]


def _scene_sample_tokens(nusc: NuScenes, scene_token: str) -> list[str]:
    scene = nusc.get("scene", scene_token)
    toks, cur = [], scene["first_sample_token"]
    while cur:
        toks.append(cur)
        cur = nusc.get("sample", cur)["next"]
    return toks


def _new_run_dir(project_root: Path, exp: str = "outdoor_proposal_recall") -> Path:
    date = time.strftime("%Y-%m-%d")
    existing = list(project_root.glob(f"results/{date}_{exp}_v*"))
    n = len(existing) + 1
    run = project_root / "results" / f"{date}_{exp}_v{n:02d}"
    (run / "outputs").mkdir(parents=True, exist_ok=True)
    return run


def _resolve_cache(cli_path: Optional[str], default_under_results: str,
                   project_root: Path) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    return (project_root / "results" / default_under_results).resolve()


def main():
    project_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=None,
                    help="γ-native CenterPoint proposal-pickle directory.")
    ap.add_argument("--detguided-cache", default=None,
                    help="detguided proposal-pickle directory.")
    ap.add_argument("--hybrid-cache", default=None,
                    help="Hybrid Proposal v2 pickle directory (<token>.hybrid.pkl).")
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--scene-limit", type=int, default=0,
                    help="If > 0, only the first N val scenes are evaluated.")
    ap.add_argument("--skip-oracle-map", action="store_true",
                    help="Compute only Recall_box + Recall_iou3d; skip the "
                         "DetectionEval pass.")
    ap.add_argument("--sources", default="gamma,detguided",
                    help="Comma list of sources to evaluate.")
    ap.add_argument("--output-name", default=None,
                    help="Override the auto-versioned run dir name (used by "
                         "tests; production should let the script auto-pick).")
    ap.add_argument(
        "--iou-mode",
        choices=("iou3d", "bev"),
        default="iou3d",
        help="IoU metric used by both Recall_iou3d and the maxiou oracle. "
             "'iou3d' (default) = true 3D volume IoU (BEV intersection × z "
             "overlap). 'bev' = legacy BEV-only IoU.",
    )
    ap.add_argument(
        "--oracle-mode",
        choices=("maxiou", "nearest"),
        default="maxiou",
        help="Oracle relabel policy. 'maxiou' (default) = highest-IoU GT "
             "with GT deduplication (true per-proposal ceiling). 'nearest' "
             "= legacy BEV-centre argmin with 4 m cutoff and NO dedup.",
    )
    ap.add_argument(
        "--oracle-iou-min",
        type=float,
        default=0.0,
        help="Minimum IoU (per --iou-mode) required for an oracle proposal "
             "→ GT assignment under --oracle-mode maxiou. 0.0 lets any "
             "positive overlap count; >0 enforces a stricter ceiling.",
    )
    ap.add_argument(
        "--skip-geometry-diag",
        action="store_true",
        help="Disable the additive geometry diagnostic (per-GT BEV-argmax "
             "center/size errors, z-overlap, IoU histograms, recall curves, "
             "and the z-corrected counterfactual). Enabled by default.",
    )
    args = ap.parse_args()

    gamma_cache = _resolve_cache(
        args.gamma_cache,
        "outdoor_native_temporal_cpcache_thr000_single",
        project_root,
    )
    detguided_cache = _resolve_cache(
        args.detguided_cache,
        "outdoor_detguided_cpcache_thr000_full150",
        project_root,
    )
    hybrid_cache = _resolve_cache(
        args.hybrid_cache,
        "outdoor_hybrid_cpcache_gravity_nusc10",
        project_root,
    )
    print(f"[probe] γ cache       : {gamma_cache}")
    print(f"[probe] detguided     : {detguided_cache}")
    print(f"[probe] hybrid        : {hybrid_cache}")
    sources_req = [s.strip() for s in args.sources.split(",") if s.strip()]
    need = {"gamma": gamma_cache, "detguided": detguided_cache, "hybrid": hybrid_cache}
    for s in sources_req:
        c = need.get(s)
        if c is not None and not c.exists():
            raise SystemExit(f"cache directory missing for source {s!r}: {c}")

    out_dir = (project_root / "results" / args.output_name
               if args.output_name else _new_run_dir(project_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] output dir    : {out_dir}")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    accumulators: dict[str, RecallAccumulator] = {
        s: RecallAccumulator(
            iou_mode=args.iou_mode,
            oracle_mode=args.oracle_mode,
            oracle_iou_min=float(args.oracle_iou_min),
            geometry_diag=(not args.skip_geometry_diag),
        )
        for s in sources
    }
    print(
        f"[probe] iou_mode={args.iou_mode!r} oracle_mode={args.oracle_mode!r} "
        f"oracle_iou_min={args.oracle_iou_min}",
        flush=True,
    )

    print("[probe] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version,
                    dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[probe] val scenes    : {len(val_scenes)}", flush=True)

    t0 = time.time()
    for si, sc_tok in enumerate(val_scenes):
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            # Common per-sample geometry (ego_pose, lidar↔ego).
            sample = nusc.get("sample", sa_tok)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"],
                                        Quaternion(ego_rec["rotation"]))
            T_lidar_to_ego = transform_matrix(lidar_cs["translation"],
                                              Quaternion(lidar_cs["rotation"]))
            ego_quat = Quaternion(matrix=ego_pose[:3, :3])
            lidar_to_ego_q = Quaternion(matrix=T_lidar_to_ego[:3, :3])

            gts = _gt_records(nusc, sa_tok, ego_pose)
            for src in sources:
                if src == "gamma":
                    raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
                elif src == "detguided":
                    raw = _load_cached_proposals(detguided_cache, sa_tok, suffix=".detguided")
                elif src == "hybrid":
                    raw = _load_cached_proposals(hybrid_cache, sa_tok, suffix=".hybrid")
                else:
                    raise SystemExit(f"unknown --sources entry: {src!r}")
                if raw is None:
                    accumulators[src].n_samples_missing_cache += 1
                    continue
                # Count unlabeled before filtering (so we can report the
                # detguided OV class loss separately from geometry).
                accumulators[src].n_proposals_unlabeled += sum(
                    1 for p in raw if p.get("cls_name") not in NUSC_10_SET
                )
                props = _proposals_to_global(raw, ego_pose, lidar_to_ego_q, ego_quat)
                accumulators[src].record_sample(props, gts, ego_pose, sa_tok)
        if (si + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[probe] scene {si+1}/{len(val_scenes)} done — "
                  f"elapsed {elapsed:.0f}s", flush=True)

    # Aggregate summaries.
    payload: dict = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "detguided_cache": str(detguided_cache),
            "n_val_scenes": len(val_scenes),
            "scene_limit": int(args.scene_limit),
            "box_distance_thresholds_m": list(BOX_DIST_THRESHOLDS_M),
            "iou3d_thresholds": list(IOU3D_THRESHOLDS),
            "distance_bins": [[lo, hi] for lo, hi in DIST_BINS],
            "oracle_match_radius_m": ORACLE_MATCH_RADIUS_M,
            "iou_mode": args.iou_mode,
            "oracle_mode": args.oracle_mode,
            "oracle_iou_min": float(args.oracle_iou_min),
            "geometry_diag": (not args.skip_geometry_diag),
        },
        "per_source": {},
    }
    for src, acc in accumulators.items():
        s = acc.summary()
        if not args.skip_oracle_map:
            try:
                s["oracle_map"] = _eval_oracle_map(acc, out_dir, src)
            except Exception as exc:
                s["oracle_map"] = {"error": repr(exc)}
        if not args.skip_geometry_diag:
            s["geometry_diag"] = acc.geometry_summary()
        payload["per_source"][src] = s

    out_file = out_dir / "recall_probe.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[probe] wrote {out_file}")

    # Console summary (head-line only).
    print("\n=== Recall probe head-line ===")
    for src in sources:
        s = payload["per_source"][src]
        print(f"  {src}: micro@2m={s['recall_box']['thr_2.0m']['micro_overall']!r} "
              f"micro@IoU0.5={s['recall_iou3d']['thr_0.50']['micro_overall']!r} "
              f"oracle_mAP={s.get('oracle_map', {}).get('mAP', 'n/a')}")
        gd = s.get("geometry_diag")
        if gd:
            rc = gd["recall_curves"]
            ez = gd["error_stats"]["overall"]["ez"]
            print(f"        geom: ez_mean={ez.get('mean')!r} ez_p50={ez.get('p50')!r} "
                  f"| recall@IoU0.5  bev={rc['bev'].get('0.50')!r} "
                  f"iou3d={rc['iou3d'].get('0.50')!r} "
                  f"iou3d_zcorr={rc['iou3d_zcorr'].get('0.50')!r}")


if __name__ == "__main__":
    main()
