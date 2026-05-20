"""Streaming nuScenes evaluator — Task 2.1 Outdoor Stage A.

Parallel to :class:`StreamingScanNetEvaluator`, but adapted to:
- nuScenes per-sample stream (~2 Hz) instead of per-frame ScanNet sequence
- γ CenterPoint LiDAR proposals (per-sample, no built-in tracking)
- YOLO-World labels from 6 surround cameras
- Same temporal layer (M11 / M12-fixed hooks via hooks_streaming)
- nuScenes-devkit DetectionEval for mean AP / NDS

Key design departures from Indoor (documented in
`docs/task_2_1_outdoor_stage_a_notes.md`):
1. Cross-sample association: CenterPoint produces fresh proposals each
   sample. We assign stable ``global_id`` via greedy centroid-distance
   + class-match within ``association_distance_m`` (default 2.0).
   Without stable IDs, M11/M12 ("confirmed after K appearances")
   cannot work.
2. YOLO-World labeling: project the 3D proposal centroid into each of
   6 cameras, find the closest YOLO-World 2D bbox per in-frame camera,
   take the highest-score match across cameras. Only nuScenes-10 class
   names survive — open-vocab extensions (tree, pole, ...) are dropped
   because they have no GT counterparts in nuScenes detection.
3. 3D box geometry (size, rotation, velocity) comes from CenterPoint;
   only the class label is overridden by YOLO-World.
4. Per-sample pred_history snapshot format matches Indoor exactly:
   ``list[dict[global_id, class_idx]]`` so Indoor's
   ``metrics.label_switch_count`` / ``metrics.time_to_confirm`` apply
   unchanged.
"""
from __future__ import annotations

import json
import os
import os.path as osp
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from pyquaternion import Quaternion

from dataloaders.nuscenes_loader import NuScenesLoader
from method_scannet.streaming.hooks_streaming import (
    install_method_streaming,
    uninstall_all_streaming,
)
from method_scannet.streaming.metrics import label_switch_count, time_to_confirm

# CenterPoint is loaded lazily so that import doesn't pull mmdet3d on a
# CPU-only login node.
NUSC_10 = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)
NUSC_10_SET = set(NUSC_10)
DEFAULT_ASSOC_DIST_M = 2.0
# Per-scene global-id stride: scene k's ids live in [k*STRIDE, (k+1)*STRIDE).
# 1e6 comfortably exceeds any per-scene proposal-instance count.
SCENE_ID_STRIDE = 1_000_000
DEFAULT_IOU_THRESHOLD = 0.3


def _set_mm_scope(name: str) -> None:
    """Switch mmengine's default registry scope.

    OpenYolo3D loads mmyolo (sets active scope to mmyolo); mmdet3d
    CenterPoint inference requires mmdet3d scope to resolve
    LoadPointsFromFile etc. We toggle scope at every inference
    boundary. Idempotent / safe to call repeatedly.
    """
    try:
        from mmengine.registry import init_default_scope
        init_default_scope(name)
        return
    except Exception:
        pass
    # Fallback path for mmengine versions where init_default_scope
    # cannot switch an existing active scope.
    try:
        from mmengine.registry import DefaultScope
        DefaultScope._instance_dict.pop("_default_scope", None)
        DefaultScope.get_instance("_default_scope", scope_name=name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-sample evaluator state — parallel to RunningInstanceLabeler.
# ---------------------------------------------------------------------------
class NuScenesRunningLabeler:
    """Simple per-global_id label histogram + snapshot.

    Stores votes from each (global_id, frame_class_assignment). Snapshot
    returns the argmax-count class for each requested id. No vertex
    masks, no projection — purely on assigned-label-per-sample inputs.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = int(num_classes)
        self.counts: dict[int, np.ndarray] = {}

    def add_vote(self, global_id: int, cls_idx: int, weight: float = 1.0) -> None:
        if cls_idx < 0 or cls_idx >= self.num_classes:
            return
        h = self.counts.setdefault(int(global_id),
                                   np.zeros(self.num_classes, dtype=np.float64))
        h[int(cls_idx)] += float(weight)

    def snapshot(self, instance_ids) -> dict[int, int]:
        out: dict[int, int] = {}
        for iid in instance_ids:
            iid = int(iid)
            h = self.counts.get(iid)
            if h is None or float(h.sum()) == 0.0:
                out[iid] = -1
            else:
                out[iid] = int(np.argmax(h))
        return out

    def best_label(self, global_id: int) -> int:
        h = self.counts.get(int(global_id))
        if h is None or float(h.sum()) == 0.0:
            return -1
        return int(np.argmax(h))

    def reset(self) -> None:
        self.counts.clear()


# ---------------------------------------------------------------------------
# Cross-sample tracker.
# ---------------------------------------------------------------------------
class CentroidAssociator:
    """Greedy centroid-distance, class-aware proposal tracker.

    Maintains the set of active global_ids with their last-seen centroid
    (ego frame) and class. For each new proposal, finds the closest
    active id of the same class within ``threshold_m`` and reuses it;
    otherwise allocates a fresh global_id.

    State is per-scene (caller resets between scenes). Active ids carry
    forward across consecutive samples; an id that misses too many
    samples is dropped from the active set so it doesn't shadow new
    objects, but its identity is *not* re-issued.
    """

    def __init__(self, threshold_m: float = DEFAULT_ASSOC_DIST_M,
                 max_age: int = 5, id_offset: int = 0) -> None:
        self.threshold_m = float(threshold_m)
        self.max_age = int(max_age)
        self.id_offset = int(id_offset)
        self._active: dict[int, dict] = {}
        self._next_id: int = int(id_offset)

    def reset(self) -> None:
        self._active.clear()
        self._next_id = int(self.id_offset)

    def step(self, proposals: list[dict]) -> list[int]:
        """Assign global_ids for one sample's proposals.

        ``proposals`` is the list returned by CenterPointProposalGenerator
        (each dict has ``cls_name`` and ``centroid_ego``).
        Returns: list[int] of global_ids parallel to ``proposals``.
        """
        # Age all active ids by 1; drop stale.
        for gid in list(self._active.keys()):
            self._active[gid]["age"] += 1
            if self._active[gid]["age"] > self.max_age:
                self._active.pop(gid, None)

        if not proposals:
            return []

        gid_assignments: list[Optional[int]] = [None] * len(proposals)
        used: set[int] = set()

        # Greedy match: order proposals by descending score so high-confidence
        # detections claim their identity first.
        order = sorted(range(len(proposals)),
                       key=lambda i: -proposals[i].get("score", 0.0))
        for j in order:
            p = proposals[j]
            cls = p["cls_name"]
            c = np.asarray(p["centroid_ego"], dtype=np.float64)
            best_gid, best_d = None, self.threshold_m + 1e-9
            for gid, st in self._active.items():
                if gid in used:
                    continue
                if st["cls"] != cls:
                    continue
                d = float(np.linalg.norm(c[:2] - st["centroid"][:2]))
                if d < best_d:
                    best_d = d
                    best_gid = gid
            if best_gid is not None:
                gid_assignments[j] = best_gid
                used.add(best_gid)
                self._active[best_gid]["centroid"] = c
                self._active[best_gid]["age"] = 0
            else:
                new_gid = self._next_id
                self._next_id += 1
                gid_assignments[j] = new_gid
                self._active[new_gid] = {"cls": cls, "centroid": c, "age": 0}

        return [int(g) for g in gid_assignments]  # type: ignore


# ---------------------------------------------------------------------------
# YOLO-World per-camera label assignment.
# ---------------------------------------------------------------------------
def _project_to_camera(centroid_ego: np.ndarray,
                       cam_to_ego: np.ndarray,
                       intrinsic: np.ndarray,
                       image_hw: tuple[int, int]) -> Optional[tuple[float, float]]:
    """Project an ego-frame centroid into a camera. Return (px, py) or
    None if behind the camera or outside image bounds.
    """
    T_ego_to_cam = np.linalg.inv(cam_to_ego)
    p_h = np.concatenate([centroid_ego[:3], [1.0]])
    p_cam = T_ego_to_cam @ p_h
    z = float(p_cam[2])
    if z <= 0.1:
        return None
    pix = intrinsic @ p_cam[:3]
    u = float(pix[0] / pix[2])
    v = float(pix[1] / pix[2])
    H, W = image_hw
    if not (0.0 <= u < W and 0.0 <= v < H):
        return None
    return (u, v)


def _box_corners_ego(centroid_ego: np.ndarray, size: tuple[float, float, float],
                     box_q_ego: "Quaternion") -> np.ndarray:
    """8 corners of an oriented 3D box in ego frame. size = (dx, dy, dz)
    matching CenterPoint's bbox ordering."""
    dx, dy, dz = float(size[0]), float(size[1]), float(size[2])
    xs = np.array([dx, dx, dx, dx, -dx, -dx, -dx, -dx]) / 2.0
    ys = np.array([dy, dy, -dy, -dy, dy, dy, -dy, -dy]) / 2.0
    zs = np.array([dz, -dz, dz, -dz, dz, -dz, dz, -dz]) / 2.0
    corners_local = np.stack([xs, ys, zs], axis=1)  # (8,3)
    R = box_q_ego.rotation_matrix                    # (3,3)
    return (R @ corners_local.T).T + np.asarray(centroid_ego[:3], dtype=np.float64)


def _project_corners_to_2dbbox(corners_ego: np.ndarray, cam_to_ego: np.ndarray,
                               intrinsic: np.ndarray,
                               image_hw: tuple[int, int]) -> Optional[list[float]]:
    """Project 8 ego-frame corners into a camera; return the axis-aligned
    2D bbox [x1,y1,x2,y2] clipped to the image, or None if too few corners
    are in front of the camera / inside the image."""
    T_ego_to_cam = np.linalg.inv(cam_to_ego)
    H, W = image_hw
    pts2d = []
    for c in corners_ego:
        p_h = np.concatenate([c[:3], [1.0]])
        p_cam = T_ego_to_cam @ p_h
        z = float(p_cam[2])
        if z <= 0.1:
            continue
        pix = intrinsic @ p_cam[:3]
        u = float(pix[0] / pix[2])
        v = float(pix[1] / pix[2])
        pts2d.append((u, v))
    if len(pts2d) < 2:
        return None
    arr = np.asarray(pts2d, dtype=np.float64)
    x1, y1 = float(arr[:, 0].min()), float(arr[:, 1].min())
    x2, y2 = float(arr[:, 0].max()), float(arr[:, 1].max())
    # Reject boxes entirely off-image.
    if x2 < 0 or y2 < 0 or x1 >= W or y1 >= H:
        return None
    x1 = max(0.0, x1); y1 = max(0.0, y1)
    x2 = min(float(W - 1), x2); y2 = min(float(H - 1), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou_2d(a: list[float], b: np.ndarray) -> float:
    """IoU between axis-aligned 2D boxes a=[x1,y1,x2,y2], b=[x1,y1,x2,y2]."""
    ix1 = max(a[0], float(b[0])); iy1 = max(a[1], float(b[1]))
    ix2 = min(a[2], float(b[2])); iy2 = min(a[3], float(b[3]))
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _yolo_label_for_proposal_iou(box2d_per_cam: dict, cam_outputs: dict,
                                 text_prompts: list[str],
                                 iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                                 ) -> tuple[Optional[str], float, float]:
    """IoU-based YOLO label fusion (Fix 3, Task 2.2).

    For each camera where the proposal's projected 3D-box footprint is
    available, compute IoU against every YOLO 2D bbox; keep the match
    with the highest IoU across all cameras (above ``iou_threshold``).
    Returns (class_name_or_None, yolo_score, best_iou).
    """
    best_iou, best_score, best_cls = iou_threshold, -1.0, None
    for cam_name, box2d in box2d_per_cam.items():
        if box2d is None:
            continue
        out = cam_outputs.get(cam_name)
        if out is None:
            continue
        bboxes = out["bbox"]
        labels = out["labels"]
        scores = out["scores"]
        if bboxes is None or len(bboxes) == 0:
            continue
        bb = np.asarray(bboxes, dtype=np.float64)
        sc = np.asarray(scores, dtype=np.float64)
        lb = np.asarray(labels)
        for k in range(bb.shape[0]):
            iou = _iou_2d(box2d, bb[k])
            if iou < best_iou:
                continue
            cls_idx = int(lb[k])
            if cls_idx < 0 or cls_idx >= len(text_prompts):
                continue
            # Tie-break by IoU first, then by YOLO score.
            if iou > best_iou or (abs(iou - best_iou) < 1e-9 and float(sc[k]) > best_score):
                best_iou = iou
                best_score = float(sc[k])
                best_cls = text_prompts[cls_idx]
    if best_cls is None:
        return None, -1.0, 0.0
    return best_cls, best_score, best_iou


def _yolo_label_for_proposal(centroid_ego: np.ndarray,
                             cam_outputs: dict,
                             text_prompts: list[str],
                             pixel_match_radius: float = 80.0) -> tuple[Optional[str], float]:
    """Assign a YOLO-World class to a proposal.

    For each in-frame camera, find the closest YOLO 2D bbox center to
    the projected proposal centroid (within ``pixel_match_radius`` px),
    then keep the highest-score match across cameras.
    Returns: (class_name_or_None, score). Class name comes from
    ``text_prompts`` and is filtered to nuScenes-10 by the caller.
    """
    best_score, best_cls = -1.0, None
    for cam_name, out in cam_outputs.items():
        if out is None:
            continue
        proj = out["projection"]   # (px, py) or None
        if proj is None:
            continue
        u, v = proj
        bboxes = out["bbox"]           # (n, 4) [x1, y1, x2, y2]
        labels = out["labels"]         # (n,) int
        scores = out["scores"]         # (n,) float
        if bboxes is None or len(bboxes) == 0:
            continue
        bb = np.asarray(bboxes, dtype=np.float64)
        cx = 0.5 * (bb[:, 0] + bb[:, 2])
        cy = 0.5 * (bb[:, 1] + bb[:, 3])
        dist = np.hypot(cx - u, cy - v)
        within = dist <= pixel_match_radius
        if not within.any():
            continue
        idxs = np.where(within)[0]
        # Of matching bboxes, take the one with highest YOLO score.
        sc = np.asarray(scores, dtype=np.float64)
        best_local = int(idxs[np.argmax(sc[idxs])])
        s = float(sc[best_local])
        cls_idx = int(np.asarray(labels)[best_local])
        if cls_idx < 0 or cls_idx >= len(text_prompts):
            continue
        cls_name = text_prompts[cls_idx]
        if s > best_score:
            best_score = s
            best_cls = cls_name
    return best_cls, best_score


def _detection_box_dict(global_id: int,
                        sample_token: str,
                        bbox_lidar: list[float],
                        centroid_global: np.ndarray,
                        ego_translation: np.ndarray,
                        rotation_global_wxyz: list[float],
                        detection_name: str,
                        score: float) -> dict:
    """Build a DetectionBox-shaped dict for nuScenes-devkit eval.

    bbox_lidar: [x, y, z, dx, dy, dz, yaw, (vx, vy)] from CenterPoint.
    We use dx, dy, dz as size in nuScenes convention (w, l, h) — exactly
    matches CenterPoint output ordering.
    """
    if len(bbox_lidar) >= 9:
        vx, vy = float(bbox_lidar[7]), float(bbox_lidar[8])
    else:
        vx, vy = 0.0, 0.0
    attr_default = "vehicle.moving" if detection_name in (
        "car", "truck", "bus", "trailer", "construction_vehicle", "motorcycle",
        "bicycle"
    ) else ""
    return {
        "sample_token": sample_token,
        "translation": [float(x) for x in centroid_global],
        "size": [float(bbox_lidar[3]), float(bbox_lidar[4]), float(bbox_lidar[5])],
        "rotation": rotation_global_wxyz,  # list[float] length 4
        "velocity": [vx, vy],
        "ego_translation": [float(x) for x in ego_translation],
        "num_pts": 1,  # placeholder; eval doesn't use this if range filter passes
        "detection_name": detection_name,
        "detection_score": float(score),
        "attribute_name": attr_default,
    }


# ---------------------------------------------------------------------------
# Stage B — β1 (HDBSCAN) proposal conversion + deployable hybrid union.
# ---------------------------------------------------------------------------
def _beta1_clusters_to_proposals(beta1_out: dict) -> list[dict]:
    """Convert LiDARProposalGenerator output (ego-frame, axis-aligned,
    class-agnostic clusters) into the proposal-dict shape the evaluator
    consumes. β1 has no class (YOLO labels downstream) and no detection
    score (we use a point-count pseudo-score). Boxes are axis-aligned in
    ego (yaw=0); downstream applies the small lidar→ego tilt.
    """
    centroids = beta1_out.get("cluster_centroids")
    bboxes = beta1_out.get("cluster_bbox")
    sizes = beta1_out.get("cluster_sizes")
    props: list[dict] = []
    if centroids is None or len(centroids) == 0:
        return props
    for k in range(centroids.shape[0]):
        mn = np.asarray(bboxes[k][:3], dtype=np.float64)
        mx = np.asarray(bboxes[k][3:], dtype=np.float64)
        center = (mn + mx) / 2.0
        dims = np.clip(mx - mn, 1e-3, None)
        npts = int(sizes[k]) if sizes is not None else 0
        props.append({
            "cls_name": "object",       # class assigned by YOLO-World downstream
            "cls_idx": -1,
            "score": float(min(1.0, npts / 100.0)),  # point-count pseudo-score
            "bbox_lidar": [float(center[0]), float(center[1]), float(center[2]),
                           float(dims[0]), float(dims[1]), float(dims[2]), 0.0],
            "centroid_ego": center.tolist(),
            "_source": "beta1",
        })
    return props


def _hybrid_distance_aware_union(gamma_props: list[dict], beta1_props: list[dict],
                                 threshold_m: float = 35.0,
                                 dedup_dist_m: float = 2.0) -> list[dict]:
    """Deployable (GT-free) distance-aware union of γ and β1 proposals.

    Principle from the May α diagnosis, runtime version:
      - near (< threshold_m ego-xy): γ preferred — keep all γ; add a β1
        proposal only if no γ proposal is within dedup_dist_m.
      - far  (>= threshold_m):       β1 preferred — keep all β1; add a γ
        proposal only if no β1 proposal is within dedup_dist_m.
    NOT the oracle strategy_distance_aware (which needs GT). Will not
    reproduce the 46.7% oracle M-rate.
    """
    for p in gamma_props:
        p.setdefault("_source", "gamma")

    def _xy(p):
        c = p["centroid_ego"]
        return np.array([c[0], c[1]], dtype=np.float64)

    def _near_any(p, others):
        pxy = _xy(p)
        for o in others:
            if float(np.linalg.norm(pxy - _xy(o))) < dedup_dist_m:
                return True
        return False

    merged: list[dict] = []
    for p in gamma_props:
        d = float(np.hypot(p["centroid_ego"][0], p["centroid_ego"][1]))
        if d < threshold_m:
            merged.append(p)                       # γ preferred near
        elif not _near_any(p, beta1_props):
            merged.append(p)                       # γ fills β1 gaps far
    for p in beta1_props:
        d = float(np.hypot(p["centroid_ego"][0], p["centroid_ego"][1]))
        if d >= threshold_m:
            merged.append(p)                       # β1 preferred far
        elif not _near_any(p, gamma_props):
            merged.append(p)                       # β1 fills γ gaps near
    return merged


# ---------------------------------------------------------------------------
# Main evaluator.
# ---------------------------------------------------------------------------
class StreamingNuScenesEvaluator:
    """Streaming evaluator for one or more nuScenes scenes (axis-installed).

    Construct once, set axis via ``install_method_streaming`` (Indoor
    hooks_streaming module), call ``run_scene(scene_token)`` per scene.
    Pred history + raw predictions are accumulated on ``self``; call
    ``finalize()`` to dump per-axis JSONs.
    """

    def __init__(
        self,
        nuscenes_loader: NuScenesLoader,
        cp_proposals,                  # CenterPointProposalGenerator
        oy3d,                          # OpenYolo3D with network_2d available
        text_prompts: list[str],
        association_threshold_m: float = DEFAULT_ASSOC_DIST_M,
        association_max_age: int = 5,
        pixel_match_radius: float = 80.0,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        proposal_source: str = "gamma",      # gamma | beta1 | hybrid
        beta1_generator=None,                # LiDARProposalGenerator (hybrid/beta1)
        hybrid_threshold_m: float = 35.0,
        tmp_dir: Optional[str] = None,
    ) -> None:
        self.loader = nuscenes_loader
        self.cp = cp_proposals
        self.oy3d = oy3d
        self.beta1 = beta1_generator
        self.proposal_source = str(proposal_source)
        self.hybrid_threshold_m = float(hybrid_threshold_m)
        if self.proposal_source in ("beta1", "hybrid") and self.beta1 is None:
            raise ValueError(
                f"proposal_source={self.proposal_source} requires beta1_generator")
        self.text_prompts = list(text_prompts)
        self.num_classes = len(self.text_prompts)
        self.association_threshold_m = float(association_threshold_m)
        self.association_max_age = int(association_max_age)
        self.pixel_match_radius = float(pixel_match_radius)
        self.iou_threshold = float(iou_threshold)
        self.tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="t21_outdoor_")
        os.makedirs(self.tmp_dir, exist_ok=True)
        # State (per-scene).
        self.method_11 = None
        self.method_12 = None
        self.associator: Optional[CentroidAssociator] = None
        self.running_labeler: Optional[NuScenesRunningLabeler] = None
        self.pred_history: list[dict[int, int]] = []
        # Cross-scene accumulators (per axis).
        self.per_sample_pred_boxes: dict[str, list[dict]] = {}
        self.per_sample_gt_boxes: dict[str, list[dict]] = {}
        self.last_axis_walltime_s: float = 0.0

    # -- axis lifecycle ---------------------------------------------------
    def install_axis(self, axis_name: str, method_id: Optional[str] = None,
                     **kwargs) -> None:
        # Use Indoor's hook installer directly — it just sets attrs on us.
        # ``method_id`` lets display-axis names (e.g. M12_thr080) map to a
        # real installer id (M12) with custom kwargs.
        uninstall_all_streaming(self)
        mid = method_id or axis_name
        if mid != "baseline":
            install_method_streaming(self, mid, **kwargs)

    def begin_axis(self) -> None:
        self.per_sample_pred_boxes = {}
        self.per_sample_gt_boxes = {}
        self.pred_history = []
        self.last_axis_walltime_s = 0.0

    # -- per-scene --------------------------------------------------------
    def setup_scene(self, scene_offset: int = 0) -> None:
        # Fix 1 (Task 2.2): scene-offset global ids so the same physical
        # object never shares an id across scenes in the concatenated
        # pred_history (otherwise time_to_confirm conflates distinct
        # instances) and so the M11/M12 gate's per-id counters cannot
        # collide across scenes.
        self.associator = CentroidAssociator(
            threshold_m=self.association_threshold_m,
            max_age=self.association_max_age,
            id_offset=scene_offset,
        )
        self.running_labeler = NuScenesRunningLabeler(num_classes=self.num_classes)
        # Fix 1 (Task 2.2): clear the registration gate's per-instance
        # state between scenes — matches Indoor's "fresh evaluator per
        # scene" semantics. Uses the gate's own reset(); no Indoor-class
        # modification.
        if self.method_11 is not None and hasattr(self.method_11, "reset"):
            self.method_11.reset()
        if self.method_12 is not None and hasattr(self.method_12, "reset"):
            self.method_12.reset()

    def _scene_sample_tokens(self, scene_token: str) -> list[str]:
        nusc = self.loader.nusc
        scene = nusc.get("scene", scene_token)
        tokens, cur = [], scene["first_sample_token"]
        while cur:
            tokens.append(cur)
            cur = nusc.get("sample", cur)["next"]
        return tokens

    def _run_yolo_per_camera(self, images: dict, intrinsics: dict,
                             cam_to_ego: dict) -> dict:
        """Run YOLO-World on each camera image; return per-cam outputs."""
        # YOLO-World lives under mmyolo's registry scope.
        _set_mm_scope("mmyolo")
        out = {}
        for cam, img in images.items():
            # OpenYolo3D's network_2d.inference_detector expects file paths
            # OR numpy arrays — supporting numpy keeps us off disk.
            try:
                result = self.oy3d.network_2d.inference_detector([img])
            except Exception:
                # Some implementations require disk; fall back to writing.
                tmp_path = osp.join(self.tmp_dir, f"_y_{cam}.jpg")
                from PIL import Image as PILImage
                PILImage.fromarray(img.astype(np.uint8)).save(tmp_path)
                result = self.oy3d.network_2d.inference_detector([tmp_path])
            if result:
                entry = next(iter(result.values()))
            else:
                entry = None
            out[cam] = {
                "bbox": None if entry is None else entry.get("bbox"),
                "labels": None if entry is None else entry.get("labels"),
                "scores": None if entry is None else entry.get("scores"),
                "image_hw": (img.shape[0], img.shape[1]),
                "intrinsic": intrinsics[cam],
                "cam_to_ego": cam_to_ego[cam],
            }
        return out

    def _project_yolo_into_cams(self, centroid_ego: np.ndarray,
                                cam_outputs: dict) -> dict:
        per_cam = {}
        for cam, out in cam_outputs.items():
            per_cam[cam] = dict(out)
            per_cam[cam]["projection"] = _project_to_camera(
                centroid_ego,
                cam_to_ego=out["cam_to_ego"],
                intrinsic=out["intrinsic"],
                image_hw=out["image_hw"],
            )
        return per_cam

    def _yolo_class_to_text_idx(self, cls_name: str) -> int:
        if cls_name in self.text_prompts:
            return self.text_prompts.index(cls_name)
        return -1

    def step_sample(self, sample_token: str) -> dict:
        """Process one sample and snapshot pred_history."""
        if self.associator is None or self.running_labeler is None:
            raise RuntimeError("Call setup_scene() before step_sample().")

        item = self.loader._load(sample_token)
        ego_pose = item["ego_pose"]
        ego_translation = ego_pose[:3, 3]
        ego_quat = Quaternion(matrix=ego_pose[:3, :3])

        # --- γ proposals -----------------------------------------------
        pc = item["point_cloud"]
        # T_lidar_to_ego is recoverable from a fresh load; the loader
        # already returned ego-frame points, so we need the original
        # lidar→ego transform. Reload only the calibration.
        lidar_token = self.loader.nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
        lidar_sd = self.loader.nusc.get("sample_data", lidar_token)
        lidar_cs = self.loader.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        from nuscenes.utils.geometry_utils import transform_matrix
        T_lidar_to_ego = transform_matrix(
            translation=lidar_cs["translation"],
            rotation=Quaternion(lidar_cs["rotation"]),
        )
        # Fix 2 (Task 2.2): box yaw from CenterPoint is in the LIDAR frame.
        # Carry the lidar→ego rotation so box orientation reaches ego/global
        # correctly (the lidar mounting yaw is small but non-zero).
        lidar_to_ego_q = Quaternion(matrix=T_lidar_to_ego[:3, :3])
        tmp_bin = osp.join(self.tmp_dir, "_pc.bin")
        # Proposal source: gamma | beta1 | hybrid.
        gamma_props: list[dict] = []
        beta1_props: list[dict] = []
        if self.proposal_source in ("gamma", "hybrid"):
            # CenterPoint (mmdet3d) needs mmdet3d scope for LoadPointsFromFile.
            _set_mm_scope("mmdet3d")
            gamma_out = self.cp.generate(pc, T_lidar_to_ego, tmp_bin_path=tmp_bin)
            gamma_props = gamma_out["proposals"]
        if self.proposal_source in ("beta1", "hybrid"):
            beta1_out = self.beta1.generate(pc)
            beta1_props = _beta1_clusters_to_proposals(beta1_out)
        if self.proposal_source == "gamma":
            proposals = gamma_props
        elif self.proposal_source == "beta1":
            proposals = beta1_props
        else:  # hybrid
            proposals = _hybrid_distance_aware_union(
                gamma_props, beta1_props,
                threshold_m=self.hybrid_threshold_m,
                dedup_dist_m=self.association_threshold_m,
            )

        # --- cross-sample association → global ids ---------------------
        global_ids = self.associator.step(proposals)

        # --- YOLO-World inference on 6 cameras --------------------------
        cam_outputs = self._run_yolo_per_camera(
            item["images"], item["intrinsics"], item["cam_to_ego"]
        )

        # --- per-proposal: project + YOLO label -------------------------
        confirmed_ids_in: list[int] = []
        proposal_records: list[dict] = []
        for j, (p, gid) in enumerate(zip(proposals, global_ids)):
            centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
            bbox_lidar = p["bbox_lidar"]
            yaw_lidar = float(bbox_lidar[6]) if len(bbox_lidar) >= 7 else 0.0
            size = (bbox_lidar[3], bbox_lidar[4], bbox_lidar[5])
            # Fix 2: box orientation in ego frame = lidar→ego ∘ box-yaw(lidar).
            box_q_ego = lidar_to_ego_q * Quaternion(axis=(0.0, 0.0, 1.0), angle=yaw_lidar)

            # Fix 3: IoU-based label fusion. Project the oriented 3D box into
            # each camera, build its 2D footprint, IoU-match against YOLO boxes.
            corners_ego = _box_corners_ego(centroid_ego, size, box_q_ego)
            box2d_per_cam = {}
            for cam, out in cam_outputs.items():
                box2d_per_cam[cam] = _project_corners_to_2dbbox(
                    corners_ego, out["cam_to_ego"], out["intrinsic"], out["image_hw"]
                )
            yolo_cls_name, yolo_score, best_iou = _yolo_label_for_proposal_iou(
                box2d_per_cam, cam_outputs, self.text_prompts,
                iou_threshold=self.iou_threshold,
            )
            # Fallback: if no IoU match, use the centroid-radius heuristic.
            if yolo_cls_name is None:
                per_cam = self._project_yolo_into_cams(centroid_ego, cam_outputs)
                yolo_cls_name, yolo_score = _yolo_label_for_proposal(
                    centroid_ego, per_cam, self.text_prompts,
                    pixel_match_radius=self.pixel_match_radius,
                )

            # Vote for this id's label (keep only nuScenes-10).
            cls_idx_voted = -1
            chosen_name = None
            if yolo_cls_name is not None and yolo_cls_name in NUSC_10_SET:
                cls_idx_voted = self._yolo_class_to_text_idx(yolo_cls_name)
                chosen_name = yolo_cls_name
            if cls_idx_voted >= 0:
                # Weight YOLO votes by score (default 1 if absent).
                w = max(0.05, yolo_score)
                self.running_labeler.add_vote(gid, cls_idx_voted, weight=w)
                confirmed_ids_in.append(gid)
            # Stash raw record for later prediction emission. Pre-compute the
            # global box rotation here so emission uses the Fix-2 orientation.
            centroid_global = (ego_pose[:3, :3] @ centroid_ego[:3]) + ego_translation
            global_q = ego_quat * box_q_ego
            proposal_records.append({
                "global_id": gid,
                "bbox_lidar": bbox_lidar,
                "centroid_global": centroid_global,
                "rotation_global_wxyz": [float(global_q.w), float(global_q.x),
                                         float(global_q.y), float(global_q.z)],
                "yolo_cls_name": chosen_name,
                "yolo_score": float(yolo_score) if yolo_score > 0 else 0.0,
                "cp_score": float(p.get("score", 0.0)),
                "cp_cls_name": p.get("cls_name"),
            })

        # --- M11 / M12 gate -------------------------------------------
        if self.method_11 is not None:
            confirmed = list(self.method_11.gate(confirmed_ids_in))
        elif self.method_12 is not None:
            confirmed = list(self.method_12.gate(confirmed_ids_in))
        else:
            confirmed = list(confirmed_ids_in)
        confirmed_set = set(int(x) for x in confirmed)

        # --- pred_history snapshot (Indoor-format compatible) ----------
        snap = self.running_labeler.snapshot(confirmed_set)
        self.pred_history.append(snap)

        # --- emit predictions for nuScenes-devkit AP eval --------------
        sample_preds: list[dict] = []
        for rec in proposal_records:
            if rec["global_id"] not in confirmed_set:
                continue
            cls_name = rec["yolo_cls_name"]
            if cls_name is None or cls_name not in NUSC_10_SET:
                continue
            # Fix 2: rotation_global_wxyz was pre-composed with the lidar→ego
            # mounting rotation (ego_quat ∘ lidar_to_ego ∘ box_yaw).
            sample_preds.append(_detection_box_dict(
                global_id=rec["global_id"],
                sample_token=sample_token,
                bbox_lidar=rec["bbox_lidar"],
                centroid_global=rec["centroid_global"],
                ego_translation=ego_translation,
                rotation_global_wxyz=rec["rotation_global_wxyz"],
                detection_name=cls_name,
                score=rec["yolo_score"] if rec["yolo_score"] > 0 else rec["cp_score"],
            ))
        self.per_sample_pred_boxes[sample_token] = sample_preds

        # --- emit GT for this sample (for the eval call) ---------------
        gt_records: list[dict] = []
        for gt in item["gt_boxes"]:
            from nuscenes.eval.detection.utils import category_to_detection_name
            det_name = category_to_detection_name(gt["category"])
            if det_name is None or det_name not in NUSC_10_SET:
                continue
            gt_records.append({
                "sample_token": sample_token,
                "translation": [float(x) for x in gt["translation"]],
                "size": [float(x) for x in gt["size"]],
                "rotation": [float(x) for x in gt["rotation"]],
                "velocity": [0.0, 0.0],
                "ego_translation": [float(x) for x in ego_translation],
                "num_pts": int(gt.get("num_lidar_pts", 0)),
                "detection_name": det_name,
                "detection_score": -1.0,
                "attribute_name": "",
            })
        self.per_sample_gt_boxes[sample_token] = gt_records

        return {
            "sample_token": sample_token,
            "n_proposals": len(proposals),
            "n_confirmed_with_label": len(confirmed_set),
        }

    def run_scene(self, scene_token: str, scene_idx: int = 0) -> dict:
        t0 = time.time()
        self.setup_scene(scene_offset=scene_idx * SCENE_ID_STRIDE)
        tokens = self._scene_sample_tokens(scene_token)
        per_sample_info = []
        for tok in tokens:
            per_sample_info.append(self.step_sample(tok))
        wall = time.time() - t0
        return {
            "scene_token": scene_token,
            "n_samples": len(tokens),
            "wall_s": wall,
            "per_sample": per_sample_info,
        }

    # -- axis aggregation -------------------------------------------------
    def aggregate_axis_metrics(self, out_dir: Path) -> dict:
        """nuScenes-devkit detection AP + Indoor temporal metrics."""
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox

        pred_eb = EvalBoxes()
        for tok, dicts in self.per_sample_pred_boxes.items():
            if dicts:
                pred_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in dicts])
            else:
                pred_eb.add_boxes(tok, [])
        gt_eb = EvalBoxes()
        for tok, dicts in self.per_sample_gt_boxes.items():
            if dicts:
                gt_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in dicts])
            else:
                gt_eb.add_boxes(tok, [])

        out_dir.mkdir(parents=True, exist_ok=True)
        eval_summary = None
        try:
            from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate
            eval_summary = nu_evaluate(
                pred_boxes=pred_eb, gt_boxes=gt_eb,
                output_dir=str(out_dir / "nuscenes_eval"),
                config_name="detection_cvpr_2019",
            )
        except Exception as exc:
            (out_dir / "nuscenes_eval_error.txt").write_text(repr(exc))

        # Indoor-compatible temporal metrics from pred_history.
        lsc = int(label_switch_count(self.pred_history))
        ttc = time_to_confirm(self.pred_history, K=3)
        ttc_values = list(ttc.values())
        temporal = {
            "n_samples": len(self.pred_history),
            "label_switch_count_total": lsc,
            "time_to_confirm": {
                "n_instances": len(ttc_values),
                "mean": float(np.mean(ttc_values)) if ttc_values else None,
                "median": float(np.median(ttc_values)) if ttc_values else None,
                "p90": float(np.percentile(ttc_values, 90)) if ttc_values else None,
                "max": int(max(ttc_values)) if ttc_values else None,
            },
        }
        (out_dir / "temporal_metrics.json").write_text(json.dumps(temporal, indent=2))

        summary = {
            "n_samples": len(self.per_sample_pred_boxes),
            "n_pred_boxes_total": sum(len(v) for v in self.per_sample_pred_boxes.values()),
            "n_gt_boxes_total": sum(len(v) for v in self.per_sample_gt_boxes.values()),
            "axis_walltime_s": self.last_axis_walltime_s,
            "mAP": eval_summary.get("mean_ap") if eval_summary else None,
            "NDS": eval_summary.get("nd_score") if eval_summary else None,
            "temporal": temporal,
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        return summary


# ---------------------------------------------------------------------------
# CLI / main: run baseline + M11 + M12 on N scenes, dump per-axis outputs.
# ---------------------------------------------------------------------------
def _build_cp_generator(score_threshold: float = 0.10):
    from adapters.centerpoint_proposals import CenterPointProposalGenerator
    CKPT = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
            "centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_"
            "20220810_011659-04cb3a3b.pth")
    CFG = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
           "centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")
    return CenterPointProposalGenerator(
        config_path=CFG,
        checkpoint_path=CKPT,
        score_threshold=score_threshold,
        nms_iou_threshold=0.20,
        device="cuda:0",
    )


def _build_beta1_generator():
    # LiDARProposalGenerator takes individual kwargs (not a ClusteringConfig).
    # Defaults: ground z_threshold=-1.4, min_cluster_size=20, eps=0.5,
    # max_distance=100 m — the adapter's shipped config.
    from adapters.lidar_proposals import LiDARProposalGenerator
    return LiDARProposalGenerator()


def _build_loader(nuscenes_config: str) -> NuScenesLoader:
    return NuScenesLoader(config_path=nuscenes_config)


def _build_oy3d(oy3d_config: str):
    from utils import OpenYolo3D
    return OpenYolo3D(oy3d_config)


def _list_mini_scenes(loader: NuScenesLoader, limit: Optional[int] = None) -> list[str]:
    scenes = [s["token"] for s in loader.nusc.scene]
    if limit is not None:
        scenes = scenes[:limit]
    return scenes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes-config", required=True,
                        help="configs/nuscenes_baseline.yaml (v1.0-mini) or trainval")
    parser.add_argument("--oy3d-config", required=True,
                        help="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--axes", nargs="+",
                        default=["baseline", "M11", "M12"],
                        help="Subset of axes to run.")
    parser.add_argument("--scene-limit", type=int, default=10,
                        help="Number of scenes to process (default 10 for v1.0-mini).")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Optional explicit scene tokens (overrides --scene-limit).")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--association-threshold-m", type=float, default=DEFAULT_ASSOC_DIST_M)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--proposal-source", type=str, default="gamma",
                        choices=["gamma", "beta1", "hybrid"])
    parser.add_argument("--hybrid-threshold-m", type=float, default=35.0)
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = _build_loader(args.nuscenes_config)
    print(f"  scenes: {len(loader.nusc.scene)}, samples: {len(loader.nusc.sample)}", flush=True)

    cp = None
    if args.proposal_source in ("gamma", "hybrid"):
        print("Loading γ CenterPoint ...", flush=True)
        cp = _build_cp_generator(score_threshold=args.score_threshold)
    beta1 = None
    if args.proposal_source in ("beta1", "hybrid"):
        print("Loading β1 LiDARProposalGenerator (HDBSCAN) ...", flush=True)
        beta1 = _build_beta1_generator()

    print("Loading OpenYolo3D (YOLO-World) ...", flush=True)
    oy3d = _build_oy3d(args.oy3d_config)

    import yaml
    with open(args.oy3d_config) as f:
        oy3d_cfg = yaml.safe_load(f)
    text_prompts = list(oy3d_cfg["network2d"]["text_prompts"])

    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = _list_mini_scenes(loader, limit=args.scene_limit)
    print(f"  scenes to run: {len(scenes)}", flush=True)

    evaluator = StreamingNuScenesEvaluator(
        nuscenes_loader=loader, cp_proposals=cp, oy3d=oy3d,
        text_prompts=text_prompts,
        association_threshold_m=args.association_threshold_m,
        iou_threshold=args.iou_threshold,
        proposal_source=args.proposal_source,
        beta1_generator=beta1,
        hybrid_threshold_m=args.hybrid_threshold_m,
    )

    overall_summary: list[dict] = []
    for axis in args.axes:
        # Axis dispatch: display name → (installer id, kwargs).
        kwargs = {}
        method_id = axis
        if axis == "M11":
            method_id, kwargs = "M11", {"N": 3}
        elif axis == "M12":
            method_id = "M12"
            kwargs = {"prior": 0.5, "detection_likelihood": 0.8, "threshold": 0.95}
        elif axis.startswith("M12_thr"):
            thr = int(axis.split("thr")[1]) / 100.0
            method_id = "M12"
            kwargs = {"prior": 0.5, "detection_likelihood": 0.8, "threshold": thr}
        print(f"\n[axis {axis}] method_id={method_id} kwargs={kwargs}", flush=True)
        axis_dir = out_root / f"axis_{axis}"
        evaluator.install_axis(axis, method_id=method_id, **kwargs)
        evaluator.begin_axis()
        t_axis = time.time()
        for i, sc in enumerate(scenes):
            print(f"  [{i+1}/{len(scenes)}] scene {sc[:8]}...", flush=True)
            try:
                evaluator.run_scene(sc, scene_idx=i)
            except Exception as exc:
                print(f"    SCENE FAILED: {exc!r}", flush=True)
        evaluator.last_axis_walltime_s = time.time() - t_axis
        summary = evaluator.aggregate_axis_metrics(axis_dir)
        overall_summary.append({"axis": axis, **summary})
        print(f"[axis {axis}] mAP={summary.get('mAP')} NDS={summary.get('NDS')} "
              f"lsc={summary['temporal']['label_switch_count_total']} "
              f"ttc_n={summary['temporal']['time_to_confirm']['n_instances']} "
              f"wall={summary['axis_walltime_s']:.1f}s",
              flush=True)

    (out_root / "all_summaries.json").write_text(json.dumps(overall_summary, indent=2))
    print(f"\nwrote {out_root / 'all_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
