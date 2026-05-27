"""Step 2b — native γ CenterPoint baseline + temporal layer (outdoor, YOLO-bypassed).

Parallel to :mod:`nuscenes_evaluator` (the γ *pipeline*, which relabels every
CenterPoint box with YOLO-World and scored mAP 0.0526), but built on the γ
*fixed* baseline instead: native CenterPoint class + score straight to the
nuScenes-devkit eval (Task 2.5, full-val single-sweep, mAP 0.3407). The
temporal layer (M11/M12/M21/M31/M32 + phase1) is applied **on top of the
native predictions** — YOLO-World is never invoked.

Why a new file (not an --label-source flag on nuscenes_evaluator):
  - nuscenes_evaluator applies only the M11/M12 gate; M21/M31/M32 are
    installed-but-never-invoked there (the "Stage C 5-axis silent no-op").
  - The native path needs no camera/YOLO machinery at all, so emission and
    voting are simpler and the baseline axis can be made byte-equivalent to
    centerpoint_native_map_sanity.py (the 0.3407 anchor).

Protected code (imported, never modified):
  - adapters/centerpoint_proposals.py (Task 2.5 class-map fix), dataloaders/
  - method_scannet/method_11..32 (the May method classes)
  - method_scannet/streaming/metrics.py (label_switch_count / time_to_confirm)
  - nuscenes_evaluator helpers (CentroidAssociator, NuScenesRunningLabeler,
    _detection_box_dict, _list_val_scenes, _set_mm_scope, NUSC_10_SET)

Emission model (per-sample, matching nuScenes detection eval granularity):
  baseline           emit every native proposal (native cls + native score)
  M11 / M12          gate by track confirmation; emit confirmed-track boxes
  M21                emit track-voted cls (WeightedVoting.frame_weight over
                     each track's per-frame native labels); native score
  M31                per-sample class-aware vertex-set IoU NMS on the boxes
  M32                per-sample class-aware Hungarian centroid merge
  phase1             M11 (gate) -> M21 (relabel) -> M31 (merge)

Every axis writes a fire_audit.json proving the method actually ran (counts of
suppressed / relabeled / merged boxes). An axis whose mAP is bit-identical to
baseline AND whose audit counter is zero is a no-op bug and must be rejected.

M22 (CLIP EMA) is deferred: the native LiDAR head emits no per-instance image
features, so M22 has no clean definition here (see paper §6.4).
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.centerpoint_proposals import NUSC_10  # class-map-fixed index order
from method_scannet.streaming.metrics import label_switch_count, time_to_confirm
from method_scannet.streaming.nuscenes_evaluator import (
    NUSC_10_SET,
    CentroidAssociator,
    NuScenesRunningLabeler,
    SCENE_ID_STRIDE,
    _detection_box_dict,
    _list_val_scenes,
    _set_mm_scope,
)

# Method classes (May; imported unchanged).
from method_scannet.method_11_frame_counting import FrameCountingGate
from method_scannet.method_12_bayesian import BayesianGate
from method_scannet.method_21_weighted_voting import WeightedVoting
from method_scannet.method_31_iou_merging import IoUMerger
from method_scannet.method_32_hungarian_merging import HungarianMerger

CLASS_NAMES = list(NUSC_10)                 # canonical 10-class index order
NAME_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_ASSOC_DIST_M = 2.0


class ClassAgnosticAssociator(CentroidAssociator):
    """Counterfactual associator — identical to :class:`CentroidAssociator`
    with the ``st["cls"] != cls`` class gate removed (the only change).

    Used to expose proposal-level label flicker that the class-aware default
    structurally hides at ``lsc=0`` (every track is single-class by
    construction). See ``results/2026-05-21_task_3_1_lsc_diagnosis_v01``.
    """

    def step(self, proposals: list[dict]) -> list[int]:
        for gid in list(self._active.keys()):
            self._active[gid]["age"] += 1
            if self._active[gid]["age"] > self.max_age:
                self._active.pop(gid, None)
        if not proposals:
            return []
        gid_assignments: list[Optional[int]] = [None] * len(proposals)
        used: set[int] = set()
        order = sorted(range(len(proposals)),
                       key=lambda i: -proposals[i].get("score", 0.0))
        for j in order:
            p = proposals[j]
            c = np.asarray(p["centroid_ego"], dtype=np.float64)
            best_gid, best_d = None, self.threshold_m + 1e-9
            for gid, st in self._active.items():
                if gid in used:
                    continue
                # class gate intentionally omitted — only change vs base.
                d = float(np.linalg.norm(c[:2] - st["centroid"][:2]))
                if d < best_d:
                    best_d, best_gid = d, gid
            if best_gid is not None:
                gid_assignments[j] = best_gid
                used.add(best_gid)
                self._active[best_gid]["centroid"] = c
                self._active[best_gid]["age"] = 0
            else:
                new_gid = self._next_id
                self._next_id += 1
                gid_assignments[j] = new_gid
                self._active[new_gid] = {"cls": p["cls_name"], "centroid": c, "age": 0}
        return [int(g) for g in gid_assignments]  # type: ignore


# ---------------------------------------------------------------------------
# Geometry helpers for the M31 box→vertex-set IoU realization.
# ---------------------------------------------------------------------------
def _box_interior_grid(centroid_ego: np.ndarray, size, box_q_ego: Quaternion,
                       n: int = 3) -> np.ndarray:
    """n^3 sample points strictly inside an oriented box (ego frame).

    Used to feed IoUMerger (which expects boolean vertex masks) without
    modifying it: each box's interior grid guarantees a non-empty mask, and
    a vertex falling inside two boxes contributes to their set-IoU exactly as
    a shared mesh vertex would indoors.
    """
    lin = np.linspace(-0.45, 0.45, n)       # 0.9 span -> strictly interior
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    local = np.stack([gx.ravel() * float(size[0]),
                      gy.ravel() * float(size[1]),
                      gz.ravel() * float(size[2])], axis=1)   # (n^3, 3)
    R = box_q_ego.rotation_matrix
    return (R @ local.T).T + np.asarray(centroid_ego[:3], dtype=np.float64)


def _points_in_box(verts: np.ndarray, centroid_ego: np.ndarray, size,
                   box_q_ego: Quaternion) -> np.ndarray:
    """Boolean (V,) mask of which verts lie inside an oriented box."""
    R = box_q_ego.rotation_matrix
    local = (R.T @ (verts - np.asarray(centroid_ego[:3], dtype=np.float64)).T).T
    half = np.asarray([size[0] / 2.0, size[1] / 2.0, size[2] / 2.0], dtype=np.float64)
    return (np.abs(local) <= half).all(axis=1)


# ---------------------------------------------------------------------------
# Evaluator.
# ---------------------------------------------------------------------------
class NativeTemporalNuScenesEvaluator:
    """Native CenterPoint proposals + temporal layer, per-sample nuScenes eval."""

    def __init__(
        self,
        loader: NuScenesLoader,
        cp_proposals,                    # CenterPointProposalGenerator | None (cache-only)
        association_threshold_m: float = DEFAULT_ASSOC_DIST_M,
        association_max_age: int = 5,
        cp_cache_dir: Optional[str] = None,
        tmp_dir: Optional[str] = None,
        proposal_source: str = "gamma",       # gamma (CenterPoint) | detguided
        oy3d=None,                            # OpenYolo3D (network_2d) — detguided
        detguided_generator=None,            # DetectionGuidedClusterer | None (cache-only)
        text_prompts: Optional[list] = None,  # YOLO-World class names (detguided)
        proposal_score_threshold: float = 0.0,  # reliability gate (applied on read)
        class_agnostic_association: bool = False,  # task_3_1 counterfactual
    ) -> None:
        self.loader = loader
        self.cp = cp_proposals
        self.proposal_source = str(proposal_source)
        self.proposal_score_threshold = float(proposal_score_threshold)
        self.oy3d = oy3d
        self.detguided = detguided_generator
        self.class_agnostic_association = bool(class_agnostic_association)
        self.text_prompts = list(text_prompts) if text_prompts is not None else None
        if (self.proposal_source == "detguided" and self.detguided is None
                and not cp_cache_dir):
            raise ValueError(
                "proposal_source=detguided requires detguided_generator (or a complete cache)")
        self.association_threshold_m = float(association_threshold_m)
        self.association_max_age = int(association_max_age)
        self.cp_cache_dir = cp_cache_dir
        if cp_cache_dir:
            os.makedirs(cp_cache_dir, exist_ok=True)
        self.tmp_dir = tmp_dir or "/tmp/_native_temporal"
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Per-axis method hooks (None == not installed).
        self.method_11: Optional[FrameCountingGate] = None
        self.method_12: Optional[BayesianGate] = None
        self.method_21: Optional[WeightedVoting] = None
        self.method_31: Optional[IoUMerger] = None
        self.method_32: Optional[HungarianMerger] = None

        # Per-scene state.
        self.associator: Optional[CentroidAssociator] = None
        self.labeler: Optional[NuScenesRunningLabeler] = None

        # Per-axis accumulators.
        self.per_sample_pred_boxes: dict[str, list[dict]] = {}
        self.per_sample_gt_boxes: dict[str, list[dict]] = {}
        self.pred_history: list[dict[int, int]] = []
        self.audit: dict[str, int] = {}
        self.last_axis_walltime_s: float = 0.0

        # In-memory CP-proposal cache for this process (per sample_token).
        self._scene_cache: dict[str, list[dict]] = {}

    # -- axis lifecycle ---------------------------------------------------
    def install_axis(self, axis_name: str, *, m11_N: int = 3,
                     m12_threshold: float = 0.85, m31_iou: float = 0.5,
                     m32_distance: float = 2.0) -> None:
        self.method_11 = self.method_12 = self.method_21 = None
        self.method_31 = self.method_32 = None
        members = self._axis_members(axis_name)
        if "M11" in members:
            self.method_11 = FrameCountingGate(N=m11_N)
        if "M12" in members:
            self.method_12 = BayesianGate(
                prior=0.5, detection_likelihood=0.8,
                false_positive_rate=0.2, threshold=m12_threshold)
        if "M21" in members:
            self.method_21 = WeightedVoting()
        if "M31" in members:
            self.method_31 = IoUMerger(iou_threshold=m31_iou, use_kdtree=True)
        if "M32" in members:
            self.method_32 = HungarianMerger(
                spatial_alpha=0.5, distance_threshold=m32_distance,
                semantic_threshold=-1.0)   # spatial-only (no M22 features)

    @staticmethod
    def _axis_members(axis_name: str) -> set[str]:
        if axis_name == "baseline":
            return set()
        if axis_name == "phase1":
            return {"M11", "M21", "M31"}
        if axis_name.startswith("M12_thr"):
            return {"M12"}
        return {axis_name}

    def begin_axis(self) -> None:
        self.per_sample_pred_boxes = {}
        self.per_sample_gt_boxes = {}
        self.pred_history = []
        self.last_axis_walltime_s = 0.0
        self.audit = {
            "n_proposals_total": 0,
            "n_emitted_total": 0,
            "n_suppressed_by_gate": 0,   # M11/M12
            "n_m21_weighted_votes": 0,   # M21 invocation proof (independent of effect)
            "n_relabeled_by_m21": 0,     # M21 effect (relabels; 0 is legit if native labels stable)
            "n_merged_by_m31": 0,        # M31
            "n_merged_by_m32": 0,        # M32
            "n_samples": 0,
        }

    # -- per-scene --------------------------------------------------------
    def setup_scene(self, scene_offset: int) -> None:
        Associator = (ClassAgnosticAssociator if self.class_agnostic_association
                      else CentroidAssociator)
        self.associator = Associator(
            threshold_m=self.association_threshold_m,
            max_age=self.association_max_age,
            id_offset=scene_offset)
        self.labeler = NuScenesRunningLabeler(num_classes=NUM_CLASSES)
        if self.method_11 is not None:
            self.method_11.reset()
        if self.method_12 is not None:
            self.method_12.reset()

    def _scene_sample_tokens(self, scene_token: str) -> list[str]:
        nusc = self.loader.nusc
        toks, cur = [], nusc.get("scene", scene_token)["first_sample_token"]
        while cur:
            toks.append(cur)
            cur = nusc.get("sample", cur)["next"]
        return toks

    # -- data access ------------------------------------------------------
    def _load_meta(self, sample_token: str):
        """ego_pose, T_lidar_to_ego, gt_records — no images, no point file."""
        from nuscenes.eval.detection.utils import category_to_detection_name
        nusc = self.loader.nusc
        sample = nusc.get("sample", sample_token)
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))
        T_lidar_to_ego = transform_matrix(lidar_cs["translation"], Quaternion(lidar_cs["rotation"]))
        ego_translation = ego_pose[:3, 3]
        gts = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            det = category_to_detection_name(ann["category_name"])
            if det is None or det not in NUSC_10_SET:
                continue
            gts.append({
                "sample_token": sample_token,
                "translation": [float(x) for x in ann["translation"]],
                "size": [float(x) for x in ann["size"]],
                "rotation": [float(x) for x in ann["rotation"]],
                "velocity": [0.0, 0.0],
                "ego_translation": [float(x) for x in ego_translation],
                "num_pts": int(ann.get("num_lidar_pts", 0)),
                "detection_name": det,
                "detection_score": -1.0,
                "attribute_name": "",
            })
        return ego_pose, T_lidar_to_ego, gts

    def _get_proposals(self, sample_token: str) -> list[dict]:
        """Active-source proposals with the reliability score-gate applied.

        The gate filters on read (the cache always stores the unfiltered set),
        so one cache serves any threshold and the v02 cache is reused as-is.
        """
        props = self._proposals_raw(sample_token)
        if self.proposal_score_threshold > 0.0:
            props = [p for p in props
                     if float(p.get("score", 0.0)) >= self.proposal_score_threshold]
        return props

    def _proposals_raw(self, sample_token: str) -> list[dict]:
        """Standard proposal dicts from the active source, cache or live.

        Cache key is source-tagged so the γ and detguided caches never collide:
        "<token>.pkl" for gamma (unchanged, backward-compatible),
        "<token>.detguided.pkl" for detguided.
        """
        if sample_token in self._scene_cache:
            return self._scene_cache[sample_token]
        suffix = "" if self.proposal_source == "gamma" else f".{self.proposal_source}"
        if self.cp_cache_dir:
            fp = osp.join(self.cp_cache_dir, f"{sample_token}{suffix}.pkl")
            if osp.exists(fp):
                with open(fp, "rb") as f:
                    props = pickle.load(f)
                self._scene_cache[sample_token] = props
                return props
        if self.proposal_source == "gamma":
            props = self._gamma_proposals(sample_token)
        elif self.proposal_source == "detguided":
            props = self._detguided_proposals(sample_token)
        else:
            raise ValueError(f"unknown proposal_source={self.proposal_source!r}")
        self._scene_cache[sample_token] = props
        if self.cp_cache_dir:
            with open(osp.join(self.cp_cache_dir, f"{sample_token}{suffix}.pkl"), "wb") as f:
                pickle.dump(props, f)
        return props

    def _gamma_proposals(self, sample_token: str) -> list[dict]:
        """Native CenterPoint proposals (NUSC_10 only) — the 0.3407 anchor path."""
        if self.cp is None:
            raise RuntimeError(
                f"no cached proposals for {sample_token[:8]} and cp_proposals is None")
        # loader._load gives byte-identical single-sweep input to the anchor.
        item = self.loader._load(sample_token)
        pc = item["point_cloud"]
        nusc = self.loader.nusc
        lidar_sd = nusc.get("sample_data", nusc.get("sample", sample_token)["data"]["LIDAR_TOP"])
        lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        T_lidar_to_ego = transform_matrix(lidar_cs["translation"], Quaternion(lidar_cs["rotation"]))
        _set_mm_scope("mmdet3d")
        out = self.cp.generate(pc, T_lidar_to_ego,
                               tmp_bin_path=osp.join(self.tmp_dir, "_pc.bin"))
        return [p for p in out["proposals"] if p.get("cls_name") in NUSC_10_SET]

    # -- detguided (LiDAR-clustering open-vocab) proposal source ----------
    def _run_yolo_per_camera(self, images: dict, intrinsics: dict,
                             cam_to_ego: dict) -> dict:
        """YOLO-World on each camera image. Per-cam: {bbox, labels(int),
        scores, image_hw, intrinsic, cam_to_ego}. (Ported from the γ pipeline.)"""
        _set_mm_scope("mmyolo")
        out = {}
        for cam, img in images.items():
            try:
                result = self.oy3d.network_2d.inference_detector([img])
            except Exception:
                tmp_path = osp.join(self.tmp_dir, f"_y_{cam}.jpg")
                from PIL import Image as PILImage
                PILImage.fromarray(img.astype(np.uint8)).save(tmp_path)
                result = self.oy3d.network_2d.inference_detector([tmp_path])
            entry = next(iter(result.values())) if result else None
            out[cam] = {
                "bbox": None if entry is None else entry.get("bbox"),
                "labels": None if entry is None else entry.get("labels"),
                "scores": None if entry is None else entry.get("scores"),
                "image_hw": (img.shape[0], img.shape[1]),
                "intrinsic": intrinsics[cam],
                "cam_to_ego": cam_to_ego[cam],
            }
        return out

    def _detguided_proposals(self, sample_token: str) -> list[dict]:
        """Detection-guided LiDAR clustering: YOLO-World 2D dets → frustum →
        pillar foreground → HDBSCAN → per-cluster 3D proposals carrying the
        open-vocab class/score, mapped to the standard proposal-dict contract."""
        if self.detguided is None or self.oy3d is None:
            raise RuntimeError(
                f"detguided requires detguided_generator + oy3d (sample {sample_token[:8]})")
        item = self.loader._load(sample_token)          # needs images for YOLO
        pc = item["point_cloud"]
        cam_outputs = self._run_yolo_per_camera(
            item["images"], item["intrinsics"], item["cam_to_ego"])
        prompts = self.text_prompts or []

        def _name(x):
            i = int(x)
            return prompts[i] if 0 <= i < len(prompts) else f"class_{i}"

        det, intr, c2e, hw = {}, {}, {}, {}
        for cam, o in cam_outputs.items():
            intr[cam], c2e[cam], hw[cam] = o["intrinsic"], o["cam_to_ego"], o["image_hw"]
            b, l, s = o["bbox"], o["labels"], o["scores"]
            if b is None or len(b) == 0:
                det[cam] = {"xyxy": [], "labels": [], "scores": []}
                continue
            bn = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
            ln = l.cpu().numpy() if hasattr(l, "cpu") else np.asarray(l)
            sn = s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)
            det[cam] = {
                "xyxy": [[float(v) for v in bn[j]] for j in range(bn.shape[0])],
                "labels": [_name(ln[j]) for j in range(bn.shape[0])],
                "scores": [float(sn[j]) for j in range(bn.shape[0])],
            }
        dg = self.detguided.generate(pc, det, intr, c2e, hw)
        return self._detguided_to_proposals(dg, pc)

    @staticmethod
    def _detguided_to_proposals(dg_out: dict, point_cloud_ego: np.ndarray) -> list[dict]:
        """DetectionGuidedClusterer output → standard proposal dicts.

        proposals_meta carries (class, score, centroid_ego) but no box dims, so
        derive the axis-aligned (yaw=0) ego AABB from each cluster's point mask
        (mirrors _beta1_clusters_to_proposals). The open-vocab class is kept so
        it drives class-aware association + the running_labeler/M21 vote — no
        separate relabel (minimal-change fork (i))."""
        metas = dg_out.get("proposals_meta", [])
        masks = dg_out.get("proposal_masks")
        props: list[dict] = []
        if masks is None or masks.shape[1] == 0:
            return props
        for j, m in enumerate(metas):
            col = masks[:, j]
            if not col.any():
                continue
            pts = point_cloud_ego[col, :3]
            mn, mx = pts.min(axis=0), pts.max(axis=0)
            center = (mn + mx) / 2.0
            dims = np.clip(mx - mn, 1e-3, None)
            cls_name = m.get("class")
            props.append({
                "cls_name": cls_name,
                "cls_idx": NAME_TO_IDX.get(cls_name, -1),
                "score": float(m.get("score", 0.0)),
                "bbox_lidar": [float(center[0]), float(center[1]), float(center[2]),
                               float(dims[0]), float(dims[1]), float(dims[2]), 0.0],
                "centroid_ego": [float(center[0]), float(center[1]), float(center[2])],
                "_source": "detguided",
            })
        return props

    # -- per-sample -------------------------------------------------------
    def step_sample(self, sample_token: str) -> None:
        assert self.associator is not None and self.labeler is not None
        proposals = self._get_proposals(sample_token)
        ego_pose, T_lidar_to_ego, gt_records = self._load_meta(sample_token)
        ego_translation = ego_pose[:3, 3]
        ego_quat = Quaternion(matrix=ego_pose[:3, :3])
        lidar_to_ego_q = Quaternion(matrix=T_lidar_to_ego[:3, :3])

        self.audit["n_proposals_total"] += len(proposals)
        self.audit["n_samples"] += 1

        # --- association -> global ids ---------------------------------
        global_ids = self.associator.step(proposals)

        # --- per-proposal records + label votes ------------------------
        records: list[dict] = []
        gids_present: list[int] = []
        for p, gid in zip(proposals, global_ids):
            centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
            bbox_lidar = p["bbox_lidar"]
            yaw_lidar = float(bbox_lidar[6]) if len(bbox_lidar) >= 7 else 0.0
            size = (bbox_lidar[3], bbox_lidar[4], bbox_lidar[5])
            box_q_ego = lidar_to_ego_q * Quaternion(axis=(0.0, 0.0, 1.0), angle=yaw_lidar)
            centroid_global = (ego_pose[:3, :3] @ centroid_ego[:3]) + ego_translation
            global_q = ego_quat * box_q_ego
            native_idx = NAME_TO_IDX.get(p["cls_name"], -1)
            native_score = float(p.get("score", 0.0))

            # Vote weight: M21 uses WeightedVoting.frame_weight (distance from
            # ego sensor x detection confidence); otherwise native-score weight.
            if self.method_21 is not None:
                w = self.method_21.frame_weight(
                    camera_pos=(0.0, 0.0, 0.0),          # ego/LiDAR origin
                    instance_centroid=centroid_ego[:3],
                    bbox_2d_center=(1.0, 1.0),           # == image center -> w_center=1
                    image_size=(2.0, 2.0),
                    confidence=native_score)
                self.audit["n_m21_weighted_votes"] += 1
            else:
                w = max(0.05, native_score)
            if native_idx >= 0:
                self.labeler.add_vote(gid, native_idx, weight=w)
            gids_present.append(gid)
            records.append({
                "gid": gid, "native_idx": native_idx, "native_score": native_score,
                "bbox_lidar": bbox_lidar, "centroid_ego": centroid_ego,
                "size": size, "box_q_ego": box_q_ego,
                "centroid_global": centroid_global,
                "rotation_global_wxyz": [float(global_q.w), float(global_q.x),
                                         float(global_q.y), float(global_q.z)],
            })

        # --- registration gate (M11 / M12) -----------------------------
        if self.method_11 is not None:
            confirmed = set(int(x) for x in self.method_11.gate(gids_present))
        elif self.method_12 is not None:
            confirmed = set(int(x) for x in self.method_12.gate(gids_present))
        else:
            confirmed = set(int(g) for g in gids_present)
        n_gated = len(records) - sum(1 for r in records if r["gid"] in confirmed)
        self.audit["n_suppressed_by_gate"] += n_gated

        # --- build emission set (post-gate) ----------------------------
        emit = [r for r in records if r["gid"] in confirmed and r["native_idx"] >= 0]

        # --- M21 relabel (track-voted class) ---------------------------
        for r in emit:
            if self.method_21 is not None:
                voted = self.labeler.best_label(r["gid"])
                r["emit_idx"] = voted if voted >= 0 else r["native_idx"]
                if r["emit_idx"] != r["native_idx"]:
                    self.audit["n_relabeled_by_m21"] += 1
            else:
                r["emit_idx"] = r["native_idx"]

        # --- spatial merge (M31 / M32) ---------------------------------
        if self.method_31 is not None and emit:
            emit = self._apply_m31(emit)
        if self.method_32 is not None and emit:
            emit = self._apply_m32(emit)

        # --- emit detection boxes --------------------------------------
        sample_preds = []
        for r in emit:
            cls_name = CLASS_NAMES[r["emit_idx"]]
            if cls_name not in NUSC_10_SET:
                continue
            sample_preds.append(_detection_box_dict(
                global_id=r["gid"], sample_token=sample_token,
                bbox_lidar=r["bbox_lidar"], centroid_global=r["centroid_global"],
                ego_translation=ego_translation,
                rotation_global_wxyz=r["rotation_global_wxyz"],
                detection_name=cls_name, score=r["native_score"]))
        self.per_sample_pred_boxes[sample_token] = sample_preds
        self.per_sample_gt_boxes[sample_token] = gt_records
        self.audit["n_emitted_total"] += len(sample_preds)

        # --- temporal-metric snapshot (running-mode labels) ------------
        self.pred_history.append(self.labeler.snapshot(confirmed))

    # -- mergers ----------------------------------------------------------
    def _apply_m31(self, emit: list[dict]) -> list[dict]:
        """Class-aware vertex-set IoU NMS via per-box interior-point masks."""
        K = len(emit)
        verts_list, owner = [], []
        for k, r in enumerate(emit):
            g = _box_interior_grid(r["centroid_ego"], r["size"], r["box_q_ego"], n=3)
            verts_list.append(g)
            owner.append(np.full(g.shape[0], k))
        verts = np.concatenate(verts_list, axis=0)            # (V, 3)
        masks = np.zeros((verts.shape[0], K), dtype=bool)
        for k, r in enumerate(emit):
            masks[:, k] = _points_in_box(verts, r["centroid_ego"], r["size"], r["box_q_ego"])
        pred_masks = torch.from_numpy(masks)
        pred_classes = torch.tensor([r["emit_idx"] for r in emit], dtype=torch.long)
        pred_scores = torch.tensor([r["native_score"] for r in emit], dtype=torch.float32)
        kept_masks, _, _ = self.method_31.merge(
            predicted_masks=pred_masks, pred_classes=pred_classes,
            pred_scores=pred_scores, vertex_coords=verts)
        kept_cols = int(kept_masks.shape[1])
        self.audit["n_merged_by_m31"] += (K - kept_cols)
        if kept_cols == K:
            return emit
        # Map surviving columns back: IoUMerger preserves column identity via
        # boolean keep, so recompute which originals survived by IoU keep order.
        # Recover kept indices by matching column sums (stable, cheap).
        kept_idx = self._recover_kept_indices(masks, kept_masks.cpu().numpy())
        return [emit[i] for i in kept_idx]

    @staticmethod
    def _recover_kept_indices(orig_masks: np.ndarray, kept_masks: np.ndarray) -> list[int]:
        """Match kept columns back to original column indices.

        IoUMerger only drops columns (preserving ascending original order), so
        kept columns are exact copies of original columns. We map them via a
        per-column byte signature (np.packbits) — O(K·V/8), collision-free for
        boolean masks. Duplicate signatures (identical boxes) are matched FIFO.
        """
        from collections import defaultdict, deque
        orig_sig = np.packbits(orig_masks, axis=0)   # (ceil(V/8), K) uint8
        kept_sig = np.packbits(kept_masks, axis=0)
        buckets: dict[bytes, deque] = defaultdict(deque)
        for o in range(orig_sig.shape[1]):
            buckets[orig_sig[:, o].tobytes()].append(o)
        kept_idx = []
        for c in range(kept_sig.shape[1]):
            dq = buckets.get(kept_sig[:, c].tobytes())
            if dq:
                kept_idx.append(dq.popleft())
        return sorted(kept_idx)

    def _apply_m32(self, emit: list[dict]) -> list[dict]:
        """Class-aware Hungarian centroid merge (spatial-only)."""
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for k, r in enumerate(emit):
            groups[r["emit_idx"]].append(k)
        keep: set[int] = set()
        for _cls, idxs in groups.items():
            if len(idxs) == 1:
                keep.add(idxs[0])
                continue
            instance_list = [{
                "id": k, "label": emit[k]["emit_idx"],
                "centroid": np.asarray(emit[k]["centroid_ego"][:3], dtype=np.float64),
            } for k in idxs]
            merged = self.method_32.merge(instance_list, {})
            for m in merged:
                keep.add(int(m["id"]))
        self.audit["n_merged_by_m32"] += (len(emit) - len(keep))
        if len(keep) == len(emit):
            return emit
        return [emit[k] for k in sorted(keep)]

    # -- scene driver -----------------------------------------------------
    def run_scene(self, scene_token: str, scene_idx: int) -> None:
        self.setup_scene(scene_offset=scene_idx * SCENE_ID_STRIDE)
        self._scene_cache = {}   # bound process memory to one scene at a time
        for tok in self._scene_sample_tokens(scene_token):
            self.step_sample(tok)

    # -- aggregation ------------------------------------------------------
    def aggregate_axis_metrics(self, out_dir: Path, baseline_map: Optional[float]) -> dict:
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox
        from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate

        out_dir.mkdir(parents=True, exist_ok=True)
        pred_eb, gt_eb = EvalBoxes(), EvalBoxes()
        for tok, dicts in self.per_sample_pred_boxes.items():
            pred_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in dicts])
        for tok, dicts in self.per_sample_gt_boxes.items():
            gt_eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in dicts])

        eval_summary = None
        try:
            eval_summary = nu_evaluate(pred_boxes=pred_eb, gt_boxes=gt_eb,
                                       output_dir=str(out_dir / "nuscenes_eval"),
                                       config_name="detection_cvpr_2019")
        except Exception as exc:
            (out_dir / "nuscenes_eval_error.txt").write_text(repr(exc))

        lsc = int(label_switch_count(self.pred_history))
        ttc = time_to_confirm(self.pred_history, K=3)
        ttc_vals = list(ttc.values())
        temporal = {
            "n_samples": len(self.pred_history),
            "label_switch_count_total": lsc,
            "time_to_confirm": {
                "n_instances": len(ttc_vals),
                "mean": float(np.mean(ttc_vals)) if ttc_vals else None,
                "median": float(np.median(ttc_vals)) if ttc_vals else None,
                "p90": float(np.percentile(ttc_vals, 90)) if ttc_vals else None,
                "max": int(max(ttc_vals)) if ttc_vals else None,
            },
        }
        mAP = eval_summary.get("mean_ap") if eval_summary else None
        nds = eval_summary.get("nd_score") if eval_summary else None
        delta = (None if (mAP is None or baseline_map is None)
                 else float(mAP - baseline_map))
        summary = {
            "n_samples": len(self.per_sample_pred_boxes),
            "n_pred_boxes_total": sum(len(v) for v in self.per_sample_pred_boxes.values()),
            "n_gt_boxes_total": sum(len(v) for v in self.per_sample_gt_boxes.values()),
            "axis_walltime_s": self.last_axis_walltime_s,
            "mAP": mAP, "NDS": nds, "mAP_minus_baseline": delta,
            "temporal": temporal,
            "fire_audit": dict(self.audit),
        }
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "fire_audit.json").write_text(json.dumps(self.audit, indent=2))
        return summary


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _build_cp(score_threshold: float):
    from adapters.centerpoint_proposals import CenterPointProposalGenerator
    CKPT = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
            "centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_"
            "20220810_011659-04cb3a3b.pth")
    CFG = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
           "centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")
    return CenterPointProposalGenerator(config_path=CFG, checkpoint_path=CKPT,
                                        score_threshold=score_threshold, device="cuda:0")


def _build_oy3d(oy3d_config: str):
    from utils import OpenYolo3D
    return OpenYolo3D(oy3d_config)


def _build_detguided_generator():
    """DetectionGuidedClusterer = FrustumExtractor + PillarForegroundExtractor +
    HDBSCAN. The inner HDBSCAN runs on pillar-foreground points, so its own
    ground filter / distance cap are disabled (the pillar + frustum already
    bound the region)."""
    from preprocessing.detection_frustum import FrustumExtractor
    from preprocessing.pillar_foreground import PillarForegroundExtractor
    from adapters.lidar_proposals import LiDARProposalGenerator
    from proposal.detection_guided_clustering import DetectionGuidedClusterer
    frustum = FrustumExtractor(expand_ratio=0.10, min_depth=1.0, max_depth=80.0)
    pillar = PillarForegroundExtractor(pillar_size_xy=(0.5, 0.5), z_threshold=0.3,
                                       ground_estimation="ransac")
    hdb = LiDARProposalGenerator(min_cluster_size=20, min_samples=5,
                                 cluster_selection_epsilon=0.5,
                                 ground_filter=None, max_distance=None)
    return DetectionGuidedClusterer(frustum, pillar, hdb, min_points_per_frustum=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--axes", nargs="+", default=["baseline"])
    ap.add_argument("--scene-split", choices=["val", "all"], default="val")
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--score-threshold", type=float, default=0.0)
    ap.add_argument("--association-threshold-m", type=float, default=DEFAULT_ASSOC_DIST_M)
    ap.add_argument("--association-max-age", type=int, default=5)
    ap.add_argument("--cp-cache-dir", default=None,
                    help="shared proposal cache (source-tagged: <token>.pkl for gamma, "
                         "<token>.detguided.pkl for detguided).")
    ap.add_argument("--proposal-source", choices=["gamma", "detguided"], default="gamma",
                    help="gamma = native CenterPoint (0.3407 anchor); detguided = "
                         "LiDAR-clustering open-vocab (YOLO-World frustum + HDBSCAN).")
    ap.add_argument("--oy3d-config", default="configs/openyolo3d_nuscenes.yaml",
                    help="OpenYolo3D (YOLO-World) config — required for detguided.")
    ap.add_argument("--proposal-score-threshold", type=float, default=0.0,
                    help="reliability gate: drop proposals with score below this "
                         "(applied on read; cache stores the unfiltered set).")
    ap.add_argument("--m11-N", type=int, default=3)
    ap.add_argument("--m12-threshold", type=float, default=0.85)
    ap.add_argument("--m31-iou", type=float, default=0.5)
    ap.add_argument("--m32-distance", type=float, default=0.5,
                    help="M32 Hungarian merge distance gate (m). Data-driven default "
                         "0.5: the outdoor distance sweep (150 scenes) showed 0.5 m "
                         "fully recovers mAP to the γ-fixed baseline 0.3407 (vs 0.3198 "
                         "at 2.0 m); same-class centroids are never <0.42 m apart.")
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip CenterPoint init (requires a complete cp-cache).")
    ap.add_argument("--association-class-agnostic", action="store_true",
                    help="Drop the class gate in the proposal tracker "
                         "(ClassAgnosticAssociator). Exposes proposal-level "
                         "label flicker that the class-aware default "
                         "structurally hides at lsc=0 (task_3_1 finding).")
    args = ap.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    print(f"  scenes={len(loader.nusc.scene)} samples={len(loader.nusc.sample)}", flush=True)

    cp = None
    if args.proposal_source == "gamma" and not args.no_gpu:
        print("Loading CenterPoint (γ, class-map fixed) ...", flush=True)
        cp = _build_cp(score_threshold=args.score_threshold)

    oy3d = detg = text_prompts = None
    if args.proposal_source == "detguided":
        import yaml
        with open(args.oy3d_config) as f:
            text_prompts = list(yaml.safe_load(f)["network2d"]["text_prompts"])
        if not args.no_gpu:
            print("Loading OpenYolo3D (YOLO-World) + DetectionGuidedClusterer ...", flush=True)
            oy3d = _build_oy3d(args.oy3d_config)
            detg = _build_detguided_generator()

    if args.scenes:
        scenes = list(args.scenes)
    elif args.scene_split == "val":
        scenes = _list_val_scenes(loader)
    else:
        scenes = [s["token"] for s in loader.nusc.scene]
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    print(f"  source={args.proposal_source} axes={args.axes} scenes={len(scenes)} "
          f"split={args.scene_split} score_thr={args.score_threshold}", flush=True)

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=cp,
        association_threshold_m=args.association_threshold_m,
        association_max_age=args.association_max_age,
        cp_cache_dir=args.cp_cache_dir,
        proposal_source=args.proposal_source,
        oy3d=oy3d, detguided_generator=detg, text_prompts=text_prompts,
        proposal_score_threshold=args.proposal_score_threshold,
        class_agnostic_association=args.association_class_agnostic)

    # Baseline mAP anchor for the fire-audit delta (read if a sibling baseline
    # axis was already written in this run dir).
    baseline_map = None
    base_metrics = out_root / "axis_baseline" / "metrics.json"
    if base_metrics.exists():
        try:
            baseline_map = json.loads(base_metrics.read_text()).get("mAP")
        except Exception:
            baseline_map = None

    overall = []
    for axis in args.axes:
        print(f"\n[axis {axis}] installing ...", flush=True)
        ev.install_axis(axis, m11_N=args.m11_N, m12_threshold=args.m12_threshold,
                        m31_iou=args.m31_iou, m32_distance=args.m32_distance)
        ev.begin_axis()
        t0 = time.time()
        for i, sc in enumerate(scenes):
            print(f"  [{i+1}/{len(scenes)}] scene {sc[:8]} ...", flush=True)
            try:
                ev.run_scene(sc, scene_idx=i)
            except Exception as exc:
                print(f"    SCENE FAILED: {exc!r}", flush=True)
        ev.last_axis_walltime_s = time.time() - t0
        summary = ev.aggregate_axis_metrics(out_root / f"axis_{axis}", baseline_map)
        if axis == "baseline":
            baseline_map = summary.get("mAP")
        overall.append({"axis": axis, **summary})
        a = summary["fire_audit"]
        print(f"[axis {axis}] mAP={summary['mAP']} NDS={summary['NDS']} "
              f"Δbase={summary['mAP_minus_baseline']} "
              f"lsc={summary['temporal']['label_switch_count_total']} "
              f"ttc_n={summary['temporal']['time_to_confirm']['n_instances']} "
              f"| fire: gated={a['n_suppressed_by_gate']} relbl={a['n_relabeled_by_m21']} "
              f"m31={a['n_merged_by_m31']} m32={a['n_merged_by_m32']} "
              f"emit={a['n_emitted_total']}/{a['n_proposals_total']} "
              f"wall={summary['axis_walltime_s']:.0f}s", flush=True)

    (out_root / "all_summaries.json").write_text(json.dumps(overall, indent=2))
    print(f"\nwrote {out_root / 'all_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
