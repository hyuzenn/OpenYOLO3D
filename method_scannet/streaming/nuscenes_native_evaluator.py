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
import math
import os
import os.path as osp
import pickle
import random
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
GT_FRAG_RADIUS_M = 2.0   # GT-anchored fragmentation match gate (global BEV)
# Fragmentation-injection id space (controlled temporal-quality degradation):
# minted ids live far above any associator/scene id (SCENE_ID_STRIDE x 150 scenes
# << 8e9) so a broken sub-track never collides with a real track id.
FRAG_ID_BASE = 8_000_000_000
FRAG_SEED = 1234


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


class GlobalCentroidAssociator(ClassAgnosticAssociator):
    """Ego-motion-compensated variant of :class:`ClassAgnosticAssociator`.

    The matcher is byte-for-byte the same greedy / score-ordered / static /
    max-age tracker; the *only* change is the frame in which the nearest-track
    gate is evaluated. Each proposal centroid is lifted ego→global with the
    current ego_pose (set via :meth:`set_ego_pose` before every ``step``), so a
    stationary object keeps one id as the ego car drives past it — instead of
    flowing out of the 2 m ego-frame gate. This is the single "frame" knob from
    ``diagnosis/outdoor_associator_ablation_probe`` (``global_greedy_static_a5``),
    here promoted into the production streaming pipeline as a method variant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ego_R: Optional[np.ndarray] = None
        self._ego_t: Optional[np.ndarray] = None

    def set_ego_pose(self, ego_pose: np.ndarray) -> None:
        self._ego_R = np.asarray(ego_pose[:3, :3], dtype=np.float64)
        self._ego_t = np.asarray(ego_pose[:3, 3], dtype=np.float64)

    def step(self, proposals: list[dict]) -> list[int]:
        assert self._ego_R is not None, "set_ego_pose() must precede step() in global frame"
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
            ce = np.asarray(p["centroid_ego"], dtype=np.float64)
            c = self._ego_R @ ce[:3] + self._ego_t          # ego -> global
            best_gid, best_d = None, self.threshold_m + 1e-9
            for gid, st in self._active.items():
                if gid in used:
                    continue
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
        association_frame: str = "ego",      # ego | global (ego-motion-compensated)
        collect_track_metrics: bool = False,  # opt-in OV-TCS / frag / track-length probe
        frag_inject_p: float = 0.0,           # controlled fragmentation injection rate
        fuse_allow=None,                      # 2D->3D label-fusion overlay (None=off,
                                              # frozenset()=native, "ALL"=global, set=allowlist)
        fuse_tau_iou: float = 0.5,            # overlay: min match_iou to trust YOLO
        fuse_tau_score: float = 0.4,          # overlay: min YOLO score to trust YOLO
    ) -> None:
        self.loader = loader
        self.cp = cp_proposals
        self.proposal_source = str(proposal_source)
        self.proposal_score_threshold = float(proposal_score_threshold)
        self.oy3d = oy3d
        self.detguided = detguided_generator
        self.class_agnostic_association = bool(class_agnostic_association)
        self.association_frame = str(association_frame)
        if self.association_frame not in ("ego", "global"):
            raise ValueError(f"association_frame must be ego|global, got {association_frame!r}")
        self.collect_track_metrics = bool(collect_track_metrics)
        self.frag_inject_p = float(frag_inject_p)
        if not (0.0 <= self.frag_inject_p < 1.0):
            raise ValueError(f"frag_inject_p must be in [0, 1), got {frag_inject_p!r}")
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

        # Opt-in method-variant probe buffers (only filled when
        # collect_track_metrics): per-track native-label sequence (OV-TCS +
        # track length) and per-GT-instance set of matched track ids (GT-anchored
        # fragmentation). Native labels make OV-TCS a pure associator property,
        # directly comparable to outdoor_ovtcs_assoc_compare_probe.
        self._track_seq: dict[int, list[int]] = {}
        self._gt_frag: dict[str, set] = {}
        # per-track nearest-GT class votes (for per-track label correctness in
        # the OV-TCS metric-validation harvest); name -> count per track id.
        self._track_gt_cls: dict[int, dict] = {}

        # In-memory CP-proposal cache for this process (per sample_token).
        self._scene_cache: dict[str, list[dict]] = {}

        # 2D->3D label-fusion overlay (read-time, hybrid source cache only). When
        # active, the source cls_name (YOLO) is rewritten to the CP label unless
        # the YOLO evidence is tight+confident AND the YOLO target class is in the
        # allowlist; score becomes score_cp. No box dropped, geometry untouched.
        #   None         -> overlay off (read source as-is, == variant Dp)
        #   frozenset()  -> native CP labels (A0)
        #   frozenset({..}) -> class-aware allowlist (override only into these)
        #   "ALL"        -> global / class-agnostic fusion (arm F)
        self.fuse_allow = fuse_allow
        self.fuse_tau_iou = float(fuse_tau_iou)
        self.fuse_tau_score = float(fuse_tau_score)
        self._override_audit: dict[str, dict] = {}   # GT-matched conversion, per target class
        self._override_pairs: dict[str, int] = {}    # raw cp->yolo override counts (no GT)

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
        self._track_seq = {}
        self._gt_frag = {}
        self._track_gt_cls = {}
        self._override_audit = {}
        self._override_pairs = {}
        # Fragmentation injection: deterministic RNG + monotonic id minter, reset
        # once per axis so a re-run of the same variant is byte-reproducible. The
        # per-track alias map is reset per scene (tracks never cross scenes).
        self._frag_rng = random.Random(FRAG_SEED)
        self._frag_next = FRAG_ID_BASE
        self._frag_alias: dict[int, int] = {}

    # -- per-scene --------------------------------------------------------
    def setup_scene(self, scene_offset: int) -> None:
        self._frag_alias = {}
        if self.association_frame == "global":
            Associator = GlobalCentroidAssociator
        elif self.class_agnostic_association:
            Associator = ClassAgnosticAssociator
        else:
            Associator = CentroidAssociator
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
                "instance_token": ann["instance_token"],
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
        if self.fuse_allow is not None:
            props = self._apply_fusion_overlay(props)
        if self.proposal_score_threshold > 0.0:
            props = [p for p in props
                     if float(p.get("score", 0.0)) >= self.proposal_score_threshold]
        return props

    _BG_NAME = "__background__"

    def _apply_fusion_overlay(self, props: list[dict]) -> list[dict]:
        """2D->3D label-fusion overlay on the hybrid source cache (cls_name=YOLO,
        cp_cls_name=CenterPoint, score=YOLO, score_cp=CP, match_iou>0 iff a YOLO
        ROI matched this box). Rewrites cls_name->CP unless the YOLO evidence is
        tight (match_iou>=tau_iou), confident (score>=tau_score), not background,
        AND its target class is in the allowlist; score->score_cp. Nothing dropped,
        geometry fixed. fuse_allow=='ALL' is the global (class-agnostic) arm F."""
        allow = self.fuse_allow
        out: list[dict] = []
        for b in props:
            cp = b.get("cp_cls_name", b["cls_name"])
            yolo = b["cls_name"]
            s_yolo = float(b.get("score", 0.0))
            s_cp = float(b.get("score_cp", s_yolo))
            trust = (float(b.get("match_iou", 0.0)) >= self.fuse_tau_iou
                     and s_yolo >= self.fuse_tau_score and yolo != self._BG_NAME
                     and (allow == "ALL" or yolo in allow))
            new_cls = yolo if trust else cp
            overridden = trust and (new_cls != cp)
            if overridden:
                k = f"{cp}->{yolo}"
                self._override_pairs[k] = self._override_pairs.get(k, 0) + 1
            nb = dict(b)
            nb["cls_name"] = new_cls
            nb["cls_idx"] = NAME_TO_IDX.get(new_cls, -1)
            nb["score"] = s_cp
            nb["_fuse_overridden"] = overridden
            nb["_fuse_cp"] = cp
            nb["_fuse_yolo"] = yolo
            out.append(nb)
        return out

    def _audit_overrides(self, proposals: list[dict], records: list[dict],
                         gt_records: list[dict]) -> None:
        """GT-matched override-quality tally (overlay only). For each overridden
        box, match to the nearest GT in the global BEV frame within 2 m (the
        loosest nuScenes TP gate) and classify the override:
          fp_to_tp      YOLO label == GT (override enables a TP; CP was wrong)
          tp_to_fp      CP label   == GT (override broke a correct CP label)
          neutral_wrong neither matches GT
          no_gt         no GT within 2 m (override on a likely-FP region)
        override precision == fp_to_tp / (overrides matched to a GT)."""
        gts = [(g["detection_name"],
                float(g["translation"][0]), float(g["translation"][1]))
               for g in gt_records]
        R2 = 2.0 * 2.0
        for p, r in zip(proposals, records):
            if not p.get("_fuse_overridden"):
                continue
            cp, yolo = p["_fuse_cp"], p["_fuse_yolo"]
            cx, cy = float(r["centroid_global"][0]), float(r["centroid_global"][1])
            best, bestd = None, R2
            for name, gx, gy in gts:
                d = (cx - gx) ** 2 + (cy - gy) ** 2
                if d <= bestd:
                    bestd, best = d, name
            a = self._override_audit.setdefault(
                yolo, {"fp_to_tp": 0, "tp_to_fp": 0, "neutral_wrong": 0, "no_gt": 0})
            if best is None:
                a["no_gt"] += 1
            elif yolo == best:
                a["fp_to_tp"] += 1
            elif cp == best:
                a["tp_to_fp"] += 1
            else:
                a["neutral_wrong"] += 1

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
        elif self.proposal_source == "hybrid":
            # Hybrid Proposal v2 is cache-only: the boxes are CenterPoint
            # geometry relabeled offline by scripts/build_hybrid_cache.py
            # (YOLO-World per-camera + ROI->detection IoU transfer). There is
            # no live generator here — a missing file is a build gap, not a
            # recompute trigger.
            raise RuntimeError(
                f"hybrid is cache-only; missing {sample_token[:8]}.hybrid.pkl in "
                f"{self.cp_cache_dir!r} (build it with scripts/build_hybrid_cache.py)")
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
        if hasattr(self.associator, "set_ego_pose"):     # global-frame variant
            self.associator.set_ego_pose(ego_pose)
        global_ids = self.associator.step(proposals)

        # --- controlled fragmentation injection ------------------------
        # Degrade ONLY the emitted track-id signal (the underlying matcher is
        # untouched): each continuing track is broken into a fresh sub-track
        # with probability p, persisting until the next break. A brand-new id
        # cannot break (nothing to fragment yet). The remapped ids feed BOTH the
        # OV-TCS/frag/track-length probe AND the temporal layer (M11 age gate,
        # M21 voting, M31 merge), so p is a single monotone temporal-quality knob.
        if self.frag_inject_p > 0.0 and global_ids:
            remapped: list[int] = []
            for g in global_ids:
                g = int(g)
                if g not in self._frag_alias:
                    self._frag_alias[g] = g
                elif self._frag_rng.random() < self.frag_inject_p:
                    self._frag_alias[g] = self._frag_next
                    self._frag_next += 1
                remapped.append(self._frag_alias[g])
            global_ids = remapped

        # --- method-variant probe: per-track native-label sequence -----
        if self.collect_track_metrics:
            for p, gid in zip(proposals, global_ids):
                nidx = NAME_TO_IDX.get(p["cls_name"], -1)
                if nidx >= 0:
                    self._track_seq.setdefault(int(gid), []).append(nidx)

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

        # --- 2D->3D override audit (GT-matched, overlay only) ----------
        if self.fuse_allow is not None and gt_records:
            self._audit_overrides(proposals, records, gt_records)

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

        # --- method-variant probe: GT-anchored fragmentation -----------
        # Per GT instance, accumulate the set of distinct track ids assigned to
        # the nearest proposal within GT_FRAG_RADIUS_M (global BEV). |set| over
        # the scene == #fragments; matches outdoor_associator_ablation_probe.
        if self.collect_track_metrics and records and gt_records:
            prop_xy = np.asarray([r["centroid_global"][:2] for r in records],
                                 dtype=np.float64)
            prop_gid = [int(r["gid"]) for r in records]
            for g in gt_records:
                gxy = np.asarray(g["translation"][:2], dtype=np.float64)
                d = np.linalg.norm(prop_xy - gxy, axis=1)
                k = int(np.argmin(d))
                if d[k] <= GT_FRAG_RADIUS_M:
                    self._gt_frag.setdefault(g["instance_token"], set()).add(prop_gid[k])
                    # vote the nearest GT's class onto that track (per-track
                    # downstream-label target; majority resolved at harvest time).
                    cls = self._track_gt_cls.setdefault(prop_gid[k], {})
                    cls[g["detection_name"]] = cls.get(g["detection_name"], 0) + 1

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
        per_class_ap = eval_summary.get("label_aps") if eval_summary else None
        tp_errors = eval_summary.get("tp_errors") if eval_summary else None
        delta = (None if (mAP is None or baseline_map is None)
                 else float(mAP - baseline_map))
        summary = {
            "n_samples": len(self.per_sample_pred_boxes),
            "n_pred_boxes_total": sum(len(v) for v in self.per_sample_pred_boxes.values()),
            "n_gt_boxes_total": sum(len(v) for v in self.per_sample_gt_boxes.values()),
            "axis_walltime_s": self.last_axis_walltime_s,
            "association_frame": self.association_frame,
            "mAP": mAP, "NDS": nds, "mAP_minus_baseline": delta,
            "per_class_AP": per_class_ap, "tp_errors": tp_errors,
            "temporal": temporal,
            "fire_audit": dict(self.audit),
        }
        if self.fuse_allow is not None:
            summary["override_audit"] = {
                "fuse_allow": (sorted(self.fuse_allow)
                               if isinstance(self.fuse_allow, (set, frozenset))
                               else self.fuse_allow),
                "tau_iou": self.fuse_tau_iou, "tau_score": self.fuse_tau_score,
                "n_overrides_total": sum(self._override_pairs.values()),
                "pairs": dict(sorted(self._override_pairs.items(),
                                     key=lambda kv: -kv[1])),
                "by_target": self._override_audit,
            }
        if self.collect_track_metrics:
            summary["variant_metrics"] = self.compute_variant_metrics()
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        (out_dir / "fire_audit.json").write_text(json.dumps(self.audit, indent=2))
        return summary

    # -- method-variant probe aggregation ---------------------------------
    def compute_variant_metrics(self) -> dict:
        """OV-TCS (A/B/C), track-length, and GT-anchored fragmentation from the
        per-track native-label sequences this axis recorded. OV-TCS uses native
        labels grouped by associator track, so it is a pure associator property
        (axis-independent) and directly comparable to
        ``outdoor_ovtcs_assoc_compare_probe`` (ego 0.301/0.260/0.136,
        global 0.263/0.241/0.168)."""
        from collections import Counter

        def _ovtcs(seq):
            L = len(seq)
            if L == 0:
                return None
            cnt = Counter(seq)
            n = L
            H = -sum((c / n) * math.log2(c / n) for c in cnt.values())
            DR = max(cnt.values()) / L
            sw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
            CSR = (sw / (L - 1)) if L >= 2 else 0.0
            Ln = 1.0 - 1.0 / L
            Hn = H / math.log2(NUM_CLASSES) if NUM_CLASSES > 1 else 0.0
            A = Ln * (1.0 - Hn)
            B = Ln * DR
            C = Ln * (1.0 - CSR)
            return (A, B, C, L)

        A_l, B_l, C_l, lengths = [], [], [], []
        for seq in self._track_seq.values():
            r = _ovtcs(seq)
            if r is None:
                continue
            A_l.append(r[0]); B_l.append(r[1]); C_l.append(r[2]); lengths.append(r[3])
        lengths_a = np.asarray(lengths, dtype=np.float64)
        frags = np.asarray([len(s) for s in self._gt_frag.values()], dtype=np.float64)

        def _stat(a):
            if a.size == 0:
                return None
            return {
                "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max()),
            }

        return {
            "n_tracks": len(self._track_seq),
            "ov_tcs": {
                "A_mean": float(np.mean(A_l)) if A_l else None,
                "B_mean": float(np.mean(B_l)) if B_l else None,
                "C_mean": float(np.mean(C_l)) if C_l else None,
            },
            "track_length": {
                **(_stat(lengths_a) or {}),
                "singleton_frac": (float(np.mean(lengths_a == 1)) if lengths_a.size else None),
            },
            "gt_fragmentation": {
                "n_gt_instances": int(frags.size),
                "mean_fragments": float(frags.mean()) if frags.size else None,
                "median_fragments": float(np.median(frags)) if frags.size else None,
                "p90_fragments": float(np.percentile(frags, 90)) if frags.size else None,
            },
        }

    def build_track_records(self) -> list:
        """Per-track rows for OV-TCS metric validation (needs
        collect_track_metrics). Each row: gid, ovtcs_C, track_len, n_switches,
        gt_frag (worst fragmentation of any GT the track touches), gt_matched,
        correct (majority native label == nearest-GT class)."""
        from collections import Counter

        track_frag: dict[int, int] = {}
        for gids in self._gt_frag.values():
            f = len(gids)
            for gid in gids:
                track_frag[int(gid)] = max(track_frag.get(int(gid), 0), f)

        rows = []
        for gid, seq in self._track_seq.items():
            L = len(seq)
            if L == 0:
                continue
            sw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
            csr = (sw / (L - 1)) if L >= 2 else 0.0
            ov = (1.0 - 1.0 / L) * (1.0 - csr) if L >= 2 else None
            votes = self._track_gt_cls.get(int(gid))
            if votes:
                gt_name = max(votes.items(), key=lambda kv: kv[1])[0]
                pred_name = CLASS_NAMES[Counter(seq).most_common(1)[0][0]]
                correct, matched = int(pred_name == gt_name), True
            else:
                correct, matched = None, False
            rows.append({
                "gid": int(gid), "ovtcs_C": ov, "track_len": int(L),
                "n_switches": int(sw), "gt_frag": track_frag.get(int(gid)),
                "gt_matched": matched, "correct": correct,
            })
        return rows


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
    ap.add_argument("--proposal-source", choices=["gamma", "detguided", "hybrid"],
                    default="gamma",
                    help="gamma = native CenterPoint (0.3407 anchor); detguided = "
                         "LiDAR-clustering open-vocab (YOLO-World frustum + HDBSCAN); "
                         "hybrid = CenterPoint geometry relabeled by YOLO-World per-camera "
                         "(cache-only, built by scripts/build_hybrid_cache.py).")
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
    ap.add_argument("--association-frame", choices=["ego", "global"], default="ego",
                    help="Tracker matching frame. ego = production "
                         "(no ego-motion compensation); global = "
                         "GlobalCentroidAssociator (ego->global lift before the "
                         "gate). The single 'frame' method-variant knob.")
    ap.add_argument("--collect-track-metrics", action="store_true",
                    help="Also measure OV-TCS (A/B/C), track length, and "
                         "GT-anchored fragmentation on the pipeline's own tracks.")
    ap.add_argument("--frag-inject-p", type=float, default=0.0,
                    help="Controlled fragmentation-injection rate in [0,1): break "
                         "each continuing track id with this probability per frame "
                         "(degrades temporal quality without touching the matcher).")
    ap.add_argument("--fuse-allow", default=None,
                    help="2D->3D label-fusion overlay on the hybrid source cache. "
                         "Comma list of YOLO target classes to override CP->YOLO "
                         "into (e.g. 'bicycle,motorcycle'); 'none'=native CP labels; "
                         "'all'=global (class-agnostic) fusion. Omit = overlay off.")
    ap.add_argument("--fuse-tau-iou", type=float, default=0.5,
                    help="Overlay: min ROI->box match_iou to trust the YOLO label.")
    ap.add_argument("--fuse-tau-score", type=float, default=0.4,
                    help="Overlay: min YOLO 2D score to trust the YOLO label.")
    args = ap.parse_args()

    fuse_allow = None
    if args.fuse_allow is not None:
        v = args.fuse_allow.strip().lower()
        if v in ("all", "*"):
            fuse_allow = "ALL"
        elif v in ("none", "native", ""):
            fuse_allow = frozenset()
        else:
            fuse_allow = frozenset(s.strip() for s in args.fuse_allow.split(",")
                                   if s.strip())

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
        class_agnostic_association=args.association_class_agnostic,
        association_frame=args.association_frame,
        collect_track_metrics=args.collect_track_metrics,
        frag_inject_p=args.frag_inject_p,
        fuse_allow=fuse_allow,
        fuse_tau_iou=args.fuse_tau_iou,
        fuse_tau_score=args.fuse_tau_score)

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
