"""Streaming evaluation wrapper around OpenYOLO3D ScanNet pipeline.

Task 1.2a skeleton. Option (다): Mask3D per-scene (1 call at scene start) +
per-frame visibility filtering (D3) + per-frame YOLO-World inference.

Method integration (M11/12/21/22/31/32) is Task 1.4. The TODO markers in
``step_frame`` indicate the exact lines where each axis plugs in. Until
Task 1.4 lands, the wrapper runs in baseline mode and emits an
unassigned-label placeholder map (Task 1.2b fills in the real label
assignment path).

The wrapper does not touch OpenYOLO3D core or 5월 method modules. It calls
``openyolo3d.network_3d.get_class_agnostic_masks`` and
``openyolo3d.network_2d.inference_detector`` like any external user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from method_scannet.streaming.baseline import BaselineLabelAccumulator
from method_scannet.streaming.visibility import (
    compute_frame_visibility,
    compute_vertex_projection,
)


class StreamingScanNetEvaluator:
    """Frame-by-frame streaming evaluator with offline-equivalent baseline.

    Task 1.2a created the skeleton (visibility + step_frame stub).
    Task 1.2b wires in :class:`BaselineLabelAccumulator` so the final-frame
    prediction matches OpenYOLO3D's offline output. Method axes
    (M11/12/21/22/31/32) plug in at the TODO sites in ``step_frame``.
    """

    def __init__(
        self,
        openyolo3d_instance: Any,
        scene_dir: str,
        depth_scale: float = 1000.0,
        depth_threshold: float = 0.05,
        num_classes: int = 201,  # ScanNet200 = 200 classes + background
        topk: int = 40,
        topk_per_image: int = 600,
    ) -> None:
        self.openyolo3d = openyolo3d_instance
        self.scene_dir = Path(scene_dir)
        self.depth_scale = depth_scale
        self.depth_threshold = depth_threshold
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.topk_per_image = int(topk_per_image)

        color_dir = self.scene_dir / "color"
        if color_dir.exists():
            self.frame_indices = sorted(
                int(p.stem) for p in color_dir.glob("*.jpg")
            )
        else:
            self.frame_indices = []

        # Populated by setup_scene().
        self.instance_vertex_masks: Optional[np.ndarray] = None  # (K, V) bool
        self.instance_scores: Optional[np.ndarray] = None  # (K,)
        self.scene_vertices: Optional[np.ndarray] = None  # (V, 3)
        self.intrinsic: Optional[np.ndarray] = None  # (3, 3)
        self.image_resolution: Optional[tuple] = None  # (H_img, W_img)
        self.depth_resolution: Optional[tuple] = None  # (H_depth, W_depth)
        self.scaling_w: float = 1.0
        self.scaling_h: float = 1.0
        self.baseline_accumulator: Optional[BaselineLabelAccumulator] = None

        # Per-frame prediction history for Task 1.3 temporal metrics.
        self.pred_history: list[dict[int, int]] = []

        # Task 1.4a method-axis hooks. Populated by
        # ``method_scannet.streaming.hooks_streaming.install_method_streaming``.
        # ``None`` means the axis is inactive (baseline behaviour).
        self.method_11 = None  # FrameCountingGate (registration)
        self.method_12 = None  # BayesianGate (registration)
        self.method_21 = None  # WeightedVoting (label, replaces baseline votes)
        self.method_22 = None  # FeatureFusionEMA (label, visual feature)
        self.method_31 = None  # IoUMerger (post-merge)
        self.method_32 = None  # HungarianMerger (post-merge)
        # State carried between step_frame calls when methods are active.
        self._method_state: dict = {}

    # ------------------------------------------------------------------
    # Scene-level setup (Mask3D + mesh load + intrinsic)
    # ------------------------------------------------------------------

    def _load_scene_vertices(self, mesh_file: str) -> np.ndarray:
        """Default mesh vertex loader. Tests override this attribute to
        bypass real PLY parsing.
        """
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(mesh_file)
        return np.asarray(pcd.points, dtype=np.float64)

    def setup_scene(
        self,
        processed_scene_path: Optional[str] = None,
        apply_mask3d_filter: bool = False,
        mask3d_cache_path: Optional[str] = None,
    ) -> None:
        """Run Mask3D once and cache scene-level state.

        Args:
            processed_scene_path: optional path to the pre-processed point-
                cloud (``.npy``) that offline ``OpenYolo3D.predict`` feeds to
                Mask3D. Ignored when ``mask3d_cache_path`` is provided.
            apply_mask3d_filter: when True, apply the post-Mask3D score
                threshold + NMS exactly the way ``OpenYolo3D.predict`` does
                (utils/__init__.py:114-118). Default False for backwards
                compatibility with the Task 1.2a tests. Ignored when
                ``mask3d_cache_path`` is provided (cache is already filtered).
            mask3d_cache_path: optional path to a ``.pt`` file containing a
                ``(masks, scores)`` tuple already produced by
                :mod:`method_scannet.streaming.tools.generate_mask3d_cache`.
                When supplied, Mask3D inference is skipped and the cached
                tensors are loaded directly — the canonical fix for the
                Task 1.2c H6 non-determinism issue.
        """
        if mask3d_cache_path is not None and Path(mask3d_cache_path).exists():
            masks_raw, scores_raw = torch.load(mask3d_cache_path, map_location="cpu")
            ply_files = list(self.scene_dir.glob("*.ply"))
            mesh_file = str(ply_files[0]) if ply_files else None
        else:
            if processed_scene_path is not None:
                mesh_file = processed_scene_path
            else:
                mesh_files = list(self.scene_dir.glob("*.ply"))
                mesh_file = str(mesh_files[0]) if mesh_files else None

            masks_raw, scores_raw = self.openyolo3d.network_3d.get_class_agnostic_masks(
                mesh_file, datatype="mesh"
            )

            if apply_mask3d_filter:
                from utils import apply_nms

                th = self.openyolo3d.openyolo3d_config["network3d"]["th"]
                nms_iou = self.openyolo3d.openyolo3d_config["network3d"]["nms"]
                keep_score = scores_raw >= th
                if int(keep_score.sum().item()) > 0:
                    keep_nms = apply_nms(
                        masks_raw[:, keep_score].cuda(),
                        scores_raw[keep_score].cuda(),
                        nms_iou,
                    )
                    masks_raw = (
                        masks_raw.cpu().permute(1, 0)[keep_score][keep_nms].permute(1, 0)
                    )
                    scores_raw = scores_raw.cpu()[keep_score][keep_nms]

        masks_np = masks_raw.cpu().numpy() if hasattr(masks_raw, "cpu") else np.asarray(masks_raw)
        if masks_np.ndim != 2:
            raise ValueError(
                f"Expected (V, K) or (K, V) bool mask; got shape {masks_np.shape}"
            )
        # Normalize to (K, V) instance-major: heuristic — K usually ≪ V.
        if masks_np.shape[0] >= masks_np.shape[1]:
            self.instance_vertex_masks = masks_np.T.astype(bool, copy=False)
        else:
            self.instance_vertex_masks = masks_np.astype(bool, copy=False)

        scores_np = scores_raw.cpu().numpy() if hasattr(scores_raw, "cpu") else np.asarray(scores_raw)
        self.instance_scores = scores_np.astype(np.float64, copy=False)

        # When a ``.npy`` point cloud is used, the mesh-vertex layout we need
        # for projection still comes from the ``*.ply`` file; only Mask3D
        # consumed the pre-processed input.
        mesh_for_vertices = mesh_file
        if processed_scene_path is not None:
            ply_files = list(self.scene_dir.glob("*.ply"))
            if ply_files:
                mesh_for_vertices = str(ply_files[0])
        self.scene_vertices = self._load_scene_vertices(mesh_for_vertices)

        # Probe color/depth resolution from the first frame so we can rescale
        # the intrinsic exactly like OpenYOLO3D's offline path.
        if self.frame_indices:
            f0 = self.frame_indices[0]
            try:
                import imageio

                self.image_resolution = tuple(
                    imageio.imread(
                        str(self.scene_dir / "color" / f"{f0}.jpg")
                    ).shape[:2]
                )
                self.depth_resolution = tuple(
                    imageio.imread(
                        str(self.scene_dir / "depth" / f"{f0}.png")
                    ).shape[:2]
                )
            except Exception:
                self.image_resolution = self.depth_resolution = (0, 0)
        else:
            self.image_resolution = self.depth_resolution = (0, 0)

        if self.image_resolution[0] > 0 and self.depth_resolution[0] > 0:
            self.scaling_h = (
                self.depth_resolution[0] / self.image_resolution[0]
            )
            self.scaling_w = (
                self.depth_resolution[1] / self.image_resolution[1]
            )

        # Load intrinsic and rescale color → depth resolution. The raw
        # ScanNet ``intrinsics.txt`` is at color resolution (e.g. 968×1296);
        # vertex projection happens in the depth frame (e.g. 480×640) so we
        # need OpenYOLO3D's exact rescaling. Reuse the offline helper to
        # avoid math drift (Task 1.2b sanity FAIL root cause).
        intrinsic_path = self.scene_dir / "intrinsics.txt"
        full_intrinsic = np.loadtxt(intrinsic_path)
        if full_intrinsic.ndim == 2 and full_intrinsic.shape[0] >= 3 and full_intrinsic.shape[1] >= 3:
            color_intrinsic = full_intrinsic[:3, :3].astype(np.float64, copy=True)
        else:
            color_intrinsic = np.asarray(full_intrinsic, dtype=np.float64).copy()

        if (
            self.image_resolution[0] > 0
            and self.depth_resolution[0] > 0
            and self.image_resolution != self.depth_resolution
        ):
            from utils import WORLD_2_CAM

            # adjust_intrinsic doesn't read ``self``; the class binding only
            # supplies the namespace. Calling it as an unbound method with
            # ``None`` as the placeholder ``self`` works in Python 3.
            self.intrinsic = WORLD_2_CAM.adjust_intrinsic(
                None,
                color_intrinsic,
                self.image_resolution,
                self.depth_resolution,
            ).astype(np.float64, copy=False)
        else:
            self.intrinsic = color_intrinsic

        # Pre-arm the baseline accumulator. step_frame() will push raw
        # frame data into it; compute_baseline_predictions() drains it.
        depth_h, depth_w = self.depth_resolution
        self.baseline_accumulator = BaselineLabelAccumulator(
            prediction_3d_masks=torch.from_numpy(self.instance_vertex_masks.T).bool(),  # (V, K)
            num_classes=self.num_classes,
            topk=self.topk,
            topk_per_image=self.topk_per_image,
            scaling_w=self.scaling_w,
            scaling_h=self.scaling_h,
            depth_height=int(depth_h),
            depth_width=int(depth_w),
        )

    # ------------------------------------------------------------------
    # Per-frame streaming step
    # ------------------------------------------------------------------

    def _load_pose(self, frame_idx: int) -> np.ndarray:
        """Load camera-to-world pose (4×4) for the frame."""
        return np.loadtxt(self.scene_dir / "poses" / f"{frame_idx}.txt")

    def _load_depth(self, frame_idx: int) -> np.ndarray:
        """Load depth map for the frame (in meters)."""
        import imageio

        depth_raw = imageio.imread(str(self.scene_dir / "depth" / f"{frame_idx}.png"))
        return depth_raw.astype(np.float32) / self.depth_scale

    def step_frame(self, frame_idx: int) -> dict:
        """Process one frame and return per-frame snapshot.

        Returns:
            {
                "frame_idx": int,
                "visible_instances": np.ndarray (n,) int — D3-visible Mask3D
                    instance indices,
                "current_instance_map": dict[int, int] — confirmed instance
                    id → label (baseline stub assigns -1; Task 1.2b/1.4
                    fills the real label),
                "frame_preds_2d": dict — YOLO-World raw output for this frame
                    (kept for downstream label voting in Task 1.2b/1.4).
            }
        """
        if self.instance_vertex_masks is None:
            raise RuntimeError("Call setup_scene() before step_frame().")

        # --- Per-frame input load (camera-to-world → world-to-camera) ---
        pose_cam_to_world = self._load_pose(frame_idx)
        extrinsic = np.linalg.inv(pose_cam_to_world)
        depth_map = self._load_depth(frame_idx)

        # --- Step A: vertex-level projection + D3 instance visibility ----
        projection, inside_mask = compute_vertex_projection(
            self.scene_vertices,
            self.intrinsic,
            extrinsic,
            depth_map,
            depth_threshold=self.depth_threshold,
        )
        instance_visible = (self.instance_vertex_masks & inside_mask[None, :]).any(axis=1)
        visible_instances = np.where(instance_visible)[0]

        # --- Step B: per-frame 2D detection -----------------------------
        color_path = str(self.scene_dir / "color" / f"{frame_idx}.jpg")
        frame_preds_2d = self.openyolo3d.network_2d.inference_detector([color_path])
        # ``inference_detector`` returns a dict keyed by the file stem; pull
        # out the single-frame entry for the accumulator (which expects a
        # flat ``{"bbox","labels","scores"}`` dict).
        if frame_preds_2d:
            single_frame_bbox = next(iter(frame_preds_2d.values()))
        else:
            single_frame_bbox = {
                "bbox": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.long),
                "scores": torch.empty((0,)),
            }

        # --- Step C: M11/12 instance registration gate ------------------
        # Default: pass-through (baseline). M11/M12 gate visible_instances
        # against accumulated detection history.
        if self.method_11 is not None:
            confirmed_visible = np.asarray(
                self.method_11.gate(visible_instances), dtype=np.int64
            )
        elif self.method_12 is not None:
            confirmed_visible = np.asarray(
                self.method_12.gate(visible_instances), dtype=np.int64
            )
        else:
            confirmed_visible = visible_instances

        # --- Step D: feed baseline accumulator + record label snapshot --
        # M21/M22 hooks observe the per-frame data alongside the baseline
        # accumulator. They do not replace ``add_frame``; instead they keep
        # their own state and override ``compute_predictions()`` semantics
        # (resolved at end-of-scene via the hook's ``finalize`` method).
        if self.baseline_accumulator is not None:
            self.baseline_accumulator.add_frame(
                projection=projection,
                visible_mask=inside_mask,
                bbox_pred=single_frame_bbox,
            )
        if self.method_21 is not None and hasattr(self.method_21, "observe_frame"):
            self.method_21.observe_frame(
                frame_idx=frame_idx,
                visible_instances=confirmed_visible,
                projection=projection,
                inside_mask=inside_mask,
                bbox_pred=single_frame_bbox,
                instance_vertex_masks=self.instance_vertex_masks,
                scene_vertices=self.scene_vertices,
            )
        if self.method_22 is not None and hasattr(self.method_22, "observe_frame"):
            self.method_22.observe_frame(
                frame_idx=frame_idx,
                visible_instances=confirmed_visible,
                color_path=color_path,
                bbox_pred=single_frame_bbox,
            )

        # Lightweight per-frame snapshot (label-only). Confirmed visible
        # instances get a placeholder -1 label here; the final-frame label
        # comes from compute_baseline_predictions(). Task 1.2b deliberately
        # does NOT call the accumulator each frame (O(F·V·K) cost) — Task
        # 1.3 / 1.4 will add checkpoint-based per-frame mAP if needed.
        current_instance_map: dict[int, int] = {int(k): -1 for k in confirmed_visible}

        # --- Step E: M31/32 spatial merging -----------------------------
        # TODO(Task 1.4 M31/M32): IoU-based / Hungarian merge step.

        # --- Step F: history bookkeeping for Stage 3 metrics ------------
        self.pred_history.append(current_instance_map)

        return {
            "frame_idx": frame_idx,
            "visible_instances": visible_instances,
            "current_instance_map": current_instance_map,
            "frame_preds_2d": frame_preds_2d,
        }

    # ------------------------------------------------------------------
    # Final prediction (offline-equivalent)
    # ------------------------------------------------------------------

    def compute_baseline_predictions(self) -> dict:
        """Drain the baseline accumulator and return per-instance preds.

        Returns:
            ``{"pred_masks": np.ndarray (V, K_out) bool,
               "pred_classes": np.ndarray (K_out,) int,
               "pred_scores":  np.ndarray (K_out,) float}``
        Final-frame call should match ``OpenYolo3D.predict(...)`` output
        bit-for-bit modulo CPU vs CUDA float ordering.
        """
        if self.baseline_accumulator is None:
            raise RuntimeError("Call setup_scene() before compute_baseline_predictions().")
        pred_masks, pred_classes, pred_scores = (
            self.baseline_accumulator.compute_predictions()
        )
        return {
            "pred_masks": pred_masks.cpu().numpy(),
            "pred_classes": pred_classes.cpu().numpy(),
            "pred_scores": pred_scores.cpu().numpy(),
        }

    def compute_method_predictions(self) -> dict:
        """End-of-scene prediction with the currently-installed method axes.

        Pipeline order (Stage 2 §3 of Task 1.1 streaming design):
            1. Start from baseline preds.
            2. If M21 or M22 is active, ask it to ``finalize`` and override
               class assignments.
            3. If M31 or M32 is active, run ``merge`` over the final
               instance map.

        Returns ``{"pred_masks", "pred_classes", "pred_scores"}`` numpy
        arrays in the same schema as :meth:`compute_baseline_predictions`.
        """
        preds = self.compute_baseline_predictions()

        label_method = self.method_22 or self.method_21
        if label_method is not None and hasattr(label_method, "finalize"):
            try:
                overridden = label_method.finalize(
                    pred_masks=preds["pred_masks"],
                    pred_classes=preds["pred_classes"],
                    pred_scores=preds["pred_scores"],
                    instance_vertex_masks=self.instance_vertex_masks,
                )
                if overridden is not None:
                    preds = overridden
            except NotImplementedError:
                pass

        merger = self.method_32 or self.method_31
        if merger is not None and hasattr(merger, "merge"):
            try:
                merged = merger.merge(
                    pred_masks=preds["pred_masks"],
                    pred_classes=preds["pred_classes"],
                    pred_scores=preds["pred_scores"],
                    scene_vertices=self.scene_vertices,
                )
                if merged is not None:
                    preds = merged
            except NotImplementedError:
                pass

        return preds

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_streaming(self) -> list[dict]:
        """Run setup_scene() then iterate every frame in order."""
        self.setup_scene()
        return [self.step_frame(t) for t in self.frame_indices]
