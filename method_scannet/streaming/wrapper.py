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
        # M22 needs a CLIP image encoder + the inference-subset class list.
        # Installer (install_method_22) sets these alongside method_22.
        self.method_22_encoder = None
        self.method_22_class_names: Optional[list] = None
        # Per-scene camera positions in pose order — needed by M21's
        # distance-weighted voting. Populated in step_frame.
        self._camera_positions: list[np.ndarray] = []
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
        # World-frame camera position is the translation column of the
        # camera-to-world pose (needed by M21 distance weighting). Recorded
        # in the same order as baseline_accumulator._projections.
        self._camera_positions.append(
            np.asarray(pose_cam_to_world[:3, 3], dtype=np.float64).reshape(-1)
        )

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

        # --- Step D: feed baseline accumulator -------------------------
        # M21/M22 do not replace ``add_frame``; the accumulator stores the
        # raw frame data and the method adapter replays it at finalize.
        if self.baseline_accumulator is not None:
            self.baseline_accumulator.add_frame(
                projection=projection,
                visible_mask=inside_mask,
                bbox_pred=single_frame_bbox,
            )

        # M22: per-frame CLIP image-feature accumulation. For each visible
        # instance, project its vertex set, find the matching YOLO 2D bbox
        # (best IoU), encode the cropped image with CLIP, EMA-accumulate.
        if (
            self.method_22 is not None
            and self.method_22_encoder is not None
            and len(confirmed_visible) > 0
        ):
            self._method22_per_frame(
                frame_idx=frame_idx,
                color_path=color_path,
                projection=projection,
                inside_mask=inside_mask,
                bbox_pred=single_frame_bbox,
                confirmed_visible=confirmed_visible,
            )

        # Lightweight per-frame snapshot (label-only). Confirmed visible
        # instances get a placeholder -1 label here; the final-frame label
        # comes from compute_baseline_predictions(). Task 1.2b deliberately
        # does NOT call the accumulator each frame (O(F·V·K) cost) — Task
        # 1.3 / 1.4 will add checkpoint-based per-frame mAP if needed.
        current_instance_map: dict[int, int] = {int(k): -1 for k in confirmed_visible}

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

    # ------------------------------------------------------------------
    # M22 per-frame CLIP feature accumulation
    # ------------------------------------------------------------------

    def _method22_per_frame(
        self,
        frame_idx: int,
        color_path: str,
        projection: np.ndarray,
        inside_mask: np.ndarray,
        bbox_pred: dict,
        confirmed_visible: np.ndarray,
        min_match_iou: float = 0.05,
    ) -> None:
        """For each confirmed-visible Mask3D instance, find the best-matching
        YOLO 2D bbox by IoU between the instance's projected AABB and the
        YOLO bboxes, crop the image at that bbox, CLIP-encode, and feed
        the embedding to ``self.method_22.update_instance_feature``.
        """
        import imageio
        from utils import compute_iou

        boxes_2d = bbox_pred.get("bbox")
        if boxes_2d is None or len(boxes_2d) == 0:
            return
        boxes_2d_long = boxes_2d.long() if hasattr(boxes_2d, "long") else torch.as_tensor(boxes_2d).long()
        try:
            image = imageio.imread(color_path)
        except Exception:
            return
        if image.ndim != 3 or image.shape[2] != 3:
            return

        sx = self.scaling_w
        sy = self.scaling_h
        instance_masks = self.instance_vertex_masks  # (K, V) bool
        proj_int = np.asarray(projection, dtype=np.int64)

        crop_bboxes: list[list[int]] = []
        crop_prop_ids: list[int] = []
        for prop_idx in confirmed_visible:
            visible_pts = inside_mask & instance_masks[int(prop_idx)]
            n_v = int(visible_pts.sum())
            if n_v < 10:
                continue
            xy = proj_int[visible_pts]
            if xy.shape[0] < 1:
                continue
            x_l = int(xy[:, 0].min())
            x_r = int(xy[:, 0].max()) + 1
            y_t = int(xy[:, 1].min())
            y_b = int(xy[:, 1].max()) + 1
            box_img = torch.tensor(
                [x_l / sx, y_t / sy, x_r / sx, y_b / sy], dtype=torch.float32
            )
            ious = compute_iou(box_img, boxes_2d_long.float())
            iou_max = float(ious.max().item())
            if iou_max < min_match_iou:
                continue
            iou_argmax = int(ious.argmax().item())
            best_box = boxes_2d_long[iou_argmax].cpu().numpy()
            x1, y1, x2, y2 = (int(v) for v in best_box)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_bboxes.append([x1, y1, x2, y2])
            crop_prop_ids.append(int(prop_idx))

        if not crop_bboxes:
            return
        bboxes_arr = np.asarray(crop_bboxes, dtype=np.int64)
        try:
            embeddings = self.method_22_encoder.encode_cropped_bboxes(image, bboxes_arr)
        except Exception:
            return
        for prop_idx, emb in zip(crop_prop_ids, embeddings):
            self.method_22.update_instance_feature(prop_idx, emb)

    # ------------------------------------------------------------------
    # Method-axis finalize
    # ------------------------------------------------------------------

    def compute_method_predictions(self) -> dict:
        """End-of-scene prediction with the currently-installed method axes.

        Pipeline order (Task 1.4a redesign — May class signatures preserved):
            1. Label assignment: baseline / M21 / M22 (choose one).
            2. Registration filter: M11 / M12 confirmed set.
            3. Spatial merge: M31 / M32 (choose one).

        Returns ``{"pred_masks", "pred_classes", "pred_scores"}`` numpy
        arrays in the same schema as :meth:`compute_baseline_predictions`.
        """
        from method_scannet.streaming import method_adapters as _ma

        # ---- 1. Label assignment ---------------------------------------
        if self.method_21 is not None:
            camera_positions = (
                np.stack(self._camera_positions, axis=0)
                if self._camera_positions
                else np.zeros((0, 3), dtype=np.float64)
            )
            img_h, img_w = self.image_resolution
            (
                pm_t,
                pc_t,
                ps_t,
                mask_idx,
            ) = _ma.compute_predictions_method21(
                accumulator=self.baseline_accumulator,
                voter=self.method_21,
                scene_vertices=self.scene_vertices,
                camera_positions=camera_positions,
                image_width=int(img_w),
                image_height=int(img_h),
            )
            preds = {
                "pred_masks": pm_t.cpu().numpy().astype(bool),
                "pred_classes": pc_t.cpu().numpy().astype(np.int64),
                "pred_scores": ps_t.cpu().numpy().astype(np.float32),
            }
        elif self.method_22 is not None:
            (
                pm_t,
                pc_t,
                ps_t,
                mask_idx,
            ) = _ma.compute_predictions_method22(
                accumulator=self.baseline_accumulator,
                fusion=self.method_22,
                topk_per_image=self.topk_per_image,
            )
            preds = {
                "pred_masks": pm_t.cpu().numpy().astype(bool),
                "pred_classes": pc_t.cpu().numpy().astype(np.int64),
                "pred_scores": ps_t.cpu().numpy().astype(np.float32),
            }
        else:
            preds = self.compute_baseline_predictions()
            mask_idx = self.baseline_accumulator._last_mask_idx

        # ---- 2. Registration filter (M11 / M12) ------------------------
        if self.method_11 is not None:
            confirmed_set = set(int(i) for i in self.method_11._confirmed)
            preds = _ma.apply_registration_filter(preds, mask_idx, confirmed_set)
            mask_idx = self._filter_mask_idx(mask_idx, preds, confirmed_set)
        elif self.method_12 is not None:
            confirmed_set = set(int(i) for i in self.method_12._confirmed)
            preds = _ma.apply_registration_filter(preds, mask_idx, confirmed_set)
            mask_idx = self._filter_mask_idx(mask_idx, preds, confirmed_set)

        # ---- 3. Spatial merge ------------------------------------------
        if self.method_31 is not None:
            preds = _ma.apply_method31_merge(
                preds, self.method_31, self.scene_vertices
            )
        elif self.method_32 is not None:
            instance_features = (
                self.method_22.instance_features if self.method_22 is not None else None
            )
            preds = _ma.apply_method32_merge(
                preds,
                self.method_32,
                self.scene_vertices,
                mask_idx=mask_idx,
                instance_features=instance_features,
                class_aware=True,
            )

        return preds

    @staticmethod
    def _filter_mask_idx(mask_idx, preds: dict, confirmed_set: set):
        """Sync mask_idx to the filtered preds (same boolean keep order)."""
        if mask_idx is None:
            return None
        if not confirmed_set:
            return torch.zeros(0, dtype=torch.long)
        mi = mask_idx.cpu().numpy() if hasattr(mask_idx, "cpu") else np.asarray(mask_idx)
        keep = np.array([int(p) in confirmed_set for p in mi], dtype=bool)
        return torch.from_numpy(mi[keep]).long()

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run_streaming(self) -> list[dict]:
        """Run setup_scene() then iterate every frame in order."""
        self.setup_scene()
        return [self.step_frame(t) for t in self.frame_indices]
