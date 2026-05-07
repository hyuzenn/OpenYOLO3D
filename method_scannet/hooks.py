"""Runtime hooks that install METHOD_21 + METHOD_31 onto utils.OpenYolo3D
without editing the upstream file.

Usage:
    from method_scannet.hooks import install_phase1
    install_phase1()  # before constructing or calling OpenYolo3D

The installs are idempotent and reversible (uninstall_*). Originals are stashed
on the class so they can be restored without re-importing.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch

from utils import OpenYolo3D
from utils.utils_3d import Network_3D  # noqa: F401  (kept to ensure utils side-effects load)
from utils import get_visibility_mat, compute_iou

from method_scannet.method_21_weighted_voting import WeightedVoting
from method_scannet.method_31_iou_merging import IoUMerger


# ---- Per-scene caches attached to the world2cam helper -----------------------


def _ensure_vertex_coords(world2cam) -> np.ndarray:
    """Return (n_vertices, 3) world-frame vertex coords for the current scene.

    Cached on the world2cam instance to avoid re-reading the .ply for every
    instance / for METHOD_31 separately.
    """
    cached = getattr(world2cam, "_method_vertex_coords", None)
    if cached is not None:
        return cached
    coords_h, _colors = type(world2cam).load_ply(world2cam.mesh)
    coords = np.asarray(coords_h)[:, :3].astype(np.float64)
    world2cam._method_vertex_coords = coords
    return coords


def _ensure_camera_positions(world2cam) -> np.ndarray:
    """Return (n_frames, 3) camera world-positions in the same order as
    world2cam.poses (and therefore as projections_mesh_to_frame).
    """
    cached = getattr(world2cam, "_method_camera_positions", None)
    if cached is not None:
        return cached
    positions = []
    for p in world2cam.poses:
        pose = np.loadtxt(p)
        positions.append(pose[:3, 3].astype(np.float64))
    out = np.stack(positions, axis=0) if positions else np.zeros((0, 3), dtype=np.float64)
    world2cam._method_camera_positions = out
    return out


# ---- METHOD_21 patched method ------------------------------------------------


def _patched_label_3d_masks_from_label_maps(
    self,
    prediction_3d_masks,
    predictions_2d_bboxes,
    projections_mesh_to_frame,
    keep_visible_points,
    is_gt,
):
    """Drop-in replacement for OpenYolo3D.label_3d_masks_from_label_maps that
    routes per-pixel label evidence through method_scannet.WeightedVoting.

    Output contract is preserved: (predicted_masks (V, K) bool,
    pred_classes (K,) long, pred_scores (K,) float). Top-k branch behavior
    matches the original when topk_per_image != -1 and not is_gt.
    """
    voter: WeightedVoting = getattr(self, "_method21_voter")

    label_maps = self.construct_label_maps(predictions_2d_bboxes)

    visibility_matrix = get_visibility_mat(
        prediction_3d_masks.cuda().permute(1, 0),
        keep_visible_points.cuda(),
        topk=25 if is_gt else self.openyolo3d_config["openyolo3d"]["topk"],
    )
    valid_frames = visibility_matrix.sum(dim=0) >= 1

    prediction_3d_masks = prediction_3d_masks.permute(1, 0).cpu()
    prediction_3d_masks_np = prediction_3d_masks.numpy()
    projections_mesh_to_frame_v = projections_mesh_to_frame[valid_frames].cpu().numpy()
    visibility_matrix_v = visibility_matrix[:, valid_frames].cpu().numpy()
    keep_visible_points_v = keep_visible_points[valid_frames].cpu().numpy()
    label_maps_v = label_maps[valid_frames].numpy()
    bounding_boxes = list(predictions_2d_bboxes.values())
    valid_frames_cpu = valid_frames.cpu()
    bounding_boxes_valid = [bbox for (bi, bbox) in enumerate(bounding_boxes) if bool(valid_frames_cpu[bi])]
    valid_frame_idx = torch.where(valid_frames_cpu)[0].numpy()

    vertex_coords = _ensure_vertex_coords(self.world2cam)
    camera_positions_full = _ensure_camera_positions(self.world2cam)
    camera_positions_v = camera_positions_full[valid_frame_idx] if len(valid_frame_idx) > 0 else camera_positions_full[:0]

    H = self.world2cam.height
    W = self.world2cam.width
    image_size = (W, H)

    distributions = []
    class_labels = []
    class_probs = []

    img_cx = image_size[0] / 2.0
    img_cy = image_size[1] / 2.0
    inv_dist_decay = 1.0 / float(voter.distance_weight_decay)
    inv_center_decay = 1.0 / float(voter.center_weight_decay)
    alpha = float(voter.spatial_alpha)

    for mask_id, mask in enumerate(prediction_3d_masks_np):
        if mask.sum() > 0:
            instance_centroid = vertex_coords[mask].mean(axis=0)
        else:
            instance_centroid = np.zeros(3, dtype=np.float64)

        prob_normalizer = 0
        representitive_frame_ids = np.where(visibility_matrix_v[mask_id])[0]
        labels_chunks = []
        weights_chunks = []
        all_labels_for_norm = []
        iou_vals = []

        for representitive_frame_id in representitive_frame_ids:
            visible_points_mask = (
                keep_visible_points_v[representitive_frame_id].squeeze() * mask
            ).astype(bool)
            n_visible = int(visible_points_mask.sum())
            prob_normalizer += n_visible
            instance_x_y_coords = projections_mesh_to_frame_v[representitive_frame_id][
                np.where(visible_points_mask)
            ].astype(np.int64)

            boxes = bounding_boxes_valid[representitive_frame_id]["bbox"].long()
            confidence = 1.0
            if len(boxes) > 0 and len(instance_x_y_coords) > 10:
                x_l = instance_x_y_coords[:, 0].min()
                x_r = instance_x_y_coords[:, 0].max() + 1
                y_t = instance_x_y_coords[:, 1].min()
                y_b = instance_x_y_coords[:, 1].max() + 1
                box = torch.tensor(
                    [
                        x_l / self.scaling_params[1],
                        y_t / self.scaling_params[0],
                        x_r / self.scaling_params[1],
                        y_b / self.scaling_params[0],
                    ]
                )
                iou_values = compute_iou(box, boxes)
                iou_max_val = float(iou_values.max().item())
                iou_vals.append(iou_max_val)
                confidence = iou_max_val
                bbox_cx = float((x_l + x_r) / 2.0)
                bbox_cy = float((y_t + y_b) / 2.0)
            elif len(instance_x_y_coords) > 0:
                bbox_cx = float(instance_x_y_coords[:, 0].mean())
                bbox_cy = float(instance_x_y_coords[:, 1].mean())
            else:
                bbox_cx = img_cx
                bbox_cy = img_cy

            if len(instance_x_y_coords) > 0:
                selected_labels = label_maps_v[
                    representitive_frame_id,
                    instance_x_y_coords[:, 1],
                    instance_x_y_coords[:, 0],
                ]
            else:
                selected_labels = np.array([-1], dtype=np.int16)

            valid_arr = selected_labels[selected_labels != -1]
            if valid_arr.size == 0 or confidence <= 0:
                continue

            cam = camera_positions_v[representitive_frame_id]
            d3 = math.sqrt(
                (cam[0] - instance_centroid[0]) ** 2
                + (cam[1] - instance_centroid[1]) ** 2
                + (cam[2] - instance_centroid[2]) ** 2
            )
            w_dist = math.exp(-d3 * inv_dist_decay)
            d2 = math.sqrt((bbox_cx - img_cx) ** 2 + (bbox_cy - img_cy) ** 2)
            w_center = math.exp(-d2 * inv_center_decay)
            frame_w = (alpha * w_dist + (1.0 - alpha) * w_center) * confidence
            if frame_w <= 0:
                continue

            labels_chunks.append(valid_arr)
            weights_chunks.append(np.full(valid_arr.shape[0], frame_w, dtype=np.float32))
            all_labels_for_norm.append(valid_arr)

        if labels_chunks:
            all_labels = np.concatenate(labels_chunks)
            all_weights = np.concatenate(weights_chunks)
            counts = np.bincount(
                all_labels.astype(np.int64), weights=all_weights, minlength=self.num_classes
            ).astype(np.float32)
            per_inst_dist = torch.from_numpy(counts.copy())
            m = float(per_inst_dist.max().item())
            if m > 0:
                per_inst_dist = per_inst_dist / m
        else:
            per_inst_dist = torch.zeros(self.num_classes, dtype=torch.float32)
            per_inst_dist[-1] = 1.0

        if all_labels_for_norm:
            labels_distribution = np.concatenate(all_labels_for_norm)
        else:
            labels_distribution = np.array([], dtype=np.int16)

        if labels_distribution.size > 0:
            class_label = int(per_inst_dist.argmax().item())
            class_prob = float(
                (labels_distribution == class_label).sum() / max(prob_normalizer, 1)
            )
        else:
            class_label = self.num_classes - 1
            class_prob = 0.0

        if iou_vals:
            iv = np.asarray(iou_vals, dtype=np.float32)
            iv = iv[iv != 0]
            iou_prob = float(iv.mean()) if iv.size > 0 else 0.0
        else:
            iou_prob = 0.0

        class_labels.append(class_label)
        class_probs.append(class_prob * iou_prob)
        distributions.append(per_inst_dist)

    pred_classes = torch.tensor(class_labels)
    pred_scores = torch.tensor(class_probs)

    use_topk = self.openyolo3d_config["openyolo3d"]["topk_per_image"] != -1
    if use_topk:
        if len(distributions) > 0:
            distributions = torch.stack(distributions)
        else:
            distributions = torch.zeros((0, self.num_classes))

        if not is_gt:
            n_instance = distributions.shape[0]
            distributions = distributions.reshape(-1)
            labels = (
                torch.arange(self.num_classes, device=distributions.device)
                .unsqueeze(0)
                .repeat(n_instance, 1)
                .flatten(0, 1)
            )
            cur_topk = self.openyolo3d_config["openyolo3d"]["topk_per_image"]
            _, idx = torch.topk(
                distributions, k=min(cur_topk, len(distributions)), largest=True
            )
            mask_idx = torch.div(idx, self.num_classes, rounding_mode="floor")
            pred_classes = labels[idx]
            pred_scores = distributions[idx].cuda()
            prediction_3d_masks = prediction_3d_masks[mask_idx]

    return prediction_3d_masks.permute(1, 0), pred_classes, pred_scores


# ---- METHOD_31 patched method ------------------------------------------------


def _patched_label_3d_masks_from_2d_bboxes(self, scene_name, is_gt=False):
    """Run the (possibly METHOD_21-patched) labelling, then apply IoUMerger
    on the per-scene result before return.
    """
    original = OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    result = original(self, scene_name, is_gt=is_gt)
    merger: IoUMerger = getattr(self, "_method31_merger")
    masks, classes, scores = result[scene_name]
    vertex_coords = _ensure_vertex_coords(self.world2cam)
    masks_m, classes_m, scores_m = merger.merge(masks, classes, scores, vertex_coords=vertex_coords)
    self.predicted_masks = masks_m
    self.predicated_classes = classes_m
    self.predicated_scores = scores_m
    return {scene_name: (masks_m, classes_m, scores_m)}


# ---- Public install / uninstall ---------------------------------------------


_DEFAULT_VOTER = WeightedVoting()
_DEFAULT_MERGER = IoUMerger()


def install_method_21(
    voter: Optional[WeightedVoting] = None,
) -> None:
    if not hasattr(OpenYolo3D, "_original_label_3d_masks_from_label_maps"):
        OpenYolo3D._original_label_3d_masks_from_label_maps = (
            OpenYolo3D.label_3d_masks_from_label_maps
        )
    OpenYolo3D._method21_voter = voter if voter is not None else _DEFAULT_VOTER
    OpenYolo3D.label_3d_masks_from_label_maps = _patched_label_3d_masks_from_label_maps


def uninstall_method_21() -> None:
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_label_maps"):
        OpenYolo3D.label_3d_masks_from_label_maps = (
            OpenYolo3D._original_label_3d_masks_from_label_maps
        )
        del OpenYolo3D._original_label_3d_masks_from_label_maps
    if hasattr(OpenYolo3D, "_method21_voter"):
        del OpenYolo3D._method21_voter


def install_method_31(merger: Optional[IoUMerger] = None) -> None:
    if not hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        OpenYolo3D._original_label_3d_masks_from_2d_bboxes = (
            OpenYolo3D.label_3d_masks_from_2d_bboxes
        )
    OpenYolo3D._method31_merger = merger if merger is not None else _DEFAULT_MERGER
    OpenYolo3D.label_3d_masks_from_2d_bboxes = _patched_label_3d_masks_from_2d_bboxes


def uninstall_method_31() -> None:
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        OpenYolo3D.label_3d_masks_from_2d_bboxes = (
            OpenYolo3D._original_label_3d_masks_from_2d_bboxes
        )
        del OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    if hasattr(OpenYolo3D, "_method31_merger"):
        del OpenYolo3D._method31_merger


def install_phase1(
    voter: Optional[WeightedVoting] = None,
    merger: Optional[IoUMerger] = None,
) -> None:
    install_method_21(voter=voter)
    install_method_31(merger=merger)


def uninstall_phase1() -> None:
    uninstall_method_31()
    uninstall_method_21()


def install_method_31_only(merger: Optional[IoUMerger] = None) -> None:
    """Single-method ablation entry: METHOD_31 (3D IoU merging) only.

    METHOD_21 is **not** installed — the OpenYOLO3D pipeline runs the original
    pixel-mode label aggregation. The IoU merger is applied on the per-scene
    output identically to Phase 1.
    """
    install_method_31(merger=merger)


def uninstall_method_31_only() -> None:
    uninstall_method_31()


def install_method_21_only(voter: Optional[WeightedVoting] = None) -> None:
    """Single-method ablation entry: METHOD_21 (Weighted voting) only.

    METHOD_31 is **not** installed — the per-scene output passes through
    unchanged from `OpenYolo3D.label_3d_masks_from_2d_bboxes`. The
    WeightedVoting hook replaces the pixel-mode aggregation identically to
    Phase 1.
    """
    install_method_21(voter=voter)


def uninstall_method_21_only() -> None:
    uninstall_method_21()
