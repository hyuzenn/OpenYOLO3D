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


# =============================================================================
# METHOD_22 (Per-instance CLIP-image-feature label re-classification)
# =============================================================================

# Module-level cache shared across scenes (model load is expensive).
_method22_state: dict = {
    "fusion": None,           # FeatureFusionEMA instance (rebuilt per install)
    "image_encoder": None,    # CLIPImageEncoder (CPU)
    "prompt_data": None,      # raw .pt payload
    "n_inference_classes": None,
    "min_match_iou": 0.05,
    "topk_per_inst": 15,      # frames per instance for fusion (vs. config topk=40)
}


def _ensure_method22_resources(prompt_embeddings_path: str, ema_alpha: float, use_inference_subset: bool):
    """Lazy-load CLIP encoder and prompt embeddings; cache module-level so we
    don't reload on every scene. Builds a fresh FeatureFusionEMA instance.
    """
    from method_scannet.method_22_feature_fusion import FeatureFusionEMA
    from method_scannet.clip_image_encoder import CLIPImageEncoder

    if _method22_state["prompt_data"] is None:
        _method22_state["prompt_data"] = torch.load(
            prompt_embeddings_path, map_location="cpu"
        )

    pdata = _method22_state["prompt_data"]
    if use_inference_subset:
        inference_names = pdata.get("openyolo3d_inference_classes")
        if inference_names is None:
            raise KeyError(
                "openyolo3d_inference_classes missing from prompt embeddings .pt"
            )
        all_names = list(pdata["class_names"])
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        keep = [name_to_idx[n] for n in inference_names if n in name_to_idx]
        embeddings = pdata["embeddings"][keep]
        class_names = [all_names[i] for i in keep]
    else:
        embeddings = pdata["embeddings"]
        class_names = list(pdata["class_names"])

    fusion = FeatureFusionEMA(
        ema_alpha=ema_alpha,
        prompt_embeddings=embeddings,
        prompt_class_names=class_names,
    )
    _method22_state["fusion"] = fusion
    _method22_state["n_inference_classes"] = len(class_names)

    if _method22_state["image_encoder"] is None:
        _method22_state["image_encoder"] = CLIPImageEncoder(variant=pdata["clip_variant"])


def _apply_method_22(self, scene_name: str, is_gt: bool):
    """METHOD_22 re-classification.

    Runs the original `label_3d_masks_from_2d_bboxes` to seed the pipeline
    state, then re-emits per-mask predictions where each unique 3D mask
    proposal is classified via CLIP-image-feature EMA → cosine-similarity
    against pre-extracted text-prompt embeddings.

    Returns the same `{scene_name: (masks, classes, scores)}` contract.
    """
    import imageio.v2 as imageio

    state = _method22_state
    fusion = state["fusion"]
    encoder = state["image_encoder"]
    n_classes = state["n_inference_classes"]
    min_iou = state["min_match_iou"]
    topk_per_inst = state["topk_per_inst"]

    # Run original to populate self.preds_2d, self.preds_3d, self.mesh_projections
    # (predict() already populated those before label_3d_masks_from_2d_bboxes
    # was called, but we still need the original to set self.predicted_masks
    # etc. for backward compatibility with downstream code that reads them).
    original = OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    _ = original(self, scene_name, is_gt=is_gt)  # discard; we re-emit below.

    # Reset per-scene fusion
    fusion.instance_features.clear()

    # Use 3D mask proposals (pre-topk-expansion) as our unique-mask space.
    # self.preds_3d = (mask_tensor (V, n_proposals) bool, scores (n_proposals,))
    raw_masks_t = self.preds_3d[0].bool()  # (V, n_proposals)
    if raw_masks_t.dim() != 2:
        raise RuntimeError(f"preds_3d masks unexpected shape {tuple(raw_masks_t.shape)}")
    V, n_proposals = raw_masks_t.shape

    if n_proposals == 0:
        # Edge case — empty scene. Emit nothing.
        empty_masks = torch.zeros(V, 0, dtype=torch.bool)
        empty_classes = torch.zeros(0, dtype=torch.long)
        empty_scores = torch.zeros(0, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.zeros(0, dtype=torch.float32)
        self.predicted_masks = empty_masks
        self.predicated_classes = empty_classes
        self.predicated_scores = empty_scores
        return {scene_name: (empty_masks, empty_classes, empty_scores)}

    # Visibility: which frames see each proposal (top-k coverage).
    keep_visible_points = self.mesh_projections[1]  # (n_frames, V)
    proposals_VK = raw_masks_t  # (V, K=n_proposals)
    vis = get_visibility_mat(
        proposals_VK.cuda().permute(1, 0),  # (K, V) on cuda
        keep_visible_points.cuda(),
        topk=self.openyolo3d_config["openyolo3d"]["topk"],
    )  # returns (K, n_frames) bool
    valid_frames = (vis.sum(dim=0) >= 1).cpu()
    valid_frame_idx = torch.where(valid_frames)[0].numpy()

    if len(valid_frame_idx) == 0:
        return _emit_empty_predictions(self, scene_name, V)

    proj_pts = self.mesh_projections[0].cpu().numpy()  # (n_frames, V, 2) int (depth res)
    keep_vis_np = keep_visible_points.cpu().numpy()    # (n_frames, V)
    masks_np = proposals_VK.cpu().numpy()              # (V, K)

    # Frame -> 2D bbox dict (image res)
    frame_id_list = list(self.preds_2d.keys())
    bbox_per_frame = list(self.preds_2d.values())

    # Cap frames per instance to limit CLIP cost.
    vis_per_inst = vis.cpu().numpy()  # (K, n_frames)

    # Build per-frame work list:  frame_idx -> [(prop_idx, depth_aabb), ...]
    frame_work: dict = {}
    # For each proposal, pick top-N most-covered frames.
    for prop_idx in range(n_proposals):
        coverages = []
        for f_idx in valid_frame_idx:
            if not vis_per_inst[prop_idx, f_idx]:
                continue
            visible_pts = (keep_vis_np[f_idx].squeeze() & masks_np[:, prop_idx])
            n_v = int(visible_pts.sum())
            if n_v < 10:
                continue  # too few pixels to be a meaningful match
            coverages.append((f_idx, n_v, visible_pts))
        if not coverages:
            continue
        coverages.sort(key=lambda t: -t[1])
        for f_idx, n_v, visible_pts in coverages[:topk_per_inst]:
            xy = proj_pts[f_idx][np.where(visible_pts)].astype(np.int64)
            if len(xy) < 1:
                continue
            x_l, x_r = int(xy[:, 0].min()), int(xy[:, 0].max() + 1)
            y_t, y_b = int(xy[:, 1].min()), int(xy[:, 1].max() + 1)
            # depth coords; convert to image coords for bbox match
            depth_aabb = (x_l, y_t, x_r, y_b)
            frame_work.setdefault(int(f_idx), []).append((prop_idx, depth_aabb))

    # Per-frame: load image, build matching bbox per proposal, batch encode.
    color_paths = self.world2cam.color_paths
    sx = self.scaling_params[1]
    sy = self.scaling_params[0]

    for f_idx, items in frame_work.items():
        if not items:
            continue
        bboxes_2d_t = bbox_per_frame[f_idx]["bbox"].long()
        if len(bboxes_2d_t) == 0:
            continue

        try:
            image = imageio.imread(color_paths[f_idx])
        except Exception:
            continue
        if image.ndim != 3 or image.shape[2] != 3:
            continue

        # For each (prop_idx, depth_aabb) in this frame, find best 2D bbox match
        matched_bboxes = []
        matched_prop_ids = []
        for prop_idx, (x_l, y_t, x_r, y_b) in items:
            # Depth coords → image coords
            box_img = torch.tensor(
                [x_l / sx, y_t / sy, x_r / sx, y_b / sy], dtype=torch.float32
            )
            ious = compute_iou(box_img, bboxes_2d_t.float())
            iou_max, iou_argmax = float(ious.max().item()), int(ious.argmax().item())
            if iou_max < min_iou:
                continue
            best_bbox = bboxes_2d_t[iou_argmax].cpu().numpy()
            x1, y1, x2, y2 = (int(v) for v in best_bbox)
            if x2 <= x1 or y2 <= y1:
                continue
            matched_bboxes.append([x1, y1, x2, y2])
            matched_prop_ids.append(prop_idx)

        if not matched_bboxes:
            continue

        bboxes_arr = np.asarray(matched_bboxes, dtype=np.int64)
        try:
            embs = encoder.encode_cropped_bboxes(image, bboxes_arr)  # (n, 512) cpu
        except Exception:
            continue

        for prop_idx, emb in zip(matched_prop_ids, embs):
            fusion.update_instance_feature(prop_idx, emb)

    # Build (n_proposals × n_inference_classes) cosine distribution.
    # Proposals with no fusion feature get an all-zeros row → won't make top-k.
    prompts = fusion._prompt_emb_norm  # (n_classes, D)
    if prompts is None:
        return _emit_empty_predictions(self, scene_name, V)

    distribution = torch.zeros(n_proposals, n_classes, dtype=torch.float32)
    has_feature = []
    for prop_idx in range(n_proposals):
        feat = fusion.get_feature(prop_idx)
        if feat is None:
            continue
        has_feature.append(prop_idx)
        f_norm = feat.float()
        f_norm = f_norm / f_norm.norm().clamp_min(1e-12)
        sims = (f_norm.unsqueeze(0) @ prompts.t()).squeeze(0).clamp_min(0.0)
        distribution[prop_idx] = sims

    if not has_feature:
        return _emit_empty_predictions(self, scene_name, V)

    # Top-K from flattened (n_proposals × n_classes) distribution.
    topk_per_image = self.openyolo3d_config["openyolo3d"]["topk_per_image"]
    flat = distribution.reshape(-1)  # (n_proposals * n_classes,)
    if topk_per_image is None or topk_per_image == -1:
        # Fall back: one prediction per proposal (argmax cosine).
        # (Rarely needed; ScanNet config always has topk_per_image > 0.)
        n_emit = n_proposals
        idx = torch.tensor(
            [
                p * n_classes + int(distribution[p].argmax().item())
                for p in has_feature
            ],
            dtype=torch.long,
        )
    else:
        cur_topk = min(int(topk_per_image), int(flat.numel()))
        _, idx = torch.topk(flat, k=cur_topk, largest=True)

    # Decompose flat idx into (prop_idx, class_idx)
    prop_idx_t = torch.div(idx, n_classes, rounding_mode="floor")
    class_idx_t = idx - prop_idx_t * n_classes
    pred_scores_t = flat[idx]

    # Drop rows whose score is exactly 0 (no fusion feature) — these would just
    # be padding; better to emit fewer predictions than zero-score noise.
    nonzero_mask = pred_scores_t > 0
    if nonzero_mask.sum().item() == 0:
        return _emit_empty_predictions(self, scene_name, V)
    prop_idx_t = prop_idx_t[nonzero_mask]
    class_idx_t = class_idx_t[nonzero_mask]
    pred_scores_t = pred_scores_t[nonzero_mask]

    # Reassemble masks (V, K).
    pred_masks_VK = proposals_VK.cpu()[:, prop_idx_t]  # (V, K)
    pred_classes_t = class_idx_t.long()
    pred_scores_out = pred_scores_t.float()
    if torch.cuda.is_available():
        pred_scores_out = pred_scores_out.cuda()

    self.predicted_masks = pred_masks_VK
    self.predicated_classes = pred_classes_t
    self.predicated_scores = pred_scores_out
    return {scene_name: (pred_masks_VK, pred_classes_t, pred_scores_out)}


def _emit_empty_predictions(self, scene_name: str, V: int):
    empty_masks = torch.zeros(V, 0, dtype=torch.bool)
    empty_classes = torch.zeros(0, dtype=torch.long)
    empty_scores = torch.zeros(0, dtype=torch.float32)
    if torch.cuda.is_available():
        empty_scores = empty_scores.cuda()
    self.predicted_masks = empty_masks
    self.predicated_classes = empty_classes
    self.predicated_scores = empty_scores
    return {scene_name: (empty_masks, empty_classes, empty_scores)}


def _patched_label_3d_masks_from_2d_bboxes_22(self, scene_name, is_gt=False):
    """METHOD_22-only patched method."""
    return _apply_method_22(self, scene_name, is_gt)


def install_method_22_only(
    prompt_embeddings_path: str = "pretrained/scannet200_prompt_embeddings.pt",
    ema_alpha: float = 0.7,
    use_inference_subset: bool = True,
    min_match_iou: float = 0.05,
    topk_per_inst: int = 15,
) -> None:
    """METHOD_22 (FeatureFusionEMA) replaces text-prompt label decision.

    METHOD_31 / METHOD_32 are NOT installed. Refuses to install if another
    label_3d_masks_from_2d_bboxes patch is already active.
    """
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        raise RuntimeError(
            "label_3d_masks_from_2d_bboxes is already patched (METHOD_31 or "
            "METHOD_22 already installed). Uninstall first."
        )

    _ensure_method22_resources(prompt_embeddings_path, ema_alpha, use_inference_subset)
    _method22_state["min_match_iou"] = float(min_match_iou)
    _method22_state["topk_per_inst"] = int(topk_per_inst)

    OpenYolo3D._original_label_3d_masks_from_2d_bboxes = (
        OpenYolo3D.label_3d_masks_from_2d_bboxes
    )
    OpenYolo3D.label_3d_masks_from_2d_bboxes = _patched_label_3d_masks_from_2d_bboxes_22


def uninstall_method_22_only() -> None:
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        OpenYolo3D.label_3d_masks_from_2d_bboxes = (
            OpenYolo3D._original_label_3d_masks_from_2d_bboxes
        )
        del OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    _method22_state["fusion"] = None
    # Keep image_encoder cached — reload is slow.


# =============================================================================
# METHOD_32 (Hungarian-assignment merging based on spatial + semantic cost)
# =============================================================================

_method32_state: dict = {
    "merger": None,
    "class_aware": True,
    "semantic_threshold": 0.3,        # used when features are available (Phase 2)
    "no_feature_semantic_threshold": -1.0,  # for METHOD_32-only with no features
    "spatial_alpha": 0.5,
    "distance_threshold": 2.0,
}


def _build_instance_list_from_predictions(masks_VK, classes_K, vertex_coords):
    """Convert pipeline output to a list of dicts compatible with
    HungarianMerger.merge.
    """
    from method_scannet.utils_bbox import compute_instance_aabbs_batch

    masks_T = masks_VK.permute(1, 0).contiguous().bool().cpu().numpy()  # (K, V)
    aabb_dict = compute_instance_aabbs_batch(masks_T, vertex_coords)

    instance_list = []
    for k in range(masks_T.shape[0]):
        c = aabb_dict[k]["centroid"]
        if c is None:
            continue  # skip empty masks
        item = {
            "id": int(k),
            "label": int(classes_K[k].item()),
            "centroid": c,
        }
        aabb = aabb_dict[k]["aabb"]
        if aabb is not None:
            item["bbox_3d"] = aabb.reshape(2, 3)
        instance_list.append(item)
    return instance_list


def _apply_method_32(
    masks_VK,
    classes_K,
    scores_K,
    vertex_coords,
    instance_features=None,
):
    """Run HungarianMerger over the prediction tuple. Returns reduced
    (masks, classes, scores).

    `instance_features` is a dict {prediction_idx -> 1D tensor}. If None or
    empty, the merger uses `no_feature_semantic_threshold` so the spatial
    distance is the only gate.
    """
    from method_scannet.method_32_hungarian_merging import HungarianMerger
    from collections import defaultdict

    if classes_K.numel() == 0:
        return masks_VK, classes_K, scores_K

    state = _method32_state

    if instance_features is None or len(instance_features) == 0:
        eff_features = {}
        sem_thr = state["no_feature_semantic_threshold"]
    else:
        eff_features = instance_features
        sem_thr = state["semantic_threshold"]

    merger = HungarianMerger(
        spatial_alpha=state["spatial_alpha"],
        distance_threshold=state["distance_threshold"],
        semantic_threshold=sem_thr,
    )

    instance_list = _build_instance_list_from_predictions(
        masks_VK, classes_K, vertex_coords
    )
    if not instance_list:
        return masks_VK, classes_K, scores_K

    if state["class_aware"]:
        # Group by class, run Hungarian within each class.
        groups = defaultdict(list)
        for inst in instance_list:
            groups[inst["label"]].append(inst)
        kept_ids: set = set()
        for cls, insts in groups.items():
            sub_features = {
                int(i["id"]): eff_features[int(i["id"])]
                for i in insts
                if int(i["id"]) in eff_features
            }
            merged = merger.merge(insts, sub_features)
            for m in merged:
                kept_ids.add(int(m["id"]))
    else:
        merged = merger.merge(
            instance_list, {int(k): v for k, v in eff_features.items()}
        )
        kept_ids = {int(m["id"]) for m in merged}

    if not kept_ids:
        return masks_VK, classes_K, scores_K

    keep_idx = sorted(kept_ids)
    keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=masks_VK.device)
    new_masks = masks_VK[:, keep_idx_t]
    new_classes = classes_K[keep_idx_t]
    new_scores = scores_K[keep_idx_t] if scores_K.numel() else scores_K
    return new_masks, new_classes, new_scores


def _patched_label_3d_masks_from_2d_bboxes_32(self, scene_name, is_gt=False):
    """METHOD_32-only: original output → spatial-only Hungarian merge."""
    original = OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    out = original(self, scene_name, is_gt=is_gt)
    masks, classes, scores = out[scene_name]
    vertex_coords = _ensure_vertex_coords(self.world2cam)
    masks_m, classes_m, scores_m = _apply_method_32(
        masks, classes, scores, vertex_coords, instance_features=None
    )
    self.predicted_masks = masks_m
    self.predicated_classes = classes_m
    self.predicated_scores = scores_m
    return {scene_name: (masks_m, classes_m, scores_m)}


def install_method_32_only(
    spatial_alpha: float = 0.5,
    distance_threshold: float = 2.0,
    semantic_threshold: float = 0.3,
    class_aware: bool = True,
) -> None:
    """METHOD_32 (HungarianMerger) only — spatial-only when no METHOD_22
    features are present. Refuses to install if another label_3d_masks
    patch is active.
    """
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        raise RuntimeError(
            "label_3d_masks_from_2d_bboxes is already patched. Uninstall first."
        )
    _method32_state["spatial_alpha"] = float(spatial_alpha)
    _method32_state["distance_threshold"] = float(distance_threshold)
    _method32_state["semantic_threshold"] = float(semantic_threshold)
    _method32_state["class_aware"] = bool(class_aware)
    OpenYolo3D._original_label_3d_masks_from_2d_bboxes = (
        OpenYolo3D.label_3d_masks_from_2d_bboxes
    )
    OpenYolo3D.label_3d_masks_from_2d_bboxes = _patched_label_3d_masks_from_2d_bboxes_32


def uninstall_method_32_only() -> None:
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        OpenYolo3D.label_3d_masks_from_2d_bboxes = (
            OpenYolo3D._original_label_3d_masks_from_2d_bboxes
        )
        del OpenYolo3D._original_label_3d_masks_from_2d_bboxes


# =============================================================================
# Phase 2 (METHOD_22 + METHOD_32, shared CLIP features)
# =============================================================================


def _patched_label_3d_masks_from_2d_bboxes_phase2(self, scene_name, is_gt=False):
    """Phase 2: METHOD_22 re-classification → METHOD_32 Hungarian merge with
    shared per-proposal CLIP features.

    METHOD_22 emits up to topk_per_image predictions where each prediction k
    is a (proposal_idx, class) pair drawn from the per-proposal cosine
    distribution. The same proposal_idx may appear across multiple
    predictions with different class labels. METHOD_32 then merges these
    using `fusion.instance_features[proposal_idx]` as each prediction's
    visual feature.
    """
    state22 = _method22_state
    state32 = _method32_state
    fusion = state22["fusion"]
    if fusion is None:
        raise RuntimeError("Phase 2 patch active but METHOD_22 fusion not initialized")

    # 1. Run METHOD_22 re-classification (this also runs the original).
    out22 = _apply_method_22(self, scene_name, is_gt)
    masks_22, classes_22, scores_22 = out22[scene_name]

    if classes_22.numel() == 0:
        return out22

    # 2. Map each prediction k to its underlying proposal_idx by re-extracting
    # from the predicted_masks vs proposals identity. We saved no explicit
    # mapping; use mask-identity hashing on the (typically) ~600 prediction
    # columns vs the n_proposals proposal columns.
    raw_props = self.preds_3d[0].bool().cpu().numpy()  # (V, n_proposals)
    n_proposals = raw_props.shape[1]
    masks_22_np = masks_22.cpu().numpy()  # (V, K)
    K = masks_22_np.shape[1]

    # Hash each proposal mask
    prop_hashes = {raw_props[:, p].tobytes(): p for p in range(n_proposals)}
    pred_to_proposal = np.full(K, -1, dtype=np.int64)
    for k in range(K):
        h = masks_22_np[:, k].tobytes()
        pred_to_proposal[k] = prop_hashes.get(h, -1)

    # 3. Build per-prediction features by looking up fusion.instance_features
    # under the prediction's proposal_idx.
    pred_features = {}
    for k in range(K):
        p = int(pred_to_proposal[k])
        if p < 0:
            continue
        feat = fusion.get_feature(p)
        if feat is not None:
            pred_features[k] = feat

    # 4. METHOD_32 merge on the K predictions with class_aware partitioning.
    vertex_coords = _ensure_vertex_coords(self.world2cam)
    masks_m, classes_m, scores_m = _apply_method_32(
        masks_22, classes_22, scores_22, vertex_coords, instance_features=pred_features
    )
    self.predicted_masks = masks_m
    self.predicated_classes = classes_m
    self.predicated_scores = scores_m
    return {scene_name: (masks_m, classes_m, scores_m)}


def install_phase2(
    prompt_embeddings_path: str = "pretrained/scannet200_prompt_embeddings.pt",
    ema_alpha: float = 0.7,
    use_inference_subset: bool = True,
    min_match_iou: float = 0.10,
    topk_per_inst: int = 10,
    spatial_alpha: float = 0.5,
    distance_threshold: float = 2.0,
    semantic_threshold: float = 0.3,
    class_aware: bool = True,
) -> None:
    """METHOD_22 (label) + METHOD_32 (merge) with shared CLIP features."""
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        raise RuntimeError(
            "label_3d_masks_from_2d_bboxes is already patched. Uninstall first."
        )

    _ensure_method22_resources(prompt_embeddings_path, ema_alpha, use_inference_subset)
    _method22_state["min_match_iou"] = float(min_match_iou)
    _method22_state["topk_per_inst"] = int(topk_per_inst)

    _method32_state["spatial_alpha"] = float(spatial_alpha)
    _method32_state["distance_threshold"] = float(distance_threshold)
    _method32_state["semantic_threshold"] = float(semantic_threshold)
    _method32_state["class_aware"] = bool(class_aware)

    OpenYolo3D._original_label_3d_masks_from_2d_bboxes = (
        OpenYolo3D.label_3d_masks_from_2d_bboxes
    )
    OpenYolo3D.label_3d_masks_from_2d_bboxes = _patched_label_3d_masks_from_2d_bboxes_phase2


def uninstall_phase2() -> None:
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        OpenYolo3D.label_3d_masks_from_2d_bboxes = (
            OpenYolo3D._original_label_3d_masks_from_2d_bboxes
        )
        del OpenYolo3D._original_label_3d_masks_from_2d_bboxes
    _method22_state["fusion"] = None
