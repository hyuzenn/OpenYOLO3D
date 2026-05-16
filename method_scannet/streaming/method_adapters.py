"""Streaming adapters for the May method classes (M21/M22/M31/M32).

Wraps the offline May classes with the call sequences and signatures the
streaming wrapper produces. The May classes themselves are imported and
called *unchanged* — only the per-frame ↔ batch packaging differs.

Design (Task 1.4a redesign, 2026-05-14):
- M21 (WeightedVoting): batch-style — replays the accumulator's stacked
  state at finalize and feeds voter.frame_weight + vote_distribution.
- M22 (FeatureFusionEMA): streaming-style — fusion.update_instance_feature
  is called per (instance, frame); at finalize, the wrapper consumes
  fusion.instance_features for cosine classification.
- M31 (IoUMerger): single batched call at finalize with the exact May
  signature (predicted_masks, pred_classes, pred_scores, vertex_coords=).
- M32 (HungarianMerger): builds instance_list + feature dict from the
  current predictions, calls merger.merge(), filters surviving columns.
- M11/M12 registration: filters pred_masks columns whose underlying
  Mask3D proposal_idx is not in the gate's confirmed set.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# M11 / M12 — registration filter
# ---------------------------------------------------------------------------


def apply_registration_filter(
    preds: dict,
    mask_idx: torch.Tensor,
    confirmed_set: set,
) -> dict:
    """Filter (pred_masks, pred_classes, pred_scores) by keeping only
    columns whose underlying Mask3D proposal_idx is in ``confirmed_set``.

    Args:
        preds: dict with 'pred_masks' (V, K_out) bool, 'pred_classes' (K_out,),
            'pred_scores' (K_out,) — already numpy arrays.
        mask_idx: torch.Tensor (K_out,) — column k's underlying proposal idx.
        confirmed_set: set of confirmed proposal indices.
    """
    if not confirmed_set:
        # Gate confirmed nothing — emit a single empty column to avoid
        # downstream shape errors. AP will be 0 but the run completes.
        V = preds["pred_masks"].shape[0]
        return {
            "pred_masks": np.zeros((V, 0), dtype=bool),
            "pred_classes": np.zeros(0, dtype=np.int64),
            "pred_scores": np.zeros(0, dtype=np.float32),
        }
    mask_idx_np = mask_idx.cpu().numpy() if hasattr(mask_idx, "cpu") else np.asarray(mask_idx)
    keep = np.array(
        [int(p) in confirmed_set for p in mask_idx_np],
        dtype=bool,
    )
    return {
        "pred_masks": preds["pred_masks"][:, keep],
        "pred_classes": preds["pred_classes"][keep],
        "pred_scores": preds["pred_scores"][keep],
    }


# ---------------------------------------------------------------------------
# M21 — WeightedVoting batch adapter
# ---------------------------------------------------------------------------


def compute_predictions_method21(
    accumulator,
    voter,
    scene_vertices: np.ndarray,
    camera_positions: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple:
    """Replay BaselineLabelAccumulator's stacked state but route the inner
    label aggregation through ``WeightedVoting.frame_weight`` (per-frame
    weight) and a manual class histogram (vote_distribution semantics).

    The output contract matches ``BaselineLabelAccumulator.compute_predictions``:
    (pred_masks (V, K_out) bool, pred_classes (K_out,) long, pred_scores
    (K_out,) float, mask_idx (K_out,) long).
    """
    from utils import compute_iou, get_visibility_mat

    state = accumulator.stack_for_methods()
    projections_all = state["projections"]      # (F, V, 2) int32
    visible_masks_all = state["visible_masks"]  # (F, V) bool
    label_maps_all = state["label_maps"]        # (F, H, W) int16
    bbox_preds_all = state["bbox_preds"]
    pred_masks_VK = state["prediction_3d_masks"]  # (V, K) bool tensor

    num_classes = accumulator.num_classes
    topk_for_call = 25 if accumulator.is_gt else accumulator.topk
    topk_per_image = accumulator.topk_per_image
    scaling_w = accumulator.scaling_w
    scaling_h = accumulator.scaling_h

    if accumulator.n_frames == 0:
        V = pred_masks_VK.shape[0]
        return (
            torch.zeros((V, 0), dtype=torch.bool),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
        )

    pred_masks_KV = pred_masks_VK.permute(1, 0).cpu().bool()  # (K, V)
    K = int(pred_masks_KV.shape[0])

    visibility_matrix = get_visibility_mat(
        pred_masks_KV.cuda() if torch.cuda.is_available() else pred_masks_KV,
        torch.from_numpy(visible_masks_all).cuda()
        if torch.cuda.is_available()
        else torch.from_numpy(visible_masks_all),
        topk=topk_for_call,
    )
    valid_frames = (visibility_matrix.sum(dim=0) >= 1).cpu()
    if not valid_frames.any():
        V = pred_masks_VK.shape[0]
        return (
            pred_masks_VK,
            torch.full((K,), num_classes - 1, dtype=torch.long),
            torch.zeros(K, dtype=torch.float32),
            torch.arange(K, dtype=torch.long),
        )

    visibility_matrix_np = (
        visibility_matrix[:, valid_frames.to(visibility_matrix.device)]
        .cpu()
        .numpy()
    )  # (K, F')
    valid_frame_idx = torch.where(valid_frames)[0].numpy()
    projections_v = projections_all[valid_frame_idx]  # (F', V, 2)
    visible_masks_v = visible_masks_all[valid_frame_idx]  # (F', V)
    label_maps_v = label_maps_all[valid_frame_idx]      # (F', H, W)
    bbox_preds_v = [bbox_preds_all[i] for i in valid_frame_idx]
    camera_positions_v = (
        camera_positions[valid_frame_idx]
        if len(camera_positions) > 0
        else np.zeros((0, 3), dtype=np.float64)
    )

    img_cx = image_width / 2.0
    img_cy = image_height / 2.0
    inv_dist = 1.0 / float(voter.distance_weight_decay)
    inv_center = 1.0 / float(voter.center_weight_decay)
    alpha = float(voter.spatial_alpha)

    prediction_3d_masks_np = pred_masks_VK.permute(1, 0).cpu().numpy().astype(bool)  # (K, V)

    distributions: list[torch.Tensor] = []
    class_labels: list[int] = []
    class_probs: list[float] = []

    for mask_id in range(K):
        mask = prediction_3d_masks_np[mask_id]
        if mask.sum() > 0:
            instance_centroid = scene_vertices[mask].mean(axis=0)
        else:
            instance_centroid = np.zeros(3, dtype=np.float64)

        rep_frame_ids = np.where(visibility_matrix_np[mask_id])[0]
        weighted_counts = np.zeros(num_classes, dtype=np.float64)
        prob_normalizer = 0
        labels_distribution = []  # for class_prob fallback
        iou_vals: list[float] = []

        for rep_frame_id in rep_frame_ids:
            visible_points_mask = (
                visible_masks_v[rep_frame_id].squeeze() * mask
            ).astype(bool)
            n_visible = int(visible_points_mask.sum())
            prob_normalizer += n_visible
            instance_xy = (
                projections_v[rep_frame_id][np.where(visible_points_mask)].astype(np.int64)
            )

            boxes = bbox_preds_v[rep_frame_id]["bbox"].long()
            confidence = 1.0
            bbox_cx = img_cx
            bbox_cy = img_cy
            if len(boxes) > 0 and len(instance_xy) > 10:
                x_l = int(instance_xy[:, 0].min())
                x_r = int(instance_xy[:, 0].max()) + 1
                y_t = int(instance_xy[:, 1].min())
                y_b = int(instance_xy[:, 1].max()) + 1
                box = torch.tensor(
                    [
                        x_l / scaling_w,
                        y_t / scaling_h,
                        x_r / scaling_w,
                        y_b / scaling_h,
                    ],
                    dtype=torch.float32,
                )
                iou_values = compute_iou(box, boxes.float())
                iou_max_val = float(iou_values.max().item())
                iou_vals.append(iou_max_val)
                confidence = iou_max_val
                bbox_cx = float((x_l + x_r) / 2.0)
                bbox_cy = float((y_t + y_b) / 2.0)
            elif len(instance_xy) > 0:
                bbox_cx = float(instance_xy[:, 0].mean())
                bbox_cy = float(instance_xy[:, 1].mean())

            if len(instance_xy) > 0:
                selected_labels = label_maps_v[
                    rep_frame_id, instance_xy[:, 1], instance_xy[:, 0]
                ]
            else:
                selected_labels = np.array([-1], dtype=np.int16)

            valid_arr = selected_labels[selected_labels != -1]
            if valid_arr.size == 0 or confidence <= 0:
                continue

            if len(camera_positions_v) > rep_frame_id:
                cam = camera_positions_v[rep_frame_id]
                d3 = math.sqrt(
                    (cam[0] - instance_centroid[0]) ** 2
                    + (cam[1] - instance_centroid[1]) ** 2
                    + (cam[2] - instance_centroid[2]) ** 2
                )
            else:
                d3 = 0.0
            w_dist = math.exp(-d3 * inv_dist)
            d2 = math.sqrt((bbox_cx - img_cx) ** 2 + (bbox_cy - img_cy) ** 2)
            w_center = math.exp(-d2 * inv_center)
            frame_w = (alpha * w_dist + (1.0 - alpha) * w_center) * confidence
            if frame_w <= 0:
                continue

            vals, counts = np.unique(valid_arr.astype(np.int64), return_counts=True)
            for v, c in zip(vals.tolist(), counts.tolist()):
                if 0 <= v < num_classes:
                    weighted_counts[int(v)] += float(frame_w) * float(c)
            labels_distribution.append(valid_arr)

        if labels_distribution and weighted_counts.max() > 0:
            distribution = torch.from_numpy(weighted_counts.astype(np.float32))
            distribution = distribution / distribution.max()
            class_label = int(distribution.argmax().item())
            all_labels_concat = np.concatenate(labels_distribution)
            class_prob = float(
                (all_labels_concat == class_label).sum()
            ) / max(prob_normalizer, 1)
        else:
            distribution = torch.zeros(num_classes, dtype=torch.float32)
            distribution[-1] = 1.0
            class_label = num_classes - 1
            class_prob = 0.0

        if iou_vals:
            iv = np.asarray(iou_vals, dtype=np.float32)
            iv = iv[iv != 0]
            iou_prob = float(iv.mean()) if iv.size > 0 else 0.0
        else:
            iou_prob = 0.0

        distributions.append(distribution)
        class_labels.append(class_label)
        class_probs.append(class_prob * iou_prob)

    pred_classes_t = torch.tensor(class_labels, dtype=torch.long)
    pred_scores_t = torch.tensor(class_probs, dtype=torch.float32)
    pred_masks_KV_cpu = pred_masks_KV.cpu()  # (K, V)

    if topk_per_image != -1 and not accumulator.is_gt and distributions:
        distributions_stacked = torch.stack(distributions)  # (K, num_classes)
        n_inst = distributions_stacked.shape[0]
        distributions_flat = distributions_stacked.reshape(-1)
        labels_flat = (
            torch.arange(num_classes)
            .unsqueeze(0)
            .repeat(n_inst, 1)
            .flatten(0, 1)
        )
        cur_topk = min(topk_per_image, int(distributions_flat.numel()))
        _, idx = torch.topk(distributions_flat, k=cur_topk, largest=True)
        mask_idx = torch.div(idx, num_classes, rounding_mode="floor")
        pred_classes_t = labels_flat[idx]
        pred_scores_t = distributions_flat[idx]
        pred_masks_KV_cpu = pred_masks_KV_cpu[mask_idx]
    else:
        mask_idx = torch.arange(K, dtype=torch.long)

    return (
        pred_masks_KV_cpu.permute(1, 0).contiguous(),
        pred_classes_t,
        pred_scores_t,
        mask_idx.cpu().long(),
    )


# ---------------------------------------------------------------------------
# M22 — FeatureFusionEMA cosine classification adapter
# ---------------------------------------------------------------------------


def compute_predictions_method22(
    accumulator,
    fusion,
    topk_per_image: int,
) -> tuple:
    """Build (pred_masks, pred_classes, pred_scores, mask_idx) from the
    fusion accumulator's instance_features dict.

    ``fusion`` is a populated FeatureFusionEMA — populated during step_frame
    by ``wrapper._method22_per_frame``. At finalize we compute the
    (n_proposals × n_classes) cosine-similarity distribution and pick
    top-k entries.
    """
    state = accumulator.stack_for_methods()
    pred_masks_VK = state["prediction_3d_masks"]  # (V, K) bool tensor
    V, K = pred_masks_VK.shape

    n_classes = fusion._prompt_emb_norm.shape[0]
    distribution = torch.zeros(K, n_classes, dtype=torch.float32)
    # Prompt bank was loaded on CPU. CLIP image features may be on CUDA
    # when the encoder runs on GPU — match the device for the cosine
    # matmul, then move the result back to CPU for the distribution
    # tensor (which the wrapper consumes as a numpy array).
    prompts = fusion._prompt_emb_norm  # (n_classes, D)
    any_feature = False
    for prop_idx in range(K):
        feat = fusion.get_feature(prop_idx)
        if feat is None:
            continue
        any_feature = True
        f_norm = feat.float()
        f_norm = f_norm / f_norm.norm().clamp_min(1e-12)
        prompts_dev = prompts.to(f_norm.device)
        sims = (f_norm.unsqueeze(0) @ prompts_dev.t()).squeeze(0).clamp_min(0.0)
        distribution[prop_idx] = sims.detach().cpu()

    if not any_feature:
        return (
            torch.zeros((V, 0), dtype=torch.bool),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
        )

    flat = distribution.reshape(-1)
    if topk_per_image is None or topk_per_image == -1:
        # one prediction per proposal (argmax cosine)
        cur_topk = K
        _, idx = torch.topk(flat, k=min(cur_topk, int(flat.numel())), largest=True)
    else:
        cur_topk = min(int(topk_per_image), int(flat.numel()))
        _, idx = torch.topk(flat, k=cur_topk, largest=True)

    mask_idx = torch.div(idx, n_classes, rounding_mode="floor")
    class_idx = idx - mask_idx * n_classes
    pred_scores = flat[idx]

    nonzero = pred_scores > 0
    if int(nonzero.sum().item()) == 0:
        return (
            torch.zeros((V, 0), dtype=torch.bool),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.long),
        )
    mask_idx = mask_idx[nonzero]
    class_idx = class_idx[nonzero]
    pred_scores = pred_scores[nonzero]

    pred_masks_VK_out = pred_masks_VK.cpu()[:, mask_idx].contiguous()
    return (
        pred_masks_VK_out,
        class_idx.long(),
        pred_scores.float(),
        mask_idx.cpu().long(),
    )


# ---------------------------------------------------------------------------
# M31 — IoUMerger (class-aware 3D NMS) adapter
# ---------------------------------------------------------------------------


def apply_method31_merge(preds: dict, merger, scene_vertices: np.ndarray) -> dict:
    """Call the May IoUMerger with its exact kwargs and convert back to numpy."""
    pred_masks_t = torch.from_numpy(preds["pred_masks"]).bool()
    pred_classes_t = torch.from_numpy(preds["pred_classes"]).long()
    pred_scores_t = torch.from_numpy(np.asarray(preds["pred_scores"])).float()
    if pred_masks_t.shape[1] == 0:
        return preds
    kept_masks, kept_classes, kept_scores = merger.merge(
        predicted_masks=pred_masks_t,
        pred_classes=pred_classes_t,
        pred_scores=pred_scores_t,
        vertex_coords=scene_vertices,
    )
    return {
        "pred_masks": kept_masks.cpu().numpy().astype(bool),
        "pred_classes": kept_classes.cpu().numpy().astype(np.int64),
        "pred_scores": kept_scores.cpu().numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# M32 — HungarianMerger adapter
# ---------------------------------------------------------------------------


def apply_method32_merge(
    preds: dict,
    merger,
    scene_vertices: np.ndarray,
    mask_idx: Optional[torch.Tensor] = None,
    instance_features: Optional[dict] = None,
    class_aware: bool = True,
) -> dict:
    """Wrap HungarianMerger.merge(instance_list, instance_features).

    Builds instance_list from current preds — each prediction column k is
    one instance with id=k, label=pred_classes[k], centroid=mean of
    vertex coords in pred_masks[:, k]. If ``mask_idx`` is provided and
    ``instance_features`` is keyed on the underlying proposal indices,
    we look up features through mask_idx for each k.
    """
    pred_masks = preds["pred_masks"]
    pred_classes = preds["pred_classes"]
    pred_scores = preds["pred_scores"]
    V, K = pred_masks.shape
    if K == 0:
        return preds

    instance_list = []
    for k in range(K):
        col = pred_masks[:, k]
        if not col.any():
            continue
        centroid = scene_vertices[col].mean(axis=0)
        item = {
            "id": int(k),
            "label": int(pred_classes[k]),
            "centroid": centroid,
        }
        instance_list.append(item)
    if not instance_list:
        return preds

    # Features dict keyed by k (prediction column index).
    feats_for_merge: dict = {}
    if instance_features is not None and mask_idx is not None:
        mask_idx_np = mask_idx.cpu().numpy() if hasattr(mask_idx, "cpu") else np.asarray(mask_idx)
        for k in range(K):
            p = int(mask_idx_np[k])
            f = instance_features.get(p)
            if f is not None:
                feats_for_merge[k] = f

    # When the prediction set has no semantic features (e.g. M32-only
    # without M22 upstream), spatial-only merging requires the merger's
    # semantic_threshold be set to -1.0 — otherwise cosine sim against
    # zero-vector features (0) fails the default threshold (0.3) and
    # every pair is masked → no merges (the "M32-only no-op" seen in
    # the first smoke). Mirrors offline _apply_method_32 behaviour.
    if not feats_for_merge:
        from method_scannet.method_32_hungarian_merging import HungarianMerger
        merger = HungarianMerger(
            spatial_alpha=float(getattr(merger, "spatial_alpha", 0.5)),
            distance_threshold=float(getattr(merger, "distance_threshold", 2.0)),
            semantic_threshold=-1.0,
        )

    if class_aware:
        from collections import defaultdict

        groups: dict = defaultdict(list)
        for inst in instance_list:
            groups[inst["label"]].append(inst)
        kept_ids: set = set()
        for cls, insts in groups.items():
            sub_features = {
                int(i["id"]): feats_for_merge[int(i["id"])]
                for i in insts
                if int(i["id"]) in feats_for_merge
            }
            merged = merger.merge(insts, sub_features)
            for m in merged:
                kept_ids.add(int(m["id"]))
    else:
        merged = merger.merge(instance_list, feats_for_merge)
        kept_ids = {int(m["id"]) for m in merged}

    if not kept_ids:
        return preds

    keep_idx = sorted(kept_ids)
    keep_arr = np.array(keep_idx, dtype=np.int64)
    return {
        "pred_masks": pred_masks[:, keep_arr],
        "pred_classes": pred_classes[keep_arr],
        "pred_scores": pred_scores[keep_arr],
    }
