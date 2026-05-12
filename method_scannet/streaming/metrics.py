"""Temporal metrics for streaming evaluation (Task 1.1 Stage 3 spec).

Four label-only metrics (Task 1.2a stub) + mask-IoU extensions (Task 1.2b):
    - incremental_map_primary   — visible-GT-up-to-t vs prediction-at-t
    - incremental_map_secondary — full-scene-GT       vs prediction-at-t
    - id_switch_count           — Σ_i |{t: match(i,t) ≠ match(i,t−1)}|
    - label_switch_count        — Σ_k |{t: pred_class(k,t) ≠ pred_class(k,t−1)}|
    - time_to_confirm           — first t s.t. K consecutive frames have
                                  the same predicted class (C1 with K=3)
    - mask_iou_map              — mask-IoU + label-matched AP / AP_50 / AP_25
    - evaluate_scene_scannet200 — wrapper around the ScanNet200 evaluator
                                  for single-scene prediction comparison
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def incremental_map_primary(
    pred_instance_map_at_t: Mapping[int, Any],
    visible_gt_at_t: Mapping[int, Any],
) -> float:
    """Primary (a): visible GT up to t vs prediction at t.

    Stub label-matching score (per Task 1.2a). Each GT instance is "hit"
    iff at least one prediction instance has the same label; the score is
    the fraction of GT instances hit. Multi-matching is one-to-one to
    avoid double counting.

    Args:
        pred_instance_map_at_t: {pred_instance_id: label} for instances
            currently exposed by the streaming wrapper.
        visible_gt_at_t: {gt_instance_id: label} for GT instances visible
            in any frame ≤ t.

    Returns:
        Score in [0, 1]. By convention, an empty GT set returns 1.0
        (nothing to predict yet; no penalty).
    """
    if not visible_gt_at_t:
        return 1.0
    available = list(pred_instance_map_at_t.values())
    hits = 0
    for gt_label in visible_gt_at_t.values():
        if gt_label in available:
            available.remove(gt_label)
            hits += 1
    return hits / len(visible_gt_at_t)


def incremental_map_secondary(
    pred_instance_map_at_t: Mapping[int, Any],
    full_scene_gt: Mapping[int, Any],
) -> float:
    """Secondary (b): full scene GT vs prediction at t.

    Same scoring as primary, but the denominator is the full scene GT
    (does not grow with t). Curve vs t = convergence speed.
    """
    return incremental_map_primary(pred_instance_map_at_t, full_scene_gt)


def id_switch_count(
    pred_history: Sequence[Mapping[int, Any]],
    gt_matching: Mapping[int, Sequence[Any]],
) -> int:
    """ID switches per GT instance, summed across the scene.

    A "switch" is a frame t where the GT instance's matched prediction id
    differs from t−1, with neither side being ``None``/``-1``
    (i.e., disappearing or reappearing does not count, only re-routing
    between two existing prediction ids).

    Args:
        pred_history: per-frame {pred_id: label} maps (kept for symmetry
            with the rest of the API; not consumed in this stub).
        gt_matching: {gt_instance_id: [matched_pred_id_at_frame_t, ...]}.
            ``None`` (or ``-1``) at a frame means "not matched at this
            frame".

    Returns:
        Sum of switches across all GT instances.
    """
    del pred_history  # not used by the label-only stub.
    total = 0
    for _, pred_seq in gt_matching.items():
        for t in range(1, len(pred_seq)):
            prev, curr = pred_seq[t - 1], pred_seq[t]
            if prev in (None, -1) or curr in (None, -1):
                continue
            if prev != curr:
                total += 1
    return total


def label_switch_count(
    pred_history: Sequence[Mapping[int, Any]],
) -> int:
    """Label changes per prediction instance, summed across the scene.

    A "switch" is a frame t where ``pred_class[k, t] ≠ pred_class[k, t−1]``
    with neither side being ``-1`` (unassigned). Tracks per-instance.
    """
    prev_label: dict[int, Any] = {}
    total = 0
    for frame_map in pred_history:
        for pred_id, label in frame_map.items():
            if pred_id in prev_label:
                last = prev_label[pred_id]
                if last != -1 and label != -1 and last != label:
                    total += 1
            prev_label[pred_id] = label
    return total


def time_to_confirm(
    pred_history: Sequence[Mapping[int, Any]],
    K: int = 3,
) -> dict[int, int]:
    """C1 with default K=3: K consecutive frames with the same class.

    For each prediction instance ``k``:
        first_seen[k]    = first t with pred_class[k, t] != -1
        confirmed_at[k]  = first t s.t. pred_history[t-K+1..t] all carry
                           the same class for k (and that class != -1)
        time_to_confirm  = confirmed_at[k] - first_seen[k]

    Returns:
        {pred_instance_id: frames_to_confirm}. Instances that never
        reach K-consecutive stability are omitted from the result.
    """
    first_seen: dict[int, int] = {}
    confirmed_at: dict[int, int] = {}
    class_history: dict[int, list[Any]] = defaultdict(list)

    for t, frame_map in enumerate(pred_history):
        for pred_id, label in frame_map.items():
            if label == -1:
                continue
            if pred_id not in first_seen:
                first_seen[pred_id] = t
            class_history[pred_id].append(label)
            recent = class_history[pred_id][-K:]
            if (
                pred_id not in confirmed_at
                and len(recent) == K
                and all(c == recent[0] for c in recent)
            ):
                confirmed_at[pred_id] = t

    return {
        pred_id: confirmed_at[pred_id] - first_seen[pred_id]
        for pred_id in confirmed_at
    }


# ---------------------------------------------------------------------------
# Mask-IoU based AP (Task 1.2b extension)
# ---------------------------------------------------------------------------


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Vertex-level mask IoU between two boolean vectors of length V."""
    a = np.asarray(pred_mask, dtype=bool)
    b = np.asarray(gt_mask, dtype=bool)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def mask_iou_map(
    pred_instances: Mapping[int, Mapping[str, Any]],
    gt_instances: Mapping[int, Mapping[str, Any]],
    iou_thresholds: Sequence[float] = (0.5, 0.25),
) -> dict[str, float]:
    """Greedy-Hungarian AP at given mask-IoU thresholds with label match.

    Args:
        pred_instances: ``{pred_id: {"vertex_mask": np.ndarray (V,) bool,
            "label": int|str, "score": float (optional)}}``.
        gt_instances:   ``{gt_id:   {"vertex_mask": np.ndarray (V,) bool,
            "label": int|str}}``.
        iou_thresholds: thresholds to compute. The result always includes
            ``AP`` (= mean over the thresholds), ``AP_50`` (if 0.5 in the
            list) and ``AP_25`` (if 0.25 in the list).

    Returns:
        ``{"AP": float, "AP_50": float, "AP_25": float, ...}``.
    """
    out: dict[str, float] = {}
    if not gt_instances:
        # No GT to predict against → degenerate; convention: perfect score.
        for th in iou_thresholds:
            out[f"AP_{int(round(th * 100))}"] = 1.0
        out["AP"] = 1.0
        return out

    n_gt = len(gt_instances)
    sorted_preds = sorted(
        pred_instances.items(),
        key=lambda kv: -float(kv[1].get("score", 0.0)),
    )
    per_threshold: list[float] = []
    for th in iou_thresholds:
        matched_gt: set = set()
        tp = 0
        for _, pred in sorted_preds:
            best_iou = 0.0
            best_gt = None
            for gt_id, gt in gt_instances.items():
                if gt_id in matched_gt:
                    continue
                if gt["label"] != pred["label"]:
                    continue
                iou = _mask_iou(pred["vertex_mask"], gt["vertex_mask"])
                if iou >= th and iou > best_iou:
                    best_iou = iou
                    best_gt = gt_id
            if best_gt is not None:
                tp += 1
                matched_gt.add(best_gt)
        recall_at_th = tp / n_gt
        per_threshold.append(recall_at_th)
        out[f"AP_{int(round(th * 100))}"] = float(recall_at_th)
    out["AP"] = float(sum(per_threshold) / len(per_threshold)) if per_threshold else 0.0
    return out


def evaluate_scene_scannet200(
    pred_for_scene: Mapping[str, np.ndarray],
    gt_path: str,
    scene_name: str,
    pretrained_on_scannet200: bool = True,
) -> dict[str, float]:
    """Run the official ScanNet200 evaluator on a single-scene prediction.

    Args:
        pred_for_scene: ``{"pred_masks": np.ndarray (V, K_pred),
            "pred_classes": np.ndarray (K_pred,) int,
            "pred_scores": np.ndarray (K_pred,) float}``.
        gt_path: directory holding ``<scene>.txt`` GT files.
        scene_name: e.g. ``"scene0011_00"``.

    Returns:
        ``{"AP": float, "AP_50": float, "AP_25": float}`` plus head /
        common / tail per-bucket APs when available.
    """
    # Lazy import — heavy module; only loaded when running real ScanNet eval.
    from evaluate import evaluate_scannet200

    preds = {scene_name: dict(pred_for_scene)}
    output_file = "/tmp/_streaming_eval_unused.txt"
    avgs, _ar, _rc, _pcdc = evaluate_scannet200(
        preds,
        gt_path,
        output_file=output_file,
        dataset="scannet200",
        pretrained_on_scannet200=pretrained_on_scannet200,
    )

    result: dict[str, float] = {
        "AP": float(avgs.get("all_ap", float("nan"))),
        "AP_50": float(avgs.get("all_ap_50%", float("nan"))),
        "AP_25": float(avgs.get("all_ap_25%", float("nan"))),
    }
    for cat in ("head", "common", "tail"):
        if f"{cat}_ap" in avgs:
            result[f"{cat}_AP"] = float(avgs[f"{cat}_ap"])
            result[f"{cat}_AP_50"] = float(avgs.get(f"{cat}_ap50%", float("nan")))
            result[f"{cat}_AP_25"] = float(avgs.get(f"{cat}_ap25%", float("nan")))
    return result
