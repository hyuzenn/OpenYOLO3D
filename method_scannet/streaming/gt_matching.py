"""Per-frame GT↔prediction matching for the ``id_switch_count`` metric.

The streaming ablation records, for each frame ``t``, the set of *live*
prediction instances (Mask3D proposal ids) the model is currently tracking —
these are the keys of ``pred_history[t]``. To evaluate
:func:`method_scannet.streaming.metrics.id_switch_count` we must produce, for
every GT instance, the prediction id it is matched to at each frame
(``gt_matching = {gt_id: [pred_id_or_None per frame]}``). This module builds
that structure from the (already proven) per-frame history plus the fixed
scene-level masks.

Matching rule (documented for reproducibility — these choices affect the
reported numbers):

* **Prediction masks** are the fixed scene-level Mask3D proposal masks
  (``instance_vertex_masks``, shape ``(K, V)``). Proposal id == row index.
* **GT instance masks** come from the ScanNet200 per-vertex GT file whose
  value encodes ``label_id * 1000 + instance_id``. A GT instance is a unique
  value whose ``label_id`` is in ``VALID_CLASS_IDS_200_INST`` (the same
  instance-class set the official ScanNet200 AP evaluator uses).
* **IoU(g, k)** is the full-scene mask IoU between GT ``g`` and proposal
  ``k`` (constant across frames — masks are scene-level).
* At frame ``t`` the candidate predictions are the *live* ids
  ``set(pred_history[t])``. Each GT ``g`` is matched to
  ``argmax_{k in live} IoU(g, k)`` iff that IoU ≥ ``iou_threshold``, else
  ``None``. As the live set changes frame-to-frame the matched id can change.
  Per the CLEAR-MOT ID-switch definition (see ``metrics.id_switch_count``)
  only re-routing between two *present* ids counts as a switch;
  (re)appearance from/to ``None`` does not.

Note that because the per-frame live set is gated only by visibility and the
M11/M12 registration gates, axes that change *labels* (M21/M22) or only the
*finalize* merge (M31/M32) leave ``pred_history`` keys unchanged and therefore
yield the same ID-switch count as baseline by construction — this is a
property of the metric, not a bug.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np


def _valid_inst_class_ids() -> frozenset[int]:
    """Lazy import so this module stays usable (and unit-testable) without
    pulling the full ``evaluate`` package / its heavy deps."""
    from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST

    return frozenset(int(c) for c in VALID_CLASS_IDS_200_INST)


def load_gt_instance_masks(
    gt_txt_path: str,
    n_vertices: Optional[int] = None,
) -> dict[int, np.ndarray]:
    """Parse a ScanNet200 per-vertex GT ``.txt`` (``label*1000+inst``) into
    ``{gt_id: bool ndarray (V,)}`` restricted to valid instance classes.
    """
    gt = np.loadtxt(gt_txt_path, dtype=np.int64).reshape(-1)
    if n_vertices is not None and gt.shape[0] != n_vertices:
        raise ValueError(
            f"GT length {gt.shape[0]} != n_vertices {n_vertices} "
            f"({gt_txt_path}) — vertex layouts must match."
        )
    valid = _valid_inst_class_ids()
    masks: dict[int, np.ndarray] = {}
    for v in np.unique(gt):
        v = int(v)
        if v < 1000:  # 0 = unannotated; <1000 has no semantic.
            continue
        if (v // 1000) not in valid:
            continue
        masks[v] = gt == v
    return masks


def full_scene_iou(gt_mask: np.ndarray, proposal_masks: np.ndarray) -> np.ndarray:
    """IoU of one GT mask ``(V,)`` against every proposal ``(K, V)`` → ``(K,)``."""
    gm = np.asarray(gt_mask, dtype=bool)
    pm = np.asarray(proposal_masks, dtype=bool)
    inter = (pm & gm[None, :]).sum(axis=1).astype(np.float64)
    union = (pm | gm[None, :]).sum(axis=1).astype(np.float64)
    out = np.zeros(pm.shape[0], dtype=np.float64)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def build_gt_matching(
    pred_history: Sequence[Mapping[int, Any]],
    instance_vertex_masks: np.ndarray,
    gt_masks: Mapping[int, np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[int, list[Optional[int]]]:
    """Build ``{gt_id: [matched_pred_id_or_None per frame]}``.

    Args:
        pred_history: per-frame ``{pred_id: label}`` maps (live set = keys).
        instance_vertex_masks: ``(K, V)`` bool proposal masks.
        gt_masks: ``{gt_id: (V,) bool}``.
        iou_threshold: minimum full-scene IoU to accept a GT↔pred match.
    """
    ivm = np.asarray(instance_vertex_masks, dtype=bool)
    K = ivm.shape[0]
    iou = {g: full_scene_iou(m, ivm) for g, m in gt_masks.items()}
    gt_matching: dict[int, list[Optional[int]]] = {g: [] for g in gt_masks}
    for frame_map in pred_history:
        live = np.fromiter(
            (int(k) for k in frame_map.keys()), dtype=np.int64, count=len(frame_map)
        )
        live = live[(live >= 0) & (live < K)]
        for g in gt_masks:
            if live.size == 0:
                gt_matching[g].append(None)
                continue
            ious_live = iou[g][live]
            j = int(np.argmax(ious_live))
            gt_matching[g].append(int(live[j]) if float(ious_live[j]) >= iou_threshold else None)
    return gt_matching


def gt_matching_for_scene(
    pred_history: Sequence[Mapping[int, Any]],
    instance_vertex_masks: np.ndarray,
    gt_txt_path: str,
    iou_threshold: float = 0.5,
) -> tuple[dict[int, list[Optional[int]]], int]:
    """Convenience wrapper: load GT masks for a scene and build the matching.

    Returns ``(gt_matching, n_gt_instances)``.
    """
    ivm = np.asarray(instance_vertex_masks, dtype=bool)
    gt_masks = load_gt_instance_masks(gt_txt_path, n_vertices=ivm.shape[1])
    gm = build_gt_matching(pred_history, ivm, gt_masks, iou_threshold=iou_threshold)
    return gm, len(gt_masks)
