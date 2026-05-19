"""Unit tests for streaming/metrics.py mask-IoU AP extension. GPU-free."""
from __future__ import annotations

import numpy as np

from method_scannet.streaming.metrics import _mask_iou, mask_iou_map


def _inst(mask: list[bool], label: int, score: float = 1.0) -> dict:
    return {
        "vertex_mask": np.array(mask, dtype=bool),
        "label": label,
        "score": score,
    }


def test_mask_iou_helper_perfect_and_zero():
    """Sanity: identical masks → IoU=1; disjoint masks → IoU=0."""
    a = np.array([True, True, False, False])
    b = np.array([True, True, False, False])
    c = np.array([False, False, True, True])

    assert _mask_iou(a, b) == 1.0
    assert _mask_iou(a, c) == 0.0


def test_mask_iou_map_perfect_match():
    """All preds match GT with IoU=1 → AP = AP_50 = AP_25 = 1.0."""
    pred = {1: _inst([True, True, False, False], 0)}
    gt = {10: _inst([True, True, False, False], 0)}

    result = mask_iou_map(pred, gt)

    assert result["AP_50"] == 1.0
    assert result["AP_25"] == 1.0
    assert result["AP"] == 1.0


def test_mask_iou_map_zero_iou_zero_ap():
    """Disjoint masks → AP = 0."""
    pred = {1: _inst([True, True, False, False], 0)}
    gt = {10: _inst([False, False, True, True], 0)}

    result = mask_iou_map(pred, gt)

    assert result["AP_50"] == 0.0
    assert result["AP_25"] == 0.0
    assert result["AP"] == 0.0


def test_mask_iou_map_iou_50_threshold_boundary():
    """IoU = 0.5 → AP_50 should still count (>=); AP_25 also matches."""
    # Pred = [1,1,1,1,0,0]; GT = [1,1,0,0,0,0]
    # inter = 2, union = 4 → IoU = 0.5
    pred = {1: _inst([True, True, True, True, False, False], 7)}
    gt = {10: _inst([True, True, False, False, False, False], 7)}

    result = mask_iou_map(pred, gt)

    assert result["AP_50"] == 1.0
    assert result["AP_25"] == 1.0


def test_mask_iou_map_label_mismatch_zero():
    """Same mask but wrong label → no match → AP = 0."""
    pred = {1: _inst([True, True, False, False], 0)}
    gt = {10: _inst([True, True, False, False], 1)}  # different label

    result = mask_iou_map(pred, gt)

    assert result["AP"] == 0.0


def test_mask_iou_map_multi_instance_partial():
    """3 GT, 2 preds match → AP = 2/3 at AP_50."""
    pred = {
        1: _inst([True, True, False, False, False, False], 0),
        2: _inst([False, False, True, True, False, False], 1),
    }
    gt = {
        10: _inst([True, True, False, False, False, False], 0),
        11: _inst([False, False, True, True, False, False], 1),
        12: _inst([False, False, False, False, True, True], 2),
    }

    result = mask_iou_map(pred, gt)

    assert abs(result["AP_50"] - 2 / 3) < 1e-9
    assert abs(result["AP_25"] - 2 / 3) < 1e-9
