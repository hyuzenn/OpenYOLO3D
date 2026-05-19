"""Unit tests for streaming/metrics.py (4 temporal metrics). GPU-free."""
from __future__ import annotations

from method_scannet.streaming.metrics import (
    id_switch_count,
    incremental_map_primary,
    incremental_map_secondary,
    label_switch_count,
    time_to_confirm,
)


# ---------------------------------------------------------------------------
# incremental_map_primary / secondary
# ---------------------------------------------------------------------------


def test_incremental_map_primary_perfect():
    """Pred has every GT label exactly once → mAP = 1.0."""
    preds = {1: "chair", 2: "table"}
    gt = {10: "chair", 11: "table"}

    assert incremental_map_primary(preds, gt) == 1.0


def test_incremental_map_primary_empty():
    """No predictions but GT exists → mAP = 0.0."""
    preds: dict[int, str] = {}
    gt = {10: "chair", 11: "table"}

    assert incremental_map_primary(preds, gt) == 0.0


def test_incremental_map_secondary_partial():
    """Secondary uses full GT; preds covers half → mAP = 0.5."""
    preds = {1: "chair"}
    full_gt = {10: "chair", 11: "table"}

    assert incremental_map_secondary(preds, full_gt) == 0.5


# ---------------------------------------------------------------------------
# id_switch_count
# ---------------------------------------------------------------------------


def test_id_switch_no_switch():
    """GT 10 matches pred id 1 across all frames → 0 switches."""
    pred_history = [{1: "chair"}, {1: "chair"}, {1: "chair"}]
    gt_matching = {10: [1, 1, 1]}

    assert id_switch_count(pred_history, gt_matching) == 0


def test_id_switch_one_switch():
    """GT 10 matches pred 1 then pred 2 → 1 switch."""
    pred_history = [{1: "chair"}, {2: "chair"}]
    gt_matching = {10: [1, 2]}

    assert id_switch_count(pred_history, gt_matching) == 1


def test_id_switch_ignores_disappearance():
    """match → None → match: disappearance/reappearance not counted."""
    pred_history = [{1: "x"}, {}, {1: "x"}]
    gt_matching = {10: [1, None, 1]}

    assert id_switch_count(pred_history, gt_matching) == 0


# ---------------------------------------------------------------------------
# label_switch_count
# ---------------------------------------------------------------------------


def test_label_switch_no_change():
    """Same class across all frames → 0 switches."""
    history = [{1: "chair"}, {1: "chair"}, {1: "chair"}]

    assert label_switch_count(history) == 0


def test_label_switch_oscillation():
    """chair → table → chair is two switches."""
    history = [{1: "chair"}, {1: "table"}, {1: "chair"}]

    assert label_switch_count(history) == 2


# ---------------------------------------------------------------------------
# time_to_confirm (C1 with K=3)
# ---------------------------------------------------------------------------


def test_time_to_confirm_immediate():
    """class[1,0..2] = chair, chair, chair → first_seen=0, confirmed=2, TTC=2."""
    history = [{1: "chair"}, {1: "chair"}, {1: "chair"}]

    result = time_to_confirm(history, K=3)

    assert result[1] == 2


def test_time_to_confirm_never():
    """Oscillating class → never K-consecutive → instance not in result."""
    history = [
        {1: "chair"},
        {1: "table"},
        {1: "chair"},
        {1: "table"},
        {1: "chair"},
    ]

    result = time_to_confirm(history, K=3)

    assert 1 not in result
