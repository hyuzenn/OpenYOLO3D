"""Unit tests for streaming/baseline.py (BaselineLabelAccumulator). GPU-free.

Mock data: 3 instances over a tiny 5-vertex scene, 3 frames. The
accumulator stages raw per-frame data; ``compute_predictions()`` reduces
it to (masks, classes, scores) using the same algorithm as the offline
``OpenYolo3D.label_3d_masks_from_label_maps``.
"""
from __future__ import annotations

import numpy as np
import torch

from method_scannet.streaming.baseline import (
    BaselineLabelAccumulator,
    construct_label_map_single_frame,
)


def _make_accumulator(
    n_vertices: int = 5,
    n_instances: int = 3,
    num_classes: int = 4,
    height: int = 32,
    width: int = 32,
    topk: int = 40,
    topk_per_image: int = -1,  # disable global top-K filter for tests
) -> BaselineLabelAccumulator:
    masks = torch.zeros((n_vertices, n_instances), dtype=torch.bool)
    # Instance 0 owns vertices 0..1; instance 1 owns 2..3; instance 2 owns 4.
    for k, idxs in enumerate([[0, 1], [2, 3], [4]]):
        for v in idxs:
            masks[v, k] = True
    return BaselineLabelAccumulator(
        prediction_3d_masks=masks,
        num_classes=num_classes,
        topk=topk,
        topk_per_image=topk_per_image,
        depth_height=height,
        depth_width=width,
    )


def _bbox_pred(boxes, labels):
    return {
        "bbox": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "scores": torch.ones(len(boxes)),
    }


def test_baseline_first_frame_no_predictions():
    """compute_predictions() before any add_frame() returns empty arrays."""
    acc = _make_accumulator()
    masks, classes, scores = acc.compute_predictions()

    assert masks.shape == (5, 0)
    assert classes.numel() == 0
    assert scores.numel() == 0


def test_baseline_construct_label_map_paints_bbox():
    """A single bbox paints its interior with the class label."""
    bbox_pred = _bbox_pred([[5, 5, 15, 15]], [2])
    lm = construct_label_map_single_frame(bbox_pred, height=32, width=32)

    assert lm[7, 7].item() == 2
    assert lm[0, 0].item() == -1


def test_baseline_accumulate_two_frames_majority():
    """Two frames both vote class=1 for instance 0 → final label = 1."""
    acc = _make_accumulator(num_classes=4)

    # Project vertices 0, 1 of instance 0 to pixel (10, 10) — covered by the bbox.
    projection = np.zeros((5, 2), dtype=np.int32)
    projection[0] = (10, 10)
    projection[1] = (10, 10)
    inside = np.array([True, True, False, False, False])

    bbox_pred = _bbox_pred([[5, 5, 20, 20]], [1])
    acc.add_frame(projection, inside, bbox_pred)
    acc.add_frame(projection, inside, bbox_pred)

    masks, classes, scores = acc.compute_predictions()

    # Instance 0 gets class 1; instance 1 (no visible vertex) and instance 2
    # both get the background class (num_classes - 1).
    assert classes[0].item() == 1
    assert classes[1].item() == 3  # background fallback
    assert classes[2].item() == 3
    # Score depends on the IoU branch (requires >10 visible pixels in offline
    # algorithm). With only 2 visible vertices the IoU is not computed and
    # the score collapses to 0; the class assignment is the algorithmic
    # contract we're verifying here.
    assert scores[0].item() >= 0.0
    assert scores[1].item() == 0.0


def test_baseline_visible_only_voted():
    """Vertices outside inside_mask should not contribute votes."""
    acc = _make_accumulator(num_classes=4)
    projection = np.zeros((5, 2), dtype=np.int32)
    # Project instance 1's vertices (2, 3) into a class-2 bbox.
    projection[2] = (8, 8)
    projection[3] = (8, 8)
    inside = np.array([False, False, True, True, False])
    acc.add_frame(projection, inside, _bbox_pred([[5, 5, 15, 15]], [2]))

    _masks, classes, _scores = acc.compute_predictions()

    # Only instance 1 should pick up class 2; instance 0 (no visible vertex)
    # falls back to background.
    assert classes[1].item() == 2
    assert classes[0].item() == 3


def test_baseline_no_bbox_overlap_returns_background():
    """Visible vertices but no overlapping bbox → label_map says -1 → fallback."""
    acc = _make_accumulator(num_classes=4)
    projection = np.zeros((5, 2), dtype=np.int32)
    projection[0] = (1, 1)
    projection[1] = (1, 1)
    inside = np.array([True, True, False, False, False])

    # Bbox lives at the bottom-right corner; vertex pixels are at (1, 1) and
    # therefore land on label_map background (-1).
    acc.add_frame(projection, inside, _bbox_pred([[25, 25, 30, 30]], [1]))

    _masks, classes, _scores = acc.compute_predictions()

    assert classes[0].item() == 3  # background fallback because all labels = -1


def test_baseline_majority_vote_three_frames():
    """3 frames: 2 vote class=1, 1 votes class=2 → label = 1."""
    acc = _make_accumulator(num_classes=4)
    projection = np.zeros((5, 2), dtype=np.int32)
    projection[0] = (10, 10)
    projection[1] = (10, 10)
    inside = np.array([True, True, False, False, False])

    acc.add_frame(projection, inside, _bbox_pred([[5, 5, 20, 20]], [1]))
    acc.add_frame(projection, inside, _bbox_pred([[5, 5, 20, 20]], [1]))
    acc.add_frame(projection, inside, _bbox_pred([[5, 5, 20, 20]], [2]))

    _masks, classes, _scores = acc.compute_predictions()

    assert classes[0].item() == 1
