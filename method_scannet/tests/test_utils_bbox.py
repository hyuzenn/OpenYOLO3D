"""Unit tests for method_scannet/utils_bbox.py — pure numpy, GPU-free."""
from __future__ import annotations

import numpy as np
import pytest

from method_scannet.utils_bbox import (
    aabb_iou,
    aabb_volume,
    compute_aabb_from_vertex_mask,
    compute_instance_aabbs_batch,
)


def test_compute_aabb_from_simple_mask():
    pcd = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],  # selected
            [3, 3, 3],
            [4, 4, 4],  # selected
        ],
        dtype=np.float32,
    )
    mask = np.array([False, False, True, False, True])

    aabb, centroid, n = compute_aabb_from_vertex_mask(mask, pcd)
    assert n == 2
    np.testing.assert_array_almost_equal(aabb, [2, 2, 2, 4, 4, 4])
    np.testing.assert_array_almost_equal(centroid, [3, 3, 3])


def test_empty_mask_returns_none():
    pcd = np.random.rand(100, 3).astype(np.float32)
    mask = np.zeros(100, dtype=bool)
    aabb, centroid, n = compute_aabb_from_vertex_mask(mask, pcd)
    assert aabb is None
    assert centroid is None
    assert n == 0


def test_single_vertex_degenerate_aabb():
    pcd = np.array([[5, 5, 5]], dtype=np.float32)
    mask = np.array([True])
    aabb, centroid, n = compute_aabb_from_vertex_mask(mask, pcd)
    assert n == 1
    np.testing.assert_array_almost_equal(aabb, [5, 5, 5, 5, 5, 5])
    assert aabb_volume(aabb) == 0.0


def test_aabb_iou_self():
    aabb = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    assert aabb_iou(aabb, aabb) == pytest.approx(1.0)


def test_aabb_iou_disjoint():
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([5, 5, 5, 6, 6, 6])
    assert aabb_iou(a, b) == 0.0


def test_aabb_iou_partial_overlap():
    """a vol=8, b vol=8, intersection vol=1, union=15, IoU=1/15."""
    a = np.array([0, 0, 0, 2, 2, 2], dtype=np.float32)
    b = np.array([1, 1, 1, 3, 3, 3], dtype=np.float32)
    assert aabb_iou(a, b) == pytest.approx(1.0 / 15.0)


def test_batch_processing():
    rng = np.random.default_rng(0)
    pcd = rng.random((100, 3)).astype(np.float32) * 10
    masks = np.zeros((5, 100), dtype=bool)
    for i in range(5):
        masks[i, i * 20 : (i + 1) * 20] = True

    result = compute_instance_aabbs_batch(masks, pcd)
    assert len(result) == 5
    for i in range(5):
        assert result[i]["n_vertices"] == 20
        assert result[i]["aabb"] is not None
        assert result[i]["centroid"] is not None
        # AABB consistency: min <= max in every dim
        a = result[i]["aabb"]
        assert np.all(a[3:] >= a[:3])
