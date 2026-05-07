"""3D AABB helpers for METHOD_32 (HungarianMerger) integration.

OpenYOLO3D produces vertex-level boolean instance masks against the scene
mesh. METHOD_32 wants per-instance centroids and (optional) AABBs to feed
its spatial-cost term. These helpers convert vertex masks into AABBs
without touching upstream code.

NOTE: callers must pass numpy bool arrays. If the upstream pipeline yields
torch tensors (e.g. `predicted_masks.bool()` of shape `(V, K)`), convert
first with `.cpu().numpy()` and transpose to `(K, V)` for batch use. We
keep this module torch-free on purpose — pure numpy makes it cheap to call
inside post-processing loops and trivial to test.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_aabb_from_vertex_mask(
    vertex_mask: np.ndarray,
    scene_pcd: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Compute axis-aligned bounding box from an instance vertex selection.

    Args:
        vertex_mask: (N,) bool, True for vertices belonging to the instance.
        scene_pcd:   (N, 3) point cloud xyz coords.

    Returns:
        aabb       : (6,) ndarray [x_min, y_min, z_min, x_max, y_max, z_max],
                     or None if mask is empty.
        centroid   : (3,) ndarray [x, y, z], or None if mask is empty.
        n_vertices : int, count of selected vertices.

    Edge cases:
        - Empty mask  → (None, None, 0)
        - Single point → degenerate AABB with zero volume.
    """
    n_vertices = int(np.asarray(vertex_mask, dtype=bool).sum())
    if n_vertices == 0:
        return None, None, 0

    selected = scene_pcd[np.asarray(vertex_mask, dtype=bool)]
    aabb = np.concatenate([selected.min(axis=0), selected.max(axis=0)]).astype(np.float64)
    centroid = selected.mean(axis=0).astype(np.float64)
    return aabb, centroid, n_vertices


def aabb_volume(aabb: np.ndarray) -> float:
    """Volume of an axis-aligned bounding box.

    Args:
        aabb: (6,) [x_min, y_min, z_min, x_max, y_max, z_max].

    Returns:
        volume: float, >= 0. Negative dimensions are clamped to 0 so
        malformed boxes return 0 rather than raising.
    """
    arr = np.asarray(aabb, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 6:
        raise ValueError(f"aabb must have 6 elements, got shape {arr.shape}")
    dims = np.maximum(arr[3:] - arr[:3], 0.0)
    return float(dims[0] * dims[1] * dims[2])


def aabb_iou(aabb_a: np.ndarray, aabb_b: np.ndarray) -> float:
    """3D AABB IoU.

    Returns:
        iou: float in [0, 1]. 0 if either box is degenerate or boxes are
        disjoint.
    """
    a = np.asarray(aabb_a, dtype=np.float64).reshape(-1)
    b = np.asarray(aabb_b, dtype=np.float64).reshape(-1)
    if a.shape[0] != 6 or b.shape[0] != 6:
        raise ValueError("aabb_a and aabb_b must each have 6 elements")

    i_min = np.maximum(a[:3], b[:3])
    i_max = np.minimum(a[3:], b[3:])
    i_dims = np.maximum(i_max - i_min, 0.0)
    i_vol = float(i_dims[0] * i_dims[1] * i_dims[2])

    a_vol = aabb_volume(a)
    b_vol = aabb_volume(b)
    u_vol = a_vol + b_vol - i_vol
    if u_vol <= 0.0:
        return 0.0
    return float(i_vol / u_vol)


def compute_instance_aabbs_batch(
    vertex_masks: np.ndarray,
    scene_pcd: np.ndarray,
) -> dict:
    """Batch compute AABBs for multiple instances.

    Args:
        vertex_masks: (n_instances, N) bool array.
        scene_pcd  : (N, 3) point cloud.

    Returns:
        {instance_id (int) -> {'aabb': (6,)|None, 'centroid': (3,)|None,
                               'n_vertices': int}}
    """
    masks = np.asarray(vertex_masks, dtype=bool)
    if masks.ndim != 2:
        raise ValueError(
            f"vertex_masks must be 2D (n_instances, N), got shape {masks.shape}"
        )
    n_instances = masks.shape[0]
    out: dict = {}
    for i in range(n_instances):
        aabb, centroid, n_v = compute_aabb_from_vertex_mask(masks[i], scene_pcd)
        out[i] = {"aabb": aabb, "centroid": centroid, "n_vertices": n_v}
    return out
