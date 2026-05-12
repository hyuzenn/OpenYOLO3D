"""D3 frame visibility for streaming evaluation.

Per Task 1.1 Stage 2 decision: D3 (lenient, ≥1 vertex visible per instance).
The vertex-level visibility test (frustum + depth) matches OpenYOLO3D's
``WORLD_2_CAM.get_mesh_projections`` (utils/__init__.py:391), but evaluated
for a single frame and returned at instance-level granularity.

GPU-free (pure numpy). The wrapper calls this once per frame.
"""
from __future__ import annotations

import numpy as np


def compute_vertex_projection(
    scene_vertices: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    depth_map: np.ndarray,
    depth_threshold: float = 0.05,
) -> tuple:
    """Per-vertex projection + depth-consistent visibility for one frame.

    This is the vertex-level companion to :func:`compute_frame_visibility`.
    The streaming baseline accumulator (Task 1.2b) needs the per-vertex
    pixel coordinates and the per-vertex visibility mask, not just the
    instance-level D3 result.

    Returns:
        projection: (V, 2) int64 — (u, v) pixel coordinates for every
            vertex. Values are undefined for vertices that fail the
            in-front / frustum tests; the visibility mask must be used
            to gate downstream lookups.
        inside_mask: (V,) bool — True iff the vertex is in front of the
            camera, projects inside the image bounds, and the depth-map
            sample matches its camera-frame z within ``depth_threshold``.
    """
    if scene_vertices.ndim != 2 or scene_vertices.shape[1] != 3:
        raise ValueError(
            f"scene_vertices must be (V, 3); got {scene_vertices.shape}"
        )

    n_vertices = scene_vertices.shape[0]
    height, width = depth_map.shape[:2]

    vertices_h = np.concatenate(
        [scene_vertices, np.ones((n_vertices, 1), dtype=scene_vertices.dtype)],
        axis=1,
    )  # (V, 4)
    cam_coords = (camera_extrinsic[:3, :] @ vertices_h.T).T  # (V, 3)
    cam_z = cam_coords[:, 2]

    in_front = cam_z > 0
    safe_z = np.where(in_front, cam_z, 1.0)
    proj_xy = (camera_intrinsic @ cam_coords.T).T  # (V, 3)
    u = proj_xy[:, 0] / safe_z
    v = proj_xy[:, 1] / safe_z
    u_int = np.floor(u).astype(np.int64)
    v_int = np.floor(v).astype(np.int64)

    in_frustum = (
        in_front
        & (u_int >= 0)
        & (u_int < width)
        & (v_int >= 0)
        & (v_int < height)
    )

    inside_mask = np.zeros(n_vertices, dtype=bool)
    if in_frustum.any():
        idx = np.where(in_frustum)[0]
        depth_at_pixel = depth_map[v_int[idx], u_int[idx]]
        depth_match = np.abs(cam_z[idx] - depth_at_pixel) <= depth_threshold
        inside_mask[idx] = depth_match

    projection = np.stack([u_int, v_int], axis=1)  # (V, 2)
    return projection, inside_mask


def compute_frame_visibility(
    instance_vertex_masks: np.ndarray,
    scene_vertices: np.ndarray,
    camera_intrinsic: np.ndarray,
    camera_extrinsic: np.ndarray,
    depth_map: np.ndarray,
    depth_threshold: float = 0.05,
) -> np.ndarray:
    """Compute D3 instance visibility for a single frame.

    Returns:
        visible: (n_instances,) bool. ``visible[k]`` is True iff at least
            one vertex of instance ``k`` passes both the frustum test
            (pixel in image bounds) and the depth-consistency test against
            ``depth_map`` (D3 lenient gate).
    """
    if instance_vertex_masks.ndim != 2:
        raise ValueError(
            f"instance_vertex_masks must be (K, V); got {instance_vertex_masks.shape}"
        )
    if instance_vertex_masks.shape[1] != scene_vertices.shape[0]:
        raise ValueError(
            "instance_vertex_masks and scene_vertices vertex axis mismatch: "
            f"{instance_vertex_masks.shape[1]} vs {scene_vertices.shape[0]}"
        )

    _projection, inside_mask = compute_vertex_projection(
        scene_vertices, camera_intrinsic, camera_extrinsic, depth_map, depth_threshold
    )

    visible = (instance_vertex_masks & inside_mask[None, :]).any(axis=1)
    return visible
