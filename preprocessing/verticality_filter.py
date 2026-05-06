"""BEV connected-component + aspect-ratio filter.

Sits on top of β1's pillar foreground extraction. The intent is to peel
off vertical/extended structure (walls, guardrails, building facades) that
survive the height-based pillar filter — they have lots of points above
ground but their BEV footprint is a long thin strip rather than a
vehicle/pedestrian-shaped compact blob.

Pipeline:
  1. Build a BEV occupancy grid at ``pillar_size_xy``
  2. scipy.ndimage.label with 4-connectivity → connected components
  3. Per-component metrics: pillar count, BEV bounding-rect L×W, aspect ratio
  4. Classify each component:
       * ``too_small``           — n_pillars < size_min      (likely noise)
       * ``extended``            — aspect > aspect_max       (wall / guardrail)
       * ``too_large_compact``   — n_pillars > size_max AND aspect ≤ aspect_max
                                                              (large structure)
       * ``keep``                — everything else            (vehicle / pedestrian)
  5. Drop points belonging to removed components.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from scipy.ndimage import label as scipy_label


@dataclass
class VerticalityConfig:
    size_min: int = 5
    size_max: int = 100
    aspect_max: float = 5.0
    pillar_size_xy: Tuple[float, float] = (0.5, 0.5)  # match β1 best


class VerticalityFilter:
    def __init__(
        self,
        size_min: int = 5,
        size_max: int = 100,
        aspect_max: float = 5.0,
        pillar_size_xy: Tuple[float, float] = (0.5, 0.5),
        use_neighborhood: bool = True,
    ):
        # use_neighborhood is plumbed for symmetry with the spec; today only
        # 4-connectivity is implemented (the scipy default).
        del use_neighborhood
        self.config = VerticalityConfig(
            size_min=int(size_min),
            size_max=int(size_max),
            aspect_max=float(aspect_max),
            pillar_size_xy=tuple(pillar_size_xy),
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def filter(self, foreground_pcd: np.ndarray) -> dict:
        """Return the connected-component-filtered point cloud + counts.

        ``foreground_pcd`` is whatever β1 emitted in ``foreground_pcd``: shape
        (M, ≥3), columns 0:3 are x, y, z; remaining columns are preserved on
        the kept subset.
        """
        t0 = time.perf_counter()
        if foreground_pcd.ndim != 2 or foreground_pcd.shape[1] < 3:
            raise ValueError(f"foreground_pcd must be (M, ≥3); got {foreground_pcd.shape}")

        M = foreground_pcd.shape[0]
        if M == 0:
            return {
                "kept_pcd": foreground_pcd[:0],
                "kept_mask": np.zeros((0,), dtype=bool),
                "n_components_total": 0,
                "n_components_kept": 0,
                "n_components_removed_too_small": 0,
                "n_components_removed_too_large_compact": 0,
                "n_components_removed_extended": 0,
                "removed_point_ratio": 0.0,
                "n_input_points": 0,
                "n_kept_points": 0,
                "timing": {"grid_build": 0.0, "connected_components": 0.0,
                           "classify_and_mask": 0.0, "total": 0.0},
            }

        pts = foreground_pcd[:, :3]
        px, py = self.config.pillar_size_xy

        ix = np.floor(pts[:, 0] / px).astype(np.int64)
        iy = np.floor(pts[:, 1] / py).astype(np.int64)
        ix_min, ix_max = int(ix.min()), int(ix.max())
        iy_min, iy_max = int(iy.min()), int(iy.max())
        H = ix_max - ix_min + 1
        W = iy_max - iy_min + 1
        grid = np.zeros((H, W), dtype=bool)
        ix_offset = ix - ix_min
        iy_offset = iy - iy_min
        grid[ix_offset, iy_offset] = True
        t1 = time.perf_counter()

        # 4-connectivity is scipy.ndimage.label's default 2D rank-1 structure.
        labeled, n_components = scipy_label(grid)
        t2 = time.perf_counter()

        if n_components == 0:
            return {
                "kept_pcd": foreground_pcd[:0],
                "kept_mask": np.zeros((M,), dtype=bool),
                "n_components_total": 0,
                "n_components_kept": 0,
                "n_components_removed_too_small": 0,
                "n_components_removed_too_large_compact": 0,
                "n_components_removed_extended": 0,
                "removed_point_ratio": 1.0,
                "n_input_points": int(M),
                "n_kept_points": 0,
                "timing": {"grid_build": float(t1 - t0),
                           "connected_components": float(t2 - t1),
                           "classify_and_mask": 0.0,
                           "total": float(t2 - t0)},
            }

        # Per-component bounding rect + size.
        # np.bincount(labeled.ravel()) gives sizes, but we need bboxes too.
        # scipy.ndimage.find_objects returns slices — handy.
        from scipy.ndimage import find_objects
        slices = find_objects(labeled)  # list, slices[i] for component (i+1); None if missing

        size_min = self.config.size_min
        size_max = self.config.size_max
        aspect_max = self.config.aspect_max

        component_class = np.zeros(n_components + 1, dtype=np.int8)
        # 0 = sentinel (no component); 1 = keep; 2 = too_small; 3 = extended; 4 = too_large_compact
        n_too_small = n_extended = n_too_large_compact = n_keep = 0
        for cid in range(1, n_components + 1):
            sl = slices[cid - 1]
            if sl is None:
                component_class[cid] = 2  # treat as removed
                n_too_small += 1
                continue
            sub = labeled[sl] == cid
            n_pillars = int(sub.sum())
            if n_pillars < size_min:
                component_class[cid] = 2
                n_too_small += 1
                continue
            length_pillars = sl[0].stop - sl[0].start
            width_pillars = sl[1].stop - sl[1].start
            length_m = length_pillars * px
            width_m = width_pillars * py
            short = min(length_m, width_m)
            if short <= 0:
                aspect = float("inf")
            else:
                aspect = max(length_m, width_m) / short
            if aspect > aspect_max:
                component_class[cid] = 3
                n_extended += 1
            elif n_pillars > size_max:
                component_class[cid] = 4
                n_too_large_compact += 1
            else:
                component_class[cid] = 1
                n_keep += 1

        # Per-point lookup
        point_component = labeled[ix_offset, iy_offset]
        keep_mask = component_class[point_component] == 1
        t3 = time.perf_counter()

        return {
            "kept_pcd": foreground_pcd[keep_mask],
            "kept_mask": keep_mask,
            "n_components_total": int(n_components),
            "n_components_kept": int(n_keep),
            "n_components_removed_too_small": int(n_too_small),
            "n_components_removed_too_large_compact": int(n_too_large_compact),
            "n_components_removed_extended": int(n_extended),
            "n_input_points": int(M),
            "n_kept_points": int(keep_mask.sum()),
            "removed_point_ratio": float((~keep_mask).sum() / M) if M else 0.0,
            "timing": {
                "grid_build": float(t1 - t0),
                "connected_components": float(t2 - t1),
                "classify_and_mask": float(t3 - t2),
                "total": float(t3 - t0),
            },
        }
