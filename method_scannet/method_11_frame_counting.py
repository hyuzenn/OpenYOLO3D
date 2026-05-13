"""METHOD_11 — Frame-counting "wait and see" registration gate.

Streams an instance into the public map only after it has been visible in
at least ``N`` frames. Lightweight forward-only state.

Task 1.1 / 1.4 axis: registration (Phase 1, "simple" variant).
"""
from __future__ import annotations

from collections import defaultdict


class FrameCountingGate:
    """Pass-through gate that requires K consecutive (or cumulative) frames.

    Args:
        N: visibility-count threshold for confirmation.
        consecutive: if True, the count resets when an instance disappears
            in a frame; if False, counts are cumulative across the scene.
            Default cumulative (matches the "wait and see" brief).
    """

    def __init__(self, N: int = 3, consecutive: bool = False) -> None:
        if N <= 0:
            raise ValueError(f"N must be >= 1, got {N}")
        self.N = int(N)
        self.consecutive = bool(consecutive)
        self._counts: dict[int, int] = defaultdict(int)
        self._confirmed: set[int] = set()

    def reset(self) -> None:
        """Clear state between scenes."""
        self._counts.clear()
        self._confirmed.clear()

    def gate(self, visible_instances) -> list[int]:
        """Update counts for this frame and return only confirmed ids."""
        seen_this_frame = {int(i) for i in visible_instances}
        if self.consecutive:
            for k in list(self._counts):
                if k not in seen_this_frame and k not in self._confirmed:
                    self._counts[k] = 0
        for k in seen_this_frame:
            self._counts[k] += 1
            if self._counts[k] >= self.N:
                self._confirmed.add(k)
        # Emit the intersection of "currently visible" and "ever confirmed".
        return sorted(seen_this_frame & self._confirmed)

    @property
    def confirmed_count(self) -> int:
        return len(self._confirmed)
