"""Task 1.4a method-axis hooks for the streaming wrapper.

Mirrors the 5월 ``method_scannet.hooks`` pattern (offline monkey-patch),
but operates by *attribute injection* on a
:class:`method_scannet.streaming.wrapper.StreamingScanNetEvaluator`
instance — no global state, no monkey-patching of OpenYOLO3D core. Each
``install_*`` plants the method object on the evaluator's hook attribute;
``uninstall_*`` clears it.

Method ids understood by :func:`install_method_streaming`:

    'baseline'     — no hook (default)
    'M11'          — FrameCountingGate (registration, Phase 1)
    'M12'          — BayesianGate (registration, Phase 2)
    'M21'          — WeightedVoting (label, Phase 1)
    'M22'          — FeatureFusionEMA (label, Phase 2)
    'M31'          — IoUMerger (spatial merge, Phase 1)
    'M32'          — HungarianMerger (spatial merge, Phase 2)
    'phase1'       — M11 + M21 + M31
    'phase2'       — M12 + M22 + M32
    'M11+M21'      — registration + label, no merge
    'M12+M22'      — Phase 2 registration + label
    'M21+M31'      — label + merge (Phase 1 without registration)
    'M22+M32'      — label + merge (Phase 2 without registration)

The streaming wrapper does **not** call M21/M22 ``observe_frame`` /
``finalize`` unless the method exposes those methods. The Task 1.4b
implementation step will fill those out on the 5월 classes (they are
shipped today with no streaming interface), so for Task 1.4a these hooks
are wired up but the resulting predictions reduce to baseline.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Per-axis installers
# ---------------------------------------------------------------------------


def install_method_11(evaluator: Any, N: int = 3, consecutive: bool = False) -> None:
    from method_scannet.method_11_frame_counting import FrameCountingGate

    evaluator.method_11 = FrameCountingGate(N=N, consecutive=consecutive)


def uninstall_method_11(evaluator: Any) -> None:
    evaluator.method_11 = None


def install_method_12(evaluator: Any, prior: float = 0.5,
                      detection_likelihood: float = 0.8,
                      false_positive_rate: float = 0.2,
                      threshold: float = 0.95) -> None:
    from method_scannet.method_12_bayesian import BayesianGate

    evaluator.method_12 = BayesianGate(
        prior=prior,
        detection_likelihood=detection_likelihood,
        false_positive_rate=false_positive_rate,
        threshold=threshold,
    )


def uninstall_method_12(evaluator: Any) -> None:
    evaluator.method_12 = None


def install_method_21(evaluator: Any, **kwargs) -> None:
    from method_scannet.method_21_weighted_voting import WeightedVoting

    evaluator.method_21 = WeightedVoting(**kwargs)


def uninstall_method_21(evaluator: Any) -> None:
    evaluator.method_21 = None


def install_method_22(evaluator: Any, **kwargs) -> None:
    from method_scannet.method_22_feature_fusion import FeatureFusionEMA

    evaluator.method_22 = FeatureFusionEMA(**kwargs)


def uninstall_method_22(evaluator: Any) -> None:
    evaluator.method_22 = None


def install_method_31(evaluator: Any, **kwargs) -> None:
    from method_scannet.method_31_iou_merging import IoUMerger

    evaluator.method_31 = IoUMerger(**kwargs)


def uninstall_method_31(evaluator: Any) -> None:
    evaluator.method_31 = None


def install_method_32(evaluator: Any, **kwargs) -> None:
    from method_scannet.method_32_hungarian_merging import HungarianMerger

    evaluator.method_32 = HungarianMerger(**kwargs)


def uninstall_method_32(evaluator: Any) -> None:
    evaluator.method_32 = None


# ---------------------------------------------------------------------------
# Bulk by method-id
# ---------------------------------------------------------------------------


_SIMPLE_INSTALLERS = {
    "M11": install_method_11,
    "M12": install_method_12,
    "M21": install_method_21,
    "M22": install_method_22,
    "M31": install_method_31,
    "M32": install_method_32,
}

_COMPOUNDS = {
    "phase1": ("M11", "M21", "M31"),
    "phase2": ("M12", "M22", "M32"),
    "M11+M21": ("M11", "M21"),
    "M12+M22": ("M12", "M22"),
    "M21+M31": ("M21", "M31"),
    "M22+M32": ("M22", "M32"),
    "M11+M31": ("M11", "M31"),
    "M12+M32": ("M12", "M32"),
}


def install_method_streaming(
    evaluator: Any,
    method_id: str,
    **kwargs,
) -> None:
    """Install one method or a documented compound on the evaluator.

    Extra ``**kwargs`` are passed through to the underlying installer for
    single-axis ids. For compounds, kwargs are ignored (each axis uses its
    own defaults; tune individually via single-axis installs if needed).
    """
    if method_id == "baseline":
        return
    if method_id in _SIMPLE_INSTALLERS:
        _SIMPLE_INSTALLERS[method_id](evaluator, **kwargs)
        return
    if method_id in _COMPOUNDS:
        for axis in _COMPOUNDS[method_id]:
            _SIMPLE_INSTALLERS[axis](evaluator)
        return
    raise ValueError(f"Unknown method_id={method_id!r}")


def uninstall_all_streaming(evaluator: Any) -> None:
    """Reset every method hook to None."""
    uninstall_method_11(evaluator)
    uninstall_method_12(evaluator)
    uninstall_method_21(evaluator)
    uninstall_method_22(evaluator)
    uninstall_method_31(evaluator)
    uninstall_method_32(evaluator)


def list_method_ids() -> list[str]:
    return ["baseline", *_SIMPLE_INSTALLERS.keys(), *_COMPOUNDS.keys()]
