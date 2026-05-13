"""Unit tests for the Task 1.4a method-axis hooks. GPU-free.

Verifies:
- StreamingScanNetEvaluator has the six method_* attributes, all None by default.
- FrameCountingGate / BayesianGate gate behaviour.
- install_method_streaming wires each axis (single + compound) to the right slot.
- uninstall_all_streaming resets every slot.
- compute_method_predictions stays equivalent to compute_baseline_predictions
  when no method is installed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import imageio
import numpy as np
import torch

from method_scannet.method_11_frame_counting import FrameCountingGate
from method_scannet.method_12_bayesian import BayesianGate
from method_scannet.streaming.hooks_streaming import (
    install_method_streaming,
    list_method_ids,
    uninstall_all_streaming,
)
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator


# ---------------------------------------------------------------------------
# Mock scene helpers (same shape as test_wrapper.py)
# ---------------------------------------------------------------------------


def _make_mock_scene(tmpdir: Path, n_frames: int = 3, H: int = 64, W: int = 64) -> Path:
    tmpdir = Path(tmpdir)
    (tmpdir / "color").mkdir()
    (tmpdir / "depth").mkdir()
    (tmpdir / "poses").mkdir()
    intrinsic = np.array(
        [[50.0, 0.0, W / 2, 0.0], [0.0, 50.0, H / 2, 0.0],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.savetxt(tmpdir / "intrinsics.txt", intrinsic)
    for i in range(n_frames):
        imageio.imwrite(tmpdir / "color" / f"{i}.jpg", np.full((H, W, 3), 128, dtype=np.uint8))
        imageio.imwrite(tmpdir / "depth" / f"{i}.png", np.full((H, W), 1000, dtype=np.uint16))
        np.savetxt(tmpdir / "poses" / f"{i}.txt", np.eye(4, dtype=np.float64))
    (tmpdir / "scene.ply").touch()
    return tmpdir


def _mock_oy3d(n_vertices: int, n_instances: int) -> MagicMock:
    mock_oy = MagicMock()
    masks = torch.ones((n_vertices, n_instances), dtype=torch.bool)
    scores = torch.linspace(0.9, 0.5, steps=n_instances)
    mock_oy.network_3d.get_class_agnostic_masks.return_value = (masks, scores)
    mock_oy.network_2d.inference_detector.return_value = {
        "0": {
            "bbox": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.long),
            "scores": torch.empty((0,)),
        }
    }
    return mock_oy


# ---------------------------------------------------------------------------
# FrameCountingGate
# ---------------------------------------------------------------------------


def test_frame_counting_gate_confirms_after_N():
    g = FrameCountingGate(N=3)
    assert g.gate([1, 2]) == []        # count=1 each
    assert g.gate([1, 2]) == []        # count=2 each
    confirmed = g.gate([1, 2])
    assert confirmed == [1, 2]
    assert g.confirmed_count == 2


def test_frame_counting_gate_visible_only_emitted():
    g = FrameCountingGate(N=2)
    g.gate([1, 2])
    g.gate([1, 2])  # both confirmed
    confirmed = g.gate([1])  # only 1 is visible this frame
    assert confirmed == [1]


def test_frame_counting_gate_consecutive_reset():
    g = FrameCountingGate(N=3, consecutive=True)
    g.gate([1])      # count=1
    g.gate([1])      # count=2
    g.gate([])       # count resets
    g.gate([1])      # count=1 again
    confirmed = g.gate([1])  # count=2 — still not confirmed
    assert confirmed == []


# ---------------------------------------------------------------------------
# BayesianGate
# ---------------------------------------------------------------------------


def test_bayesian_gate_posterior_rises_with_observations():
    g = BayesianGate(prior=0.5, detection_likelihood=0.9, false_positive_rate=0.1, threshold=0.95)
    g.gate([7])
    p1 = g.posterior(7)
    g.gate([7])
    p2 = g.posterior(7)
    assert p1 > 0.5
    assert p2 > p1


def test_bayesian_gate_confirms_when_above_threshold():
    g = BayesianGate(prior=0.5, detection_likelihood=0.9, false_positive_rate=0.1, threshold=0.9)
    # 4 detections should clear posterior 0.9 with likelihood 0.9 / fpr 0.1.
    for _ in range(5):
        g.gate([42])
    assert 42 in set(g.gate([42]))


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------


def test_evaluator_has_all_method_slots(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    mock_oy = _mock_oy3d(n_vertices=5, n_instances=2)
    ev = StreamingScanNetEvaluator(mock_oy, str(scene_dir))
    for attr in ("method_11", "method_12", "method_21", "method_22",
                 "method_31", "method_32"):
        assert getattr(ev, attr) is None


def test_install_method_streaming_M11(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    ev = StreamingScanNetEvaluator(_mock_oy3d(5, 2), str(scene_dir))
    install_method_streaming(ev, "M11", N=4)
    assert isinstance(ev.method_11, FrameCountingGate)
    assert ev.method_11.N == 4
    assert ev.method_12 is None


def test_install_method_streaming_phase2(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    ev = StreamingScanNetEvaluator(_mock_oy3d(5, 2), str(scene_dir))
    install_method_streaming(ev, "phase2")
    assert ev.method_12 is not None
    assert ev.method_22 is not None
    assert ev.method_32 is not None
    assert ev.method_11 is None
    assert ev.method_21 is None
    assert ev.method_31 is None


def test_uninstall_all_streaming(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    ev = StreamingScanNetEvaluator(_mock_oy3d(5, 2), str(scene_dir))
    install_method_streaming(ev, "phase1")
    assert ev.method_11 is not None and ev.method_21 is not None and ev.method_31 is not None
    uninstall_all_streaming(ev)
    assert ev.method_11 is None and ev.method_21 is None and ev.method_31 is None
    assert ev.method_12 is None and ev.method_22 is None and ev.method_32 is None


def test_list_method_ids_contains_expected():
    ids = list_method_ids()
    for expected in ("baseline", "M11", "M12", "M21", "M22", "M31", "M32",
                     "phase1", "phase2", "M21+M31", "M22+M32"):
        assert expected in ids, f"missing {expected}"


def test_unknown_method_id_raises(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    ev = StreamingScanNetEvaluator(_mock_oy3d(5, 2), str(scene_dir))
    import pytest
    with pytest.raises(ValueError):
        install_method_streaming(ev, "M99")


def test_baseline_method_id_is_noop(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    ev = StreamingScanNetEvaluator(_mock_oy3d(5, 2), str(scene_dir))
    install_method_streaming(ev, "baseline")
    assert ev.method_11 is None and ev.method_21 is None


def test_compute_method_predictions_no_hook_matches_baseline(tmp_path):
    scene_dir = _make_mock_scene(tmp_path)
    n_vertices, n_instances = 6, 2
    mock_oy = _mock_oy3d(n_vertices, n_instances)
    ev = StreamingScanNetEvaluator(mock_oy, str(scene_dir))
    ev._load_scene_vertices = lambda mf: np.tile(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (n_vertices, 1)
    )
    ev.setup_scene()
    for fi in ev.frame_indices:
        ev.step_frame(fi)
    baseline = ev.compute_baseline_predictions()
    method = ev.compute_method_predictions()
    assert baseline["pred_masks"].shape == method["pred_masks"].shape
    np.testing.assert_array_equal(baseline["pred_classes"], method["pred_classes"])
    np.testing.assert_allclose(baseline["pred_scores"], method["pred_scores"])
