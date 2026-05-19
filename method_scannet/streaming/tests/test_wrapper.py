"""Unit tests for streaming/wrapper.py (StreamingScanNetEvaluator). GPU-free.

The wrapper depends on an OpenYOLO3D instance; tests inject a mock so the
real Mask3D / YOLO-World models are never loaded. The default PLY loader
is also overridden so tests do not need a real mesh file.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import imageio
import numpy as np
import pytest
import torch

from method_scannet.streaming.wrapper import StreamingScanNetEvaluator


def _make_mock_scene(
    tmpdir: Path,
    n_frames: int = 3,
    H: int = 64,
    W: int = 64,
) -> Path:
    """Create a minimal ScanNet-style scene directory layout.

    color/<i>.jpg, depth/<i>.png, poses/<i>.txt, intrinsics.txt, scene.ply
    (PLY is an empty placeholder; tests override _load_scene_vertices.)
    """
    tmpdir = Path(tmpdir)
    (tmpdir / "color").mkdir()
    (tmpdir / "depth").mkdir()
    (tmpdir / "poses").mkdir()

    intrinsic = np.array(
        [
            [50.0, 0.0, W / 2, 0.0],
            [0.0, 50.0, H / 2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.savetxt(tmpdir / "intrinsics.txt", intrinsic)

    for i in range(n_frames):
        rgb = np.full((H, W, 3), 128, dtype=np.uint8)
        imageio.imwrite(tmpdir / "color" / f"{i}.jpg", rgb)
        depth_mm = np.full((H, W), 1000, dtype=np.uint16)
        imageio.imwrite(tmpdir / "depth" / f"{i}.png", depth_mm)
        np.savetxt(tmpdir / "poses" / f"{i}.txt", np.eye(4, dtype=np.float64))

    (tmpdir / "scene.ply").touch()
    return tmpdir


def _mock_oy3d(n_vertices: int, n_instances: int) -> MagicMock:
    """Build a MagicMock OpenYOLO3D instance with the network attrs."""
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


def test_init_loads_frame_indices(tmp_path):
    """Init scans color/*.jpg and sets frame_indices in sorted order."""
    scene_dir = _make_mock_scene(tmp_path, n_frames=5)
    mock_oy = _mock_oy3d(n_vertices=10, n_instances=2)

    evaluator = StreamingScanNetEvaluator(mock_oy, str(scene_dir))

    assert evaluator.frame_indices == [0, 1, 2, 3, 4]
    # Mask3D and YOLO should NOT be called during __init__.
    mock_oy.network_3d.get_class_agnostic_masks.assert_not_called()
    mock_oy.network_2d.inference_detector.assert_not_called()


def test_setup_scene_calls_mask3d(tmp_path):
    """setup_scene() calls Mask3D exactly once and caches state."""
    scene_dir = _make_mock_scene(tmp_path)
    n_vertices, n_instances = 10, 3
    mock_oy = _mock_oy3d(n_vertices=n_vertices, n_instances=n_instances)

    evaluator = StreamingScanNetEvaluator(mock_oy, str(scene_dir))
    evaluator._load_scene_vertices = lambda mf: np.zeros(
        (n_vertices, 3), dtype=np.float64
    )
    evaluator.setup_scene()

    assert mock_oy.network_3d.get_class_agnostic_masks.call_count == 1
    assert evaluator.instance_vertex_masks is not None
    assert evaluator.instance_vertex_masks.shape == (n_instances, n_vertices)
    assert evaluator.instance_scores.shape == (n_instances,)
    assert evaluator.scene_vertices.shape == (n_vertices, 3)
    assert evaluator.intrinsic.shape == (3, 3)


def test_step_frame_returns_dict(tmp_path):
    """step_frame(t) returns dict with the documented keys and types."""
    scene_dir = _make_mock_scene(tmp_path)
    n_vertices, n_instances = 6, 2
    mock_oy = _mock_oy3d(n_vertices=n_vertices, n_instances=n_instances)

    evaluator = StreamingScanNetEvaluator(mock_oy, str(scene_dir))
    # Place all vertices at z=1m on the optical axis so they project into
    # the mock 64×64 image and match its uniform 1.0 depth map.
    evaluator._load_scene_vertices = lambda mf: np.tile(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (n_vertices, 1)
    )
    evaluator.setup_scene()

    result = evaluator.step_frame(0)

    assert isinstance(result, dict)
    assert set(result.keys()) >= {
        "frame_idx",
        "visible_instances",
        "current_instance_map",
        "frame_preds_2d",
    }
    assert result["frame_idx"] == 0
    assert isinstance(result["visible_instances"], np.ndarray)
    assert isinstance(result["current_instance_map"], dict)
    # YOLO was invoked exactly once for this frame.
    assert mock_oy.network_2d.inference_detector.call_count == 1
    # Visible instances are confirmed (baseline pass-through) so they
    # show up in the pred map.
    assert set(result["current_instance_map"].keys()) == set(
        int(k) for k in result["visible_instances"]
    )
    # Baseline stub assigns -1 until Task 1.2b implements label voting.
    assert all(v == -1 for v in result["current_instance_map"].values())


def test_run_streaming_iterates_all_frames(tmp_path):
    """run_streaming() yields one record per discovered frame."""
    scene_dir = _make_mock_scene(tmp_path, n_frames=4)
    n_vertices, n_instances = 5, 2
    mock_oy = _mock_oy3d(n_vertices=n_vertices, n_instances=n_instances)

    evaluator = StreamingScanNetEvaluator(mock_oy, str(scene_dir))
    evaluator._load_scene_vertices = lambda mf: np.tile(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (n_vertices, 1)
    )

    results = evaluator.run_streaming()

    assert len(results) == 4
    assert [r["frame_idx"] for r in results] == [0, 1, 2, 3]
    # Mask3D was called only once across the whole stream (scene-level).
    assert mock_oy.network_3d.get_class_agnostic_masks.call_count == 1
    # YOLO was called once per frame.
    assert mock_oy.network_2d.inference_detector.call_count == 4
    # pred_history grows with each frame.
    assert len(evaluator.pred_history) == 4
