"""Self-check for the retroactive-emission gate policy.

Exercises the real ``NativeTemporalNuScenesEvaluator.step_sample`` /
``_finalize_scene_emission`` / ``_emit_frame`` with stubbed I/O (no nuScenes DB,
no proposal cache, no GPU), so the deferral machinery itself is under test
rather than a copy of it.

Run: python tests/test_retro_emission.py
"""
from __future__ import annotations

import numpy as np

from method_scannet.streaming import nuscenes_native_evaluator as NE
from method_scannet.streaming.nuscenes_native_evaluator import (
    NativeTemporalNuScenesEvaluator,
)
from method_scannet.streaming.nuscenes_evaluator import NuScenesRunningLabeler


class FixedAssociator:
    """Assigns each proposal the gid carried in its ``_gid`` field."""

    def step(self, proposals):
        return [int(p["_gid"]) for p in proposals]


def _proposal(gid: int, x: float) -> dict:
    return {"_gid": gid, "cls_name": "car", "cls_idx": 0, "score": 0.9,
            "bbox_lidar": [x, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            "centroid_ego": [x, 0.0, 0.0]}


def _make_evaluator(*, N: int, retro: bool, axis: str = "M11"):
    """Bare instance: skip __init__ (needs the nuScenes DB) and wire the handful
    of attributes step_sample actually touches."""
    ev = object.__new__(NativeTemporalNuScenesEvaluator)
    ev.association_frame = "ego"
    ev.association_threshold_m = 2.0
    ev.association_max_age = 5
    ev.class_agnostic_association = False
    ev.collect_track_metrics = False
    ev.frag_inject_p = 0.0
    ev.fuse_allow = None
    ev.begin_axis()
    ev.install_axis(axis, m11_N=N, retro_emission=retro)
    _attach_stubs(ev, scene_offset=0)
    return ev


def _attach_stubs(ev, scene_offset: int) -> None:
    ev.setup_scene(scene_offset=scene_offset)
    ev.associator = FixedAssociator()
    ev.labeler = NuScenesRunningLabeler(num_classes=NE.NUM_CLASSES)


def _end_scene(ev) -> None:
    """run_scene() calls this after its frame loop; retro emits at that point."""
    ev._finalize_scene_emission()


def _new_scene(ev, scene_offset: int) -> None:
    _end_scene(ev)
    _attach_stubs(ev, scene_offset)


def _run(ev, frames, scene_idx=0):
    """frames: list of (sample_token, [proposal, ...])."""
    eye = np.eye(4)
    ev._load_meta = lambda tok: (eye.copy(), eye.copy(), [])
    for tok, props in frames:
        ev._sample_scene_idx[tok] = scene_idx
        ev._get_proposals = lambda _t, _p=props: list(_p)
        ev.step_sample(tok)


def _tokens_of(ev):
    return {t: [b["tracking_id"] for b in v]
            for t, v in ev.per_sample_pred_boxes.items()}


def _leak_ok(ev) -> bool:
    a = ev.audit
    return a["n_retro_buffered"] == (a["n_retro_flushed"] + a["n_pending_dropped"]
                                     + a["n_pending_at_axis_end"])


def test_streaming_deletes_prefix():
    ev = _make_evaluator(N=3, retro=False)
    _run(ev, [(f"tok{i}", [_proposal(7, i)]) for i in range(1, 5)])
    _end_scene(ev)
    assert _tokens_of(ev) == {"tok1": [], "tok2": [], "tok3": [7], "tok4": [7]}
    assert ev.audit["n_retro_buffered"] == 0
    assert ev.audit["n_emitted_total"] == 2


def test_retro_recovers_prefix_at_origin_tokens():
    ev = _make_evaluator(N=3, retro=True)
    _run(ev, [(f"tok{i}", [_proposal(7, i)]) for i in range(1, 5)])
    assert _tokens_of(ev) == {"tok1": [], "tok2": [], "tok3": [], "tok4": []}, \
        "nothing may be written before the scene ends"
    _end_scene(ev)
    # frames 1-2 recovered at their OWN tokens, not at the confirmation frame
    assert _tokens_of(ev) == {"tok1": [7], "tok2": [7], "tok3": [7], "tok4": [7]}
    assert ev.audit["n_retro_buffered"] == 4
    assert ev.audit["n_retro_flushed"] == 4
    assert ev.audit["n_emitted_total"] == 4
    # token order preserved == chronological (e1_gt_metrics reads dict order)
    assert list(ev.per_sample_pred_boxes) == ["tok1", "tok2", "tok3", "tok4"]
    assert _leak_ok(ev), ev.audit


def test_retro_box_keeps_origin_geometry_not_confirmation_frame():
    ev = _make_evaluator(N=3, retro=True)
    _run(ev, [(f"tok{i}", [_proposal(7, float(i))]) for i in range(1, 4)])
    _end_scene(ev)
    # x was 1,2,3 at tok1,tok2,tok3 -> each emitted box keeps its own centroid
    xs = [ev.per_sample_pred_boxes[t][0]["translation"][0] for t in
          ("tok1", "tok2", "tok3")]
    assert xs == [1.0, 2.0, 3.0], xs


def test_never_confirmed_track_emits_nothing():
    ev = _make_evaluator(N=3, retro=True)
    _run(ev, [("tok1", [_proposal(7, 1)]), ("tok2", [_proposal(7, 2)])])
    _end_scene(ev)
    assert _tokens_of(ev) == {"tok1": [], "tok2": []}
    assert ev.audit["n_emitted_total"] == 0
    assert ev.audit["n_pending_dropped"] == 2
    assert _leak_ok(ev), ev.audit


def test_no_cross_scene_emission():
    ev = _make_evaluator(N=3, retro=True)
    _run(ev, [("s0f1", [_proposal(7, 1)]), ("s0f2", [_proposal(7, 2)])])
    _new_scene(ev, NE.SCENE_ID_STRIDE)
    # same gid reappearing in scene 1 must not resurrect scene 0's frames
    _run(ev, [(f"s1f{i}", [_proposal(7, i)]) for i in range(1, 4)], scene_idx=1)
    _end_scene(ev)
    got = _tokens_of(ev)
    assert got["s0f1"] == [] and got["s0f2"] == [], got
    assert got["s1f1"] == [7] and got["s1f2"] == [7] and got["s1f3"] == [7], got
    assert ev.audit["n_pending_at_axis_end"] == 0, "parking lot must drain per scene"
    assert _leak_ok(ev), ev.audit


def test_N1_retro_equals_streaming():
    frames = [(f"tok{i}", [_proposal(7, i)]) for i in range(1, 4)]
    a, b = _make_evaluator(N=1, retro=True), _make_evaluator(N=1, retro=False)
    _run(a, frames)
    _run(b, list(frames))
    _end_scene(a)
    _end_scene(b)
    assert _tokens_of(a) == _tokens_of(b) == {"tok1": [7], "tok2": [7], "tok3": [7]}
    assert a.audit["n_pending_dropped"] == 0   # N=1 confirms on sight
    assert a.audit["n_emitted_total"] == b.audit["n_emitted_total"] == 3


def test_multi_track_recovery_does_not_clobber_live_boxes():
    ev = _make_evaluator(N=3, retro=True)
    # gid 1 present from frame 1 (confirms at frame 3); gid 2 joins at frame 3
    _run(ev, [("tok1", [_proposal(1, 1.0)]),
              ("tok2", [_proposal(1, 2.0)]),
              ("tok3", [_proposal(1, 3.0), _proposal(2, 20.0)]),
              ("tok4", [_proposal(1, 4.0), _proposal(2, 21.0)]),
              ("tok5", [_proposal(1, 5.0), _proposal(2, 22.0)])])
    _end_scene(ev)
    got = _tokens_of(ev)
    assert got["tok1"] == [1] and got["tok2"] == [1], got
    assert sorted(got["tok3"]) == [1, 2] and sorted(got["tok4"]) == [1, 2], got
    assert sorted(got["tok5"]) == [1, 2], got
    assert ev.audit["n_emitted_total"] == 8


def test_phase1_retro_still_runs_the_spatial_merge():
    """E2c isolation: retro must not bypass M31 -- _emit_frame is shared."""
    ev = _make_evaluator(N=3, retro=True, axis="phase1")
    assert ev.method_31 is not None and ev.method_21 is not None, "phase1 members"
    seen: list[int] = []
    ev._apply_m31 = lambda emit: (seen.append(len(emit)) or emit)   # record, pass through
    _run(ev, [(f"tok{i}", [_proposal(7, i)]) for i in range(1, 4)])
    assert seen == [], "merge must not run before the scene ends"
    _end_scene(ev)
    # one M31 call per frame, over that frame's own emit set
    assert seen == [1, 1, 1], seen
    assert _tokens_of(ev) == {"tok1": [7], "tok2": [7], "tok3": [7]}


def test_phase1_streaming_merge_call_pattern_is_unchanged():
    ev = _make_evaluator(N=3, retro=False, axis="phase1")
    seen: list[int] = []
    ev._apply_m31 = lambda emit: (seen.append(len(emit)) or emit)
    _run(ev, [(f"tok{i}", [_proposal(7, i)]) for i in range(1, 4)])
    _end_scene(ev)
    # streaming merges only the confirmed frame (frames 1-2 emit nothing)
    assert seen == [1], seen


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all retro-emission checks passed")
