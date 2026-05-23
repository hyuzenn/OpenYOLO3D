"""Diagnose whether the indoor (ScanNet200) streaming evaluator actually
invokes the M21 (WeightedVoting) and M31 (IoUMerger) modules on real frame
data, or whether they are silent no-ops / called-without-effect.

Motivation: the outdoor (nuScenes CenterPoint) evaluator installed
M21/M22/M31/M32/phase1 but never invoked them in step_sample (silent no-op).
Indoor 312-scene results show M21/M31 ≈ baseline (Δ ~1e-4); this script
decides — by code-level measurement, not inference — whether that is
(a) honest invocation with a tiny ScanNet200 effect, (b) the same silent
no-op artifact, or (c) invocation on trivial input with no effect.

Method (no OpenYOLO3D core edits — runtime monkey-patching only):
  - Wrap the May method-class methods AND the inline reimplementation sites
    in the streaming wrapper / adapters, recording per target:
      a) __init__ call count (instantiation),
      b) call count + input-arg shape histogram + per-call elapsed time,
      c) value-level effect:
         * M21: # instances whose final weighted-vote label differs from the
                baseline (uniform-vote) label on the same scene (= relabels),
         * M31: # proposals removed by merging (Σ input_K − output_K).
  - A patch target that fails to import or lacks the method raises
    immediately — a crash is diagnostic information, no silent fallback.

Output is raw counters/numbers only (no verdict classification, no
interpretation): per-axis JSON + stdout dump of the measured call counts,
arg-shape histograms, elapsed times, and value-effect numbers.

Usage:
  python -m diagnosis.verify_indoor_module_invocation --axis M21 --n-scenes 5
  python -m diagnosis.verify_indoor_module_invocation --axis M31 --n-scenes 5
  python -m diagnosis.verify_indoor_module_invocation --axis phase1 --n-scenes 5
"""
from __future__ import annotations

import argparse
import collections
import functools
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "2026-05-13_mask3d_cache"
DEFAULT_CONFIG = PROJECT_ROOT / "pretrained" / "config_scannet200.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "indoor_module_invocation"


# ---------------------------------------------------------------------------
# Instrumentation registry.
# ---------------------------------------------------------------------------
INSTR: dict[str, dict] = {}


def _stat(name: str) -> dict:
    return INSTR.setdefault(name, {
        "calls": 0,
        "elapsed_s": 0.0,
        "arg_shapes": collections.Counter(),
    })


def _shape_of(x):
    if x is None:
        return "None"
    if hasattr(x, "shape"):
        return "x".join(str(int(s)) for s in x.shape)
    if isinstance(x, (list, tuple)):
        return f"len{len(x)}"
    return type(x).__name__


def _wrap(owner, attr: str, key: str, pre=None, post=None):
    """Wrap owner.attr (class method or module function). Raise if missing."""
    if not hasattr(owner, attr):
        raise AttributeError(
            f"instrumentation target {key}: {owner!r} has no attribute {attr!r}")
    orig = getattr(owner, attr)

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        st = _stat(key)
        st["calls"] += 1
        if pre is not None:
            pre(st, args, kwargs)
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        st["elapsed_s"] += time.perf_counter() - t0
        if post is not None:
            post(st, args, kwargs, out)
        return out

    setattr(owner, attr, wrapped)


# ---------------------------------------------------------------------------
# Axis-specific instrumentation install.
# ---------------------------------------------------------------------------
def install_instrumentation(axis: str) -> dict:
    """Patch targets for the requested axis. Returns the target-key map."""
    want_m21 = axis in ("M21", "phase1")
    want_m31 = axis in ("M31", "phase1")
    if not (want_m21 or want_m31):
        raise ValueError(f"--axis must be M21 | M31 | phase1, got {axis!r}")

    keys = {"m21_methods": [], "m21_inline": [], "m31_methods": [], "m31_inline": []}

    if want_m21:
        from method_scannet import method_21_weighted_voting as m21mod
        from method_scannet.streaming import running_labeler as rlmod
        from method_scannet.streaming import method_adapters as mamod
        WV = m21mod.WeightedVoting

        def _init_post(st, a, k, out):
            st["arg_shapes"]["instantiated"] += 1
        _wrap(WV, "__init__", "WeightedVoting.__init__", post=_init_post)

        def _fw_pre(st, a, k):
            # frame_weight(self, camera_pos, instance_centroid, bbox_2d_center, image_size, confidence=)
            cam = a[1] if len(a) > 1 else k.get("camera_pos")
            ctr = a[2] if len(a) > 2 else k.get("instance_centroid")
            st["arg_shapes"][f"cam={_shape_of(cam)};ctr={_shape_of(ctr)}"] += 1
        _wrap(WV, "frame_weight", "WeightedVoting.frame_weight", pre=_fw_pre)

        def _vote_pre(st, a, k):
            labels = a[1] if len(a) > 1 else k.get("per_instance_frame_labels")
            st["arg_shapes"][f"n_inst={_shape_of(labels)}"] += 1
        _wrap(WV, "vote_distribution", "WeightedVoting.vote_distribution", pre=_vote_pre)
        _wrap(WV, "vote_label", "WeightedVoting.vote_label", pre=_vote_pre)
        keys["m21_methods"] = ["WeightedVoting.frame_weight",
                               "WeightedVoting.vote_distribution",
                               "WeightedVoting.vote_label"]

        # Inline reimplementation sites (streaming wrapper / adapter — not core).
        def _cmw_pre(st, a, k):
            xy = k.get("xy")
            if xy is None and len(a) > 3:
                xy = a[3]
            st["arg_shapes"][f"xy={_shape_of(xy)}"] += 1
        _wrap(rlmod.RunningInstanceLabeler, "_compute_m21_weight",
              "RunningInstanceLabeler._compute_m21_weight", pre=_cmw_pre)
        _wrap(mamod, "compute_predictions_method21",
              "method_adapters.compute_predictions_method21")
        keys["m21_inline"] = ["RunningInstanceLabeler._compute_m21_weight",
                              "method_adapters.compute_predictions_method21"]

    if want_m31:
        from method_scannet import method_31_iou_merging as m31mod
        from method_scannet.streaming import method_adapters as mamod
        IM = m31mod.IoUMerger

        def _init31_post(st, a, k, out):
            st["arg_shapes"]["instantiated"] += 1
        _wrap(IM, "__init__", "IoUMerger.__init__", post=_init31_post)

        def _merge_pre(st, a, k):
            pm = k.get("predicted_masks", a[1] if len(a) > 1 else None)
            pc = k.get("pred_classes", a[2] if len(a) > 2 else None)
            in_k = int(pc.shape[0]) if (pc is not None and hasattr(pc, "shape")) else -1
            st["arg_shapes"][f"masks={_shape_of(pm)}"] += 1
            st.setdefault("_in_k", []).append(in_k)

        def _merge_post(st, a, k, out):
            # out = (kept_masks (V,K'), kept_classes (K',), kept_scores (K',))
            out_k = int(out[1].shape[0]) if (out and hasattr(out[1], "shape")) else -1
            st.setdefault("_out_k", []).append(out_k)
            in_k = st["_in_k"][-1] if st.get("_in_k") else -1
            if in_k >= 0 and out_k >= 0:
                st["merges"] = st.get("merges", 0) + (in_k - out_k)
        _wrap(IM, "merge", "IoUMerger.merge", pre=_merge_pre, post=_merge_post)

        _wrap(mamod, "apply_method31_merge", "method_adapters.apply_method31_merge")
        keys["m31_methods"] = ["IoUMerger.merge"]
        keys["m31_inline"] = ["method_adapters.apply_method31_merge"]

    return keys


# ---------------------------------------------------------------------------
# Scene runner.
# ---------------------------------------------------------------------------
def run_one_scene(oy3d, cfg, cache_dir: Path, scene_name: str, axis: str) -> dict:
    """Build a fresh evaluator, run the streaming loop with `axis` installed
    ('baseline' = no methods), return the final all-instance label snapshot."""
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
    from method_scannet.streaming.hooks_streaming import (
        install_method_streaming, uninstall_all_streaming)

    scene_dir = PROJECT_ROOT / "data" / "scannet200" / scene_name
    cache_path = cache_dir / f"{scene_name}.pt"

    ev = StreamingScanNetEvaluator(
        openyolo3d_instance=oy3d,
        scene_dir=str(scene_dir),
        depth_scale=cfg["openyolo3d"]["depth_scale"],
        depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
        num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
        topk=int(cfg["openyolo3d"].get("topk", 40)),
        topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
    )
    frequency = int(cfg["openyolo3d"].get("frequency", 10))
    ev.frame_indices = [f for f in ev.frame_indices if f % frequency == 0]
    ev.setup_scene(mask3d_cache_path=str(cache_path))

    uninstall_all_streaming(ev)
    if axis != "baseline":
        install_method_streaming(ev, axis)

    for fi in ev.frame_indices:
        ev.step_frame(fi)
    # Trigger the finalize path (compute_predictions_method21 / apply_method31_merge).
    ev.compute_method_predictions()

    K = int(ev.instance_vertex_masks.shape[0])
    final_snapshot = ev.running_labeler.snapshot(list(range(K)))
    return {"n_frames": len(ev.frame_indices), "K": K, "snapshot": final_snapshot}


def relabel_count(axis_snap: dict, base_snap: dict) -> tuple[int, int]:
    """# instances whose label differs (both valid) and # compared."""
    diff = compared = 0
    for iid, la in axis_snap.items():
        lb = base_snap.get(iid, -1)
        if la == -1 or lb == -1:
            continue
        compared += 1
        if la != lb:
            diff += 1
    return diff, compared


# ---------------------------------------------------------------------------
# Raw counter summary.
# ---------------------------------------------------------------------------
def _summarize(key: str) -> dict:
    st = INSTR.get(key, {"calls": 0, "elapsed_s": 0.0, "arg_shapes": collections.Counter()})
    calls = st["calls"]
    out = {
        "calls": calls,
        "total_elapsed_s": round(st["elapsed_s"], 6),
        "mean_elapsed_ms": round(1000.0 * st["elapsed_s"] / calls, 4) if calls else 0.0,
        "arg_shapes": dict(st["arg_shapes"]),
    }
    if "merges" in st:
        out["merges"] = int(st["merges"])  # M31 only
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=["M21", "M31", "phase1"])
    ap.add_argument("--n-scenes", type=int, default=5)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    want_m21 = args.axis in ("M21", "phase1")
    want_m31 = args.axis in ("M31", "phase1")

    # ---- instrument BEFORE importing/constructing anything that uses them ---
    keys = install_instrumentation(args.axis)

    from evaluate import SCENE_NAMES_SCANNET200
    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    scenes = [s for s in SCENE_NAMES_SCANNET200
              if (cache_dir / f"{s}.pt").exists()
              and (PROJECT_ROOT / "data" / "scannet200" / s).is_dir()][: args.n_scenes]
    if not scenes:
        raise SystemExit("no usable scenes (cache + data) found")

    print(f"=== indoor module-invocation diagnostic ===", flush=True)
    print(f"axis={args.axis} n_scenes={len(scenes)} scenes={scenes}", flush=True)
    cfg = load_yaml(args.config)
    print("constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)

    per_scene = []
    total_relabel = total_compared = 0
    for i, sc in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] {sc} (axis) ...", flush=True)
        axis_res = run_one_scene(oy3d, cfg, cache_dir, sc, args.axis)
        rec = {"scene": sc, "n_frames": axis_res["n_frames"], "K": axis_res["K"]}
        if want_m21:
            print(f"           {sc} (baseline pass for relabel) ...", flush=True)
            base_res = run_one_scene(oy3d, cfg, cache_dir, sc, "baseline")
            d, c = relabel_count(axis_res["snapshot"], base_res["snapshot"])
            rec["m21_relabels"] = d
            rec["m21_compared"] = c
            total_relabel += d
            total_compared += c
        per_scene.append(rec)

    # ---- build report ------------------------------------------------------
    report = {"axis": args.axis, "n_scenes": len(scenes), "scenes": scenes,
              "per_scene": per_scene, "targets": {}}

    if want_m21:
        m21 = {k: _summarize(k) for k in keys["m21_methods"] + keys["m21_inline"]}
        report["targets"]["M21"] = m21
        report["m21_value_effect"] = {
            "relabels_total": total_relabel,
            "instances_compared": total_compared,
            "relabel_rate": round(total_relabel / total_compared, 6) if total_compared else None,
            "definition": "final weighted-vote label != baseline uniform-vote label, per instance",
        }
        report["targets"]["WeightedVoting.__init__"] = _summarize("WeightedVoting.__init__")

    if want_m31:
        m31 = {k: _summarize(k) for k in keys["m31_methods"] + keys["m31_inline"]}
        report["targets"]["M31"] = m31
        merges = INSTR.get("IoUMerger.merge", {}).get("merges", 0)
        in_k = INSTR.get("IoUMerger.merge", {}).get("_in_k", [])
        out_k = INSTR.get("IoUMerger.merge", {}).get("_out_k", [])
        report["m31_value_effect"] = {
            "merges_total": int(merges),
            "merge_calls": len(in_k),
            "input_proposals_total": int(sum(in_k)) if in_k else 0,
            "output_proposals_total": int(sum(out_k)) if out_k else 0,
            "definition": "proposals removed by IoU NMS = Σ(input_K − output_K) over merge calls",
        }
        report["targets"]["IoUMerger.__init__"] = _summarize("IoUMerger.__init__")

    out_path = OUT_DIR / f"{args.axis}_invocation.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ---- stdout summary (raw counters only) --------------------------------
    print("\n==================== RAW COUNTERS ====================", flush=True)
    if want_m21:
        init_c = report["targets"]["WeightedVoting.__init__"]["calls"]
        fw = report["targets"]["M21"]["WeightedVoting.frame_weight"]["calls"]
        vd = report["targets"]["M21"]["WeightedVoting.vote_distribution"]["calls"]
        vl = report["targets"]["M21"]["WeightedVoting.vote_label"]["calls"]
        cmw = report["targets"]["M21"]["RunningInstanceLabeler._compute_m21_weight"]["calls"]
        cp21 = report["targets"]["M21"]["method_adapters.compute_predictions_method21"]["calls"]
        ve = report["m21_value_effect"]
        print(f"  M21: __init__={init_c} | class methods: frame_weight={fw} "
              f"vote_distribution={vd} vote_label={vl}", flush=True)
        print(f"       inline reimpl: _compute_m21_weight={cmw} compute_predictions_method21={cp21}", flush=True)
        print(f"       value effect: relabels={ve['relabels_total']}/{ve['instances_compared']} "
              f"(rate={ve['relabel_rate']})", flush=True)
    if want_m31:
        init_c = report["targets"]["IoUMerger.__init__"]["calls"]
        mc = report["targets"]["M31"]["IoUMerger.merge"]["calls"]
        am = report["targets"]["M31"]["method_adapters.apply_method31_merge"]["calls"]
        ve = report["m31_value_effect"]
        print(f"  M31: __init__={init_c} | merge={mc} apply_method31_merge={am}", flush=True)
        print(f"       value effect: merges={ve['merges_total']} "
              f"(in={ve['input_proposals_total']} -> out={ve['output_proposals_total']} "
              f"over {ve['merge_calls']} calls)", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
