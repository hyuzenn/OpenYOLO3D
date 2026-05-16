"""Task 1.4a redesign — 1-scene integration smoke test.

Runs the streaming wrapper on a single ScanNet200 scene under each of the
10 method configurations (baseline, M11, M12, M21, M22, M31, M32,
phase1, phase2, M21+M31, M22+M32) using the cached Mask3D output, then
checks two PASS conditions:

    (a) No crash → run completes and produces a non-empty preds dict.
    (b) Output differs from baseline → method is not silently a no-op.
        Difference is measured against (n_predictions, pred_classes,
        pred_scores) — any divergence counts as PASS.

This is the regression that caught the first-attempt no-op: unit tests
exercised the hook plumbing with mock data, but real-scene inference
with the May class signatures was never run.

Usage:

    CUDA_VISIBLE_DEVICES=0 python -m method_scannet.streaming.tests.test_integration_real_scene \\
        --scene scene0011_00 \\
        --output results/2026-05-14_task14a_smoke_v01

Smoke results are written to ``<output>/smoke_results.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


SMOKE_CONFIGS_FULL: tuple[tuple[str, str, dict], ...] = (
    # (label, install_method_id, kwargs)
    ("baseline", "baseline", {}),
    ("M11", "M11", {"N": 3}),
    ("M12", "M12", {}),
    ("M21", "M21", {}),
    ("M22", "M22", {}),
    ("M31", "M31", {}),
    ("M32", "M32", {}),
    ("phase1", "phase1", {}),
    ("phase2", "phase2", {}),
    ("M21+M31", "M21+M31", {}),
    ("M22+M32", "M22+M32", {}),
)

# Re-run subset for the M32 standalone fix verification.
SMOKE_CONFIGS_M32_FIX: tuple[tuple[str, str, dict], ...] = (
    ("baseline", "baseline", {}),
    ("M32", "M32", {}),
)

SMOKE_CONFIGS = SMOKE_CONFIGS_FULL


def _summarize_preds(preds: dict) -> dict:
    """Compact fingerprint of a preds dict, for log + diff."""
    pm = preds["pred_masks"]
    pc = np.asarray(preds["pred_classes"]).astype(np.int64)
    ps = np.asarray(preds["pred_scores"]).astype(np.float64)
    return {
        "n_predictions": int(pm.shape[1]) if pm.ndim == 2 else 0,
        "n_unique_classes": int(np.unique(pc).size),
        "class_mode": int(np.bincount(pc).argmax()) if pc.size else -1,
        "scores_mean": float(ps.mean()) if ps.size else 0.0,
        "scores_std": float(ps.std()) if ps.size else 0.0,
        "scores_max": float(ps.max()) if ps.size else 0.0,
        "masks_total_vertices": int(pm.sum()) if pm.size else 0,
    }


def _preds_differ(a: dict, b: dict) -> bool:
    """True if the two summaries differ beyond numerical noise."""
    if a["n_predictions"] != b["n_predictions"]:
        return True
    if a["n_unique_classes"] != b["n_unique_classes"]:
        return True
    if a["masks_total_vertices"] != b["masks_total_vertices"]:
        return True
    if abs(a["scores_mean"] - b["scores_mean"]) > 1e-6:
        return True
    if abs(a["scores_max"] - b["scores_max"]) > 1e-6:
        return True
    return False


def _run_one_axis(
    oy3d,
    cfg: dict,
    scene_dir: Path,
    cache_path: Path,
    method_label: str,
    method_id: str,
    method_kwargs: dict,
    frequency: int,
) -> dict:
    from method_scannet.streaming.hooks_streaming import (
        install_method_streaming,
        uninstall_all_streaming,
    )
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator

    t0 = time.time()
    evaluator = StreamingScanNetEvaluator(
        openyolo3d_instance=oy3d,
        scene_dir=str(scene_dir),
        depth_scale=cfg["openyolo3d"]["depth_scale"],
        depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
        num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
        topk=int(cfg["openyolo3d"].get("topk", 40)),
        topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
    )
    evaluator.frame_indices = [
        f for f in evaluator.frame_indices if f % frequency == 0
    ]
    evaluator.setup_scene(mask3d_cache_path=str(cache_path))

    uninstall_all_streaming(evaluator)
    if method_id != "baseline":
        install_method_streaming(evaluator, method_id, **method_kwargs)

    error = None
    summary = None
    try:
        for fi in evaluator.frame_indices:
            evaluator.step_frame(fi)
        preds = evaluator.compute_method_predictions()
        summary = _summarize_preds(preds)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc!s}"
    walltime = time.time() - t0
    out = {
        "label": method_label,
        "method_id": method_id,
        "method_kwargs": method_kwargs,
        "walltime_sec": float(walltime),
        "n_frames_streamed": int(len(evaluator.frame_indices)),
    }
    if error is not None:
        out["status"] = "crash"
        out["error"] = error
    else:
        out["status"] = "ok"
        out.update(summary)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0011_00", type=str)
    parser.add_argument(
        "--cache-dir",
        default="results/2026-05-13_mask3d_cache",
        type=str,
    )
    parser.add_argument(
        "--config",
        default="pretrained/config_scannet200.yaml",
        type=str,
    )
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--subset",
        choices=["full", "m32_fix"],
        default="full",
        help="full = 11 configs; m32_fix = baseline + M32 only (verify fix).",
    )
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "run.log"
    log = log_path.open("w")

    def log_print(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    log_print(f"=== Task 1.4a — 1-scene integration smoke ===")
    log_print(f"scene  : {args.scene}")
    log_print(f"cache  : {args.cache_dir}")
    log_print(f"output : {out_root}")

    scene_dir = Path("data/scannet200") / args.scene
    cache_path = Path(args.cache_dir) / f"{args.scene}.pt"
    if not scene_dir.exists():
        log_print(f"[fatal] scene dir missing: {scene_dir}")
        sys.exit(2)
    if not cache_path.exists():
        log_print(f"[fatal] mask3d cache missing: {cache_path}")
        sys.exit(2)

    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    cfg = load_yaml(args.config)
    frequency = int(cfg["openyolo3d"].get("frequency", 10))

    log_print("[init] Constructing OpenYolo3D ...")
    t = time.time()
    oy3d = OpenYolo3D(args.config)
    log_print(f"[init] OpenYolo3D ready in {time.time() - t:.1f}s")

    configs = SMOKE_CONFIGS_M32_FIX if args.subset == "m32_fix" else SMOKE_CONFIGS_FULL
    log_print(f"subset : {args.subset} ({len(configs)} configs)")

    all_results: list[dict] = []
    for label, method_id, kwargs in configs:
        log_print(f"\n--- [{label}] method_id={method_id} kwargs={kwargs} ---")
        r = _run_one_axis(
            oy3d, cfg, scene_dir, cache_path,
            label, method_id, kwargs, frequency,
        )
        if r["status"] == "ok":
            log_print(
                f"  [{label}] OK in {r['walltime_sec']:.1f}s  "
                f"n_pred={r['n_predictions']}  "
                f"n_cls={r['n_unique_classes']}  "
                f"mean_score={r['scores_mean']:.4f}  "
                f"max_score={r['scores_max']:.4f}"
            )
        else:
            log_print(f"  [{label}] CRASH  {r['error']}")
        all_results.append(r)

    baseline_summary = next(
        (r for r in all_results if r["label"] == "baseline" and r["status"] == "ok"),
        None,
    )
    if baseline_summary is None:
        log_print("\n[fatal] baseline run failed — cannot compute PASS table")
        (out_root / "smoke_results.json").write_text(json.dumps(all_results, indent=2))
        sys.exit(3)

    pass_count = 0
    fail_count = 0
    log_print("\n=== SMOKE PASS TABLE ===")
    log_print(f"{'method':<10}  {'status':<7}  {'n_pred':>6}  "
              f"{'n_cls':>5}  {'mean':>8}  {'max':>8}  PASS?")
    for r in all_results:
        if r["status"] == "crash":
            verdict = "CRASH"
            fail_count += 1
        elif r["label"] == "baseline":
            verdict = "ref"
        else:
            differs = _preds_differ(r, baseline_summary)
            verdict = "PASS" if differs else "NO-OP"
            if differs:
                pass_count += 1
            else:
                fail_count += 1
        if r["status"] == "ok":
            log_print(
                f"{r['label']:<10}  {r['status']:<7}  {r['n_predictions']:>6}  "
                f"{r['n_unique_classes']:>5}  {r['scores_mean']:>8.4f}  "
                f"{r['scores_max']:>8.4f}  {verdict}"
            )
        else:
            log_print(
                f"{r['label']:<10}  {r['status']:<7}  {'-':>6}  "
                f"{'-':>5}  {'-':>8}  {'-':>8}  {verdict}"
            )

    n_methods = len(configs) - 1  # exclude baseline
    log_print(f"\nSmoke: {pass_count}/{n_methods} PASS  (FAIL/CRASH: {fail_count})")

    (out_root / "smoke_results.json").write_text(json.dumps(all_results, indent=2))
    log_print(f"\nResults written to {out_root / 'smoke_results.json'}")
    log.close()

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
