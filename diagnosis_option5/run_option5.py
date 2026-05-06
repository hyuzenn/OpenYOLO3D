"""Option 5 orchestrator — 2D detection-guided clustering sweep.

Stage 1 — load 50 W1.5 samples (with full 6-cam image set).
Stage 2 — run 6-cam YOLO-World **once per sample** and cache detections.
Stage 3 — sweep 27 (expand_ratio × min_points × min_depth) combos using the
          cached detections; β1 best pillar + W1.5 best HDBSCAN are locked.
Stage 4 — pick best by tiered selection rule, regress β1 spot-check, write
          aggregate + 7 figures + report.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import time

import numpy as np
import yaml
from PIL import Image

from dataloaders.nuscenes_loader import NuScenesLoader
from preprocessing.detection_frustum import FrustumExtractor
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from proposal.detection_guided_clustering import DetectionGuidedClusterer
from diagnosis_option5.measurements import (
    BETA1_BEST_PILLAR, HDBSCAN_BEST, SIX_CAMERAS,
    SampleTimeout, measure_with_timeout, PER_SAMPLE_TIMEOUT_S,
)
from diagnosis_option5.aggregate import aggregate_option5, render_all_option5


GRID = {
    "expand_ratio": [0.0, 0.1, 0.2],
    "min_points_per_frustum": [5, 10, 20],
    "min_depth": [0.5, 1.0, 2.0],
}


def _combo_id(combo: dict) -> str:
    return (f"er{combo['expand_ratio']:g}_minp{combo['min_points_per_frustum']}_"
            f"mind{combo['min_depth']:g}")


def _build_loader(version, data_config_path, out_dir, label):
    """Like diagnosis_w1._build_loader but keeps all 6 cameras (we need every
    cam image for YOLO + every intrinsic for frustum projection)."""
    with open(data_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["nuscenes"]["version"] = version
    cfg["nuscenes"]["cameras"] = list(SIX_CAMERAS)
    tmp = osp.join(out_dir, f"_data_config_{label}.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"  loading {version} ...")
    t0 = time.time()
    loader = NuScenesLoader(config_path=tmp)
    print(f"    ready in {time.time() - t0:.1f}s ({len(loader)} samples)")
    return loader


def _cache_with_six_cam(loader, tokens, source_label):
    tok_to_idx = {t: i for i, t in enumerate(loader.sample_tokens)}
    cache = {}
    for tok in tokens:
        if tok not in tok_to_idx:
            continue
        item = loader[tok_to_idx[tok]]
        intrinsics = {c: item["intrinsics"][c] for c in SIX_CAMERAS if c in item["intrinsics"]}
        cam_to_ego = {c: item["cam_to_ego"][c] for c in SIX_CAMERAS if c in item["cam_to_ego"]}
        image_hw = {c: item["images"][c].shape[:2] for c in SIX_CAMERAS if c in item["images"]}
        images = {c: item["images"][c] for c in SIX_CAMERAS if c in item["images"]}
        cache[tok] = {
            "source": source_label,
            "sample_token": tok,
            "pc_ego": item["point_cloud"],
            "gt_boxes": item["gt_boxes"],
            "ego_pose": item["ego_pose"],
            "images": images,
            "intrinsics_per_cam": intrinsics,
            "cam_to_ego_per_cam": cam_to_ego,
            "image_hw_per_cam": image_hw,
        }
    return cache


def _save_six_images_for_yolo(images_dict: dict, work_root: str, sample_token: str):
    sample_dir = osp.join(work_root, sample_token)
    os.makedirs(sample_dir, exist_ok=True)
    paths = []
    for i, cam in enumerate(SIX_CAMERAS):
        if cam not in images_dict:
            continue
        p = osp.join(sample_dir, f"{i}.jpg")
        Image.fromarray(images_dict[cam]).save(p, quality=95)
        paths.append(p)
    return sample_dir, paths


def _yolo_to_per_cam(yolo_preds: dict, text_prompts):
    """Network_2D returns dict keyed by frame_id (basename without ext). We
    saved 6 cams as 0.jpg..5.jpg, so frame_id "i" maps to SIX_CAMERAS[i]."""
    per_cam = {}
    for i, cam in enumerate(SIX_CAMERAS):
        pred = yolo_preds.get(str(i))
        if pred is None:
            per_cam[cam] = {"xyxy": [], "labels": [], "scores": []}
            continue
        b = pred["bbox"]
        l = pred["labels"]
        s = pred["scores"]
        bn = b.cpu().numpy() if hasattr(b, "cpu") else np.asarray(b)
        ln = l.cpu().numpy() if hasattr(l, "cpu") else np.asarray(l)
        sn = s.cpu().numpy() if hasattr(s, "cpu") else np.asarray(s)
        xyxys, labels, scores = [], [], []
        for j in range(bn.shape[0]):
            xyxys.append([float(v) for v in bn[j]])
            li = int(ln[j])
            labels.append(text_prompts[li] if 0 <= li < len(text_prompts) else f"class_{li}")
            scores.append(float(sn[j]))
        per_cam[cam] = {"xyxy": xyxys, "labels": labels, "scores": scores}
    return per_cam


def _run_yolo_cache(cache, oy3d_cfg, work_root):
    """Run YOLO-World 6-cam per sample, cache result, free image memory."""
    print(f"\nInitializing YOLO-World ...")
    t_init = time.time()
    from utils.utils_2d import Network_2D
    network = Network_2D(oy3d_cfg)
    print(f"  YOLO-World ready in {time.time() - t_init:.1f}s")
    text_prompts = oy3d_cfg["network2d"]["text_prompts"]
    yolo_cache = {}

    for i, tok in enumerate(sorted(cache.keys())):
        rec = cache[tok]
        sample_dir, paths = _save_six_images_for_yolo(rec["images"], work_root, tok)
        try:
            preds = network.get_bounding_boxes(paths, text=text_prompts)
            yolo_cache[tok] = _yolo_to_per_cam(preds, text_prompts)
        finally:
            shutil.rmtree(sample_dir, ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"  YOLO cached {i + 1}/{len(cache)}")
        # We don't need the raw images anymore; drop them to free memory.
        rec["images"] = None
    print(f"  YOLO cached {len(yolo_cache)}/{len(cache)} samples")
    del network
    gc.collect()
    return yolo_cache


def _run_combo(combo, cache, yolo_cache, hdbscan_gen, pillar_extractor, out_dirs):
    cid = _combo_id(combo)
    combo_dir = osp.join(out_dirs["per_sample_per_config"], cid)
    os.makedirs(combo_dir, exist_ok=True)

    frustum = FrustumExtractor(
        expand_ratio=combo["expand_ratio"],
        min_depth=combo["min_depth"],
    )
    clusterer = DetectionGuidedClusterer(
        frustum_extractor=frustum,
        pillar_extractor=pillar_extractor,
        hdbscan_generator=hdbscan_gen,
        min_points_per_frustum=combo["min_points_per_frustum"],
    )

    per_sample = []
    timeouts = errors = 0
    for tok in sorted(cache.keys()):
        try:
            rec = measure_with_timeout(clusterer, cache[tok], yolo_cache[tok])
            with open(osp.join(combo_dir, f"{tok}.json"), "w") as f:
                json.dump(rec, f, indent=2,
                          default=lambda o: float(o) if hasattr(o, "item") else str(o))
            per_sample.append(rec)
        except SampleTimeout:
            timeouts += 1
        except Exception as e:
            errors += 1
            print(f"  [combo {cid} | {tok}] error: {e}")

    if not per_sample:
        return {"combo": combo, "combo_id": cid,
                "n_samples_succeeded": 0, "n_timeouts": timeouts, "n_errors": errors}

    M = [r["M_rate"] for r in per_sample]
    L = [r["L_rate"] for r in per_sample]
    D = [r["D_rate"] for r in per_sample]
    miss = [r["miss_rate"] for r in per_sample]
    n_prop = [r["n_proposals_total"] for r in per_sample]
    timing = [r["timing_total_s"] for r in per_sample]
    recall = [r["two_d_recall"] for r in per_sample]
    M_in_det = [r["M_rate_within_detected"] for r in per_sample]
    M_in_undet = [r["M_rate_within_undetected"] for r in per_sample]

    return {
        "combo": combo,
        "combo_id": cid,
        "n_samples_succeeded": len(per_sample),
        "n_timeouts": timeouts,
        "n_errors": errors,
        "mean_n_proposals": float(np.mean(n_prop)),
        "median_n_proposals": float(np.median(n_prop)),
        "mean_M_rate": float(np.mean(M)),
        "mean_L_rate": float(np.mean(L)),
        "mean_D_rate": float(np.mean(D)),
        "mean_miss_rate": float(np.mean(miss)),
        "mean_2d_recall": float(np.mean(recall)),
        "mean_M_within_detected": float(np.mean([m for m in M_in_det if m is not None])) if M_in_det else 0.0,
        "mean_M_within_undetected": float(np.mean([m for m in M_in_undet if m is not None])) if M_in_undet else 0.0,
        "median_timing_total_s": float(np.median(timing)),
        "p95_timing_total_s": float(np.percentile(timing, 95)),
    }


def _select_best(summaries):
    valid = [c for c in summaries if c["n_samples_succeeded"] >= 45]
    if not valid:
        return None, "NO_VALID_COMBOS"
    f1 = [c for c in valid if c["mean_M_rate"] >= 0.50
          and c["mean_n_proposals"] <= 30
          and c["median_timing_total_s"] < 3.0]
    if f1:
        return max(f1, key=lambda c: c["mean_M_rate"]), "TIER1_STRONG"
    f2 = [c for c in valid if c["mean_M_rate"] >= 0.42
          and c["mean_n_proposals"] <= 50
          and c["median_timing_total_s"] < 4.0]
    if f2:
        return max(f2, key=lambda c: c["mean_M_rate"]), "TIER2_PARTIAL"
    f3 = [c for c in valid if abs(c["mean_M_rate"] - 0.36) < 0.03
          and c["mean_n_proposals"] <= 50]
    if f3:
        return max(f3, key=lambda c: c["mean_M_rate"]), "TIER3_PLATEAU"
    return max(valid, key=lambda c: c["mean_M_rate"]), "TIER4_FAIL"


def _verify_beta1_regression(cache, hdbscan_gen, pillar_extractor) -> dict:
    """Acceptance #7 — sample 5 samples and run β1-only (no frustum) → verify
    M_rate matches β1's saved 0.3612.
    """
    from diagnosis_w1.measurements import match_gt_to_clusters
    rng = np.random.default_rng(42)
    tokens = sorted(cache.keys())
    pick = sorted(rng.choice(len(tokens), size=min(5, len(tokens)), replace=False).tolist())
    M_rates = []
    n_cl = []
    for i in pick:
        rec = cache[tokens[i]]
        fg = pillar_extractor.extract(rec["pc_ego"])
        if fg["foreground_pcd"].shape[0] == 0:
            M_rates.append(0.0); n_cl.append(0)
            continue
        h = hdbscan_gen.generate(fg["foreground_pcd"])
        per_gt, cases = match_gt_to_clusters(
            rec["gt_boxes"], rec["ego_pose"],
            fg["foreground_pcd"][:, :3], h["cluster_ids"],
        )
        n_gt = len(rec["gt_boxes"])
        M_rates.append(cases["M"] / n_gt if n_gt else 0.0)
        n_cl.append(int(h["n_clusters"]))
    mean_M = float(np.mean(M_rates))
    mean_n = float(np.mean(n_cl))
    return {
        "checked_samples": len(M_rates),
        "sample_indices": pick,
        "mean_M_rate": mean_M,
        "expected_M_rate_full50": 0.3612,
        "mean_n_clusters": mean_n,
        "expected_n_clusters_full50": 182.12,
        # 5-sample subset will not exactly match the 50-sample mean; we check
        # whether it's in the same neighbourhood (Δ < 0.10) and that
        # individual sample n_clusters are sane.
        "passes": (abs(mean_M - 0.3612) < 0.10) and all(c > 0 for c in n_cl),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-step-a-samples",
                        default="results/diagnosis_step_a/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--openyolo-config", default="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_option5")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "per_sample_per_config": osp.join(out_dir, "per_sample_per_config"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(args.use_step_a_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_step_a_samples, osp.join(out_dir, "samples_used.json"))

    n_combos = (len(GRID["expand_ratio"]) * len(GRID["min_points_per_frustum"])
                * len(GRID["min_depth"]))
    print("=" * 60)
    print(f"Option 5 — 2D-detection-guided clustering")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens) + len(trainval_tokens)} samples")
    print(f"  sweep: {n_combos} combos")
    print("=" * 60)

    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini_o5")
    mini_cache = _cache_with_six_cam(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval_o5")
    trainval_cache = _cache_with_six_cam(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples (with 6-cam images)")

    # Load OpenYOLO3D config (for YOLO + Mask3D-side params we don't use)
    with open(args.openyolo_config) as f:
        oy3d_cfg = yaml.safe_load(f)

    work_root = tempfile.mkdtemp(prefix="option5_imgs_")
    try:
        yolo_cache = _run_yolo_cache(cache, oy3d_cfg, work_root)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    # ---- regression check (acceptance #7) ----
    pillar_extractor = PillarForegroundExtractor(**BETA1_BEST_PILLAR)
    hdbscan_gen = LiDARProposalGenerator(**HDBSCAN_BEST)
    print("\nβ1 regression spot-check (5 samples, no frustum) ...")
    reg = _verify_beta1_regression(cache, hdbscan_gen, pillar_extractor)
    print(f"  {reg['checked_samples']} samples, mean_M={reg['mean_M_rate']:.4f}, "
          f"mean_n_cl={reg['mean_n_clusters']:.4f} → "
          f"{'PASS' if reg['passes'] else 'FAIL'}")
    with open(osp.join(out_dir, "beta1_regression.json"), "w") as f:
        json.dump(reg, f, indent=2)
    if not reg["passes"]:
        print("  ABORT — β1 baseline drift.")
        sys.exit(2)

    # ---- sweep ----
    combos = []
    for er, mp, md in itertools.product(
        GRID["expand_ratio"], GRID["min_points_per_frustum"], GRID["min_depth"]
    ):
        combos.append({"expand_ratio": er, "min_points_per_frustum": mp, "min_depth": md})

    print(f"\n=== sweep: {len(combos)} combos × {len(cache)} samples ===")
    t_sweep_start = time.time()
    summaries = []
    for i, combo in enumerate(combos):
        cid = _combo_id(combo)
        t_c = time.time()
        s = _run_combo(combo, cache, yolo_cache, hdbscan_gen, pillar_extractor, out_dirs)
        elapsed = time.time() - t_c
        summaries.append(s)
        if s["n_samples_succeeded"] == 0:
            print(f"  [{i+1}/{len(combos)}] {cid}: FAILED")
        else:
            print(f"  [{i+1}/{len(combos)}] {cid}: "
                  f"M={s['mean_M_rate']*100:.1f}%, "
                  f"n_prop={s['mean_n_proposals']:.1f}, "
                  f"miss={s['mean_miss_rate']*100:.1f}%, "
                  f"2D_recall={s['mean_2d_recall']*100:.1f}%, "
                  f"t={s['median_timing_total_s']:.2f}s "
                  f"({s['n_samples_succeeded']}/{len(cache)} ok, {elapsed:.1f}s)")
    print(f"  sweep finished in {time.time() - t_sweep_start:.1f}s")

    # ---- best ----
    best, verdict = _select_best(summaries)
    sweep_record = {
        "n_samples": len(cache),
        "grid": GRID,
        "n_combos": len(combos),
        "results": summaries,
        "selection_verdict": verdict,
        "best": best,
        "selection_rule": ("tier1: M≥0.50, n_prop≤30, t<3.0s; "
                           "tier2: M≥0.42, n_prop≤50, t<4.0s; "
                           "tier3 (PLATEAU): |M−0.36|<0.03 and n_prop≤50; "
                           "tier4 (FAIL): otherwise"),
        "fixed_pillar_config": BETA1_BEST_PILLAR,
        "fixed_hdbscan_config": HDBSCAN_BEST,
    }
    with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
        json.dump(sweep_record, f, indent=2)

    if best is None:
        print("  No valid combo. Skipping aggregate.")
        return 0

    print(f"\n  selection verdict: {verdict}")
    print(f"  best: {best['combo_id']} "
          f"(M={best['mean_M_rate']*100:.1f}%, n_prop={best['mean_n_proposals']:.1f}, "
          f"miss={best['mean_miss_rate']*100:.1f}%, "
          f"2D_recall={best['mean_2d_recall']*100:.1f}%, "
          f"t={best['median_timing_total_s']:.2f}s)")

    agg = aggregate_option5(summaries, best, sweep_record, reg, cache, out_dirs)
    with open(osp.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    render_all_option5(summaries, best, sweep_record, reg, agg,
                       out_dirs["figures"], osp.join(out_dir, "report.md"))
    print(f"  → {out_dir}/aggregate.json")
    print(f"  → {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
