"""Cosine-distribution probe for setting M22/M32 thresholds from data.

Record-only monkey-patches (no OpenYOLO3D core edits) on:
  - FeatureFusionEMA.predict_label  -> image<->text cosine: per-call max cosine
    and top1-top2 margin over the prompt bank (drives the M22 confidence/margin
    gate thresholds tau, m).
  - HungarianMerger.build_cost_matrix -> image<->image cosine among same-class
    instances and their pairwise centroid distances (drives M32 semantic_threshold
    and distance_threshold).

Axis to run:
  --axis M22+M32  : real CLIP image features flow into M32 -> image<->image sims
                    are meaningful; also exercises M22 predict (image<->text).
  --axis M32      : M32-only is spatial (zero features) -> sims ~0; use it only
                    for the same-class pairwise DISTANCE distribution.

Outputs percentiles + coarse histograms (not raw arrays) to
results/cosine_probe/<axis>_cosine_probe.json + stdout.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "2026-05-13_mask3d_cache"
DEFAULT_CONFIG = PROJECT_ROOT / "pretrained" / "config_scannet200.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "cosine_probe"

PROBE: dict = {
    "m22_maxcos": [], "m22_margin": [], "m22_top1_minus_mean": [],
    "m32_cand_sim": [], "m32_all_pair_dist": [], "m32_cand_pair_dist": [],
    "n_classes": None, "n_predict_calls": 0, "n_buildcost_calls": 0,
}


def _pct(arr):
    if not arr:
        return None
    a = np.asarray(arr, dtype=np.float64)
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n": int(a.size),
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()), "std": float(a.std()),
        "pct": {str(q): float(np.percentile(a, q)) for q in qs},
    }


def _hist(arr, lo, hi, bins=20):
    if not arr:
        return None
    counts, edges = np.histogram(np.asarray(arr, dtype=np.float64), bins=bins, range=(lo, hi))
    return {"edges": [float(x) for x in edges], "counts": [int(c) for c in counts]}


def install_probes():
    from method_scannet.method_22_feature_fusion import FeatureFusionEMA, _l2_normalize
    from method_scannet.method_31_iou_merging import IoUMerger  # noqa: F401 (sanity import)
    from method_scannet import method_32_hungarian_merging as m32mod

    # ---- M22 predict_label: image<->text cosine max + margin ----
    if not hasattr(FeatureFusionEMA, "predict_label"):
        raise AttributeError("FeatureFusionEMA.predict_label missing")
    _orig_predict = FeatureFusionEMA.predict_label

    @functools.wraps(_orig_predict)
    def predict_label_probe(self, instance_id):
        feat = self.instance_features.get(int(instance_id))
        if feat is not None and self._prompt_emb_norm is not None:
            import torch
            prompts = self._prompt_emb_norm.to(feat.device)
            f = _l2_normalize(feat.unsqueeze(0).float())
            sims = (f @ prompts.t()).squeeze(0)
            PROBE["n_classes"] = int(sims.numel())
            vals, _ = torch.sort(sims, descending=True)
            top1 = float(vals[0].item())
            top2 = float(vals[1].item()) if vals.numel() > 1 else float(vals[0].item())
            PROBE["m22_maxcos"].append(top1)
            PROBE["m22_margin"].append(top1 - top2)
            PROBE["m22_top1_minus_mean"].append(top1 - float(sims.mean().item()))
            PROBE["n_predict_calls"] += 1
        return _orig_predict(self, instance_id)

    FeatureFusionEMA.predict_label = predict_label_probe

    # ---- M32 build_cost_matrix: image<->image sim + same-class pair dist ----
    HM = m32mod.HungarianMerger
    if not hasattr(HM, "build_cost_matrix"):
        raise AttributeError("HungarianMerger.build_cost_matrix missing")
    _orig_bcm = HM.build_cost_matrix
    _cos_mat = m32mod._cosine_similarity_matrix

    @functools.wraps(_orig_bcm)
    def build_cost_matrix_probe(self, centroids, features):
        c = np.asarray(centroids, dtype=np.float64)
        n = c.shape[0]
        if n >= 2:
            diff = c[:, None, :] - c[None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            sim = _cos_mat(features)
            dthr = float(self.distance_threshold)
            iu = np.triu_indices(n, k=1)
            for di, si in zip(dist[iu], sim[iu]):
                PROBE["m32_all_pair_dist"].append(float(di))
                if di <= dthr:
                    PROBE["m32_cand_pair_dist"].append(float(di))
                    PROBE["m32_cand_sim"].append(float(si))
            PROBE["n_buildcost_calls"] += 1
        return _orig_bcm(self, centroids, features)

    HM.build_cost_matrix = build_cost_matrix_probe


def run_axis_scene(oy3d, cfg, cache_dir: Path, scene_name: str, axis: str):
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
    from method_scannet.streaming.hooks_streaming import (
        install_method_streaming, uninstall_all_streaming)
    scene_dir = PROJECT_ROOT / "data" / "scannet200" / scene_name
    ev = StreamingScanNetEvaluator(
        openyolo3d_instance=oy3d, scene_dir=str(scene_dir),
        depth_scale=cfg["openyolo3d"]["depth_scale"],
        depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
        num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
        topk=int(cfg["openyolo3d"].get("topk", 40)),
        topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
    )
    frequency = int(cfg["openyolo3d"].get("frequency", 10))
    ev.frame_indices = [f for f in ev.frame_indices if f % frequency == 0]
    ev.setup_scene(mask3d_cache_path=str(cache_dir / f"{scene_name}.pt"))
    uninstall_all_streaming(ev)
    install_method_streaming(ev, axis)
    for fi in ev.frame_indices:
        ev.step_frame(fi)
    ev.compute_method_predictions()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=["M22+M32", "M32"])
    ap.add_argument("--n-scenes", type=int, default=5)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    install_probes()

    from evaluate import SCENE_NAMES_SCANNET200
    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    scenes = [s for s in SCENE_NAMES_SCANNET200
              if (cache_dir / f"{s}.pt").exists()
              and (PROJECT_ROOT / "data" / "scannet200" / s).is_dir()][: args.n_scenes]
    if not scenes:
        raise SystemExit("no usable scenes")

    print(f"=== cosine-distribution probe axis={args.axis} scenes={scenes} ===", flush=True)
    cfg = load_yaml(args.config)
    print("constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)
    for i, sc in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] {sc} ...", flush=True)
        run_axis_scene(oy3d, cfg, cache_dir, sc, args.axis)

    report = {
        "axis": args.axis, "n_scenes": len(scenes), "scenes": scenes,
        "n_classes": PROBE["n_classes"],
        "n_predict_calls": PROBE["n_predict_calls"],
        "n_buildcost_calls": PROBE["n_buildcost_calls"],
        "m22_image_text": {
            "max_cosine": _pct(PROBE["m22_maxcos"]),
            "max_cosine_hist": _hist(PROBE["m22_maxcos"], 0.0, 0.6, 24),
            "top1_minus_top2_margin": _pct(PROBE["m22_margin"]),
            "margin_hist": _hist(PROBE["m22_margin"], 0.0, 0.3, 30),
            "top1_minus_mean": _pct(PROBE["m22_top1_minus_mean"]),
        },
        "m32_image_image": {
            "candidate_pair_cosine": _pct(PROBE["m32_cand_sim"]),  # dist<=distance_threshold
            "candidate_sim_hist": _hist(PROBE["m32_cand_sim"], -0.2, 1.0, 24),
            "same_class_pair_dist_all": _pct(PROBE["m32_all_pair_dist"]),
            "candidate_pair_dist": _pct(PROBE["m32_cand_pair_dist"]),
            "cand_dist_hist": _hist(PROBE["m32_cand_pair_dist"], 0.0, 2.0, 20),
        },
    }
    out_path = OUT_DIR / f"{args.axis.replace('+','_')}_cosine_probe.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n==================== COSINE DISTRIBUTIONS ====================", flush=True)
    print(f"n_classes(prompt bank)={PROBE['n_classes']} predict_calls={PROBE['n_predict_calls']} "
          f"buildcost_calls={PROBE['n_buildcost_calls']}", flush=True)
    mt = report["m22_image_text"]["max_cosine"]
    mg = report["m22_image_text"]["top1_minus_top2_margin"]
    if mt:
        print(f"M22 image-text max-cosine: mean={mt['mean']:.4f} p10={mt['pct']['10']:.4f} "
              f"p50={mt['pct']['50']:.4f} p90={mt['pct']['90']:.4f}", flush=True)
    if mg:
        print(f"M22 top1-top2 margin:      mean={mg['mean']:.4f} p10={mg['pct']['10']:.4f} "
              f"p50={mg['pct']['50']:.4f} p90={mg['pct']['90']:.4f}", flush=True)
    cs = report["m32_image_image"]["candidate_pair_cosine"]
    cd = report["m32_image_image"]["candidate_pair_dist"]
    if cs:
        print(f"M32 candidate-pair image-image cosine: n={cs['n']} mean={cs['mean']:.4f} "
              f"p50={cs['pct']['50']:.4f} p90={cs['pct']['90']:.4f} p99={cs['pct']['99']:.4f}", flush=True)
    if cd:
        print(f"M32 candidate-pair distance (m): n={cd['n']} mean={cd['mean']:.4f} "
              f"p50={cd['pct']['50']:.4f} p90={cd['pct']['90']:.4f}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
