"""W1 entry point — HDBSCAN proposal gate validation.

Two stages run sequentially in one process to share the costly NuScenes
loaders (mini ~5s + trainval ~60s init):

  1. Sweep — 4×3×3 = 36 (min_cluster_size, min_samples, epsilon) combos
             scored by mean(n_clusters) over 40 cached samples. The best
             config inside [5, 15] closest to 10 is selected and recorded
             to parameter_sweep.json.
  2. Validate — re-runs that single best config on the same 40 samples,
             this time computing GT-cluster matching (M/L/D/miss) and
             writing per_sample/{token}.json + figures + report.md.

Spec § run flow allows separating these into two invocations; we keep one
process so the loaders are paid for once.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import signal
import sys
import time
import traceback

import numpy as np
import yaml

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_w1.measurements import match_gt_to_clusters, cluster_extents
from diagnosis_w1.aggregate import (
    aggregate_w1,
    render_all_w1,
    plot_parameter_sweep,
)


PER_SAMPLE_TIMEOUT_S = 60


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


# ----------------------------- sample loading -----------------------------

def _build_loader(version, data_config_path, out_dir, label):
    with open(data_config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["nuscenes"]["version"] = version
    # CAM_FRONT only is enough — we don't need images for HDBSCAN, but the
    # loader still loads them. Use minimum cams to cut load time.
    cfg["nuscenes"]["cameras"] = ["CAM_FRONT"]
    tmp = osp.join(out_dir, f"_data_config_{label}.yaml")
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"  loading {version} ...")
    t0 = time.time()
    loader = NuScenesLoader(config_path=tmp)
    print(f"    ready in {time.time() - t0:.1f}s ({len(loader)} samples)")
    return loader


def _cache_samples(loader, tokens, source_label):
    """Eagerly load the requested samples into a dict keyed by token.

    We strip out cached image arrays we don't need to keep memory low —
    point cloud + GT boxes + ego pose are the only fields W1 consumes.
    """
    tok_to_idx = {tok: i for i, tok in enumerate(loader.sample_tokens)}
    cache = {}
    missing = []
    for tok in tokens:
        if tok not in tok_to_idx:
            missing.append(tok)
            continue
        item = loader[tok_to_idx[tok]]
        cache[tok] = {
            "source": source_label,
            "pc_ego": item["point_cloud"],          # (N, 4)
            "gt_boxes": item["gt_boxes"],
            "ego_pose": item["ego_pose"],
            "sample_token": tok,
        }
    return cache, missing


# ----------------------------- sweep -----------------------------

def _run_sweep(cache, grid, out_dir):
    """Run HDBSCAN with each (min_cluster_size, min_samples, epsilon) on every
    cached sample. Score by mean(n_clusters). Returns full results + best.
    """
    tokens = sorted(cache.keys())
    results = []  # list of {config, mean_n, std_n, per_sample_n, valid_count}
    for mcs in grid["min_cluster_size"]:
        for ms in grid["min_samples"]:
            for eps in grid["cluster_selection_epsilon"]:
                gen = LiDARProposalGenerator(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    cluster_selection_epsilon=eps,
                )
                ns = []
                for tok in tokens:
                    pc = cache[tok]["pc_ego"]
                    out = gen.generate(pc)
                    ns.append(out["n_clusters"])
                mean_n = float(np.mean(ns))
                results.append({
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "cluster_selection_epsilon": eps,
                    "mean_n_clusters": mean_n,
                    "median_n_clusters": float(np.median(ns)),
                    "std_n_clusters": float(np.std(ns)),
                    "min_n_clusters": int(np.min(ns)),
                    "max_n_clusters": int(np.max(ns)),
                    "per_sample_n_clusters": ns,
                })

    # Selection: 5 ≤ mean ≤ 15, then closest to 10. Falls back to closest-to-10
    # overall if no combo lands in the band (so we still report a "best" plus
    # explicit reason it's outside the gate).
    in_band = [r for r in results if 5.0 <= r["mean_n_clusters"] <= 15.0]
    if in_band:
        best = min(in_band, key=lambda r: abs(r["mean_n_clusters"] - 10.0))
        best["selection"] = "in-band"
    else:
        best = min(results, key=lambda r: abs(r["mean_n_clusters"] - 10.0))
        best["selection"] = "out-of-band-fallback"

    sweep_record = {
        "n_samples_swept": len(tokens),
        "grid": grid,
        "results": results,
        "best": {k: best[k] for k in [
            "min_cluster_size", "min_samples", "cluster_selection_epsilon",
            "mean_n_clusters", "median_n_clusters", "std_n_clusters",
            "selection",
        ]},
        "selection_rule": "argmin |mean_n_clusters - 10| within 5..15; "
                          "falls back to global argmin when no combo enters the band.",
    }
    with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
        json.dump(sweep_record, f, indent=2)
    plot_parameter_sweep(results, grid, osp.join(out_dir, "figures",
                                                  "parameter_sweep_heatmap.png"))
    return sweep_record


# ----------------------------- validate -----------------------------

def _validate_with_config(cache, best_cfg, out_dir):
    per_sample_dir = osp.join(out_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    gen = LiDARProposalGenerator(
        min_cluster_size=best_cfg["min_cluster_size"],
        min_samples=best_cfg["min_samples"],
        cluster_selection_epsilon=best_cfg["cluster_selection_epsilon"],
    )

    succeeded, failed = [], []
    signal.signal(signal.SIGALRM, _alarm)

    tokens = sorted(cache.keys())
    for i, tok in enumerate(tokens):
        rec = cache[tok]
        print(f"[{i+1}/{len(tokens)}] {tok} ({rec['source']})")
        signal.alarm(PER_SAMPLE_TIMEOUT_S)
        t0 = time.time()
        try:
            pc = rec["pc_ego"]
            out = gen.generate(pc)
            cluster_ids = out["cluster_ids"]
            per_gt, cases = match_gt_to_clusters(
                rec["gt_boxes"], rec["ego_pose"], pc[:, :3], cluster_ids,
            )
            ext = cluster_extents(out["cluster_centroids"], out["cluster_bbox"], out["cluster_sizes"])
            n_total = len(rec["gt_boxes"])
            payload = {
                "sample_token": tok,
                "source": rec["source"],
                "config": gen.config_dict,
                "n_input_points": int(out["n_input_points"]),
                "n_clusters": int(out["n_clusters"]),
                "cluster_sizes": out["cluster_sizes"].tolist(),
                "cluster_extent_xy": ext["extent_xy"].tolist(),
                "cluster_extent_z": ext["extent_z"].tolist(),
                "cluster_distance_xy": ext["distance"].tolist(),
                "cluster_centroids_ego": out["cluster_centroids"].tolist(),
                "noise_ratio": out["noise_ratio"],
                "ground_filtered_ratio": out["ground_filtered_ratio"],
                "distance_filtered_ratio": out["distance_filtered_ratio"],
                "timing": out["timing"],
                "n_gt_total": int(n_total),
                "case_counts": cases,
                "per_gt": per_gt,
                "wall_seconds": float(time.time() - t0),
                "status": "ok",
            }
            signal.alarm(0)
            with open(osp.join(per_sample_dir, f"{tok}.json"), "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  ✓ {payload['wall_seconds']:.2f}s — {out['n_clusters']} clusters, "
                  f"M/L/D/miss = {cases['M']}/{cases['L']}/{cases['D']}/{cases['miss']}")
            succeeded.append(payload)
        except SampleTimeout:
            signal.alarm(0)
            failed.append({"sample_token": tok, "reason": "timeout",
                           "wall_seconds": float(time.time() - t0)})
            print(f"  ✗ timeout")
        except Exception as e:
            signal.alarm(0)
            tb = traceback.format_exc()
            print(f"  ✗ {e}\n{tb}")
            failed.append({"sample_token": tok, "reason": str(e),
                           "traceback": tb,
                           "wall_seconds": float(time.time() - t0)})

    return succeeded, failed


# ----------------------------- main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "sweep", "validate"], default="both")
    parser.add_argument("--num-mini", type=int, default=20)
    parser.add_argument("--num-trainval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--mini-samples-source", default="results/diagnosis/samples_used.json")
    parser.add_argument("--trainval-samples-source",
                        default="results/diagnosis_tier2_trainval/samples_used.json")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/w1_clustering_check")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(osp.join(out_dir, "per_sample"), exist_ok=True)
    os.makedirs(osp.join(out_dir, "figures"), exist_ok=True)

    # Load canonical sample tokens.
    with open(args.mini_samples_source) as f:
        mini_meta = json.load(f)
    with open(args.trainval_samples_source) as f:
        trainval_meta = json.load(f)

    mini_tokens = mini_meta["sample_tokens"][: args.num_mini]
    trainval_tokens = trainval_meta["sample_tokens"][: args.num_trainval]

    print("=" * 60)
    print(f"W1 HDBSCAN gate — {len(mini_tokens)} mini + {len(trainval_tokens)} trainval samples, seed={args.seed}")
    print("=" * 60)

    # Load both loaders.
    print("\nStep 0: load samples")
    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini")
    mini_cache, mini_missing = _cache_samples(mini_loader, mini_tokens, "mini")
    if mini_missing:
        print(f"  WARNING: {len(mini_missing)} mini tokens not found")
    del mini_loader  # release nusc handle to free memory

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval")
    trainval_cache, tv_missing = _cache_samples(trainval_loader, trainval_tokens, "trainval")
    if tv_missing:
        print(f"  WARNING: {len(tv_missing)} trainval tokens not found")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    used_tokens = list(cache.keys())
    print(f"  cached {len(cache)} samples in memory")

    with open(osp.join(out_dir, "samples_used.json"), "w") as f:
        json.dump({
            "seed": args.seed,
            "n_mini": len(mini_cache),
            "n_trainval": len(trainval_cache),
            "mini_source": args.mini_samples_source,
            "trainval_source": args.trainval_samples_source,
            "tokens": used_tokens,
            "tokens_by_source": {
                "mini": [t for t in used_tokens if cache[t]["source"] == "mini"],
                "trainval": [t for t in used_tokens if cache[t]["source"] == "trainval"],
            },
        }, f, indent=2)

    # ---- Step 1: sweep ----
    grid = {
        "min_cluster_size": [10, 20, 30, 50],
        "min_samples": [3, 5, 10],
        "cluster_selection_epsilon": [0.0, 0.5, 1.0],
    }

    if args.mode in ("both", "sweep"):
        print("\nStep 1: parameter sweep (4×3×3 = 36 combos)")
        t0 = time.time()
        sweep_record = _run_sweep(cache, grid, out_dir)
        b = sweep_record["best"]
        print(f"  sweep finished in {time.time() - t0:.1f}s")
        print(f"  best config: min_cluster_size={b['min_cluster_size']}, "
              f"min_samples={b['min_samples']}, eps={b['cluster_selection_epsilon']}, "
              f"mean_n={b['mean_n_clusters']:.2f} ({b['selection']})")
    else:
        with open(osp.join(out_dir, "parameter_sweep.json")) as f:
            sweep_record = json.load(f)

    if args.mode == "sweep":
        return 0

    # ---- Step 2: validate ----
    print("\nStep 2: validate with best config (write per_sample, figures, report)")
    succeeded, failed = _validate_with_config(cache, sweep_record["best"], out_dir)

    print("\n" + "=" * 60)
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed")
    print("=" * 60)

    # Spec § acceptance: ≥35/40 success.
    if len(succeeded) >= 35:
        agg = aggregate_w1(succeeded, sweep_record)
        with open(osp.join(out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        render_all_w1(succeeded, agg, sweep_record, failed,
                      osp.join(out_dir, "figures"),
                      osp.join(out_dir, "report.md"))
        print(f"  → {out_dir}/aggregate.json")
        print(f"  → {out_dir}/report.md")
    else:
        print(f"  Only {len(succeeded)} succeeded — below the ≥35 acceptance bar.")
        with open(osp.join(out_dir, "report.md"), "w") as f:
            f.write(f"# W1 — INCOMPLETE ({len(succeeded)}/{len(succeeded)+len(failed)} succeeded)\n")

    with open(osp.join(out_dir, "_failed.json"), "w") as f:
        json.dump(failed, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
