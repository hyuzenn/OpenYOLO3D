"""Step A orchestrator — pillar resolution sweep with too_small spatial analysis.

12 combos = 4 pillar sizes × 3 z_thresholds. Ground estimation fixed to
β1 best (percentile, p=10). HDBSCAN config locked to W1.5 best. Verticality
filter NOT applied — we want resolution effect in isolation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np

from diagnosis_w1.run_clustering_check import _build_loader, _cache_samples
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_step_a.measurements import (
    HDBSCAN_BEST, SampleTimeout, measure_with_timeout, PER_SAMPLE_TIMEOUT_S,
)
from diagnosis_step_a.aggregate import aggregate_step_a, render_all_step_a


GRID = {
    "pillar_size_xy": [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    "z_threshold": [0.2, 0.3, 0.5],
    # ground_estimation, percentile_p locked to β1 best
}
GROUND_FIXED = {"ground_estimation": "percentile", "percentile_p": 10.0}


def _combo_id(combo: dict) -> str:
    px, py = combo["pillar_size_xy"]
    return f"pillar{px:g}x{py:g}_zthr{combo['z_threshold']:g}"


def _verify_beta1_regression_combo(summaries) -> dict:
    """Acceptance #7 — find the (pillar=(0.5,0.5), z=0.3) combo and check its
    M_rate matches β1's saved 0.3612 to 4 decimals.
    """
    target = next(
        (s for s in summaries
         if tuple(s["combo"]["pillar_size_xy"]) == (0.5, 0.5)
         and s["combo"]["z_threshold"] == 0.3),
        None,
    )
    if target is None:
        return {"passes": False, "error": "regression combo not present in sweep"}
    M = target["mean_M_rate"]
    n = target["mean_n_clusters"]
    return {
        "combo_id": target["combo_id"],
        "M_rate": M,
        "expected_M_rate": 0.3612,
        "delta_M": abs(M - 0.3612),
        "n_clusters": n,
        "expected_n_clusters": 182.12,
        "delta_n_clusters": abs(n - 182.12),
        "passes": (abs(M - 0.3612) < 1e-3) and (abs(n - 182.12) < 0.5),
    }


def _run_combo(combo: dict, cache: dict, hdbscan_gen, out_dirs: dict) -> dict:
    cid = _combo_id(combo)
    combo_dir = osp.join(out_dirs["per_sample_per_config"], cid)
    os.makedirs(combo_dir, exist_ok=True)

    extractor = PillarForegroundExtractor(
        pillar_size_xy=combo["pillar_size_xy"],
        z_threshold=combo["z_threshold"],
        **GROUND_FIXED,
    )

    per_sample = []
    timeouts = errors = 0
    for tok in sorted(cache.keys()):
        try:
            rec = measure_with_timeout(extractor, hdbscan_gen, cache[tok])
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

    n_cl = [r["n_clusters"] for r in per_sample]
    M = [r["M_rate"] for r in per_sample]
    L = [r["L_rate"] for r in per_sample]
    D = [r["D_rate"] for r in per_sample]
    miss = [r["miss_rate"] for r in per_sample]
    timing = [r["timing_total"] for r in per_sample]
    fg = [r["foreground"]["ratio"] for r in per_sample]

    # Sum component breakdown counts across samples for this combo
    too_small_in = sum(r["components"]["too_small_inside_gt"] for r in per_sample)
    too_small_out = sum(r["components"]["too_small_outside_gt"] for r in per_sample)
    kept_in = sum(r["components"]["kept_inside_gt"] for r in per_sample)
    kept_out = sum(r["components"]["kept_outside_gt"] for r in per_sample)
    n_too_small = too_small_in + too_small_out
    n_kept = kept_in + kept_out

    avg_pillars = [r["avg_pillars_per_gt_box"] for r in per_sample]

    return {
        "combo": combo,
        "combo_id": cid,
        "n_samples_succeeded": len(per_sample),
        "n_timeouts": timeouts,
        "n_errors": errors,
        "mean_n_clusters": float(np.mean(n_cl)),
        "median_n_clusters": float(np.median(n_cl)),
        "mean_M_rate": float(np.mean(M)),
        "mean_L_rate": float(np.mean(L)),
        "mean_D_rate": float(np.mean(D)),
        "mean_miss_rate": float(np.mean(miss)),
        "mean_foreground_ratio": float(np.mean(fg)),
        "median_timing_total": float(np.median(timing)),
        "p95_timing_total": float(np.percentile(timing, 95)),
        "components_summed": {
            "too_small_inside_gt": int(too_small_in),
            "too_small_outside_gt": int(too_small_out),
            "kept_inside_gt": int(kept_in),
            "kept_outside_gt": int(kept_out),
            "n_too_small": int(n_too_small),
            "n_kept": int(n_kept),
            "too_small_inside_gt_ratio": (too_small_in / n_too_small) if n_too_small else 0.0,
            "kept_inside_gt_ratio": (kept_in / n_kept) if n_kept else 0.0,
        },
        "mean_avg_pillars_per_gt_box": float(np.mean(avg_pillars)),
    }


def _select_best(summaries):
    valid = [c for c in summaries if c["n_samples_succeeded"] >= 45]
    if not valid:
        return None, "NO_VALID_COMBOS"
    f1 = [c for c in valid if c["mean_M_rate"] >= 0.45
          and 10 <= c["mean_n_clusters"] <= 100
          and c["median_timing_total"] < 3.0]
    if f1:
        return max(f1, key=lambda c: c["mean_M_rate"]), "TIER1_STRONG"
    f2 = [c for c in valid if c["mean_M_rate"] >= 0.40
          and c["median_timing_total"] < 4.0]
    if f2:
        return max(f2, key=lambda c: c["mean_M_rate"]), "TIER2_PARTIAL"
    best_overall = max(valid, key=lambda c: c["mean_M_rate"])
    if abs(best_overall["mean_M_rate"] - 0.3612) < 0.03:
        return best_overall, "TIER3_PLATEAU"
    return best_overall, "TIER4_FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-w1-5-samples",
                        default="results/w1_5_diagnostic_sweep/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_step_a")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "per_sample_per_config": osp.join(out_dir, "per_sample_per_config"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(args.use_w1_5_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_w1_5_samples, osp.join(out_dir, "samples_used.json"))

    n_combos = len(GRID["pillar_size_xy"]) * len(GRID["z_threshold"])
    print("=" * 60)
    print(f"Step A — pillar resolution sweep")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens) + len(trainval_tokens)} samples (W1.5 set)")
    print(f"  sweep: {n_combos} combos (4 pillar × 3 z_thr; ground=percentile p=10 fixed)")
    print("=" * 60)

    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini_sa")
    mini_cache, _ = _cache_samples(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval_sa")
    trainval_cache, _ = _cache_samples(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")

    hdbscan_gen = LiDARProposalGenerator(**HDBSCAN_BEST)

    combos = []
    for ps, zt in itertools.product(GRID["pillar_size_xy"], GRID["z_threshold"]):
        combos.append({"pillar_size_xy": ps, "z_threshold": zt})

    print(f"\n=== sweep: {len(combos)} combos × {len(cache)} samples ===")
    t_sweep = time.time()
    summaries = []
    for i, combo in enumerate(combos):
        cid = _combo_id(combo)
        t_c = time.time()
        s = _run_combo(combo, cache, hdbscan_gen, out_dirs)
        elapsed = time.time() - t_c
        summaries.append(s)
        if s["n_samples_succeeded"] == 0:
            print(f"  [{i+1}/{len(combos)}] {cid}: FAILED")
        else:
            cs = s["components_summed"]
            print(f"  [{i+1}/{len(combos)}] {cid}: "
                  f"M={s['mean_M_rate']*100:.1f}%, "
                  f"miss={s['mean_miss_rate']*100:.1f}%, "
                  f"n_cl={s['mean_n_clusters']:.1f}, "
                  f"too_small_in_gt={cs['too_small_inside_gt_ratio']*100:.1f}%, "
                  f"avg_pillars/gt={s['mean_avg_pillars_per_gt_box']:.2f}, "
                  f"t={s['median_timing_total']:.2f}s "
                  f"({s['n_samples_succeeded']}/{len(cache)} ok, {elapsed:.1f}s)")
    print(f"  sweep finished in {time.time() - t_sweep:.1f}s")

    # ---- regression check (acceptance #7) ----
    reg = _verify_beta1_regression_combo(summaries)
    if reg.get("passes"):
        print(f"\n  β1 regression: M={reg['M_rate']:.4f} (Δ={reg['delta_M']:.6f}), "
              f"n_cl={reg['n_clusters']:.4f} (Δ={reg['delta_n_clusters']:.6f}) → PASS")
    else:
        print(f"\n  β1 regression: FAIL → {reg}")

    with open(osp.join(out_dir, "beta1_regression.json"), "w") as f:
        json.dump(reg, f, indent=2)
    if not reg.get("passes"):
        print("  ABORT — β1 baseline drift.")
        sys.exit(2)

    # ---- best ----
    best, verdict = _select_best(summaries)
    sweep_record = {
        "n_samples": len(cache),
        "grid": {"pillar_size_xy": [list(p) for p in GRID["pillar_size_xy"]],
                  "z_threshold": GRID["z_threshold"]},
        "ground_fixed": GROUND_FIXED,
        "n_combos": len(combos),
        "results": summaries,
        "selection_verdict": verdict,
        "best": best,
        "selection_rule": ("tier1: M≥0.45, 10≤n_cl≤100, t<3.0s; "
                           "tier2: M≥0.40, t<4.0s; "
                           "tier3 (PLATEAU): |M−0.3612| < 0.03; "
                           "tier4 (FAIL): otherwise"),
        "fixed_hdbscan_config": HDBSCAN_BEST,
    }
    with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
        json.dump(sweep_record, f, indent=2)
    if best is None:
        print("  No valid combo. Skipping aggregate.")
        return 0
    print(f"\n  selection verdict: {verdict}")
    print(f"  best: {best['combo_id']} "
          f"(M={best['mean_M_rate']*100:.1f}%, n_cl={best['mean_n_clusters']:.1f}, "
          f"miss={best['mean_miss_rate']*100:.1f}%, "
          f"too_small_in_gt={best['components_summed']['too_small_inside_gt_ratio']*100:.1f}%, "
          f"t={best['median_timing_total']:.2f}s)")

    # ---- aggregate + report ----
    agg = aggregate_step_a(summaries, best, sweep_record, reg, cache, out_dirs)
    with open(osp.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    render_all_step_a(summaries, best, sweep_record, reg, agg,
                       out_dirs["figures"], osp.join(out_dir, "report.md"))
    print(f"  → {out_dir}/aggregate.json")
    print(f"  → {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
