"""W1.5 orchestrator — runs Phase A → B → C in sequence.

Single Python process, single PBS job. NuScenes loaders are paid for once.
Sample cache is shared across all three phases for consistency. Each phase
writes its own verdict.json under results/.../phase_{a,b,c}/. The unified
report is written by aggregate.write_unified_report.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np

# Reuse W1 loader + cache helpers verbatim — no point reimplementing.
from diagnosis_w1.run_clustering_check import _build_loader, _cache_samples
from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_w1_5.phase_a_ground_filter import run_phase_a
from diagnosis_w1_5.phase_b_extended_sweep import run_phase_b
from diagnosis_w1_5.phase_c_distance_stratified import run_phase_c
from diagnosis_w1_5.aggregate import write_unified_report


def _verify_w1_regression(cache: dict, w1_per_sample_dir: str) -> dict:
    """Acceptance #8 — adapter backward-compat. For every cached mini token whose
    W1 per-sample JSON exists, compute n_clusters with the adapter (W1 best
    config) and verify it matches the saved value exactly. Any mismatch is fatal.
    """
    gen = LiDARProposalGenerator(min_cluster_size=50, min_samples=10,
                                  cluster_selection_epsilon=1.0)
    matches, mismatches = [], []
    checked = 0
    for tok, rec in cache.items():
        saved_path = osp.join(w1_per_sample_dir, f"{tok}.json")
        if not osp.exists(saved_path):
            continue
        with open(saved_path) as f:
            saved = json.load(f)
        new = gen.generate(rec["pc_ego"])
        checked += 1
        if int(new["n_clusters"]) == int(saved["n_clusters"]):
            matches.append(tok)
        else:
            mismatches.append({"tok": tok, "new": new["n_clusters"],
                               "saved": saved["n_clusters"]})
    return {"checked": checked, "matches": len(matches),
            "mismatches": mismatches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-mini", type=int, default=20)
    parser.add_argument("--num-trainval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--mini-samples-source",
                        default="results/diagnosis/samples_used.json")
    parser.add_argument("--trainval-samples-source",
                        default="results/diagnosis_tier2_trainval/samples_used.json")
    parser.add_argument("--w1-per-sample-dir",
                        default="results/w1_clustering_check/per_sample")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/w1_5_diagnostic_sweep")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    t_total_start = time.time()

    # ---- load samples (once, shared across phases) ----
    with open(args.mini_samples_source) as f:
        mini_meta = json.load(f)
    with open(args.trainval_samples_source) as f:
        trainval_meta = json.load(f)
    mini_tokens = mini_meta["sample_tokens"][: args.num_mini]
    trainval_tokens = trainval_meta["sample_tokens"][: args.num_trainval]

    print("=" * 60)
    print(f"W1.5 — {len(mini_tokens)} mini + {len(trainval_tokens)} trainval, seed={args.seed}")
    print("=" * 60)

    print("\nStep 0: load samples")
    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini_w15")
    mini_cache, _ = _cache_samples(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval_w15")
    trainval_cache, _ = _cache_samples(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")

    provenance = {
        "seed": args.seed,
        "n_mini": len(mini_cache),
        "n_trainval": len(trainval_cache),
        "mini_source": args.mini_samples_source,
        "trainval_source": args.trainval_samples_source,
        "tokens": list(cache.keys()),
        "tokens_by_source": {
            "mini": [t for t in cache if cache[t]["source"] == "mini"],
            "trainval": [t for t in cache if cache[t]["source"] == "trainval"],
        },
    }
    with open(osp.join(out_dir, "samples_used.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    # ---- Acceptance #8 sanity (adapter backward-compat) ----
    print("\nStep 0.5: W1 regression sanity (adapter backward-compat)")
    reg = _verify_w1_regression(cache, args.w1_per_sample_dir)
    print(f"  checked {reg['checked']} samples, "
          f"matches={reg['matches']}, mismatches={len(reg['mismatches'])}")
    if reg["mismatches"]:
        print("  ABORT — adapter backward-compat broken:")
        for m in reg["mismatches"][:5]:
            print(f"    {m['tok']}: new={m['new']}, saved={m['saved']}")
        sys.exit(2)
    with open(osp.join(out_dir, "w1_regression.json"), "w") as f:
        json.dump(reg, f, indent=2)

    # ---- Phase A ----
    phase_a = run_phase_a(cache, out_dir)

    # ---- Phase B ----
    phase_b = run_phase_b(cache, phase_a, out_dir)

    # ---- Phase C ----
    phase_c = run_phase_c(cache, phase_b, out_dir)

    # ---- unified report ----
    walltime = time.time() - t_total_start
    write_unified_report(out_dir, provenance, phase_a, phase_b, phase_c, walltime)
    print(f"\n→ report.md written. Total walltime {walltime/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
