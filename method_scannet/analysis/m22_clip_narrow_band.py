"""Step 3 analysis for METHOD_22 — quantify the CLIP narrow-band hypothesis.

Loads the cached ScanNet200 prompt embeddings (``v1`` bare "{class}" and
``v2`` "a photo of a {class}"), computes the 200×200 cosine-similarity
matrix, and identifies the high-confusion class pairs that explain the
−0.0285 AP regression observed in the May 2026 single-axis ablation
(results/2026-05-08_scannet_method_22_only_v02).

CPU only; ~5 s.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate.scannet200.scannet_constants import (
    HEAD_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
)


def _load_embeddings(path: Path) -> tuple[np.ndarray, list[str], str]:
    blob = torch.load(str(path), map_location="cpu")
    emb = blob["embeddings"]
    if isinstance(emb, torch.Tensor):
        emb_np = emb.detach().cpu().numpy()
    else:
        emb_np = np.asarray(emb)
    # L2 normalise so dot product == cosine.
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_np = emb_np / np.clip(norms, 1e-9, None)
    return emb_np, list(blob["class_names"]), str(blob.get("prompt_template", "?"))


def _bucket_of(name: str) -> str:
    if name in TAIL_CATS_SCANNET_200:
        return "tail"
    if name in COMMON_CATS_SCANNET_200:
        return "common"
    if name in HEAD_CATS_SCANNET_200:
        return "head"
    return "other"


def analyse(emb: np.ndarray, names: list[str], variant: str, out_dir: Path) -> dict:
    n = emb.shape[0]
    sim = emb @ emb.T  # (n, n), values in [-1, 1]
    # Mask diagonal (self-similarity = 1).
    sim_off = sim.copy()
    np.fill_diagonal(sim_off, -np.inf)
    nearest = sim_off.argmax(axis=1)
    nearest_scores = sim_off.max(axis=1)

    # Off-diagonal flat statistics.
    iu = np.triu_indices(n, k=1)
    off_vals = sim[iu]
    stats = {
        "variant": variant,
        "n_classes": n,
        "off_diag_min": float(off_vals.min()),
        "off_diag_max": float(off_vals.max()),
        "off_diag_mean": float(off_vals.mean()),
        "off_diag_median": float(np.median(off_vals)),
        "off_diag_p1": float(np.percentile(off_vals, 1)),
        "off_diag_p99": float(np.percentile(off_vals, 99)),
        "nearest_neighbour_score_min": float(nearest_scores.min()),
        "nearest_neighbour_score_max": float(nearest_scores.max()),
        "nearest_neighbour_score_mean": float(nearest_scores.mean()),
    }

    # Top-30 confused pairs by nearest-neighbour similarity.
    order = np.argsort(-nearest_scores)
    top_pairs = []
    for k in order[:30]:
        a = names[k]
        b = names[int(nearest[k])]
        ba, bb = _bucket_of(a), _bucket_of(b)
        top_pairs.append(
            {
                "class_a": a,
                "class_b": b,
                "bucket_a": ba,
                "bucket_b": bb,
                "cosine": float(nearest_scores[k]),
                "bucket_pair": f"{ba}-{bb}",
            }
        )

    # Bucket-aggregated confused-pair density: how often does the
    # nearest neighbour share / cross a bucket?
    bucket_pair_counts: dict[str, int] = {}
    bucket_diagonal: dict[str, int] = {}  # nearest is in the same bucket
    for i in range(n):
        a = _bucket_of(names[i])
        b = _bucket_of(names[int(nearest[i])])
        key = f"{a}->{b}"
        bucket_pair_counts[key] = bucket_pair_counts.get(key, 0) + 1
        if a == b:
            bucket_diagonal[a] = bucket_diagonal.get(a, 0) + 1
    stats["bucket_pair_counts"] = bucket_pair_counts
    stats["nearest_in_same_bucket"] = bucket_diagonal

    # --- Figures -------------------------------------------------------
    # Heatmap (downsampled order: head → common → tail for readability).
    bucket_order = {"head": 0, "common": 1, "tail": 2, "other": 3}
    perm = sorted(range(n), key=lambda i: (bucket_order[_bucket_of(names[i])], names[i]))
    sim_ordered = sim[np.ix_(perm, perm)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_ordered, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_title(f"CLIP class similarity ({variant})")
    ax.set_xlabel("class index (head -> common -> tail)")
    ax.set_ylabel("class index")
    # Bucket borders
    cumulative = 0
    for b in ("head", "common", "tail"):
        cumulative += sum(1 for nm in names if _bucket_of(nm) == b)
        ax.axhline(cumulative - 0.5, color="white", linewidth=0.5)
        ax.axvline(cumulative - 0.5, color="white", linewidth=0.5)
    plt.tight_layout()
    fig_path = out_dir / f"m22_confusion_heatmap_{variant.replace(' ', '_').replace('{', '').replace('}', '')}.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)

    # Histogram of off-diagonal similarities.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(off_vals, bins=80, color="steelblue", edgecolor="none")
    ax.axvline(off_vals.mean(), color="orange", linestyle="--", label=f"mean={off_vals.mean():.3f}")
    ax.axvline(np.percentile(off_vals, 99), color="red", linestyle="--",
               label=f"p99={np.percentile(off_vals, 99):.3f}")
    ax.set_xlabel("pairwise cosine similarity (off-diagonal)")
    ax.set_ylabel("count")
    ax.set_title(f"M22 narrow-band hypothesis ({variant})")
    ax.legend()
    plt.tight_layout()
    hist_path = out_dir / f"m22_offdiag_histogram_{variant.replace(' ', '_').replace('{', '').replace('}', '')}.png"
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)

    return {
        "stats": stats,
        "top_confused_pairs": top_pairs,
        "figures": {
            "heatmap": str(fig_path.relative_to(out_dir.parent.parent)),
            "histogram": str(hist_path.relative_to(out_dir.parent.parent)),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1", default="pretrained/scannet200_prompt_embeddings.pt", type=str
    )
    parser.add_argument(
        "--v2", default="pretrained/scannet200_prompt_embeddings_v2.pt", type=str
    )
    parser.add_argument(
        "--output",
        default="results/2026-05-13_step3_m22_m32_failures",
        type=str,
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    print("=== Step 3 — M22 CLIP narrow-band analysis ===")
    out: dict = {}
    for label, path in [("v1", args.v1), ("v2", args.v2)]:
        emb, names, template = _load_embeddings(Path(path))
        print(f"  {label}: template={template!r} n={len(names)}")
        analysis = analyse(emb, names, label, out_dir / "figures")
        out[label] = analysis
        s = analysis["stats"]
        print(f"    off-diag: mean={s['off_diag_mean']:.3f} median={s['off_diag_median']:.3f} "
              f"p99={s['off_diag_p99']:.3f}")
        print(f"    nearest-neighbour cosine: mean={s['nearest_neighbour_score_mean']:.3f} "
              f"max={s['nearest_neighbour_score_max']:.3f}")
        print(f"    top-5 confused pairs:")
        for p in analysis["top_confused_pairs"][:5]:
            print(f"      {p['class_a']} <-> {p['class_b']}  cos={p['cosine']:.3f} "
                  f"[{p['bucket_pair']}]")

    (out_dir / "m22_clip_narrow_band.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'm22_clip_narrow_band.json'}")


if __name__ == "__main__":
    main()
