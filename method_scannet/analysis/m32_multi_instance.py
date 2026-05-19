"""Step 3 analysis for METHOD_32 — quantify the multi-instance
absorption hypothesis on ScanNet200 ground truth.

Parses every val-scene's ``ground_truth/<scene>.txt`` to build the class
× instance-count distribution, identifies same-class instance pairs that
fall within the 2 m centroid distance threshold used by the May 2026
METHOD_32 (and would therefore be absorbed by the class-aware Hungarian
merger), and renders the supporting figures.

CPU only; ~30-60 s.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import open3d as o3d

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluate import SCENE_NAMES_SCANNET200
from evaluate.scannet200.scannet_constants import (
    CLASS_LABELS_200,
    HEAD_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    VALID_CLASS_IDS_200,
)

_LABEL_TO_NAME = {int(VALID_CLASS_IDS_200[i]): CLASS_LABELS_200[i] for i in range(len(VALID_CLASS_IDS_200))}


def _bucket_of(name: str) -> str:
    if name in TAIL_CATS_SCANNET_200:
        return "tail"
    if name in COMMON_CATS_SCANNET_200:
        return "common"
    if name in HEAD_CATS_SCANNET_200:
        return "head"
    return "other"


def _instance_centroids(vertices: np.ndarray, gt: np.ndarray) -> dict[int, dict]:
    """Return ``{inst_uid: {"label": class_id, "centroid": (3,), "n_verts": int}}``."""
    insts: dict[int, dict] = {}
    for uid in np.unique(gt[gt > 0]):
        mask = gt == uid
        if mask.sum() < 10:
            continue
        c = vertices[mask].mean(axis=0)
        insts[int(uid)] = {
            "label": int(uid // 1000),
            "centroid": c.astype(np.float32),
            "n_verts": int(mask.sum()),
        }
    return insts


def analyse(scene_root: Path, gt_root: Path, distance_threshold_m: float = 2.0) -> dict:
    # Per-class instance count (across the val set).
    class_instance_counts: Counter = Counter()
    # Same-class within-scene pairs and how many fall under the threshold.
    pairs_total_by_class: Counter = Counter()
    pairs_under_threshold_by_class: Counter = Counter()
    scenes_with_multi_instance: dict[str, list[str]] = defaultdict(list)
    sample_close_pairs: list[dict] = []

    for scene_name in SCENE_NAMES_SCANNET200:
        gt_path = gt_root / f"{scene_name}.txt"
        ply_path = next((scene_root / scene_name).glob("*.ply"), None)
        if not gt_path.exists() or ply_path is None:
            continue
        gt = np.loadtxt(gt_path, dtype=np.int64)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] != gt.shape[0]:
            continue  # vertex mismatch (would have been flagged by Option B)
        insts = _instance_centroids(pts, gt)

        # Per-class instance count this scene.
        by_label: dict[int, list[int]] = defaultdict(list)
        for uid, info in insts.items():
            by_label[info["label"]].append(uid)
        for label, uids in by_label.items():
            class_instance_counts[label] += len(uids)
            if len(uids) >= 2:
                cname = _LABEL_TO_NAME.get(label, "?")
                scenes_with_multi_instance[cname].append(scene_name)
                # Pairwise distance check.
                for a, b in combinations(uids, 2):
                    d = float(np.linalg.norm(insts[a]["centroid"] - insts[b]["centroid"]))
                    pairs_total_by_class[label] += 1
                    if d <= distance_threshold_m:
                        pairs_under_threshold_by_class[label] += 1
                        if len(sample_close_pairs) < 30:
                            sample_close_pairs.append(
                                {
                                    "scene": scene_name,
                                    "class": cname,
                                    "bucket": _bucket_of(cname),
                                    "n_instances_in_scene": len(uids),
                                    "pair_distance_m": d,
                                }
                            )

    # Aggregate stats.
    classes_with_any_pair = sorted(
        ((label, pairs_total_by_class[label], pairs_under_threshold_by_class.get(label, 0))
         for label in pairs_total_by_class),
        key=lambda t: -t[1],
    )
    rows = []
    for label, n_pairs, n_under in classes_with_any_pair:
        cname = _LABEL_TO_NAME.get(label, "?")
        rows.append(
            {
                "class": cname,
                "bucket": _bucket_of(cname),
                "total_within_scene_pairs": n_pairs,
                "pairs_within_threshold": n_under,
                "absorption_ratio": float(n_under) / n_pairs if n_pairs > 0 else 0.0,
            }
        )

    summary = {
        "distance_threshold_m": distance_threshold_m,
        "n_classes_with_multi_instance": sum(1 for r in rows if r["total_within_scene_pairs"] > 0),
        "n_pairs_total": int(sum(r["total_within_scene_pairs"] for r in rows)),
        "n_pairs_within_threshold": int(sum(r["pairs_within_threshold"] for r in rows)),
        "absorption_ratio_overall": (
            float(sum(r["pairs_within_threshold"] for r in rows))
            / max(sum(r["total_within_scene_pairs"] for r in rows), 1)
        ),
        "per_class_rows": rows,
        "sample_close_pairs": sample_close_pairs,
        "scenes_with_multi_instance_per_class_top10": {
            cname: sorted(set(sc_list))[:10]
            for cname, sc_list in sorted(scenes_with_multi_instance.items(), key=lambda kv: -len(kv[1]))[:10]
        },
    }
    return summary


def render_figures(summary: dict, out_dir: Path) -> dict[str, str]:
    rows = summary["per_class_rows"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, str] = {}

    # Top-25 multi-instance classes — total pairs vs absorbed pairs.
    rows_sorted = sorted(rows, key=lambda r: -r["total_within_scene_pairs"])[:25]
    labels = [r["class"] for r in rows_sorted]
    total = [r["total_within_scene_pairs"] for r in rows_sorted]
    absorbed = [r["pairs_within_threshold"] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, total, width=0.36, label="all within-scene pairs", color="steelblue")
    ax.bar(x + 0.18, absorbed, width=0.36, label="<=2 m (M32 absorbs)", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=80, fontsize=8)
    ax.set_ylabel("pairs (across val scenes)")
    ax.set_title("Top-25 classes by multi-instance pair count — absorption under M32 2 m")
    ax.legend()
    plt.tight_layout()
    fpath = fig_dir / "m32_multi_instance_top25.png"
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    figs["multi_instance_top25"] = str(fpath.relative_to(out_dir))

    # Absorption-ratio histogram (per class).
    ratios = [r["absorption_ratio"] for r in rows if r["total_within_scene_pairs"] > 0]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(ratios, bins=20, color="darkorange", edgecolor="none")
    ax.axvline(np.mean(ratios), color="black", linestyle="--",
               label=f"mean={np.mean(ratios):.2f}")
    ax.set_xlabel("absorption ratio per class")
    ax.set_ylabel("class count")
    ax.set_title("M32 absorption ratio distribution")
    ax.legend()
    plt.tight_layout()
    fpath = fig_dir / "m32_absorption_ratio_distribution.png"
    fig.savefig(fpath, dpi=120)
    plt.close(fig)
    figs["absorption_ratio_distribution"] = str(fpath.relative_to(out_dir))

    return figs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root", default="data/scannet200", type=str
    )
    parser.add_argument(
        "--gt-root", default="data/scannet200/ground_truth", type=str
    )
    parser.add_argument(
        "--output", default="results/2026-05-13_step3_m22_m32_failures", type=str
    )
    parser.add_argument("--distance-threshold-m", type=float, default=2.0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Step 3 — M32 multi-instance absorption analysis ===")
    summary = analyse(
        Path(args.scene_root), Path(args.gt_root),
        distance_threshold_m=args.distance_threshold_m,
    )
    print(f"  classes with multi-instance pairs: {summary['n_classes_with_multi_instance']}")
    print(f"  total within-scene pairs         : {summary['n_pairs_total']}")
    print(f"  pairs within 2 m threshold        : {summary['n_pairs_within_threshold']}")
    print(f"  overall absorption ratio          : {summary['absorption_ratio_overall']:.3f}")

    figs = render_figures(summary, out)
    (out / "m32_multi_instance.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  figures: {figs}")
    print(f"  wrote {out / 'm32_multi_instance.json'}")


if __name__ == "__main__":
    main()
