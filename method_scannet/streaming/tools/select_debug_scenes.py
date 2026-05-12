"""Pick 1-2 ScanNet200 scenes with high common/tail GT instance content for
Task 1.2c Option E frame-level streaming-vs-offline debugging.

CPU only, ~30s.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate import SCENE_NAMES_SCANNET200
from evaluate.scannet200.scannet_constants import (
    CLASS_LABELS_200,
    HEAD_CATS_SCANNET_200,
    COMMON_CATS_SCANNET_200,
    TAIL_CATS_SCANNET_200,
    VALID_CLASS_IDS_200,
)


# Map class_id (raw ScanNet label_id) → bucket name.
_LABEL_TO_NAME = {int(VALID_CLASS_IDS_200[i]): CLASS_LABELS_200[i] for i in range(len(VALID_CLASS_IDS_200))}
_HEAD = set(HEAD_CATS_SCANNET_200)
_COMMON = set(COMMON_CATS_SCANNET_200)
_TAIL = set(TAIL_CATS_SCANNET_200)


def _bucket_of(label_id: int) -> str | None:
    name = _LABEL_TO_NAME.get(int(label_id))
    if name is None:
        return None
    if name in _TAIL:
        return "tail"
    if name in _COMMON:
        return "common"
    if name in _HEAD:
        return "head"
    return None


def scene_gt_summary(gt_path: Path) -> dict:
    """Per-vertex GT in ScanNet format is ``label_id*1000 + instance_id``.
    Background vertices = 0. Return per-bucket instance counts.
    """
    arr = np.loadtxt(gt_path, dtype=np.int64)
    insts = np.unique(arr[arr > 0])
    bucket_counts: Counter = Counter()
    class_counts: Counter = Counter()
    for inst_uid in insts:
        label_id = int(inst_uid // 1000)
        bucket = _bucket_of(label_id)
        if bucket is None:
            continue
        bucket_counts[bucket] += 1
        class_counts[_LABEL_TO_NAME[label_id]] += 1
    total = sum(bucket_counts.values())
    return {
        "n_instances": int(total),
        "n_head": int(bucket_counts.get("head", 0)),
        "n_common": int(bucket_counts.get("common", 0)),
        "n_tail": int(bucket_counts.get("tail", 0)),
        "fraction_common_or_tail": (
            (bucket_counts.get("common", 0) + bucket_counts.get("tail", 0)) / total
            if total > 0 else 0.0
        ),
        "class_counts": dict(class_counts.most_common(10)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-scene-dir",
        default="results/2026-05-12_scannet_streaming_baseline_v01/per_scene",
        type=str,
    )
    parser.add_argument(
        "--output",
        default="results/2026-05-13_streaming_debug_E",
        type=str,
    )
    parser.add_argument("--n-pick", default=2, type=int)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_scene_dir = Path(args.per_scene_dir)
    gt_root = Path("data/scannet200/ground_truth")

    print(f"=== Scene selection for Task 1.2c Option E ===")
    candidates: list[dict] = []
    for scene_name in SCENE_NAMES_SCANNET200:
        gt_path = gt_root / f"{scene_name}.txt"
        per_scene_path = per_scene_dir / f"{scene_name}.json"
        if not gt_path.exists() or not per_scene_path.exists():
            continue
        gt_summary = scene_gt_summary(gt_path)
        per_scene = json.loads(per_scene_path.read_text())
        candidates.append({
            "scene_name": scene_name,
            **gt_summary,
            "n_frames_streamed": int(per_scene.get("n_frames_streamed", 0)),
            "n_mask3d_instances_after_filter": int(
                per_scene.get("n_mask3d_instances_after_filter", 0)
            ),
            "d3_median": float(
                per_scene.get("d3_visibility", {}).get("median", 0.0)
            ),
            "walltime_seconds": float(per_scene.get("walltime_seconds", 0.0)),
        })

    # Selection criteria:
    #   - at least 3 common/tail instances (so the regression is visible)
    #   - n_frames_streamed in [40, 120] (debug speed)
    #   - K_mask3d after filter >= 8 (enough proposals to fight over)
    filtered = [
        c for c in candidates
        if (c["n_common"] + c["n_tail"]) >= 3
        and 40 <= c["n_frames_streamed"] <= 120
        and c["n_mask3d_instances_after_filter"] >= 8
    ]
    # Rank: high common+tail fraction first; tie-break with shorter wall time.
    filtered.sort(
        key=lambda c: (-c["fraction_common_or_tail"], c["walltime_seconds"])
    )

    picked = filtered[: args.n_pick]

    summary = {
        "n_candidates_with_data": len(candidates),
        "n_filtered_by_criteria": len(filtered),
        "picked_scenes": [c["scene_name"] for c in picked],
        "criteria": {
            "min_common_plus_tail": 3,
            "n_frames_range": [40, 120],
            "min_K_mask3d": 8,
        },
        "picked_details": picked,
        "next_5_alternatives": [
            c["scene_name"] for c in filtered[args.n_pick : args.n_pick + 5]
        ],
    }
    (out_dir / "scene_selection.json").write_text(json.dumps(summary, indent=2))

    print(f"  candidates with data         : {len(candidates)}")
    print(f"  satisfying criteria          : {len(filtered)}")
    print(f"  picked (top {args.n_pick})              : {[c['scene_name'] for c in picked]}")
    for c in picked:
        print(
            f"    {c['scene_name']}: "
            f"n_inst={c['n_instances']} (h={c['n_head']}, c={c['n_common']}, t={c['n_tail']}) "
            f"frac_c+t={c['fraction_common_or_tail']:.2f} "
            f"frames={c['n_frames_streamed']} K_m3d={c['n_mask3d_instances_after_filter']} "
            f"wt={c['walltime_seconds']:.1f}s"
        )
    print(f"\nwrote {out_dir / 'scene_selection.json'}")


if __name__ == "__main__":
    main()
