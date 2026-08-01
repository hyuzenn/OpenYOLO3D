"""E2b: build a track-count-matched control cell.

Pre-registration: experiments/preregistration_E2b_2026-07-31.md.
From the ungated baseline tracks, keep exactly K_s track IDs per scene, where
K_s = number of distinct track IDs the retro-gate (E2c cell) emits in scene s.
Selection is deliberately non-temporal: top-K_s by mean detection score
(--mode topk, decision-bearing) or uniform random (--mode random --seed S,
sanity arms). All boxes of a kept track are emitted; nothing else is filtered.

Usage: python scripts/e2b_trackmatch.py --retro-cell <e2c retro cell> \
           --base-cell <grid .../axis_baseline> --out <ctrl cell> --mode topk
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def track_ids_per_scene(tj: dict) -> dict:
    scene_of = tj["sample_scene_idx"]
    ids: dict = defaultdict(set)
    for tok, boxes in tj["pred"].items():
        s = scene_of[tok]
        for b in boxes:
            ids[s].add(b["tracking_id"])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retro-cell", required=True, help="E2c retro cell dir (axis_*/tracks.json)")
    ap.add_argument("--base-cell", required=True, help="baseline axis dir containing tracks.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["topk", "random"], default="topk")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    retro_tj = json.loads(next(iter(sorted(Path(a.retro_cell).glob("axis_*/tracks.json")))).read_text())
    base_tj = json.loads((Path(a.base_cell) / "tracks.json").read_text())

    K = {s: len(v) for s, v in track_ids_per_scene(retro_tj).items()}

    scene_of = base_tj["sample_scene_idx"]
    # scene -> tid -> [score_sum, n_boxes]
    stat: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for tok, boxes in base_tj["pred"].items():
        s = scene_of[tok]
        for b in boxes:
            e = stat[s][b["tracking_id"]]
            e[0] += b["detection_score"]
            e[1] += 1

    rng = np.random.default_rng(a.seed)
    keep: dict = {}
    for s in stat:
        k = K.get(s, 0)
        cand = stat[s]
        # prereg §4: gate tracks are a subset of baseline tracks, so this must hold
        assert len(cand) >= k, f"scene {s}: {len(cand)} baseline tracks < K={k} — abort per prereg"
        if a.mode == "topk":
            order = sorted(cand, key=lambda t: (-cand[t][0] / cand[t][1], -cand[t][1], t))
            keep[s] = set(order[:k])
        else:
            keep[s] = set(rng.choice(sorted(cand), size=k, replace=False).tolist())

    pred = {tok: [b for b in boxes if b["tracking_id"] in keep.get(scene_of[tok], set())]
            for tok, boxes in base_tj["pred"].items()}

    out = Path(a.out) / "axis_baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracks.json").write_text(json.dumps(
        {"sample_scene_idx": base_tj["sample_scene_idx"], "pred": pred, "gt": base_tj["gt"]}))

    n_boxes = sum(len(v) for v in pred.values())
    kept = {s: len(v) for s, v in keep.items()}
    match_ok = all(kept.get(s, 0) == K.get(s, 0) for s in set(K) | set(kept))
    (out / "match_stats.json").write_text(json.dumps(
        {"mode": a.mode, "seed": a.seed, "n_pred_boxes_total": n_boxes,
         "n_tracks_total": sum(kept.values()), "n_tracks_target": sum(K.values()),
         "per_scene_K": {str(s): K.get(s, 0) for s in sorted(K)},
         "track_match_exact": match_ok}, indent=2))
    print(f"[e2b] {a.mode} seed={a.seed}: {sum(kept.values())} tracks "
          f"(target {sum(K.values())}, exact={match_ok}), {n_boxes} boxes -> {out}")
    assert match_ok, "track-count match not exact — abort per prereg"


if __name__ == "__main__":
    main()
