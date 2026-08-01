"""E2: pick the baseline score threshold that matches a target box count.

The baseline axis emits every cached proposal (score unchanged, see
nuscenes_native_evaluator._detection_box_dict(score=native_score)), and
--proposal-score-threshold filters on read with `score >= t`. So count(t) is
exactly the number of emitted baseline boxes with score >= t: the matched
threshold is the K-th largest emitted score, and a binary search would
converge to the same value. Ties are resolved upward so count(t) <= K.

Usage: python scripts/e2_thrmatch_pick.py --tracks <baseline tracks.json>
                                          --target 503619 --out t.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pick_threshold(scores: np.ndarray, target: int) -> tuple[float, int]:
    """Smallest t with |{s >= t}| <= target, using only observed score values."""
    s = np.sort(scores)[::-1]
    if target >= s.size:
        return float(s[-1]), int(s.size)
    t = float(s[target - 1])          # K-th largest
    n = int((scores >= t).sum())      # >= target when ties straddle t
    if n > target:                    # ties: step up to the next distinct value
        higher = s[s > t]
        t = float(higher[-1]) if higher.size else float(s[0]) + 1.0
        n = int((scores >= t).sum())
    return t, n


def _selftest() -> None:
    a = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert pick_threshold(a, 3) == (0.7, 3)
    b = np.array([0.9, 0.5, 0.5, 0.5, 0.4])          # tie block at the cut
    t, n = pick_threshold(b, 3)
    assert n <= 3 and t == 0.9 and n == 1, (t, n)    # never exceeds the target
    assert pick_threshold(a, 99)[1] == 5
    print("[e2-pick] selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="baseline cell tracks.json")
    ap.add_argument("--target", type=int, required=True, help="target box count")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()

    pred = json.loads(Path(args.tracks).read_text())["pred"]
    scores = np.fromiter((b["detection_score"] for bs in pred.values() for b in bs),
                         dtype=np.float64)
    t, n = pick_threshold(scores, args.target)
    rec = {"tracks": args.tracks, "n_baseline_boxes": int(scores.size),
           "target": args.target, "threshold": t, "predicted_count": n,
           "rel_err": (n - args.target) / args.target}
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(f"[e2-pick] target={args.target} t={t:.6f} predicted={n} "
          f"rel_err={rec['rel_err']:+.4%}", flush=True)


if __name__ == "__main__":
    main()
