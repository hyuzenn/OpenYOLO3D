"""Regression check: instance-level (W1-style) metrics must be identical
between v1 and v2.

Stage 4 acceptance criterion #5. v2 only re-runs nuScenes-devkit
accumulate / calc_ap. The instance-level loader (instance_metrics.py)
and predictions are unchanged, so M_rate, L_rate, D_rate, miss_rate and
the per-distance breakdown must match v1 exactly.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    with open("results/diagnosis_beta_baseline/aggregate.json") as f:
        v1 = json.load(f)["instance_level"]
    with open("results/diagnosis_beta_baseline_v2/aggregate.json") as f:
        v2 = json.load(f)["instance_level"]

    failures = []

    def cmp_field(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in set(a) | set(b):
                cmp_field(path + [k], a.get(k), b.get(k))
            return
        if isinstance(a, float) and isinstance(b, float):
            ok = abs(a - b) < 1e-12
        else:
            ok = a == b
        if not ok:
            failures.append((".".join(map(str, path)), a, b))

    cmp_field([], v1, v2)

    if failures:
        print("REGRESSION_FAIL")
        for path, a, b in failures:
            print(f"  {path}: v1={a} v2={b}")
        return 1
    print(f"REGRESSION_OK — instance-level metrics identical (n_GT={v1['n_gt_total']}, M={v1['case_counts']['M']}, miss={v1['case_counts']['miss']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
