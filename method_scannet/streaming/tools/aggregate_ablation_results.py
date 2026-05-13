"""Build a 12-way streaming-ablation comparison table from per-axis
``summary.json`` files. Mirrors the May-2026 ``results/experiment_tracker.md``
schema so paper writing can drop the table in directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-root",
        required=True,
        type=str,
        help="results/<date>_streaming_ablation_v01/ produced by "
             "method_scannet.streaming.eval_streaming_ablation",
    )
    parser.add_argument(
        "--baseline-ap",
        type=float,
        default=None,
        help="Reference baseline AP (e.g. Task 1.2c Option G streaming AP) "
             "to compute Δ columns. If omitted, the ``axis_baseline`` row is "
             "used as reference automatically.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.ablation_root)
    axes_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("axis_"))
    if not axes_dirs:
        raise SystemExit(f"No axis_* directories in {root}")

    rows: list[dict] = []
    for p in axes_dirs:
        summary_path = p / "summary.json"
        if not summary_path.exists():
            rows.append({"axis": p.name.replace("axis_", ""), "error": "missing summary.json"})
            continue
        rows.append(json.loads(summary_path.read_text()))

    # Find baseline reference.
    baseline_ap = args.baseline_ap
    if baseline_ap is None:
        for r in rows:
            if r.get("axis") == "baseline":
                baseline_ap = r.get("AP")
                break

    lines: list[str] = []
    lines.append("# Streaming ablation summary")
    lines.append("")
    header = (
        "| Axis | method_id | AP | AP_50 | AP_25 | head_AP | common_AP | tail_AP | "
        "Δ AP vs baseline | walltime(s) |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        ap = r.get("AP")
        delta = (ap - baseline_ap) if (ap is not None and baseline_ap is not None) else None
        lines.append(
            f"| {r.get('axis', '?')} | {r.get('method_id', '?')} "
            f"| {ap if ap is None else f'{ap:.4f}'} "
            f"| {r.get('AP_50') if r.get('AP_50') is None else f'{r.get(\"AP_50\"):.4f}'} "
            f"| {r.get('AP_25') if r.get('AP_25') is None else f'{r.get(\"AP_25\"):.4f}'} "
            f"| {r.get('head_AP') if r.get('head_AP') is None else f'{r.get(\"head_AP\"):.4f}'} "
            f"| {r.get('common_AP') if r.get('common_AP') is None else f'{r.get(\"common_AP\"):.4f}'} "
            f"| {r.get('tail_AP') if r.get('tail_AP') is None else f'{r.get(\"tail_AP\"):.4f}'} "
            f"| {'' if delta is None else f'{delta:+.4f}'} "
            f"| {'' if r.get('axis_walltime_seconds') is None else f'{r.get(\"axis_walltime_seconds\"):.0f}'} |"
        )

    table = "\n".join(lines)
    print(table)

    out_path = Path(args.output) if args.output else root / "ablation_table.md"
    out_path.write_text(table + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
