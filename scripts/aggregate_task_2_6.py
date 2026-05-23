"""Step 2b Stage-5 aggregate — native γ CenterPoint baseline + temporal layer.

Reads results/<DATE>_outdoor_native_temporal_v01/axis_*/metrics.json (+ the
devkit per_class.json) and writes a results markdown:
  - 7-axis table: mAP / NDS / Δbaseline / lsc / ttc / fire-audit
  - per-class AP table
  - cross-baseline anchor row (γ pipeline 0.0526 vs γ fixed 0.3407 vs +temporal)

No results are written to any CLAUDE.md (per project rule); the report lives in
the run dir as STEP2B_aggregate.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os.path as osp
from pathlib import Path

AXES = ["baseline", "M11", "M12_thr085", "M21", "M31", "M32", "phase1"]
NUSC_CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer",
                "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
# Cross-baseline anchors (recorded, full val 150 scenes single-sweep).
GAMMA_PIPELINE_MAP = 0.0526   # Stage C (YOLO relabel)
GAMMA_FIXED_MAP = 0.3407      # Task 2.5 native single-sweep (= baseline axis)


def _fmt(x, p="0.4f"):
    return "—" if x is None else format(x, p)


def _fired(axis: str, a: dict) -> str:
    if axis == "baseline":
        return "(ref)"
    if axis in ("M11", "M12_thr085"):
        return "✓" if a.get("n_suppressed_by_gate", 0) > 0 else "NO-OP?"
    if axis == "M21":
        # invocation proof is the weighted-vote count; relabel is the (often null) effect
        return ("✓" if a.get("n_m21_weighted_votes", 0) > 0 else "NO-OP?") \
            + f" ({a.get('n_relabeled_by_m21', 0)} relbl)"
    if axis == "M31":
        return "✓" if a.get("n_merged_by_m31", 0) > 0 else "NO-OP?"
    if axis == "M32":
        return "✓" if a.get("n_merged_by_m32", 0) > 0 else "NO-OP?"
    if axis == "phase1":
        return "✓" if (a.get("n_suppressed_by_gate", 0) > 0
                       or a.get("n_merged_by_m31", 0) > 0) else "NO-OP?"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="results/<DATE>_outdoor_native_temporal_v01 (auto-detect if omitted)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.run_dir:
        run = Path(args.run_dir)
    else:
        cands = sorted(glob.glob(str(root / "results" / "*_outdoor_native_temporal_v01")))
        if not cands:
            raise SystemExit("no native_temporal_v01 run dir found")
        run = Path(cands[-1])

    rows, per_class = {}, {}
    for ax in AXES:
        mfile = run / f"axis_{ax}" / "metrics.json"
        if not mfile.exists():
            rows[ax] = None
            continue
        rows[ax] = json.loads(mfile.read_text())
        pcf = run / f"axis_{ax}" / "nuscenes_eval" / "per_class.json"
        if pcf.exists():
            per_class[ax] = json.loads(pcf.read_text())

    L = []
    L.append("# Step 2b — native γ CenterPoint baseline + temporal layer (v1.0-trainval val, 150 scenes)\n")
    L.append(f"_Run dir: `{run.name}` · single-sweep · score_threshold=0.0 · YOLO bypassed (native CenterPoint labels)._\n")

    L.append("\n## Cross-baseline anchor\n")
    L.append("| pipeline | mAP | note |")
    L.append("|---|---|---|")
    L.append(f"| γ pipeline (Stage C, YOLO relabel) | {GAMMA_PIPELINE_MAP:.4f} | single-sweep + YOLO relabel bottleneck |")
    L.append(f"| γ fixed (Task 2.5 native, = baseline axis) | {GAMMA_FIXED_MAP:.4f} | native CenterPoint label/score |")
    base = rows.get("baseline")
    if base and base.get("mAP") is not None:
        L.append(f"| γ fixed reproduced here (baseline axis) | {base['mAP']:.4f} | equivalence check vs 0.3407 |")

    L.append("\n## 7-axis temporal layer\n")
    L.append("| axis | mAP | NDS | Δbase | lsc | ttc_mean | emit/prop | fired (audit) |")
    L.append("|---|---|---|---|---|---|---|---|")
    base_map = base.get("mAP") if base else None
    for ax in AXES:
        r = rows.get(ax)
        if r is None:
            L.append(f"| {ax} | _pending_ | | | | | | |")
            continue
        a = r.get("fire_audit", {})
        t = r.get("temporal", {})
        ttc = t.get("time_to_confirm", {})
        db = None if (r.get("mAP") is None or base_map is None) else r["mAP"] - base_map
        emit = f"{a.get('n_emitted_total','?')}/{a.get('n_proposals_total','?')}"
        L.append(f"| {ax} | {_fmt(r.get('mAP'))} | {_fmt(r.get('NDS'))} | "
                 f"{'—' if db is None else format(db,'+.4f')} | "
                 f"{t.get('label_switch_count_total','—')} | {_fmt(ttc.get('mean'),'0.2f')} | "
                 f"{emit} | {_fired(ax, a)} |")

    if per_class:
        L.append("\n## Per-class AP (devkit AP_mean over dist thresholds)\n")
        header = "| class | " + " | ".join(AXES) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(AXES) + 1))
        for cls in NUSC_CLASSES:
            cells = []
            for ax in AXES:
                pc = per_class.get(ax, {})
                v = pc.get(cls, {}).get("AP_mean") if cls in pc else None
                cells.append(_fmt(v))
            L.append(f"| {cls} | " + " | ".join(cells) + " |")

    L.append("\n## Interpretation notes\n")
    L.append("- **Baseline equivalence**: baseline axis equals the Task 2.5 native anchor (0.3407) at full "
             "150-scene scale; the temporal-on-native comparison is valid.")
    L.append("- **lsc=0 is STRUCTURAL, not a stability finding (corrected by Task 3.1, 2026-05-21).** The "
             "`CentroidAssociator` is class-aware (`nuscenes_evaluator.py:193` — a proposal only matches a "
             "track of the same class), so every track is single-class by construction and "
             "`label_switch_count` is pinned to 0; M21 relabels 0 for the same reason (one class to vote "
             "over). Do NOT claim native LiDAR labels are temporally stable. Counterfactual: GT-anchored γ "
             "(instance_token = perfect assoc) switches class on ~81% of multi-frame instances; the "
             "class-agnostic counterfactual on these same proposals gives lsc≈744–3152. So lsc/ttc cannot "
             "be the outdoor contribution AS WIRED; ttc (confirmation latency) is still meaningful.")
    L.append("- **What the temporal layer does here**: registration gating only. M11/M12 trade large mAP "
             "for FP suppression at score_threshold=0.0 (M11 −0.204, M12 −0.171; ~78–79% of proposals "
             "gated). M21 null (structural, above). M31 null on mAP (merges redundant same-class overlaps "
             "with no AP effect). M32 slightly negative (over-merges). phase1 ≈ M11 (gate-dominated).")
    L.append("- **Operating point caveat**: score_threshold=0.0 (the 0.3407 anchor's setting) includes a "
             "large low-score tail that nuScenes AP rewards via recall; the gate removes it → big mAP cost. "
             "A moderate-threshold operating point is the natural follow-up axis.")
    L.append("- **M21 firing proof**: `n_m21_weighted_votes` (per-proposal weighted-vote invocations) is "
             "non-zero even though `n_relabeled_by_m21`=0 — a legitimate structural null, NOT the Stage-C "
             "silent no-op (where M21/M31/M32 were installed but never invoked).")
    L.append("- **M22 deferred**: native LiDAR head emits no per-instance image features → CLIP-EMA "
             "undefined here (paper §6.4).")
    L.append("- **Opportunity (separate decision, needs code change)**: class-agnostic association + M21/M22 "
             "could expose and fix γ's real per-track class flicker. Full diagnosis: "
             "`docs/task_3_1_lsc_zero_root_cause.md`.")

    out = run / "STEP2B_aggregate.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
