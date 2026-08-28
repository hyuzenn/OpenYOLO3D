"""Dry-run the phase-4 aggregator against the OLD single-sweep cells.

Purpose: prove the collection, comparison and sanity-check code works before
the 10-sweep data exists, so phase 4 is not the place we discover a typo.

It must (a) read all eight published rows, (b) reproduce the published mAP/NDS
to ~1e-3, since those cells ARE what the paper reported, and (c) fire the
"single-sweep artifact" and "NDS < mAP" flags, which are exactly the defects the
old table has. Writes nothing into the real artifact paths.
"""
from __future__ import annotations

import os.path as osp
import sys
from pathlib import Path

ROOT = Path("/home/rintern16/OpenYOLO3D")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit/table1_regen_2026-08-28"))

import aggregate_table1 as A

OLD_GRID = ROOT / "results/2026-07-18_e1_grid_v01/cells"
OLD_E2C = ROOT / "results/2026-07-30_e2c_retro_thrmatch_v01/cells"
OLD_CB = ROOT / "audit/cbmot/cells"

A.ROWS = [
    ("Sensor", "Baseline (unfiltered)",  OLD_GRID / "gamma_ego"),
    ("Sensor", "Control (threshold)",    OLD_E2C / "ctrl_ego"),
    ("Sensor", "Control (accumulation)", OLD_CB / "cbmot_retro_ego_N3_parallel_addition_noise0.05"),
    ("Sensor", "Confirmation (N=3)",     OLD_E2C / "retro_ego"),
    ("World",  "Baseline (unfiltered)",  OLD_GRID / "gamma_global"),
    ("World",  "Control (threshold)",    OLD_E2C / "ctrl_global"),
    ("World",  "Control (accumulation)", OLD_CB / "cbmot_retro_global_N3_parallel_addition_noise0.05"),
    ("World",  "Confirmation (N=3)",     OLD_E2C / "retro_global"),
]

rows = A.collect()
checks = A.sanity(rows)

print("\n=== smoke: reproduce published mAP/NDS from the old cells ===")
bad = 0
for r in rows:
    pub = A.PUBLISHED.get((r["frame"], r["arm"]), {})
    for k in ("mAP", "NDS"):
        o, n = pub.get(k), r.get(k)
        if o is None or n is None:
            continue
        ok = abs(n - o) < 1.5e-3
        print(f"  [{'OK ' if ok else 'BAD'}] {r['frame']:6s} {r['arm']:22s} "
              f"{k}: published {o:.4f} vs cell {n:.6f}")
        bad += not ok

md = A.render(rows, checks)
print(f"\n=== rendered report: {len(md.splitlines())} lines, "
      f"{len(A.problems)} flags ===")
print("\n".join(md.splitlines()[:18]))

print("\n=== smoke expectations ===")
n_ok = sum(1 for r in rows if r.get("status") == "OK")
print(f"  rows collected OK      : {n_ok}/8")
single_flag = any("single-sweep" in p for p in A.problems)
nds_flag = any("NDS" in p and "< mAP" in p for p in A.problems)
print(f"  single-sweep flag fired: {single_flag}")
print(f"  NDS<mAP flag fired     : {nds_flag}")
fail = (bad > 0) or n_ok != 8 or not single_flag or not nds_flag
print("\n=== SMOKE " + ("FAIL" if fail else "PASS") + " ===")
sys.exit(1 if fail else 0)
