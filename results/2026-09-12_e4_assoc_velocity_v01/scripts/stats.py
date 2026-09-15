"""Associator sensitivity: confirmation minus its detection-budget-matched control,
under the constant-velocity matcher, in both association frames.

Reuses scripts/e2_thrmatch_report.py unchanged (same combine/bootstrap, SEED 20260718,
10k scene resamples) -- no new statistics. Delta = confirmation - control, i.e. the
same sign convention as Tab. 1, so the canonical CenterPoint reference values are
  sensor +0.0625 [+0.0471,+0.0785]   world +0.0148 [+0.0098,+0.0203]
PASS criterion (fixed before the run): the AssA delta's CI excludes zero in BOTH frames.
"""
import json, sys
from pathlib import Path
import numpy as np
R = Path("/home/rintern16/OpenYOLO3D"); sys.path.insert(0, str(R / "scripts"))
from e2_thrmatch_report import (load_cell, row, scene_arrays, combine, per_scene_metrics,
                                paired_stats, boot_ci_combined, METRICS, SEED, N_BOOT)
OUT = Path(sys.argv[1])
REF = {"ego": {"AssA": 0.0625, "ci": [0.0471, 0.0785]},
       "global": {"AssA": 0.0148, "ci": [0.0098, 0.0203]}}
rep = {"seed": SEED, "n_boot": N_BOOT, "delta": "confirmation - control",
       "canonical_reference_static_associator": REF, "frames": {}}
for frame in ("ego", "global"):
    cells = {"confirmation": OUT / f"cells/retrovel_{frame}",
             "control": OUT / f"cells/ctrlvel_{frame}"}
    cd = {k: load_cell(v) for k, v in cells.items()}
    scenes = sorted(set(cd["confirmation"]["perscene"]["trackeval"])
                    & set(cd["control"]["perscene"]["trackeval"]))
    ps = {k: per_scene_metrics(cd[k]["perscene"], scenes) for k in cd}
    arr = {k: scene_arrays(cd[k]["perscene"], scenes) for k in cd}
    ci = boot_ci_combined(arr["control"], arr["confirmation"], len(scenes))
    f = {"n_scenes": len(scenes),
         "arms": {k: {"boxes": cd[k]["det"]["n_pred_boxes_total"],
                      "n_tracks": cd[k]["e1"]["n_tracks"],
                      **{m: cd[k]["e1"]["gt_based"][m] for m in ("AssA", "HOTA", "IDF1", "DetA")}}
                  for k in cd},
         "paired_stats": {m: paired_stats(ps["control"][m], ps["confirmation"][m]) for m in METRICS},
         "bootstrap_ci_combined_delta": ci}
    f["exact_box_match"] = f["arms"]["confirmation"]["boxes"] == f["arms"]["control"]["boxes"]
    lo, hi = ci["AssA"]
    f["AssA_delta"] = f["arms"]["confirmation"]["AssA"] - f["arms"]["control"]["AssA"]
    f["AssA_ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    rep["frames"][frame] = f
rep["PASS"] = all(rep["frames"][f]["AssA_ci_excludes_zero"] and rep["frames"][f]["AssA_delta"] > 0
                  for f in ("ego", "global"))
(OUT / "result.json").write_text(json.dumps(rep, indent=1))
for frame in ("ego", "global"):
    f = rep["frames"][frame]; a = f["arms"]
    print("%-6s boxes %s/%s exact=%s | AssA conf %.4f ctrl %.4f delta %+.4f CI %s | canonical %+.4f" % (
        frame, a["confirmation"]["boxes"], a["control"]["boxes"], f["exact_box_match"],
        a["confirmation"]["AssA"], a["control"]["AssA"], f["AssA_delta"],
        [round(x, 4) for x in f["bootstrap_ci_combined_delta"]["AssA"]], REF[frame]["AssA"]))
print("PASS:", rep["PASS"])
