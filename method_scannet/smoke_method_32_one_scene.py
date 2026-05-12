"""Single-scene smoke test for METHOD_32 (and Phase 2)."""
from __future__ import annotations

import json
import os
import sys


SCENE = "scene0011_00"
SCENE_PATH = "data/scannet200/scene0011_00"
PROCESSED = "data/scannet200/scene0011_00/0011_00.npy"
MASKS_DIR = "output/scannet200/scannet200_masks"
CONFIG = "pretrained/config_scannet200.yaml"


def _run(install_fn, uninstall_fn, label: str):
    from utils import OpenYolo3D

    install_fn()
    o = OpenYolo3D(CONFIG)
    pred = o.predict(
        path_2_scene_data=SCENE_PATH,
        depth_scale=1000.0,
        datatype="mesh",
        processed_scene=PROCESSED,
        path_to_3d_masks=MASKS_DIR,
        is_gt=False,
    )
    masks, classes, scores = pred[SCENE]
    K = int(classes.shape[0])
    out = {
        "label": label,
        "K": K,
        "min_class": int(classes.min().item()) if K > 0 else None,
        "max_class": int(classes.max().item()) if K > 0 else None,
        "min_score": float(scores.min().item()) if K > 0 else None,
        "max_score": float(scores.max().item()) if K > 0 else None,
        "unique_classes": int(classes.unique().shape[0]) if K > 0 else 0,
    }
    if K == 0:
        out["sanity"] = "FAIL: K=0"
    else:
        out["sanity"] = "OK"
    print(f"[{label}] {json.dumps(out)}", flush=True)
    uninstall_fn()
    return out


def main() -> int:
    from method_scannet.hooks import (
        install_method_32_only,
        uninstall_method_32_only,
        install_phase2,
        uninstall_phase2,
    )

    results = {}
    results["method_32_only"] = _run(
        install_method_32_only, uninstall_method_32_only, "method_32_only"
    )
    results["phase2"] = _run(install_phase2, uninstall_phase2, "phase2")

    out_path = os.environ.get("SMOKE_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    all_ok = all(r["sanity"] == "OK" for r in results.values())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
