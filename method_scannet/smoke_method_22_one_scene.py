"""Single-scene smoke test for METHOD_22.

Runs `install_method_22_only` and predicts a single scene to verify the
patched function produces sensible output (non-empty K, classes in
[0..197] for the inference subset, scores in [0, 1]). Should complete in
~3-5 minutes on an A100.

Writes a JSON report to $SMOKE_OUT (or stdout if unset).
"""
from __future__ import annotations

import json
import os
import sys


SCENE = "scene0011_00"
SCENE_PATH = "data/scannet200/scene0011_00"
PROCESSED = "data/scannet200/scene0011_00/0011_00.npy"
MASKS_DIR = "output/scannet200/scannet200_masks"
CONFIG = "pretrained/config_scannet200.yaml"


def main() -> int:
    from method_scannet.hooks import install_method_22_only, uninstall_method_22_only
    from utils import OpenYolo3D

    install_method_22_only()
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
        "scene": SCENE,
        "K": K,
        "n_vertices": int(masks.shape[0]) if masks.numel() else 0,
        "min_class": int(classes.min().item()) if K > 0 else None,
        "max_class": int(classes.max().item()) if K > 0 else None,
        "min_score": float(scores.min().item()) if K > 0 else None,
        "max_score": float(scores.max().item()) if K > 0 else None,
        "mean_score": float(scores.mean().item()) if K > 0 else None,
        "unique_classes": int(classes.unique().shape[0]) if K > 0 else 0,
    }

    # Sanity checks
    if K == 0:
        out["sanity"] = "FAIL: K=0"
    else:
        ok_class_range = 0 <= out["min_class"] <= out["max_class"] <= 197
        ok_score_range = 0.0 <= out["min_score"] <= out["max_score"] <= 1.0
        out["sanity"] = "OK" if (ok_class_range and ok_score_range) else (
            f"FAIL: class_range={ok_class_range}, score_range={ok_score_range}"
        )

    print(json.dumps(out, indent=2), flush=True)

    out_path = os.environ.get("SMOKE_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    uninstall_method_22_only()
    return 0 if out["sanity"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
