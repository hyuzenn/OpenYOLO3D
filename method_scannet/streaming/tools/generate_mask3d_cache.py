"""Generate per-scene Mask3D cache for Task 1.2c Option G.

Each output is a torch ``.pt`` file containing the same tuple that
``OpenYolo3D.predict`` would compute internally:

    (masks: bool[V, K_post_filter], scores: float[K_post_filter])

This is the format ``OpenYolo3D.predict(path_to_3d_masks=cache_dir)``
expects (utils/__init__.py:119-120). The streaming wrapper picks up the
same cache via ``setup_scene(mask3d_cache_path=cache_dir/<scene>.pt)``.

GPU required. ~10-15s per scene * 312 = ~1-1.5h.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluate import SCENE_NAMES_SCANNET200
from utils import OpenYolo3D, apply_nms
from utils.utils_2d import load_yaml


def generate_one(oy3d: OpenYolo3D, scene_name: str, cache_dir: Path, cfg: dict) -> dict:
    out_path = cache_dir / f"{scene_name}.pt"
    if out_path.exists():
        return {"scene": scene_name, "status": "exists"}

    scene_dir = Path("data/scannet200") / scene_name
    scene_id = scene_name.replace("scene", "")
    processed_npy = scene_dir / f"{scene_id}.npy"
    mesh_input = str(processed_npy) if processed_npy.exists() else str(
        next(scene_dir.glob("*.ply"))
    )

    t = time.time()
    masks_raw, scores_raw = oy3d.network_3d.get_class_agnostic_masks(
        mesh_input, datatype="mesh"
    )

    # OpenYolo3D.predict post-filter (utils/__init__.py:116-118).
    th = cfg["network3d"]["th"]
    nms_iou = cfg["network3d"]["nms"]
    keep_score = scores_raw >= th
    if int(keep_score.sum().item()) > 0:
        keep_nms = apply_nms(
            masks_raw[:, keep_score].cuda(),
            scores_raw[keep_score].cuda(),
            nms_iou,
        )
        masks_filtered = (
            masks_raw.cpu().permute(1, 0)[keep_score][keep_nms].permute(1, 0)
        )
        scores_filtered = scores_raw.cpu()[keep_score][keep_nms]
    else:
        masks_filtered = masks_raw.cpu()
        scores_filtered = scores_raw.cpu()

    torch.save((masks_filtered, scores_filtered), str(out_path))
    return {
        "scene": scene_name,
        "status": "written",
        "K": int(masks_filtered.shape[1]),
        "V": int(masks_filtered.shape[0]),
        "wall_seconds": time.time() - t,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        default="results/2026-05-13_mask3d_cache",
        type=str,
    )
    parser.add_argument(
        "--config",
        default="pretrained/config_scannet200.yaml",
        type=str,
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)

    print(f"=== Task 1.2c Option G — Mask3D cache generation ===")
    print(f"cache  : {cache_dir}")
    print(f"config : {args.config}")
    print(f"scenes : {len(SCENE_NAMES_SCANNET200)}")
    print()

    t_total = time.time()
    print("Constructing OpenYolo3D ...", flush=True)
    t0 = time.time()
    oy3d = OpenYolo3D(args.config)
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    written = 0
    skipped = 0
    failed = 0
    for i, scene_name in enumerate(SCENE_NAMES_SCANNET200):
        try:
            rec = generate_one(oy3d, scene_name, cache_dir, cfg)
            if rec["status"] == "exists":
                skipped += 1
            else:
                written += 1
        except Exception as exc:
            print(f"  [{i + 1}/{len(SCENE_NAMES_SCANNET200)}] {scene_name} FAILED: {exc!r}",
                  flush=True)
            failed += 1
            continue

        if (i + 1) % 20 == 0 or i == len(SCENE_NAMES_SCANNET200) - 1:
            elapsed = time.time() - t_total
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(SCENE_NAMES_SCANNET200) - i - 1) / max(rate, 1e-6)
            print(
                f"  [{i + 1}/{len(SCENE_NAMES_SCANNET200)}] "
                f"written={written} skipped={skipped} failed={failed} "
                f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
                flush=True,
            )

    print(
        f"\n=== Done: written={written} skipped={skipped} failed={failed} "
        f"total_wall={(time.time() - t_total)/60:.1f}min ==="
    )


if __name__ == "__main__":
    main()
