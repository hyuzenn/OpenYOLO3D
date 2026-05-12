"""Compare vertex counts of the Mask3D `.npy` input vs the mesh `.ply`
for every ScanNet200 validation scene. Direct test of Task 1.2c
diagnosis hypothesis H1 (Mask3D output vs projection-index misalignment).

CPU only; runs in ~1-2 min over 312 scenes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from evaluate import SCENE_NAMES_SCANNET200


def _scene_npy_path(scene_dir: Path) -> Path:
    """OpenYOLO3D pre-processing convention: <scene_id>.npy where
    scene_id strips the 'scene' prefix (e.g. scene0011_00 → 0011_00).
    """
    scene_id = scene_dir.name.replace("scene", "")
    return scene_dir / f"{scene_id}.npy"


def _scene_ply_path(scene_dir: Path) -> Path:
    plys = list(scene_dir.glob("*.ply"))
    if not plys:
        return scene_dir / "missing.ply"
    return plys[0]


def compare(scene_root: Path, scene_names: list[str]) -> dict:
    stats: dict[str, dict] = {}
    mismatches: list[str] = []
    missing: list[str] = []
    t_start = time.time()

    for i, name in enumerate(scene_names):
        scene_dir = scene_root / name
        npy_p = _scene_npy_path(scene_dir)
        ply_p = _scene_ply_path(scene_dir)

        rec: dict[str, int | bool | str] = {}
        if not npy_p.exists() or not ply_p.exists():
            rec["status"] = "missing_input"
            rec["npy_exists"] = npy_p.exists()
            rec["ply_exists"] = ply_p.exists()
            missing.append(name)
        else:
            try:
                arr = np.load(npy_p, mmap_mode="r")
                n_npy = int(arr.shape[0])
                pcd = o3d.io.read_point_cloud(str(ply_p))
                n_ply = int(len(pcd.points))
                rec["n_npy"] = n_npy
                rec["n_ply"] = n_ply
                rec["match"] = bool(n_npy == n_ply)
                rec["delta"] = int(n_npy - n_ply)
                if not rec["match"]:
                    mismatches.append(name)
            except Exception as exc:
                rec["status"] = f"error: {exc!r}"
                missing.append(name)
        stats[name] = rec

        if (i + 1) % 50 == 0 or i == len(scene_names) - 1:
            elapsed = time.time() - t_start
            print(
                f"  [{i + 1}/{len(scene_names)}] elapsed={elapsed:.1f}s "
                f"mismatches={len(mismatches)} missing={len(missing)}",
                flush=True,
            )

    summary = {
        "total_scenes": len(scene_names),
        "matched_scenes": len(scene_names) - len(mismatches) - len(missing),
        "mismatched_scenes": len(mismatches),
        "missing_input_scenes": len(missing),
        "mismatch_ratio": len(mismatches) / max(len(scene_names), 1),
        "mismatch_list": mismatches,
        "missing_list": missing,
    }
    return {"summary": summary, "per_scene": stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root", default="data/scannet200", type=str
    )
    parser.add_argument(
        "--output",
        default="results/2026-05-13_npy_ply_comparison",
        type=str,
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== Task 1.2c diagnosis — Option B: .npy vs .ply vertex count ===")
    print(f"scene root: {args.scene_root}")
    print(f"output    : {out}")
    print(f"scenes    : {len(SCENE_NAMES_SCANNET200)}")
    print()

    result = compare(Path(args.scene_root), list(SCENE_NAMES_SCANNET200))

    (out / "per_scene.json").write_text(json.dumps(result["per_scene"], indent=2))
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2))

    s = result["summary"]
    print()
    print("=== Result ===")
    print(f"  total           : {s['total_scenes']}")
    print(f"  matched         : {s['matched_scenes']}")
    print(f"  mismatched      : {s['mismatched_scenes']}  ({s['mismatch_ratio']*100:.2f}%)")
    print(f"  missing_input   : {s['missing_input_scenes']}")
    if s["mismatch_list"]:
        print(f"  first 10 mismatch: {s['mismatch_list'][:10]}")
    print(f"\nwrote {out / 'summary.json'}")
    print(f"wrote {out / 'per_scene.json'}")


if __name__ == "__main__":
    main()
