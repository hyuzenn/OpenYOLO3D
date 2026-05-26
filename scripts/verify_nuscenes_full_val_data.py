"""Pre-flight data integrity check for the nuScenes 150-scene Full-Val temporal run.

Walks every val scene end-to-end with the nuScenes devkit and verifies that
every file the native_evaluator (detguided source, single-sweep forced) will
actually open exists on disk:

  - samples/LIDAR_TOP/*.pcd.bin            (keyframe lidar — single sweep)
  - samples/CAM_FRONT*/CAM_BACK*/*.jpg     (YOLO-World input for detguided)
  - v1.0-trainval/{scene,sample,...}.json  (devkit metadata)

Optional sweeps coverage report (informational only; multi_sweep=False in the
evaluator so sweep files are not required, but we report coverage anyway so
the user knows the Full-Sweep blobs landed cleanly).

Writes a JSON report next to this script and exits non-zero on any missing
keyframe file (a true blocker).
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter
from pathlib import Path


CAMS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/rintern16/OpenYOLO3D/data/nuscenes")
    ap.add_argument("--version",  default="v1.0-trainval")
    ap.add_argument("--report",   default="/home/rintern16/OpenYOLO3D/results/"
                                          "2026-05-26_nusc_full150_dataverify_v01/report.json")
    args = ap.parse_args()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    from nuscenes import NuScenes
    from nuscenes.utils.splits import val as VAL_SCENES

    print(f"[verify] dataroot={args.dataroot} version={args.version}", flush=True)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"[verify] scenes={len(nusc.scene)} samples={len(nusc.sample)}", flush=True)

    name2scene = {s["name"]: s for s in nusc.scene}
    missing_scenes = [n for n in VAL_SCENES if n not in name2scene]
    val_tokens = [name2scene[n]["token"] for n in VAL_SCENES if n in name2scene]
    print(f"[verify] val scenes resolved: {len(val_tokens)}/{len(VAL_SCENES)}"
          f"  (missing-scene-names={missing_scenes})", flush=True)

    per_scene = []
    missing_files: list[str] = []
    sweep_total = 0
    sweep_present = 0
    n_keyframe_total = 0

    for sc_token in val_tokens:
        sc = nusc.get("scene", sc_token)
        scene_name = sc["name"]
        scene_missing: list[str] = []
        scene_sweep_total = 0
        scene_sweep_present = 0
        scene_keyframes = 0

        cur = sc["first_sample_token"]
        while cur:
            sample = nusc.get("sample", cur)
            scene_keyframes += 1

            # Keyframe LIDAR (single sweep — what the run actually reads).
            lidar_sd_tok = sample["data"]["LIDAR_TOP"]
            lidar_sd = nusc.get("sample_data", lidar_sd_tok)
            lp = osp.join(args.dataroot, lidar_sd["filename"])
            if not osp.exists(lp):
                scene_missing.append(lp)

            # Keyframe cameras (detguided needs all 6).
            for cam in CAMS:
                cam_sd = nusc.get("sample_data", sample["data"][cam])
                cp = osp.join(args.dataroot, cam_sd["filename"])
                if not osp.exists(cp):
                    scene_missing.append(cp)

            # Walk the non-keyframe LIDAR_TOP sweeps under this sample's lidar
            # chain (informational — multi_sweep=False, so no blocker).
            sd_tok = lidar_sd["next"]
            while sd_tok:
                sd = nusc.get("sample_data", sd_tok)
                if sd["is_key_frame"]:
                    break
                scene_sweep_total += 1
                fp = osp.join(args.dataroot, sd["filename"])
                if osp.exists(fp):
                    scene_sweep_present += 1
                sd_tok = sd["next"]

            cur = sample["next"]

        n_keyframe_total += scene_keyframes
        sweep_total += scene_sweep_total
        sweep_present += scene_sweep_present
        missing_files.extend(scene_missing)
        per_scene.append({
            "scene": scene_name,
            "token": sc_token,
            "keyframes": scene_keyframes,
            "sweeps_referenced": scene_sweep_total,
            "sweeps_present": scene_sweep_present,
            "n_missing_keyframe_files": len(scene_missing),
        })

    keyframe_ok = (len(missing_files) == 0)
    sweep_cov = (sweep_present / sweep_total) if sweep_total else 0.0

    report = {
        "dataroot": args.dataroot,
        "version": args.version,
        "n_val_scenes_resolved": len(val_tokens),
        "n_val_scenes_missing_from_db": len(missing_scenes),
        "missing_scene_names": missing_scenes,
        "n_keyframes_total": n_keyframe_total,
        "n_missing_keyframe_files_total": len(missing_files),
        "missing_keyframe_examples": missing_files[:10],
        "sweep_refs_total": sweep_total,
        "sweep_files_present": sweep_present,
        "sweep_coverage_ratio": round(sweep_cov, 4),
        "per_scene": per_scene,
        "blocker": not keyframe_ok,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"[verify] wrote {args.report}", flush=True)

    # Stdout summary.
    print("---- SUMMARY ----")
    print(f"  val scenes:           {len(val_tokens)}/{len(VAL_SCENES)}")
    print(f"  total keyframes:      {n_keyframe_total}")
    print(f"  missing key files:    {len(missing_files)}  (blocker={not keyframe_ok})")
    print(f"  sweep refs / present: {sweep_total} / {sweep_present}  "
          f"({sweep_cov*100:.2f}%)  [informational]")
    if not keyframe_ok:
        sample = Counter(p.rsplit("/", 2)[1] for p in missing_files).most_common(5)
        print(f"  missing-by-channel (top5): {sample}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
