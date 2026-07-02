"""Verify the nuScenes metadata tables are present and loadable on this server.

The shards carry only pixels/lidar + a thin meta.json; the 13 nuScenes table
JSONs must be copied here separately. Run this after rsync'ing v1.0-trainval/
to confirm the 'local metadata' dependency is satisfied before the smoke test.

  python verify_metadata.py --dataroot data/nuscenes --version v1.0-trainval
"""

import argparse
import os.path as osp

# nuScenes v1.0 schema: all tables the API loads at construction time.
REQUIRED_TABLES = [
    "attribute", "calibrated_sensor", "category", "ego_pose", "instance",
    "log", "map", "sample", "sample_annotation", "sample_data", "scene",
    "sensor", "visibility",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    args = ap.parse_args()

    meta_dir = osp.join(args.dataroot, args.version)
    print(f"checking {meta_dir}/ ...")
    missing = [t for t in REQUIRED_TABLES if not osp.isfile(osp.join(meta_dir, f"{t}.json"))]
    if missing:
        print(f"  MISSING tables: {', '.join(t + '.json' for t in missing)}")
        print("  -> copy them from the machine holding full nuScenes (see rsync cmd).")
        raise SystemExit(1)
    print(f"  all {len(REQUIRED_TABLES)} table files present.")

    # actually load via the API (catches truncated/corrupt copies, not just presence)
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"  NuScenes API loaded OK: {len(nusc.sample)} samples, {len(nusc.scene)} scenes.")
    print("metadata dependency satisfied — safe to run run_nuscenes_stream.py / run_stream_diff.py")


if __name__ == "__main__":
    main()
