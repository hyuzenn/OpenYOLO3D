"""Bit-identity check: StreamingNuScenesLoader (stream) vs NuScenesLoader (disk).

Pulls samples from the remote stream and, for each one, loads the SAME
sample_token from disk via NuScenesLoader, then compares every field of the
output dict. Reports per-field PASS/FAIL and exits non-zero if anything differs.

This must be run ONCE while the heavy samples/ + sweeps/ blobs are STILL on
disk (the disk loader needs them). After it passes, the blobs can be deleted
and the stream becomes the source of truth.

Usage:
  python run_stream_diff.py --num 1
  python run_stream_diff.py --num 5 \
      --stream-config configs/nuscenes_stream.yaml \
      --disk-config   configs/nuscenes_trainval.yaml
"""

import argparse

import numpy as np

from dataloaders.nuscenes_loader import NuScenesLoader
from dataloaders.nuscenes_stream_loader import StreamingNuScenesLoader


def _cmp_array(name, a, b, results):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        results.append((name, a is b or a == b, "non-array"))
        return
    if a.shape != b.shape or a.dtype != b.dtype:
        results.append((name, False, f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"))
        return
    exact = np.array_equal(a, b)
    if exact:
        results.append((name, True, "exact"))
    else:
        # surface how far off, so a float-precision diff is distinguishable
        # from an actual data mismatch
        maxabs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        close = np.allclose(a, b, rtol=0, atol=0)
        results.append((name, close, f"NOT exact (max|Δ|={maxabs:.3e})"))


def _cmp_gt_boxes(stream_boxes, disk_boxes, results):
    if len(stream_boxes) != len(disk_boxes):
        results.append(("gt_boxes.len", False, f"{len(stream_boxes)} vs {len(disk_boxes)}"))
        return
    results.append(("gt_boxes.len", True, f"{len(disk_boxes)}"))
    all_ok = True
    for i, (sb, db) in enumerate(zip(stream_boxes, disk_boxes)):
        for k in ("category", "instance_token", "num_lidar_pts"):
            if sb[k] != db[k]:
                all_ok = False
                results.append((f"gt_boxes[{i}].{k}", False, f"{sb[k]} vs {db[k]}"))
        for k in ("translation", "size", "rotation"):
            if not np.array_equal(sb[k], db[k]):
                all_ok = False
                results.append((f"gt_boxes[{i}].{k}", False, "array mismatch"))
    results.append(("gt_boxes.contents", all_ok, "all boxes match" if all_ok else "see above"))


def compare(stream_item, disk_item):
    results = []
    # scalars
    results.append(("sample_token", stream_item["sample_token"] == disk_item["sample_token"],
                    stream_item["sample_token"]))
    results.append(("timestamp", stream_item["timestamp"] == disk_item["timestamp"],
                    str(disk_item["timestamp"])))
    # arrays
    _cmp_array("point_cloud", stream_item["point_cloud"], disk_item["point_cloud"], results)
    _cmp_array("ego_pose", stream_item["ego_pose"], disk_item["ego_pose"], results)
    # per-camera dicts
    cams = sorted(disk_item["images"].keys())
    if sorted(stream_item["images"].keys()) != cams:
        results.append(("cameras", False,
                        f"{sorted(stream_item['images'].keys())} vs {cams}"))
    else:
        for cam in cams:
            _cmp_array(f"images[{cam}]", stream_item["images"][cam], disk_item["images"][cam], results)
            _cmp_array(f"intrinsics[{cam}]", stream_item["intrinsics"][cam], disk_item["intrinsics"][cam], results)
            _cmp_array(f"cam_to_ego[{cam}]", stream_item["cam_to_ego"][cam], disk_item["cam_to_ego"][cam], results)
    # gt boxes
    _cmp_gt_boxes(stream_item["gt_boxes"], disk_item["gt_boxes"], results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-config", default="configs/nuscenes_stream.yaml")
    parser.add_argument("--disk-config", default="configs/nuscenes_trainval.yaml")
    parser.add_argument("--num", type=int, default=1, help="number of streamed samples to verify")
    args = parser.parse_args()

    stream_loader = StreamingNuScenesLoader(config_path=args.stream_config)
    disk_loader = NuScenesLoader(config_path=args.disk_config)
    disk_index = {tok: i for i, tok in enumerate(disk_loader.sample_tokens)}

    all_pass = True
    n = 0
    for stream_item in stream_loader:
        token = stream_item["sample_token"]
        if token not in disk_index:
            print(f"[{n}] {token}: SKIP (not in disk loader's sample list)")
            continue
        disk_item = disk_loader[disk_index[token]]

        results = compare(stream_item, disk_item)
        sample_pass = all(ok for _, ok, _ in results)
        all_pass = all_pass and sample_pass
        print(f"\n=== sample {n}  token={token}  -> {'PASS' if sample_pass else 'FAIL'} ===")
        for name, ok, detail in results:
            print(f"  [{'OK ' if ok else 'XX '}] {name:28s} {detail}")

        n += 1
        if n >= args.num:
            break

    print(f"\n{'='*60}\nOVERALL: {'PASS (bit-identical)' if all_pass else 'FAIL'} over {n} sample(s)")
    raise SystemExit(0 if all_pass and n > 0 else 1)


if __name__ == "__main__":
    main()
