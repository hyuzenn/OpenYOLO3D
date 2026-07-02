"""Smoke test for the hybrid streaming nuScenes loader.

Pulls ONE sample from the remote WebDataset stream, enriches it with local
nuScenes metadata, and prints types/shapes of each key. Mirrors run_nuscenes.py
so the streaming loader's output can be compared field-for-field against the
disk loader. Does NOT invoke OpenYOLO3D inference.

Prereqs:
  - reverse tunnel up and serving shards (see test_wds_loader.py preflight)
  - nuScenes <version>/*.json metadata tables present under dataroot
"""

import argparse

from dataloaders.nuscenes_stream_loader import StreamingNuScenesLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_stream.yaml")
    args = parser.parse_args()

    loader = StreamingNuScenesLoader(config_path=args.config)
    print(f"StreamingNuScenesLoader initialized: {len(loader)} samples "
          f"(version={loader.version}, shards={len(loader.urls)})")

    item = next(iter(loader))

    print(f"\nsample_token : {item['sample_token']}")
    print(f"timestamp    : {item['timestamp']}")
    print(f"point_cloud  : shape={item['point_cloud'].shape} dtype={item['point_cloud'].dtype}")
    print(f"ego_pose     : shape={item['ego_pose'].shape}")
    print(f"gt_boxes     : {len(item['gt_boxes'])} annotations")
    if item["gt_boxes"]:
        print(f"               e.g. {item['gt_boxes'][0]['category']}")

    print("\nper-camera:")
    for cam in sorted(item["images"].keys()):
        img = item["images"][cam]
        print(
            f"  {cam:18s} "
            f"img={img.shape}/{img.dtype}  "
            f"K={item['intrinsics'][cam].shape}  "
            f"T_cam2ego={item['cam_to_ego'][cam].shape}"
        )


if __name__ == "__main__":
    main()
