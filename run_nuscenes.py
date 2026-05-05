"""Smoke test entry point for the nuScenes dataloader.

Loads ONE sample from the configured split and prints types/shapes of
each key. Does NOT invoke any OpenYOLO3D inference. Pipeline integration
is a future task.
"""

import argparse

from dataloaders.nuscenes_loader import NuScenesLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_baseline.yaml")
    args = parser.parse_args()

    loader = NuScenesLoader(config_path=args.config)
    print(f"NuScenesLoader initialized: {len(loader)} samples (version={loader.version})")

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
