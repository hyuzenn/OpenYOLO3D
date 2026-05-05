"""LiDAR → camera projection sanity check.

Projects every LiDAR point in the sample into the chosen camera and
reports counts. Sanity criterion: at least `min_in_bounds` points must
land inside the image. With ~30k LiDAR points and even the narrowest
~70° FoV camera, getting fewer than ~500 in-bounds points means the
transform chain is wrong (typical correct sweeps yield several thousand
per camera).

Why we don't gate on in_bounds/in_front ratio: nuScenes LiDAR is 360°
but each camera only covers a 70° (or 110° for CAM_BACK) horizontal
FoV, so the in-front half-space contains many points outside the FoV.
A 17–30% ratio is the *expected* value for the ~70° cameras and not a
red flag.

The `--n` flag still controls how many points are also drawn as a
random subset for additional reporting (matches the original task spec
"projects 100 LiDAR points").
"""

import argparse
import os

import numpy as np

from .nuscenes_loader import NuScenesLoader


def _project_all(item, cam):
    pc_ego = item["point_cloud"][:, :3]
    K = item["intrinsics"][cam]
    T_cam_to_ego = item["cam_to_ego"][cam]
    H, W = item["images"][cam].shape[:2]

    pts_h = np.concatenate([pc_ego, np.ones((pc_ego.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]

    in_front_mask = pts_cam[:, 2] > 0.1
    pts_cam_front = pts_cam[in_front_mask]
    if pts_cam_front.shape[0] == 0:
        return {"total": int(pc_ego.shape[0]), "in_front": 0, "in_bounds": 0, "image_hw": (H, W)}

    uv = (K @ pts_cam_front.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    in_bounds = int(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).sum())
    return {
        "total": int(pc_ego.shape[0]),
        "in_front": int(pts_cam_front.shape[0]),
        "in_bounds": in_bounds,
        "image_hw": (H, W),
    }


def project_lidar_to_camera(item, cam="CAM_FRONT", n_points=100, seed=0):
    """Original spec: project a random N-point subset and report counts."""
    pc_ego = item["point_cloud"][:, :3]
    K = item["intrinsics"][cam]
    T_cam_to_ego = item["cam_to_ego"][cam]
    H, W = item["images"][cam].shape[:2]

    rng = np.random.default_rng(seed)
    n = min(n_points, pc_ego.shape[0])
    idx = rng.choice(pc_ego.shape[0], size=n, replace=False)
    pts_ego = pc_ego[idx]

    pts_h = np.concatenate([pts_ego, np.ones((n, 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]

    in_front_mask = pts_cam[:, 2] > 0.1
    pts_cam_front = pts_cam[in_front_mask]
    if pts_cam_front.shape[0] == 0:
        return {"sampled": n, "in_front": 0, "in_bounds": 0, "image_hw": (H, W)}

    uv = (K @ pts_cam_front.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    in_bounds = int(((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)).sum())
    return {
        "sampled": n,
        "in_front": int(pts_cam_front.shape[0]),
        "in_bounds": in_bounds,
        "image_hw": (H, W),
    }


def save_overlay(item, cam, output_path):
    """Save a PNG with in-bounds LiDAR points overlaid on the camera image,
    colored by camera-frame depth (close = red, far = blue)."""
    import matplotlib.pyplot as plt

    pc_ego = item["point_cloud"][:, :3]
    K = item["intrinsics"][cam]
    T_cam_to_ego = item["cam_to_ego"][cam]
    img = item["images"][cam]
    H, W = img.shape[:2]

    pts_h = np.concatenate([pc_ego, np.ones((pc_ego.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]

    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    if pts_cam.shape[0] == 0:
        return None

    depth = pts_cam[:, 2].copy()
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[in_bounds]
    depth = depth[in_bounds]
    if uv.shape[0] == 0:
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(img)
    ax.scatter(uv[:, 0], uv[:, 1], c=depth, cmap="jet_r", s=4, alpha=0.8, edgecolors="none")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return output_path, int(uv.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--cam", default="CAM_FRONT")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-in-bounds", type=int, default=500,
                        help="Pass threshold for full-cloud in-bounds count")
    parser.add_argument("--overlay-out", default="results/sanity_overlay.png",
                        help="Path to save LiDAR-on-image overlay PNG (uses --cam)")
    args = parser.parse_args()

    loader = NuScenesLoader(config_path=args.config)
    item = next(iter(loader))

    full = _project_all(item, args.cam)
    subset = project_lidar_to_camera(item, cam=args.cam, n_points=args.n, seed=args.seed)

    H, W = full["image_hw"]
    print(f"sample_token  : {item['sample_token']}")
    print(f"camera        : {args.cam} (image {H}x{W})")
    print()
    print(f"random {args.n}-point subset (matches original spec):")
    print(f"  in front    : {subset['in_front']}  (z>0.1 in cam frame)")
    print(f"  in bounds   : {subset['in_bounds']}  (0 ≤ u<W and 0 ≤ v<H)")
    print()
    print(f"full sweep ({full['total']} points):")
    print(f"  in front    : {full['in_front']}")
    print(f"  in bounds   : {full['in_bounds']}")
    print()

    if full["in_front"] == 0:
        print("→ ✗ no points landed in front of the camera; check transforms")
    elif full["in_bounds"] >= args.min_in_bounds:
        print(f"→ ✓ projection looks sane (≥ {args.min_in_bounds} in-bounds points)")
    else:
        print(f"→ ✗ only {full['in_bounds']} in-bounds points (< {args.min_in_bounds}); verify cam-to-ego extrinsic")

    overlay = save_overlay(item, args.cam, args.overlay_out)
    if overlay is not None:
        path, n = overlay
        print(f"\noverlay saved : {path}  ({n} in-bounds points, colored by depth)")


if __name__ == "__main__":
    main()
