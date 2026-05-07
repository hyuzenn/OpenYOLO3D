"""METHOD_31 single-method ablation, iou_threshold = 0.7.

Mirrors `eval_method_31_only.py` but passes a custom `IoUMerger` instance
with `iou_threshold = 0.7` to `install_method_31_only`. No edits to hooks.py
or method_31_iou_merging.py are required — `install_method_31_only` already
accepts a `merger=` keyword.
"""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_method_31_only
    from method_scannet.method_31_iou_merging import IoUMerger

    # Only iou_threshold differs from Step 1.5a / Phase 1. Every other knob
    # is held identical so the comparison is a clean threshold sweep.
    merger = IoUMerger(
        iou_threshold=0.7,
        use_kdtree=True,
        kdtree_neighbor_radius=2.0,
        same_class_only=True,
    )
    install_method_31_only(merger=merger)

    from run_evaluation import test_pipeline_full

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="scannet200", type=str)
    parser.add_argument(
        "--path_to_3d_masks",
        default="./output/scannet200/scannet200_masks",
        type=str,
    )
    parser.add_argument(
        "--is_gt", default=False, action=argparse.BooleanOptionalAction
    )
    opt = parser.parse_args()
    test_pipeline_full(opt.dataset_name, opt.path_to_3d_masks, opt.is_gt)


if __name__ == "__main__":
    main()
