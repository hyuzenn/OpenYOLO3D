"""Phase 1 evaluation entry: METHOD_21 + METHOD_31 active.

Mirrors run_evaluation.py's CLI, but installs the method hooks first so the
upstream pipeline's call sites land in our patched implementations. Metrics
are dumped to $RUN_DIR/metrics.json by run_evaluation._maybe_dump_metrics.
"""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_phase1
    from method_scannet.method_21_weighted_voting import WeightedVoting
    from method_scannet.method_31_iou_merging import IoUMerger

    voter = WeightedVoting(
        distance_weight_decay=10.0,
        center_weight_decay=300.0,
        spatial_alpha=0.5,
    )
    merger = IoUMerger(
        iou_threshold=0.5,
        use_kdtree=True,
        kdtree_neighbor_radius=2.0,
        same_class_only=True,
    )
    install_phase1(voter=voter, merger=merger)

    # Import after hooks so the patched class object is used.
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
