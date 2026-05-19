"""Single-method ablation entry: METHOD_31 (3D IoU merging) only.

Mirrors run_evaluation.py's CLI but installs only the METHOD_31 hook so the
upstream OpenYOLO3D label-aggregation runs unchanged. Metrics are dumped to
$RUN_DIR/metrics.json by run_evaluation._maybe_dump_metrics.
"""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_method_31_only
    from method_scannet.method_31_iou_merging import IoUMerger

    # IDENTICAL hyperparameters to Phase 1 — required for the comparison to be
    # an isolation of METHOD_21's contribution.
    merger = IoUMerger(
        iou_threshold=0.5,
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
