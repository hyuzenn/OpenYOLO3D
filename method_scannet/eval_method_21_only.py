"""Single-method ablation entry: METHOD_21 (Weighted voting) only.

Mirrors run_evaluation.py's CLI but installs only the METHOD_21 hook so the
per-scene output passes through unchanged. Metrics are dumped to
$RUN_DIR/metrics.json by run_evaluation._maybe_dump_metrics.
"""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_method_21_only
    from method_scannet.method_21_weighted_voting import WeightedVoting

    # IDENTICAL hyperparameters to Phase 1 — required for the comparison to be
    # an isolation of METHOD_31's contribution.
    voter = WeightedVoting(
        distance_weight_decay=10.0,
        center_weight_decay=300.0,
        spatial_alpha=0.5,
    )
    install_method_21_only(voter=voter)

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
