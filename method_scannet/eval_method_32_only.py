"""METHOD_32 (HungarianMerger) standalone eval.

No METHOD_22, no METHOD_31. The merger sees no per-instance visual
features so the semantic-similarity gate is disabled internally and
spatial distance is the only merge signal (modulated by class_aware).
"""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_method_32_only

    install_method_32_only(
        spatial_alpha=0.5,
        distance_threshold=2.0,
        semantic_threshold=0.3,
        class_aware=True,
    )

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
