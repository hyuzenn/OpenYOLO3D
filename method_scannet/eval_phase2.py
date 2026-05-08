"""Phase 2 (METHOD_22 + METHOD_32) standalone eval."""
from __future__ import annotations

import argparse


def main() -> None:
    from method_scannet.hooks import install_phase2

    install_phase2(
        prompt_embeddings_path="pretrained/scannet200_prompt_embeddings.pt",
        ema_alpha=0.7,
        use_inference_subset=True,
        min_match_iou=0.15,
        topk_per_inst=5,
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
