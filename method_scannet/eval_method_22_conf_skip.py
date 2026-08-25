"""METHOD_22 + Task 1 confidence-skip CLI entry.

Sibling to ``eval_method_22_only_v2.py`` — same v2 prompt embeddings, same
``test_pipeline_full`` delegation, but adds the Task-1 / professor-feedback
C-variant knobs so a Tau sweep can be driven from the shell without ever
touching the production entry. Default flags reproduce the v2 baseline
byte-for-byte (``--conf-mode none``).

CLI:
    python -m method_scannet.eval_method_22_conf_skip \\
        --dataset_name scannet200 \\
        --path_to_3d_masks ./output/scannet200/scannet200_masks \\
        --conf-mode skip --tau-skip 0.20

Optional scene filtering (additive only — patches the module-level scene
list before ``test_pipeline_full`` reads it; does not modify
``run_evaluation.py``):
    --scene-limit N        evaluate only the first N scenes
    --scenes NAME[,NAME..] evaluate only the listed scenes (exact match)
    --seed-scenes K        deterministically shuffle by hash and take K
                           (compatibility with non-contiguous smoke sets).
"""
from __future__ import annotations

import argparse
import os


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", default="scannet200", type=str)
    parser.add_argument(
        "--path_to_3d_masks",
        default="./output/scannet200/scannet200_masks",
        type=str,
    )
    parser.add_argument(
        "--is_gt",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--prompt-embeddings-path",
        default="pretrained/scannet200_prompt_embeddings_v2.pt",
        type=str,
    )
    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--min-match-iou", type=float, default=0.15)
    parser.add_argument("--topk-per-inst", type=int, default=5)
    parser.add_argument(
        "--conf-mode",
        choices=("none", "skip", "weighted"),
        default="none",
        help="EMA confidence mode. 'none' (default) preserves the v2 "
             "baseline byte-for-byte. 'skip' drops EMA updates whose "
             "matched-YOLO-bbox confidence is < --tau-skip. 'weighted' "
             "applies the confidence-weighted EMA "
             "F_new=(1-alpha*c)*F_old+(alpha*c)*F_match (alpha=--ema-alpha is "
             "the incoming-feature weight here; c is the matched-bbox "
             "confidence clamped to [0,1], or 1.0 when absent). 'weighted' "
             "also honours --tau-skip.",
    )
    parser.add_argument(
        "--tau-skip",
        type=float,
        default=0.0,
        help="Confidence threshold for --conf-mode skip and weighted: a "
             "match with confidence < tau_skip is dropped before the EMA "
             "update. Ignored when --conf-mode is 'none'.",
    )
    parser.add_argument(
        "--conf-strict",
        action="store_true",
        default=False,
        help="Fail-Loud retrofit for --conf-mode skip. Asserts that every "
             "score plumbing step (preds_2d['scores'] presence, length "
             "match to bbox tensor, per-element coercibility) succeeds; "
             "raises immediately on defect instead of silently falling "
             "back to confidence=None. Ignored under --conf-mode none "
             "so the v2 baseline anchor remains byte-identical.",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N scenes (slice of the canonical "
             "SCENE_NAMES_SCANNET200 list).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="",
        help="Comma-separated explicit scene-name list; takes precedence "
             "over --scene-limit.",
    )
    return parser.parse_args()


def _maybe_filter_scenes(opt: argparse.Namespace) -> None:
    """Monkey-patch ``evaluate.SCENE_NAMES_SCANNET200`` BEFORE
    ``run_evaluation.test_pipeline_full`` imports it. This module owns no
    state of its own; the production module is left untouched.
    """
    explicit = [s.strip() for s in opt.scenes.split(",") if s.strip()]
    if not explicit and opt.scene_limit <= 0:
        return

    import evaluate as _eval_mod
    full = list(_eval_mod.SCENE_NAMES_SCANNET200)
    if explicit:
        missing = [s for s in explicit if s not in full]
        if missing:
            raise SystemExit(
                f"--scenes lists scenes not in SCENE_NAMES_SCANNET200: {missing[:5]}..."
            )
        subset = explicit
    else:
        subset = full[: int(opt.scene_limit)]
    _eval_mod.SCENE_NAMES_SCANNET200 = subset

    # run_evaluation.py does `from evaluate import SCENE_NAMES_SCANNET200, ...`
    # at module load — patch the imported name there too if already loaded.
    import sys
    if "run_evaluation" in sys.modules:
        sys.modules["run_evaluation"].SCENE_NAMES_SCANNET200 = subset
    print(
        f"[eval_method_22_conf_skip] filtered scene list -> "
        f"{len(subset)} scene(s): {subset[:5]}{'...' if len(subset) > 5 else ''}"
    )


def main() -> None:
    opt = _parse_args()
    _maybe_filter_scenes(opt)

    from method_scannet.hooks import install_method_22_only

    install_method_22_only(
        prompt_embeddings_path=opt.prompt_embeddings_path,
        ema_alpha=opt.ema_alpha,
        use_inference_subset=True,
        min_match_iou=opt.min_match_iou,
        topk_per_inst=opt.topk_per_inst,
        conf_mode=opt.conf_mode,
        tau_skip=opt.tau_skip,
        conf_strict=opt.conf_strict,
    )

    # Best-effort diagnostic print so the run.log self-documents the config.
    print(
        f"[eval_method_22_conf_skip] conf_mode={opt.conf_mode!r} "
        f"tau_skip={opt.tau_skip:.3f} conf_strict={opt.conf_strict} "
        f"ema_alpha={opt.ema_alpha} min_match_iou={opt.min_match_iou} "
        f"topk_per_inst={opt.topk_per_inst} "
        f"RUN_DIR={os.environ.get('RUN_DIR', '<unset>')}"
    )

    from run_evaluation import test_pipeline_full
    test_pipeline_full(opt.dataset_name, opt.path_to_3d_masks, opt.is_gt)


if __name__ == "__main__":
    main()
