"""
PHASE 2 INTEGRATION DRAFT - NOT YET ACTIVE.

This file is a draft of METHOD_22 (and forward-stubbed Phase 2 / Mix)
integration. It will be merged into method_scannet/hooks.py after:

    1. The METHOD_31 iou=0.7 ablation result is received.
    2. User reviews and resolves the decision items listed at the
       bottom of this docstring.

The METHOD_22-only path is implemented with the recommended approach
(patch point B: re-classify after the original mask decision, using
cropped-bbox CLIP image features fused per-instance via EMA). The
init-side prep (load prompt embeddings, build FeatureFusionEMA, build
CLIPImageEncoder) is real working code so the dependency chain is
validated. The actual monkey-patch is staged as a pseudocode block, and
the function raises NotImplementedError before any global state is
written. METHOD_32-only / Phase 2 / Mix-A / Mix-B are stub installers
that raise NotImplementedError immediately.

Do NOT import this from active eval scripts. The intended consumers are:
    - test_phase2_integration_draft.py (validates the contract).
    - The future hooks.py merge that copies + uncomments the wrapper.

Decision items still open (must be resolved before merge):
    [ ] OpenYolo3D's exact patch target function (best guess:
        utils.OpenYolo3D.label_3d_masks_from_2d_bboxes).
    [ ] Frame-data interface inside the patched call (where do we get
        per-frame RGB image + 2D bboxes + per-bbox instance id?).
    [ ] use_inference_subset = True/False (True = drop wall/floor).
    [ ] ema_alpha (default 0.7; sweep [0.5, 0.7, 0.9]).
    [ ] METHOD_32 class_aware vs class_agnostic.
    [ ] METHOD_22 / METHOD_32 visual-feature sharing (avoid 2x compute).
    [ ] Hyperparameter exposure (config_scannet200.yaml vs hooks default).
"""
from __future__ import annotations

import os

# Belt-and-suspenders CPU forcing (effective only if torch is not yet
# imported by the caller). The classes below also default to device='cpu'.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from typing import Dict, Optional

import numpy as np
import torch

from method_scannet.clip_image_encoder import CLIPImageEncoder
from method_scannet.method_22_feature_fusion import FeatureFusionEMA
from method_scannet.utils_bbox import compute_instance_aabbs_batch  # noqa: F401  (re-exported convenience)


# ----------------------------------------------------------------------
# GLOBAL STATE (will move to hooks.py upon merge — kept here for shape
# documentation only; not populated while the install_* function raises).
# ----------------------------------------------------------------------

_method_22_state: Dict[str, Optional[object]] = {
    "fusion": None,         # FeatureFusionEMA instance
    "image_encoder": None,  # CLIPImageEncoder instance
    "prompts_data": None,   # raw .pt payload
    "original_func": None,  # backup of patched OpenYolo3D method
}


# ----------------------------------------------------------------------
# METHOD_22 ONLY  (recommended approach — pseudocode + raise)
# ----------------------------------------------------------------------


def install_method_22_only(
    prompt_embeddings_path: str = "pretrained/scannet200_prompt_embeddings.pt",
    ema_alpha: float = 0.7,
    use_inference_subset: bool = True,
):
    """METHOD_22 (FeatureFusionEMA) replaces text-prompt label decision.

    Approach: patch point B. The original mask decision logic stays
    intact — only the label assigned to each mask is recomputed using
    cross-frame CLIP image features (cropped bbox → ViT-B/32 → EMA →
    cosine vs prompt embeddings).

    Args:
        prompt_embeddings_path: path to .pt with text embeddings (built
            by method_scannet/extract_prompt_embeddings.py).
        ema_alpha: EMA decay coefficient.
        use_inference_subset: if True, drop wall/floor (matches
            config_scannet200.yaml's inference set); else use all 200.

    Open TODOs (resolved at merge time):
        - Confirm exact OpenYolo3D method to patch (best guess
          OpenYolo3D.label_3d_masks_from_2d_bboxes; need to read the
          method's signature & return contract).
        - Confirm per-frame access to (rgb_image, bboxes_2d, instance_ids)
          from inside the wrapper. May require also caching the 2D-frame
          stage's output on `self`.
        - Confirm output contract (caller expects dict / tensor / list?).
        - If ScanNet200 evaluator includes wall/floor as background,
          use_inference_subset=True is safe; otherwise revisit.
    """
    # --- prep work runs (validates dependency chain) -----------------
    prompts_data = torch.load(prompt_embeddings_path, map_location="cpu")

    if use_inference_subset:
        inference_names = prompts_data.get("openyolo3d_inference_classes")
        if inference_names is None:
            raise KeyError(
                "openyolo3d_inference_classes missing from "
                f"{prompt_embeddings_path}; rebuild via extract_prompt_embeddings.py"
            )
        all_names = list(prompts_data["class_names"])
        name_to_idx = {n: i for i, n in enumerate(all_names)}
        keep_indices = [name_to_idx[n] for n in inference_names if n in name_to_idx]
        embeddings = prompts_data["embeddings"][keep_indices]
        class_names = [all_names[i] for i in keep_indices]
    else:
        embeddings = prompts_data["embeddings"]
        class_names = list(prompts_data["class_names"])

    # Validate METHOD_22 can be constructed against this prompt set.
    fusion = FeatureFusionEMA(
        ema_alpha=ema_alpha,
        prompt_embeddings=embeddings,
        prompt_class_names=class_names,
    )

    # Validate same-variant CLIP image encoder loads.
    image_encoder = CLIPImageEncoder(variant=prompts_data["clip_variant"])

    # Local-only — we deliberately do NOT yet stash to _method_22_state.
    # Stashing happens together with the actual monkey-patch; both belong
    # in hooks.py at merge time.
    _ = fusion, image_encoder, prompts_data

    # --- pseudocode for hook installation (uncomment at merge) -------
    #
    # from utils import OpenYolo3D
    # _method_22_state["fusion"] = fusion
    # _method_22_state["image_encoder"] = image_encoder
    # _method_22_state["prompts_data"] = prompts_data
    # _method_22_state["original_func"] = OpenYolo3D.label_3d_masks_from_2d_bboxes
    #
    # def _patched(self, *args, **kwargs):
    #     # 1. Run original to obtain (mask, default_label, score) triples.
    #     out = _method_22_state["original_func"](self, *args, **kwargs)
    #
    #     # 2. Per frame, encode bbox crops and EMA-update per instance.
    #     #    Frame-data access TBD — likely from cached 2D stage on `self`.
    #     for frame_idx, frame_data in enumerate(self._cached_2d_frames):
    #         image = frame_data["rgb"]                      # (H,W,3) uint8
    #         bboxes = frame_data["bboxes_2d"]               # (n,4)
    #         instance_ids = frame_data["instance_ids"]      # (n,) int
    #         per_inst = extract_per_frame_visual_embeddings(
    #             image, bboxes, instance_ids,
    #             _method_22_state["image_encoder"],
    #         )
    #         for iid, emb in per_inst.items():
    #             _method_22_state["fusion"].update_instance_feature(iid, emb)
    #
    #     # 3. Re-determine each instance's label from its accumulated feature.
    #     fusion = _method_22_state["fusion"]
    #     # rewrite `out` labels using fusion.predict_label(inst_id)
    #     return out
    #
    # OpenYolo3D.label_3d_masks_from_2d_bboxes = _patched
    # ------------------------------------------------------------------

    raise NotImplementedError(
        "phase2_integration_draft.install_method_22_only is staged code; "
        "merge into method_scannet/hooks.py after resolving the open "
        "decision items listed in the module docstring."
    )


def uninstall_method_22_only() -> None:
    """Restore original label decision logic. Stub until install pairs up."""
    _method_22_state["fusion"] = None
    _method_22_state["image_encoder"] = None
    _method_22_state["prompts_data"] = None
    _method_22_state["original_func"] = None
    raise NotImplementedError("Pair with install_method_22_only.")


# ----------------------------------------------------------------------
# PER-FRAME VISUAL EMBEDDING HELPER  (working code, exercised by tests)
# ----------------------------------------------------------------------


@torch.no_grad()
def extract_per_frame_visual_embeddings(
    image: np.ndarray,
    bboxes_2d: np.ndarray,
    instance_ids: np.ndarray,
    image_encoder: CLIPImageEncoder,
) -> Dict[int, torch.Tensor]:
    """Encode each detection bbox in a single frame and group by instance.

    If multiple bboxes in this frame map to the same instance (rare but
    possible — e.g. a re-detection across overlapping crops), the
    embeddings are summed and re-normalized so the per-frame contribution
    stays a unit vector (i.e. mean direction on the unit sphere).

    Args:
        image:        (H, W, 3) uint8 RGB.
        bboxes_2d:    (n, 4) integer pixel coords [x1, y1, x2, y2].
        instance_ids: (n,) int — which 3D instance each bbox belongs to.
        image_encoder: CLIPImageEncoder pinned to the same variant as the
            text prompt embeddings.

    Returns:
        {instance_id: (embed_dim,) torch.float32, L2-normalized, on CPU}.
        Empty input → empty dict.
    """
    bboxes_arr = np.asarray(bboxes_2d).reshape(-1, 4) if len(np.asarray(bboxes_2d)) else np.zeros((0, 4))
    ids_arr = np.asarray(instance_ids).reshape(-1)
    if bboxes_arr.shape[0] == 0:
        return {}
    if ids_arr.shape[0] != bboxes_arr.shape[0]:
        raise ValueError(
            f"instance_ids length {ids_arr.shape[0]} does not match "
            f"bboxes count {bboxes_arr.shape[0]}"
        )

    embeddings = image_encoder.encode_cropped_bboxes(image, bboxes_arr)  # (n, D)

    result: Dict[int, torch.Tensor] = {}
    for inst_id, emb in zip(ids_arr.tolist(), embeddings):
        iid = int(inst_id)
        if iid not in result:
            result[iid] = emb.clone()
        else:
            summed = result[iid] + emb
            result[iid] = summed / summed.norm().clamp_min(1e-12)
    return result


# ----------------------------------------------------------------------
# METHOD_32 ONLY  (stub — pending bbox_3d format + feature-sharing decision)
# ----------------------------------------------------------------------


def install_method_32_only(*args, **kwargs):
    """METHOD_32 (HungarianMerger) post-merge.

    Open TODOs (resolved at merge time):
        - bbox_3d input format from METHOD_31's output point. utils_bbox.
          compute_instance_aabbs_batch handles vertex-level masks.
        - Where to get per-instance visual features. If METHOD_22 is also
          active, share fusion.instance_features (1x compute, aligned
          semantics). If running solo, METHOD_32 needs its own visual
          provider — typically not worth it.
        - class_aware vs class_agnostic (plan §4.1). Recommended:
          class_aware to start.
    """
    raise NotImplementedError(
        "phase2_integration_draft.install_method_32_only requires bbox_3d "
        "format confirmation and feature-sharing decision (plan §4)."
    )


def uninstall_method_32_only():
    raise NotImplementedError("Pair with install_method_32_only.")


# ----------------------------------------------------------------------
# PHASE 2 (METHOD_22 + METHOD_32)  /  MIX-A  /  MIX-B  — stubs
# ----------------------------------------------------------------------


def install_phase2(**kwargs):
    """METHOD_22 (label) + METHOD_32 (merge) simultaneous install."""
    raise NotImplementedError(
        "phase2_integration_draft.install_phase2 depends on both "
        "install_method_22_only and install_method_32_only completion."
    )


def uninstall_phase2():
    raise NotImplementedError("Pair with install_phase2.")


def install_mix_a(**kwargs):
    """Mix-A: METHOD_21 (label) + METHOD_32 (merge).

    Reuses hooks.py's install_method_21_hook and adds METHOD_32 post-merge.
    """
    raise NotImplementedError("Mix-A: pending hooks.py merge of METHOD_32.")


def uninstall_mix_a():
    raise NotImplementedError("Pair with install_mix_a.")


def install_mix_b(**kwargs):
    """Mix-B: METHOD_22 (label) + METHOD_31 (merge).

    Reuses hooks.py's METHOD_31 hook and adds METHOD_22 label rewrite.
    """
    raise NotImplementedError("Mix-B: pending hooks.py merge of METHOD_22.")


def uninstall_mix_b():
    raise NotImplementedError("Pair with install_mix_b.")
