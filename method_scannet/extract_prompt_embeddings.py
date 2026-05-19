"""Extract ScanNet200 text-prompt embeddings using the same CLIP text encoder
that OpenYOLO3D's YOLO-World v2-X uses, and cache them to disk for METHOD_22.

CPU-forced: this script must not touch GPU. Run with

    CUDA_VISIBLE_DEVICES="" python method_scannet/extract_prompt_embeddings.py

Sources of truth:
    - Class list  : evaluate/scannet200/scannet_constants.CLASS_LABELS_200 (200 entries).
    - CLIP variant: openai/clip-vit-base-patch32, used by
      models/YOLO-World/yolo_world/models/backbones/mm_backbone.py
      (HuggingCLIPLanguageBackbone). It uses
      transformers.CLIPTextModelWithProjection + AutoTokenizer and
      L2-normalizes text_embeds.
    - Prompt template: bare class name, mirroring
      pretrained/config_scannet200.yaml's `network2d.text_prompts` (which
      lists 199 instance classes + a single ' ' background — no
      "a photo of a" wrapper).

Output: pretrained/scannet200_prompt_embeddings.pt with keys
    embeddings        (200, embed_dim) torch.float32, L2-normalized
    class_names       list[str] of length 200 (matches embeddings rows)
    embed_dim         int
    clip_variant      'openai/clip-vit-base-patch32'
    prompt_template   '{class_name}' (bare, no wrapper)
    valid_class_ids   list[int] of length 200 (from VALID_CLASS_IDS_200)
    head_cats / common_cats / tail_cats : list[str] convenience subsets
    openyolo3d_inference_classes : list[str] (length 199; CLASS_LABELS_200
                                              minus 'wall', 'floor', matching
                                              config_scannet200.yaml)
"""
from __future__ import annotations

import os

# Force CPU before any torch / transformers import — METHOD_31 ablation may be
# using the GPU in another terminal.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import sys

import torch


def _load_scannet_constants():
    """Import scannet_constants without triggering evaluate/__init__.py side
    effects (which transitively imports torch/scipy and is slow at module
    load).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    src = os.path.join(repo_root, "evaluate", "scannet200", "scannet_constants.py")
    spec = importlib.util.spec_from_file_location("scannet_constants", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        # Should never happen because of CUDA_VISIBLE_DEVICES, but defend
        # against accidental device acquisition just in case.
        raise RuntimeError(
            "CUDA appears available; this script must run with CUDA_VISIBLE_DEVICES=''"
        )

    consts = _load_scannet_constants()
    class_names = list(consts.CLASS_LABELS_200)
    valid_ids = list(consts.VALID_CLASS_IDS_200)
    if len(class_names) != 200:
        raise RuntimeError(f"expected 200 class names, got {len(class_names)}")
    if len(valid_ids) != 200:
        raise RuntimeError(f"expected 200 valid ids, got {len(valid_ids)}")

    # OpenYOLO3D's inference time prompt set (config_scannet200.yaml) excludes
    # 'wall' and 'floor' (stuff classes — not predicted as instances).
    inference_set = [c for c in class_names if c not in ("wall", "floor")]
    if len(inference_set) != 198:
        # The full 200 minus 'wall' minus 'floor' = 198. Note: the YAML lists
        # 199 prompts because it duplicates one entry — keep this empirical
        # subset documented but don't fail on it.
        print(
            f"[note] inference_set length is {len(inference_set)}, not 198; "
            "this is informational only.",
            flush=True,
        )

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    model_name = "openai/clip-vit-base-patch32"
    print(f"[1/4] Loading tokenizer + text encoder: {model_name} (CPU)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    text_model.eval()

    prompt_template = "{class_name}"  # bare; matches config_scannet200.yaml
    prompts = [prompt_template.format(class_name=n) for n in class_names]

    print(f"[2/4] Tokenizing 200 prompts (CPU)", flush=True)
    enc = tokenizer(text=prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    print(f"[3/4] Forward pass (CPU)", flush=True)
    with torch.no_grad():
        out = text_model(**enc)
        embeds = out.text_embeds  # (200, embed_dim)
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = embeds.float().cpu().contiguous()

    embed_dim = int(embeds.shape[-1])
    print(f"[4/4] Saving (shape {tuple(embeds.shape)}, dtype {embeds.dtype})", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    out_dir = os.path.join(repo_root, "pretrained")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scannet200_prompt_embeddings.pt")

    payload = {
        "embeddings": embeds,
        "class_names": class_names,
        "embed_dim": embed_dim,
        "clip_variant": model_name,
        "prompt_template": prompt_template,
        "valid_class_ids": valid_ids,
        "head_cats": list(consts.HEAD_CATS_SCANNET_200),
        "common_cats": list(consts.COMMON_CATS_SCANNET_200),
        "tail_cats": list(consts.TAIL_CATS_SCANNET_200),
        "openyolo3d_inference_classes": inference_set,
    }
    torch.save(payload, out_path)

    size_kb = os.path.getsize(out_path) / 1024.0
    norms = embeds.norm(dim=-1)
    print(
        f"\nSaved {out_path}\n"
        f"  shape       : {tuple(embeds.shape)}\n"
        f"  dtype       : {embeds.dtype}\n"
        f"  file size   : {size_kb:.1f} KB\n"
        f"  unit-norm   : min={norms.min().item():.6f}, max={norms.max().item():.6f}\n"
        f"  first class : {class_names[0]!r}\n"
        f"  last class  : {class_names[-1]!r}\n"
        f"  variant     : {model_name}\n"
        f"  template    : {prompt_template!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
