"""Extract ScanNet200 text-prompt embeddings with prompt template
"a photo of a {class_name}" — variant of extract_prompt_embeddings.py
that uses the canonical CLIP prompt wrapper instead of the bare class
name.

CPU-forced. Reuses the same CLIP variant
(`openai/clip-vit-base-patch32`), the same class list
(`evaluate.scannet200.scannet_constants.CLASS_LABELS_200`), the same
inference subset (drop wall, floor), and the same payload schema as
the v1 extractor. Only the prompt template and the output filename
differ.

Output: pretrained/scannet200_prompt_embeddings_v2.pt
"""
from __future__ import annotations

import os

# Force CPU before any torch / transformers import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import sys

import torch


def _load_scannet_constants():
    """Mirror v1's scannet_constants loader to avoid evaluate/__init__.py
    side effects."""
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
        raise RuntimeError(
            "CUDA appears available; this script must run with CUDA_VISIBLE_DEVICES=''"
        )

    consts = _load_scannet_constants()
    class_names = list(consts.CLASS_LABELS_200)
    valid_ids = list(consts.VALID_CLASS_IDS_200)
    if len(class_names) != 200 or len(valid_ids) != 200:
        raise RuntimeError(
            f"expected 200 class names + 200 valid ids, got {len(class_names)} / {len(valid_ids)}"
        )

    # OpenYOLO3D's inference time prompt set excludes 'wall' and 'floor'.
    inference_set = [c for c in class_names if c not in ("wall", "floor")]

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    model_name = "openai/clip-vit-base-patch32"
    print(f"[1/4] Loading tokenizer + text encoder: {model_name} (CPU)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    text_model.eval()

    prompt_template = "a photo of a {class_name}"  # canonical CLIP wrapper
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
    out_path = os.path.join(out_dir, "scannet200_prompt_embeddings_v2.pt")

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
        f"  variant     : {model_name}\n"
        f"  template    : {prompt_template!r}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
