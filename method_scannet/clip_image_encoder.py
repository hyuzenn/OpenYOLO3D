"""METHOD_22 visual feature provider — CLIP image encoder, CPU-forced.

Per-frame visual embeddings for FeatureFusionEMA come from cropping each
2D bbox out of its source image and pushing the crop through the CLIP
*image* tower of the same variant whose *text* tower already produced
`pretrained/scannet200_prompt_embeddings.pt`. Same projection space →
cosine similarity is meaningful between text prompts and image crops
without further alignment.

CPU is enforced two ways: (1) the module sets `CUDA_VISIBLE_DEVICES=""`
on import as belt-and-suspenders, and (2) the class defaults to
`device='cpu'` and routes everything through `torch.device(device)`.
Set the env var *before* importing torch in any caller to make item (1)
take effect on torch's CUDA initialization.
"""
from __future__ import annotations

import os

# Force CPU before torch is imported by callers that import this module
# first. (No-op if torch was already imported with CUDA visible — in that
# case the class-level `device='cpu'` is the actual safeguard.)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from typing import Optional

import numpy as np
import torch


class CLIPImageEncoder:
    """CLIP image encoder pinned to the same variant as the text encoder
    used for ScanNet200 prompt embeddings.
    """

    EXPECTED_VARIANT: str = "openai/clip-vit-base-patch32"
    EXPECTED_EMBED_DIM: int = 512

    def __init__(
        self,
        variant: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor  # lazy import

        variant = variant or self.EXPECTED_VARIANT
        self.variant = variant
        self.device = torch.device(device)

        self.model = CLIPModel.from_pretrained(variant).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(variant)
        self.model.eval()

        self.embed_dim = int(self.model.config.projection_dim)
        if self.embed_dim != self.EXPECTED_EMBED_DIM:
            raise AssertionError(
                f"embed_dim mismatch: got {self.embed_dim}, "
                f"expected {self.EXPECTED_EMBED_DIM} for variant {variant!r}"
            )

    @torch.no_grad()
    def encode_cropped_bboxes(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
    ) -> torch.Tensor:
        """Encode each bbox crop into an L2-normalized CLIP embedding.

        Args:
            image:  (H, W, 3) uint8 RGB.
            bboxes: (n, 4) integer pixel coords [x1, y1, x2, y2].

        Returns:
            (n, embed_dim) L2-normalized CPU float tensor.

        Edge cases handled:
            - Empty bboxes → (0, embed_dim) tensor.
            - Coords outside image → clamped to image bounds.
            - Zero-area crop → falls back to a 1x1 patch at the bbox origin.
        """
        bboxes = np.asarray(bboxes).reshape(-1, 4) if len(np.asarray(bboxes)) else np.zeros((0, 4))
        if bboxes.shape[0] == 0:
            return torch.zeros(0, self.embed_dim)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) RGB, got shape {image.shape}"
            )
        H, W = image.shape[:2]

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box — fall back to a single-pixel patch at the
                # clamped origin so the encoder still gets a valid input.
                x2 = min(W, x1 + 1)
                y2 = min(H, y1 + 1)
                if x2 <= x1 or y2 <= y1:
                    # Box was at the bottom/right edge of the image; pull the
                    # origin one pixel up/left.
                    x1 = max(0, x2 - 1)
                    y1 = max(0, y2 - 1)
            crops.append(image[y1:y2, x1:x2])

        inputs = self.processor(images=crops, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values)
        norms = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        features = features / norms
        return features.float().cpu().contiguous()

    @torch.no_grad()
    def encode_full_image(self, image: np.ndarray) -> torch.Tensor:
        """Single full-image encoding, no cropping.

        Returns:
            (1, embed_dim) L2-normalized CPU tensor.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must be (H, W, 3) RGB, got shape {image.shape}"
            )
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values)
        norms = features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        features = features / norms
        return features.float().cpu().contiguous()
