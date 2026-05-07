"""METHOD_22 — Per-instance visual feature fusion via EMA, classified against
pre-extracted text-prompt embeddings (CLIP-style label assignment).

Pipeline:
    For each (instance, frame) we collect a visual embedding (e.g. cropped-bbox
    CLIP image feature). EMA-accumulate per instance:

        f_t = alpha * f_{t-1} + (1 - alpha) * f_current      (alpha in [0, 1])

    On the first frame we initialize f_0 = f_current. After all frames, every
    instance has a single accumulated embedding. Final label is

        argmax_c  cos( f_instance, prompt_embedding[c] )

This module is *intentionally* hooks-free: it owns no global state, takes
prompt embeddings at construction, and is safe to instantiate per-scene. CPU
or GPU tensors both work — all ops follow the input device.
"""
from __future__ import annotations

from typing import Optional

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize along the last dimension. Safe for zero-vectors (returns zeros)."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / n


class FeatureFusionEMA:
    def __init__(
        self,
        ema_alpha: float = 0.7,
        prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_class_names: Optional[list] = None,
    ) -> None:
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")
        self.ema_alpha = float(ema_alpha)

        if prompt_embeddings is not None and prompt_class_names is not None:
            if prompt_embeddings.shape[0] != len(prompt_class_names):
                raise ValueError(
                    "prompt_embeddings.shape[0] must equal len(prompt_class_names) "
                    f"({prompt_embeddings.shape[0]} vs {len(prompt_class_names)})"
                )

        self.prompt_class_names: Optional[list] = (
            list(prompt_class_names) if prompt_class_names is not None else None
        )
        # Cache an L2-normalized copy so cosine-sim is one matmul.
        self._prompt_emb: Optional[torch.Tensor] = None
        self._prompt_emb_norm: Optional[torch.Tensor] = None
        if prompt_embeddings is not None:
            self.set_prompt_embeddings(prompt_embeddings, prompt_class_names)

        # instance_id -> accumulated feature (1D tensor on the device of the
        # first update for that instance).
        self.instance_features: dict = {}

    # --- prompt setup --------------------------------------------------------

    def set_prompt_embeddings(
        self,
        prompt_embeddings: torch.Tensor,
        prompt_class_names: Optional[list] = None,
    ) -> None:
        if prompt_embeddings.dim() != 2:
            raise ValueError(
                f"prompt_embeddings must be 2D (n_classes, dim), got shape {tuple(prompt_embeddings.shape)}"
            )
        self._prompt_emb = prompt_embeddings.detach()
        self._prompt_emb_norm = _l2_normalize(self._prompt_emb.float())
        if prompt_class_names is not None:
            if len(prompt_class_names) != prompt_embeddings.shape[0]:
                raise ValueError(
                    "prompt_class_names length must match prompt_embeddings.shape[0]"
                )
            self.prompt_class_names = list(prompt_class_names)

    # --- EMA accumulation ----------------------------------------------------

    def update_instance_feature(
        self,
        instance_id: int,
        frame_visual_embedding: torch.Tensor,
    ) -> None:
        """EMA-accumulate a per-frame visual embedding for an instance.

        First update for an id seeds the running feature; subsequent updates
        apply f_t = alpha * f_{t-1} + (1 - alpha) * f_current.
        """
        if frame_visual_embedding.dim() != 1:
            frame_visual_embedding = frame_visual_embedding.reshape(-1)
        cur = frame_visual_embedding.detach().float()

        prev = self.instance_features.get(int(instance_id))
        if prev is None:
            self.instance_features[int(instance_id)] = cur.clone()
            return
        if prev.shape != cur.shape:
            raise ValueError(
                f"feature dim mismatch for id={instance_id}: prev {tuple(prev.shape)} "
                f"vs new {tuple(cur.shape)}"
            )
        self.instance_features[int(instance_id)] = (
            self.ema_alpha * prev + (1.0 - self.ema_alpha) * cur
        )

    def update_batch(self, items) -> None:
        """Convenience: pass an iterable of (instance_id, embedding) pairs."""
        for iid, emb in items:
            self.update_instance_feature(iid, emb)

    # --- prediction ----------------------------------------------------------

    def _check_ready_for_predict(self) -> None:
        if self._prompt_emb_norm is None:
            raise RuntimeError(
                "prompt_embeddings not set — call set_prompt_embeddings() first."
            )

    def predict_label(self, instance_id: int):
        """Return (class_name_or_index, confidence) for one instance.

        confidence is the max cosine similarity in [-1, 1]. If
        `prompt_class_names` was provided, the first element is the class
        string; otherwise it's the integer class index.
        """
        self._check_ready_for_predict()
        feat = self.instance_features.get(int(instance_id))
        if feat is None:
            raise KeyError(f"no accumulated feature for instance_id={instance_id}")

        prompts = self._prompt_emb_norm.to(feat.device)
        f_norm = _l2_normalize(feat.unsqueeze(0).float())  # (1, D)
        sims = (f_norm @ prompts.t()).squeeze(0)  # (n_classes,)
        idx = int(torch.argmax(sims).item())
        conf = float(sims[idx].item())
        label = self.prompt_class_names[idx] if self.prompt_class_names is not None else idx
        return label, conf

    def predict_all(self) -> dict:
        """Return {instance_id: (class_name_or_index, confidence)} for every
        accumulated instance.
        """
        self._check_ready_for_predict()
        return {iid: self.predict_label(iid) for iid in self.instance_features.keys()}

    # --- introspection -------------------------------------------------------

    def num_instances(self) -> int:
        return len(self.instance_features)

    def get_feature(self, instance_id: int) -> Optional[torch.Tensor]:
        return self.instance_features.get(int(instance_id))
