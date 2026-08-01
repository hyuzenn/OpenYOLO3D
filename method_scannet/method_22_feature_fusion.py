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
        normalize_per_frame: bool = False,
        margin: float = 0.0,
        conf_mode: str = "none",
        tau_skip: float = 0.0,
        conf_strict: bool = False,
        record_feature_trace: bool = False,
        const_k: float = 0.335,
    ) -> None:
        if not (0.0 <= ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in [0, 1], got {ema_alpha}")
        self.ema_alpha = float(ema_alpha)
        # Fix knobs (defaults preserve original behaviour):
        #   normalize_per_frame: L2-normalize each per-frame embedding before
        #     the EMA so high-norm noisy crops can't dominate the running mean.
        #   margin: confidence-margin gate on predict_label — switch an
        #     instance's label only when (top1 - top2) cosine >= margin,
        #     otherwise hold its previous label (kills per-frame argmax flips
        #     in the dense open-vocab prompt space). margin=0.0 -> no gate.
        #   conf_mode: "none" (default) ignores the per-frame `confidence`
        #     argument to `update_instance_feature` entirely — byte-for-byte
        #     baseline behaviour. "skip" drops EMA updates whose confidence
        #     falls below `tau_skip` (Task 1 / professor-feedback C variant).
        #     "weighted" applies the professor's confidence-weighted EMA
        #     (see update_instance_feature): F_new = (1 - alpha*c)*F_old +
        #     (alpha*c)*F_match, with c=clamp(confidence,0,1) and c=1.0 when
        #     confidence is None (reduces to a plain EMA whose INCOMING-feature
        #     weight is alpha). `tau_skip` is honoured in this mode too — a
        #     match with confidence < tau_skip is dropped before the update.
        #   tau_skip: confidence threshold used by conf_mode="skip" AND
        #     conf_mode="weighted"; the update is silently dropped when
        #     `confidence < tau_skip` (a non-None confidence is required to
        #     trigger the drop).
        #   conf_strict: Fail-Loud retrofit for the C variant. Active ONLY
        #     when conf_mode == "skip". With conf_strict=True, any
        #     update_instance_feature call that arrives without a usable
        #     `confidence` value (i.e. None) raises a ValueError instead of
        #     silently bypassing the gate. Default False preserves the
        #     pre-Task-1 conservative behaviour (`None` ⇒ "no gate" ⇒
        #     update applied). conf_mode=="none" ignores this flag entirely.
        if conf_mode not in ("none", "skip", "weighted", "ovtcs_scale", "const_scale"):
            raise ValueError(
                "conf_mode must be 'none', 'skip', 'weighted', 'ovtcs_scale' or "
                f"'const_scale', got {conf_mode!r}"
            )
        #   conf_mode "ovtcs_scale": OV-TCS-aware confidence-weighted EMA. Same
        #     incoming-weight convention as "weighted", but the incoming weight
        #     is additionally scaled by the *causal online* OV-TCS_C of the
        #     instance computed from its raw per-frame argmax-label sequence:
        #         w = clamp(alpha * c * OVTCS_C, 0, 1)
        #     with OVTCS_C = (1 - 1/L) * (1 - switches/(L-1)) for L>=2 else 0.
        #     The label sequence is the per-frame raw crop argmax (NOT the
        #     EMA-committed label) so the control signal is independent of the
        #     memory it modulates (no EMA->label->OVTCS->EMA feedback loop).
        #   record_feature_trace: when True, additionally log per-applied-update
        #     (OVTCS_t, drift_t, F_t) so ovtcs_diagnostics() can report the
        #     OVTCS<->drift and OVTCS<->cosine-to-final-feature correlations.
        #     Read-only instrumentation: it never changes the EMA math, so a
        #     "weighted" arm with record_feature_trace=True stays numerically
        #     identical to plain "weighted".
        self.normalize_per_frame = bool(normalize_per_frame)
        self.margin = float(margin)
        self.conf_mode = str(conf_mode)
        self.tau_skip = float(tau_skip)
        self.conf_strict = bool(conf_strict)
        self.record_feature_trace = bool(record_feature_trace)
        #   conf_mode "const_scale": attribution control for ovtcs_scale —
        #     w = clamp(alpha * c * const_k, 0, 1) with a FIXED const_k (default
        #     0.335 = the observed mean online OV-TCS). Same average step-shrink
        #     as ovtcs_scale but with NO per-update OV-TCS variation, so
        #     comparing the two isolates whether OV-TCS's per-update signal adds
        #     anything beyond a smaller effective alpha.
        self.const_k = float(const_k)
        self._last_label: dict = {}
        # Diagnostic counters — zero side-effects on labelling; only populated
        # when conf_mode != "none". Useful for confirming the skip path
        # actually fires during sweep runs.
        self.n_updates_attempted: int = 0
        self.n_updates_skipped: int = 0

        # --- OV-TCS online bookkeeping (active only when ovtcs is tracked) ----
        # _track_ovtcs gates ALL of the new per-frame work so the
        # none/skip/weighted feature math stays byte-for-byte unchanged unless a
        # caller opts in via conf_mode="ovtcs_scale" or record_feature_trace.
        self._track_ovtcs: bool = (
            self.conf_mode == "ovtcs_scale" or self.record_feature_trace
        )
        self._raw_L: dict = {}          # id -> #observations (track length L)
        self._raw_switches: dict = {}   # id -> #consecutive raw-label changes
        self._raw_last: dict = {}       # id -> last raw argmax label idx
        self._applied_count: dict = {}  # id -> #EMA writes (incl. seed)
        self._observed_count: dict = {} # id -> #observations reaching the EMA
        self._ovtcs_values: list = []   # per-observation OVTCS_C (L>=2)
        self._drift_values: list = []   # per-applied-update ||F_new-F_old||2
        # Per-applied-update trace (only when record_feature_trace); kept
        # per-instance so cos-to-final can be resolved at harvest time.
        self._feat_trace: dict = {}     # id -> list of (ovtcs, drift, F_t clone)

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
        confidence: Optional[float] = None,
    ) -> None:
        """EMA-accumulate a per-frame visual embedding for an instance.

        First update for an id seeds the running feature; subsequent updates
        apply f_t = alpha * f_{t-1} + (1 - alpha) * f_current.

        `confidence` is the optional per-frame detection confidence (e.g.
        YOLO-World score for the matched 2D bbox). It is consulted ONLY when
        ``conf_mode != "none"``; in default mode it is ignored, preserving
        byte-for-byte baseline behaviour.
        """
        if self.conf_mode != "none":
            self.n_updates_attempted += 1
        # conf_mode="skip": drop low-confidence updates entirely (Task 1 / C variant).
        # Fail-Loud (conf_strict=True): treat `confidence is None` as a
        # plumbing bug — every skip-mode caller MUST forward a real score, so
        # raise instead of silently falling through. Fail-Silent default
        # (conf_strict=False) retains the conservative interpretation where
        # callers that did not opt into confidence plumbing apply the update.
        if self.conf_mode == "skip" and confidence is None and self.conf_strict:
            raise ValueError(
                f"FeatureFusionEMA(conf_mode='skip', conf_strict=True): "
                f"update_instance_feature(instance_id={instance_id}) received "
                f"confidence=None. Every skip-mode call site must plumb a real "
                f"YOLO-World score; check the matched_scores list in the caller."
            )
        if (
            self.conf_mode in ("skip", "weighted")
            and confidence is not None
            and float(confidence) < self.tau_skip
        ):
            self.n_updates_skipped += 1
            return

        if frame_visual_embedding.dim() != 1:
            frame_visual_embedding = frame_visual_embedding.reshape(-1)
        cur = frame_visual_embedding.detach().float()
        if self.normalize_per_frame:
            cur = _l2_normalize(cur)

        iid = int(instance_id)
        # --- OV-TCS online bookkeeping (gated; no-op for none/skip/weighted
        # unless record_feature_trace) -------------------------------------
        ovtcs = None
        if self._track_ovtcs:
            ovtcs = self._update_raw_ovtcs(iid, cur)
            self._observed_count[iid] = self._observed_count.get(iid, 0) + 1
            if self._raw_L.get(iid, 0) >= 2:
                self._ovtcs_values.append(float(ovtcs))

        prev = self.instance_features.get(iid)
        if prev is None:
            self.instance_features[iid] = cur.clone()
            if self._track_ovtcs:
                self._applied_count[iid] = self._applied_count.get(iid, 0) + 1
            return
        if prev.shape != cur.shape:
            raise ValueError(
                f"feature dim mismatch for id={instance_id}: prev {tuple(prev.shape)} "
                f"vs new {tuple(cur.shape)}"
            )
        if self.conf_mode == "ovtcs_scale":
            # OV-TCS-aware weighted EMA: same incoming-weight convention as
            # "weighted" but the incoming weight is scaled by the causal online
            # OV-TCS_C, down-weighting (or zeroing, when OVTCS_C=0) updates that
            # arrive while the raw label sequence is temporally inconsistent.
            c = 1.0 if confidence is None else float(confidence)
            c = min(1.0, max(0.0, c))
            o = 1.0 if ovtcs is None else float(ovtcs)
            w = self.ema_alpha * c * o
            w = min(1.0, max(0.0, w))
            new = (1.0 - w) * prev + w * cur
        elif self.conf_mode == "const_scale":
            # Matched-average-shrink control: fixed const_k in place of OVTCS.
            c = 1.0 if confidence is None else float(confidence)
            c = min(1.0, max(0.0, c))
            w = self.ema_alpha * c * self.const_k
            w = min(1.0, max(0.0, w))
            new = (1.0 - w) * prev + w * cur
        elif self.conf_mode == "weighted":
            # Professor's confidence-weighted EMA. Here `ema_alpha` is the
            # weight of the INCOMING feature (F_match), the opposite of the
            # none/skip convention where it weights F_old — see the class
            # docstring. Missing confidence falls back to c=1.0 (NEVER 0,
            # which would freeze the memory), so this reduces to a plain EMA
            # with incoming-weight alpha. Both factors are clamped to [0,1]:
            # ema_alpha is validated in __init__ and c is clamped here, hence
            # w = alpha*c in [0,1] and the (1-w), w coefficients stay in [0,1].
            c = 1.0 if confidence is None else float(confidence)
            c = min(1.0, max(0.0, c))
            w = self.ema_alpha * c
            new = (1.0 - w) * prev + w * cur
        else:
            new = self.ema_alpha * prev + (1.0 - self.ema_alpha) * cur
        self.instance_features[iid] = new

        if self._track_ovtcs:
            self._applied_count[iid] = self._applied_count.get(iid, 0) + 1
            drift = float((new - prev).norm().item())
            self._drift_values.append(drift)
            # Trace only well-defined points (L>=2, both OVTCS and drift exist).
            # NOTE: on the "ovtcs_scale" arm drift is mechanically tied to the
            # OVTCS weight (circular) — the OVTCS<->drift correlation should be
            # read off the "weighted" baseline arm, where w is independent of
            # OVTCS. We still record both for completeness.
            if self.record_feature_trace and self._raw_L.get(iid, 0) >= 2:
                self._feat_trace.setdefault(iid, []).append(
                    (float(ovtcs), drift, new.detach().clone())
                )

    def update_batch(self, items) -> None:
        """Convenience: pass an iterable of (instance_id, embedding) pairs.

        NOTE: production routes via update_instance_feature; this helper is
        test-only. Confidence plumbing is therefore NOT supported here —
        callers that need conf_mode="skip" must invoke
        ``update_instance_feature(..., confidence=...)`` directly.
        """
        for iid, emb in items:
            self.update_instance_feature(iid, emb)

    # --- OV-TCS online signal ------------------------------------------------

    def _update_raw_ovtcs(self, iid: int, cur: torch.Tensor) -> float:
        """Update the instance's raw per-frame argmax-label running stats with
        this frame, and return its *causal* online OV-TCS_C.

        OV-TCS_C = (1 - 1/L) * (1 - CSR), CSR = switches/(L-1), computed over the
        raw per-frame crop argmax labels seen so far (this frame inclusive). L=1
        -> 0.0 (L_norm=0). Mirrors ``diagnosis/outdoor_ov_tcs_probe.py``.

        argmax(cos) is invariant to positive scaling of ``cur`` so no separate
        normalization of the current embedding is required. If prompts are not
        set (shouldn't happen on the install path), returns 1.0 so ovtcs_scale
        degrades gracefully to plain weighted.
        """
        prompts = self._prompt_emb_norm
        if prompts is None:
            return 1.0
        prompts = prompts.to(cur.device)
        sims = cur.float().unsqueeze(0) @ prompts.t()  # (1, n_classes)
        lbl = int(sims.squeeze(0).argmax().item())
        L = self._raw_L.get(iid, 0) + 1
        self._raw_L[iid] = L
        if L >= 2 and self._raw_last.get(iid) != lbl:
            self._raw_switches[iid] = self._raw_switches.get(iid, 0) + 1
        self._raw_last[iid] = lbl
        if L < 2:
            return 0.0
        sw = self._raw_switches.get(iid, 0)
        return (1.0 - 1.0 / L) * (1.0 - sw / (L - 1))

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
        if sims.numel() >= 2:
            top2v, top2i = torch.topk(sims, 2)
            idx = int(top2i[0].item())
            conf = float(top2v[0].item())
            margin = float(top2v[0].item() - top2v[1].item())
        else:
            idx = int(torch.argmax(sims).item())
            conf = float(sims[idx].item())
            margin = float("inf")
        label = self.prompt_class_names[idx] if self.prompt_class_names is not None else idx

        # Confidence-margin gate: when the top-1/top-2 cosine separation is
        # below `margin`, the argmax is a near-tie (per-frame noise) — hold the
        # instance's previously committed label instead of flipping. The first
        # observation always seeds (no previous label to hold).
        iid = int(instance_id)
        if self.margin > 0.0 and iid in self._last_label and margin < self.margin:
            return self._last_label[iid], conf
        self._last_label[iid] = label
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

    # --- OV-TCS instrumentation ----------------------------------------------

    def reset(self) -> None:
        """Clear accumulated features and all OV-TCS bookkeeping. Defensive —
        the streaming harness already builds a fresh instance per scene."""
        self.instance_features = {}
        self._last_label = {}
        self.n_updates_attempted = 0
        self.n_updates_skipped = 0
        self._raw_L = {}
        self._raw_switches = {}
        self._raw_last = {}
        self._applied_count = {}
        self._observed_count = {}
        self._ovtcs_values = []
        self._drift_values = []
        self._feat_trace = {}

    def ovtcs_diagnostics(self) -> dict:
        """Harvest per-scene OV-TCS / EMA-update diagnostics.

        Returns aggregate scalars plus, when ``record_feature_trace`` was set,
        the pooled per-applied-update trace arrays (OVTCS_t, drift_t,
        cos_to_final_t) for the OVTCS<->drift / OVTCS<->stability correlations.
        All values are plain Python floats/lists (JSON-serializable). Returns an
        empty-ish dict with n_instances=0 when OV-TCS tracking was inactive.
        """
        import math

        applied = list(self._applied_count.values())
        observed = list(self._observed_count.values())
        ovv = list(self._ovtcs_values)
        drift = list(self._drift_values)

        def _stats(xs: list) -> dict:
            if not xs:
                return {"n": 0, "mean": None, "std": None}
            n = len(xs)
            m = sum(xs) / n
            var = sum((x - m) ** 2 for x in xs) / n
            return {"n": n, "mean": float(m), "std": float(math.sqrt(var))}

        out = {
            "n_instances": len(self._raw_L),
            "updates_applied_per_track": _stats([float(a) for a in applied]),
            "observations_per_track": _stats([float(o) for o in observed]),
            "n_updates_applied_total": int(sum(applied)),
            "n_observations_total": int(sum(observed)),
            "feature_drift": _stats(drift),
            "online_ovtcs": _stats(ovv),
        }

        # Per-applied-update trace + cos-to-final (only with feature trace).
        if self.record_feature_trace and self._feat_trace:
            tr_ovtcs: list = []
            tr_drift: list = []
            tr_cos: list = []
            for iid, entries in self._feat_trace.items():
                final = self.instance_features.get(iid)
                if final is None:
                    continue
                fn = final.float()
                fn = fn / fn.norm().clamp_min(1e-12)
                for o, d, ft in entries:
                    g = ft.float()
                    g = g / g.norm().clamp_min(1e-12)
                    tr_ovtcs.append(float(o))
                    tr_drift.append(float(d))
                    tr_cos.append(float((g * fn).sum().item()))
            out["trace"] = {
                "ovtcs": tr_ovtcs,
                "drift": tr_drift,
                "cos_to_final": tr_cos,
            }
        return out
