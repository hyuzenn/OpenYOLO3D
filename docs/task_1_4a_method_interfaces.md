# Task 1.4a — May method class signatures (verified 2026-05-14)

Authoritative reference for how the streaming wrapper must call each May
class. All signatures grepped from the unmodified May files
(`method_scannet/method_*.py`) and cross-checked against `method_scannet/hooks.py`
(May's offline-pipeline install).

The May classes are NOT modified by this task. The streaming adapter
matches these exact signatures.

---

## M11 — `FrameCountingGate`  (file: `method_11_frame_counting.py`)

```python
class FrameCountingGate:
    def __init__(self, N: int = 3, consecutive: bool = False) -> None: ...
    def reset(self) -> None: ...
    def gate(self, visible_instances) -> list[int]:
        """Update counts for this frame and return only confirmed ids."""
    @property
    def confirmed_count(self) -> int: ...
```

State (private): `self._counts: dict[int, int]`, `self._confirmed: set[int]`.

Streaming usage:
- Per-frame: `confirmed_now = gate.gate(visible_instances)`.
- At finalize: read `gate._confirmed` for the full set ever confirmed in the scene; use it to filter final predictions.

---

## M12 — `BayesianGate`  (file: `method_12_bayesian.py`)

```python
class BayesianGate:
    def __init__(self, prior=0.5, detection_likelihood=0.8,
                 false_positive_rate=0.2, threshold=0.95) -> None: ...
    def reset(self) -> None: ...
    def gate(self, visible_instances) -> list[int]: ...
    def posterior(self, instance_id: int) -> float: ...
    @property
    def confirmed_count(self) -> int: ...
```

State (private): `_posteriors: dict[int, float]`, `_confirmed: set[int]`.
Same finalize semantics as M11.

---

## M21 — `WeightedVoting`  (file: `method_21_weighted_voting.py`)

```python
class WeightedVoting:
    def __init__(self, distance_weight_decay=10.0, center_weight_decay=300.0,
                 spatial_alpha=0.5) -> None: ...

    def frame_weight(self, camera_pos, instance_centroid, bbox_2d_center,
                     image_size, confidence: float = 1.0) -> float:
        # spatial = alpha * exp(-d3D/D) + (1-alpha) * exp(-d2D/C)
        # return  spatial * confidence

    def vote_distribution(self, per_instance_frame_labels: list,
                          per_instance_frame_meta: list,
                          num_classes: int) -> torch.Tensor:
        # (n_inst, num_classes) max-normalized weighted histogram.
        # Each meta dict must match frame_weight kwargs.

    def vote_label(self, per_instance_frame_labels: list,
                   per_instance_frame_meta: list,
                   num_classes: int) -> list: ...
```

**Pattern in May `hooks.py` `_patched_label_3d_masks_from_label_maps`:**
The May offline path *inlines* the frame-weight formula (lines 184-193 of hooks.py)
rather than calling `voter.frame_weight()` on every iteration. It still
uses `voter.distance_weight_decay`, `voter.center_weight_decay`, and
`voter.spatial_alpha` as the source of truth for the weights.

**Streaming adapter (`method_adapters.compute_predictions_method21`):**
Replays the same offline algorithm: iterates over Mask3D proposals × their
representative frames, computes the inlined frame weight using the
voter's hyperparameters, builds a weighted class histogram per proposal,
then does the topk-per-image expansion. Does **not** call
`vote_distribution`/`vote_label` (those are batch APIs whose pre-built
input list would itself require the loop). Hyperparameter respect
is the same as offline.

---

## M22 — `FeatureFusionEMA`  (file: `method_22_feature_fusion.py`)

```python
class FeatureFusionEMA:
    def __init__(self, ema_alpha=0.7,
                 prompt_embeddings: torch.Tensor | None = None,
                 prompt_class_names: list | None = None) -> None: ...

    def set_prompt_embeddings(self, prompt_embeddings, prompt_class_names=None): ...

    def update_instance_feature(self, instance_id: int,
                                frame_visual_embedding: torch.Tensor) -> None:
        # EMA: f_t = alpha * f_{t-1} + (1-alpha) * f_current

    def predict_label(self, instance_id: int):
        # (class_name_or_idx, cosine_confidence) for one instance.

    def predict_all(self) -> dict: ...
    def num_instances(self) -> int: ...
    def get_feature(self, instance_id: int) -> torch.Tensor | None: ...
```

State: `instance_features: dict[int, torch.Tensor]` (1D feature per id).

**Streaming pattern (this task):**
Per-frame in `step_frame`:
1. For each confirmed-visible Mask3D proposal, project its vertex set into
   the depth frame to get the 2D AABB.
2. Match AABB → best 2D YOLO bbox by image-resolution IoU.
3. Crop image at that bbox; CLIP-encode with `method_22_encoder.encode_cropped_bboxes`.
4. `fusion.update_instance_feature(proposal_idx, embedding)`.

At finalize (`method_adapters.compute_predictions_method22`):
- Build `(n_proposals × n_classes)` cosine distribution against `_prompt_emb_norm`.
- Top-k expansion over the flattened distribution → `(pred_masks, pred_classes, pred_scores)`.

This mirrors offline `_apply_method_22` in `hooks.py` (lines 434-644), differing only in that the per-frame encode loop is driven by `StreamingScanNetEvaluator._method22_per_frame` instead of the offline per-frame work list reconstructed at finalize.

---

## M31 — `IoUMerger`  (file: `method_31_iou_merging.py`)

```python
class IoUMerger:
    def __init__(self, iou_threshold=0.5, use_kdtree=True,
                 kdtree_neighbor_radius=2.0, same_class_only=True) -> None: ...

    def merge(self,
              predicted_masks: torch.Tensor,    # (n_vertices, K) bool
              pred_classes:    torch.Tensor,    # (K,) long
              pred_scores:     torch.Tensor,    # (K,) float
              vertex_coords:   np.ndarray | None = None,  # (n_vertices, 3)
              ):
        return (kept_masks, kept_classes, kept_scores)
```

Streaming usage: a single batched call at finalize. The kwargs are
**positional or named exactly as above**.

> **First-attempt bug**: the old wrapper called
> `merger.merge(pred_masks=…, scene_vertices=…)` — *both* kwargs are
> misnamed (`pred_masks` vs `predicted_masks`, `scene_vertices` vs
> `vertex_coords`). That raised `TypeError` and would have crashed,
> but it was wrapped in `try…except NotImplementedError`, so the actual
> exception propagated up and matched the "crash expected" half of
> Task 1.4b's 8 failing axes.

---

## M32 — `HungarianMerger`  (file: `method_32_hungarian_merging.py`)

```python
class HungarianMerger:
    def __init__(self, spatial_alpha=0.5, distance_threshold=2.0,
                 semantic_threshold=0.3) -> None: ...

    def build_cost_matrix(self, centroids: np.ndarray, features: torch.Tensor) -> np.ndarray:
        ...

    def merge(self,
              instance_list: list,         # [{"id", "label", "centroid", "bbox_3d?"}, ...]
              instance_features: dict,     # {id: 1D torch.Tensor}
              ) -> list:
        # Returns merged instance dicts {"id", "label", "centroid",
        # "merged_from", "bbox_3d?"}, sorted by surviving id.
```

Streaming usage at finalize (`method_adapters.apply_method32_merge`):
- Build `instance_list` from the current predictions: id = column index,
  label = `pred_classes[k]`, centroid = mean of vertex coords in
  `pred_masks[:, k]`.
- Build `instance_features` keyed by the same column index, looking up
  the underlying proposal index via the `mask_idx` returned from M22
  (or empty dict for spatial-only merging when M22 is not active).
- Class-aware grouping (matches May offline `_apply_method_32`).

> **First-attempt bug**: same as M31 — the wrapper called `merger.merge(pred_masks=…, scene_vertices=…)`, which Hungarian does not accept either, plus it never built `instance_list` at all.

---

## M11/M12 finalize semantics (this task's fix)

In the May offline hooks there is no analog: M11/M12 are entirely new (designed for streaming, Phase 1 / Phase 2 of the registration axis). Their effect is to filter which Mask3D proposals are admitted to the final prediction set.

**First-attempt bug**: the old wrapper updated `_confirmed` during step_frame but never used it at finalize, so the final prediction came from `BaselineLabelAccumulator.compute_predictions()` over every proposal regardless of gate state → identical AP to baseline.

**Fix (`method_adapters.apply_registration_filter`)**: At finalize, take the `mask_idx` mapping (output column k → underlying proposal idx) returned by the accumulator, intersect with `gate._confirmed`, and slice the prediction columns down.

---

## Call-sequence cheat sheet for the streaming wrapper

```
step_frame(t):
    # ... A: D3 visibility, B: YOLO per-frame  ...

    if M11:                 confirmed_visible = M11.gate(visible_instances)
    elif M12:               confirmed_visible = M12.gate(visible_instances)
    else:                   confirmed_visible = visible_instances

    baseline_accumulator.add_frame(...)              # always
    if M22 and M22_encoder:  wrapper._method22_per_frame(...)  # CLIP encode + fusion.update_instance_feature

compute_method_predictions():
    if M21:    (preds, mask_idx) = method_adapters.compute_predictions_method21(accumulator, voter, ...)
    elif M22:  (preds, mask_idx) = method_adapters.compute_predictions_method22(accumulator, fusion, ...)
    else:      preds = compute_baseline_predictions()
               mask_idx = accumulator._last_mask_idx

    if M11:    preds, mask_idx = filter_by_confirmed(preds, mask_idx, M11._confirmed)
    elif M12:  preds, mask_idx = filter_by_confirmed(preds, mask_idx, M12._confirmed)

    if M31:    preds = IoUMerger.merge(predicted_masks=preds.pred_masks, pred_classes=..., pred_scores=..., vertex_coords=scene_vertices)
    elif M32:  preds = apply_method32_merge(preds, HungarianMerger, scene_vertices, mask_idx=mask_idx, instance_features=M22.instance_features if M22 else None)
    return preds
```
