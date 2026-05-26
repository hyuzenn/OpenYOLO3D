# Task 3.0 — Indoor method implementation audit (M21/M22/M31/M32)

Date: 2026-05-21 · Scope: **read-only**, zero code changes.
Goal: for each method, separate *documented intent* (docstring / prior task
docs — cited, never guessed) from *actual code*, judge streaming suitability,
hyperparameter sensitivity, Outdoor applicability, and assign a category:

- **(A)** implementation problem → fix has high value, re-measure
- **(B)** implementation correct, hyperparameter mistuned → sweep
- **(C)** implementation correct, fundamental limit → report as negative ablation

> Reported Indoor numbers (M21 lsc +1.04%; M22 standalone AP −27.2% / lsc
> +110%; M22+M32 AP −49% / lsc +342%; M22 offline −0.0285 → streaming
> −0.0532; M32 absorption) are taken from the user's Task 1.4b/1.4d ablation
> and the user's own analysis scripts (`method_scannet/analysis/`). They are
> cited as given; this audit explains them at the code level.

**Top-line finding:** at the *class-implementation* level, none of
M21/M22/M31/M32 is category (A) — each matches its docstring and was
verified non-no-op in the Indoor 1.4a smoke (10/10 PASS,
`docs/task_1_4a_redesign_notes.md`). The only category-(A) defect is in the
**Outdoor evaluator wiring** (§2.5), a different layer. So the Indoor
negatives are genuine behaviour of correct code → (B)/(C), not bugs.

---

## 2.1 M21 — `WeightedVoting` (label axis, Step D)

**Documented intent** (`method_21_weighted_voting.py:1-15`): replace
OpenYOLO3D's uniform pixel-mode aggregation with a per-(instance,frame)
histogram weighted by
`w = (α·exp(-d3D/D) + (1-α)·exp(-d2D_center/C)) · conf`, D=10 m, C=300 px,
α=0.5. Goal (axis = "stabilize labels", `pipeline_analysis §7`): down-weight
far / off-center / low-IoU observations.

**Actual code:** `frame_weight` (`:35-54`) implements the formula exactly;
`vote_distribution` (`:56-94`) builds the max-normalized weighted histogram
with the OpenYOLO3D background fallback. The streaming adapter
`compute_predictions_method21` (`method_adapters.py:74-299`) and
`RunningInstanceLabeler._compute_m21_weight` (`running_labeler.py:152-212`)
both **inline the identical formula** using the live voter's
hyperparameters — matching offline `hooks.py`.

**Intent vs actual:** **match.** Code = docstring formula, hyperparameters
respected on both the batch and per-frame paths.

**Streaming suitability — the source of the wrong-sign lsc.** Per-frame
snapshots take the argmax of the *cumulative* weighted histogram
(`running_labeler.snapshot`). Under uniform voting, early votes accumulate
and the argmax locks in. Under distance-weighting, a single **late, near,
high-weight** frame can outweigh many early far votes and **flip** the
cumulative argmax → a label switch. So weighting can *increase* lsc rather
than decrease it — consistent with the reported **lsc +1.04%** (small, and
the wrong direction for a "stabilizer"). This is an inherent tension of
weighted-cumulative-argmax in streaming, not a coding error.

**Hyperparameter sensitivity — high.** The decay constants set how "spiky"
weights are. Large D, C → weights flatten → reduces to baseline (no late
flips, lsc → ~0). Small D, C → late high-weight frames dominate → more
flips. The default (D=10 m, C=300 px) is an *indoor offline* tuning; it has
never been tuned against the streaming lsc objective.

**Outdoor applicability:** the weighting (3D camera↔centroid distance,
2D center offset, IoU conf) is well-defined for the LiDAR+6-camera setup. In
the native classifier path (Mode 2) there is no per-frame pixel-label vote,
so M21 would have to attach to the YOLO-relabel path (Mode 1), where it is
currently a no-op (§2.5).

**Category: (B)** — correct implementation, decay hyperparameters mistuned
for the streaming stabilization objective (and a structural late-flip
tension worth noting). A D/C/α sweep against lsc is the natural test.

---

## 2.2 M22 — `FeatureFusionEMA` (label axis, Step D)

**Documented intent** (`method_22_feature_fusion.py:1-18`): per-instance
CLIP visual-feature EMA `f_t = α·f_{t-1} + (1-α)·f_cur` (α=0.7), final label
= `argmax_c cos(f_inst, prompt_emb[c])`.

**Actual code:** `update_instance_feature` (`:85-110`) is a correct EMA;
`predict_label` (`:125-143`) is correct L2-normalized cosine argmax. The
streaming adapter `compute_predictions_method22` (`method_adapters.py:307-382`)
builds the (K×n_classes) cosine distribution + top-k expansion;
`wrapper._method22_per_frame` (`wrapper.py:489-560`) does the per-frame crop
→ CLIP-encode → `update_instance_feature`.

**Intent vs actual:** **match** on the EMA + cosine mechanism.

**Two compounding limits (both grounded in the user's own analysis):**

1. **CLIP narrow band — fundamental.** `analysis/m22_clip_narrow_band.py`
   computes the 200×200 prompt-embedding cosine matrix and shows the
   off-diagonal band is high (poor inter-class separation; the ~0.759
   nearest-neighbour cosine the user cites). When classes are barely
   separable in CLIP text-image space, cosine argmax mislabels regardless of
   feature quality. This caps M22 AP and is **(C)** — a property of the
   embedding geometry, not the code.

2. **Streaming EMA dilution — adaptation gap.** Offline M22 used a top-15
   best-frame cap (`analysis` / `hooks.py` note); the streaming path EMAs
   **every** visible frame's crop (`_method22_per_frame` has no
   quality/confidence gate). Poor crops drift the running feature →
   worse-than-offline (reported **offline −0.0285 → streaming −0.0532**) and
   the cosine argmax flips frame-to-frame (**lsc +110%**). α=0.7 is also a
   weak smoother for a per-frame stream. This part is **(B)** — fixable by a
   confidence-gated update and/or higher α.

**Streaming suitability:** poor as-is — no frame selection, low α → drift +
instability. The EMA is *structurally* streamable (it's the offline top-K
selection that didn't port).

**Outdoor applicability:** **N/A / undefined** for the native LiDAR
classifier (Mode 2 has no CLIP visual feature per instance). This is the
code-level basis for the Step 2b "defer / N/A" decision on M22 outdoor —
confirmed: M22 requires a CLIP image-feature source the native path lacks.

**Category: (C) primary (CLIP narrow band AP ceiling) + (B) contributing
(EMA dilution / α / no frame gate drive the −0.0532 and lsc +110%).**
Report the narrow band as a negative ablation; a frame-gate + α sweep can
recover the streaming-vs-offline gap but not the CLIP ceiling.

---

## 2.3 M31 — `IoUMerger` (spatial-merge axis, Step F)

**Documented intent** (`method_31_iou_merging.py:1-22`): class-aware 3D
vertex-set-IoU NMS — same-class predictions with mask-IoU ≥ threshold (0.5)
collapse to the higher-score one; optional KDTree centroid pre-filter.

**Actual code:** correct greedy per-class NMS (`_nms_within_group`
`:98-175`), vertex-set IoU via boolean `&`/`|`, keep-higher-score, empty
masks dropped. The adapter `apply_method31_merge` (`method_adapters.py:390-407`)
calls it with the exact signature.

**Intent vs actual:** **match.**

**Why Indoor null (lsc −0.11%):** the merge only fires on *same-class,
high-IoU (≥0.5) overlapping duplicates*. Indoor Mask3D proposals (already
NMS'd upstream) rarely contain such duplicates per scene region, so M31
seldom triggers → near-zero effect. The implementation is correct; the
**precondition is absent in indoor data**.

**Hyperparameter sensitivity — moderate.** `iou_threshold=0.5` is strict; a
lower threshold would merge more (and risk distinct objects). `same_class_only`
and `kdtree_neighbor_radius=2.0 m` also gate firing.

**Outdoor applicability — potentially higher.** Outdoor produces duplicate
boxes (multi-camera relabel, multi-sweep, far-range fragmentation) and
larger objects (vehicles ~5–10 m), so same-class overlapping duplicates are
more common than indoor → M31 may have real effect. (Native γ already has
CircleNMS in-head, so test on the YOLO-relabel/union paths.)

**Category: (C) on indoor data** (correct, but the duplicate pattern it
targets is rare indoors → null), **with a (B) iou_threshold caveat.**
Most promising as an Outdoor experiment rather than an indoor fix.

---

## 2.4 M32 — `HungarianMerger` (spatial-merge axis, Step F)

**Documented intent** (`method_32_hungarian_merging.py:1-23`): merge
fragments of one instance via Hungarian assignment on
`cost = α·||c_i-c_j|| + (1-α)·(1-cos(f_i,f_j))`, masking pairs beyond
`distance_threshold` (2 m) or below `semantic_threshold` (0.3) to +∞;
union-find clusters; smallest id wins.

**Actual code:** correct cost matrix (`build_cost_matrix:88-122`), correct
masking, `linear_sum_assignment`, union-find (`merge:126-257`). The adapter
`apply_method32_merge` (`method_adapters.py:415-506`) builds instance_list +
feature dict and, when no features exist, rebuilds the merger with
`semantic_threshold=-1.0` (mirrors offline; fixes the M32-only no-op).

**Intent vs actual:** **match.**

**Why catastrophic (standalone AP −27%, M22+M32 AP −49% / lsc +342%):** the
user's own `analysis/m32_multi_instance.py` parses ScanNet GT and shows that
at the **2 m** threshold a large fraction of *same-class, distinct*
within-scene instance pairs fall under 2 m (the ~37.6% absorption the user
cites) → M32 **merges genuinely different GT instances** → recall collapses
→ AP crashes. 2 m is simply too large for indoor object spacing
(objects legitimately sit 0.5–2 m apart). In the M22+M32 cascade, M22's
already-noisy CLIP features feed the semantic term → wrong merges compound,
and the merge re-labels/re-ids instances → lsc +342%.

**Hyperparameter sensitivity — very high, dominant.** `distance_threshold`
is the lever: at indoor scale 2 m over-merges; a smaller value (e.g.
0.3–0.5 m) would absorb far fewer distinct instances. `semantic_threshold`
and α also matter, especially in the cascade.

**Outdoor applicability — threshold-scale-dependent.** Vehicles sit
~5–10 m apart, so 2 m may *under*-merge large objects but **over**-merge
dense small objects (e.g. traffic cones / pedestrians ~0.3–1 m apart). The
threshold cannot be a single global constant across object scales; an
Outdoor port needs per-class or size-relative distance thresholds.

**Category: (B)** — correct implementation, `distance_threshold=2 m`
mistuned for indoor object density (quantified by the user's absorption
analysis). A threshold sweep (and per-class thresholds for Outdoor) is the
direct test.

---

## 2.5 Outdoor evaluator silent no-op (Step 2b finding)

**Defect (category-(A), but in the *wiring*, not the method classes):**
`StreamingNuScenesEvaluator` (`nuscenes_evaluator.py`) installs any axis via
`install_axis → install_method_streaming` (`:558-566`), so passing `"M21"`,
`"M22"`, `"M31"`, `"M32"`, `"phase1"`, or `"phase2"` **does** attach
`self.method_21/22/31/32` (the installer is generic, `hooks_streaming.py`).
But the evaluator only ever **reads** M11/M12:

- `__init__` declares only `self.method_11`, `self.method_12` (`:547-548`).
- `setup_scene` resets only those (`:591-594`).
- `step_sample` consults **only** M11/M12 for the registration gate
  (`:771-774`). Labels come from YOLO-World via
  `running_labeler.add_vote(yolo_cls_idx)` (`:743-752`) — **not M21/M22**.
  There is **no spatial-merge call** anywhere — **not M31/M32**.

**Consequence:** an axis run labeled `M21/M22/M31/M32` in the Outdoor
pipeline produces **baseline-identical** output, **with no error or
warning**. `phase1` (M11+M21+M31) reduces to M11-only; `phase2`
(M12+M22+M32) reduces to M12-only.

**Exact missing hooks** (vs the Indoor `wrapper.py:566-646` which calls all
six): `step_sample` would need, after the YOLO vote loop and before/after
the M11/M12 gate, the equivalents of (i) `compute_predictions_method21` /
`_method22` for the label axis and (ii) `apply_method31/32_merge` for the
merge axis. None are present.

**Why a May/Stage-C sanity check could miss it:** the install path is real
(the attribute gets set), the run completes cleanly, and the emitted numbers
are valid mAP — just equal to baseline. Only a **baseline-vs-M21 fingerprint
diff** (the exact check the Indoor 1.4a redesign added) would have exposed
it; a per-axis "did it run / did it crash" check passes.

**Honest §6.4 note:** any "Stage C M21/M22/M31/M32" Outdoor row is invalid —
it measures baseline, not the method. Only Outdoor M11/M12 (registration)
are validly measured in Stage C. The Indoor M21/M22/M31/M32 numbers are the
valid ones (wrapper.py fully wired).

This is **separate** from §2.1–2.4: the method *classes* are correct; the
Outdoor *evaluator* never calls them.

---

## 2.6 Verdict table + recommendations

| Method | Intent vs code | Streaming fit | Dominant lever | Category |
|--------|----------------|---------------|----------------|----------|
| M21 | match | late-flip tension | decay D/C, α | **(B)** hyperparam |
| M22 | match | poor (drift, no frame gate) | CLIP band (C) + α/gate (B) | **(C)+(B)** |
| M31 | match | n/a (batch finalize) | iou_threshold; data lacks duplicates | **(C)** indoor / (B) thresh |
| M32 | match | catastrophic over-merge | distance_threshold 2 m | **(B)** hyperparam |
| Outdoor wiring | **mismatch** | M21/22/31/32 never called | add hooks to step_sample | **(A)** (wiring, not class) |

**Cross-check with Step 2b:** no category-(A) defect exists in the method
*classes* → fixing a class would not change Step 2b. The single (A) defect is
the Outdoor evaluator wiring (§2.5), which only matters if the user wants
Outdoor label/merge axes — and even then M22 is N/A outdoor (no CLIP
features). So Step 2b can proceed on its current native-γ + M11/M12 basis;
the (B)/(C) findings inform paper framing, not Step 2b blocking.

**Recommended next steps (user decides; no fix performed here):**
1. **M32 (B)** — highest value: `distance_threshold` sweep (e.g.
   0.3/0.5/1.0/2.0 m) on Indoor; per-class/size-relative threshold for any
   Outdoor port. The user's absorption analysis predicts large gains below 2 m.
2. **M21 (B)** — D/C/α sweep against lsc; expect lsc to approach baseline as
   weights flatten (confirms the late-flip mechanism).
3. **M22** — (B) part: add a confidence/IoU gate to `_method22_per_frame`
   and raise α, re-measure the streaming-vs-offline gap; (C) part: report the
   CLIP narrow band (user's analysis figures) as a negative ablation.
   Outdoor: keep deferred / N/A.
4. **M31** — re-frame as an Outdoor experiment (duplicate boxes across
   cameras/sweeps); on Indoor, report as correct-but-null (or a quick
   iou_threshold sweep).
5. **Outdoor wiring (A)** — separate task: decide whether to wire the label
   (M21) / merge (M31/M32) axes into `nuscenes_evaluator.step_sample`. M22 is
   N/A outdoor. This is the only code-fix item, and it is gated on whether
   Outdoor label/merge ablations are wanted for the paper.

**Paper §5.4 robustness:** the Indoor M22 result rests on the user's own
narrow-band analysis (a fundamental (C) limit) plus a streaming dilution (B)
— state both so the negative is attributed correctly, not read as a bug.

---

## Acceptance (Stage 2)
- 2.1–2.4 M21/M22/M31/M32 verified (intent vs code, streaming, hyperparam, outdoor): ✓
- 2.5 Outdoor `nuscenes_evaluator.py` silent no-op diagnosed (exact lines): ✓
- 2.6 A/B/C categories assigned + recommendations: ✓
