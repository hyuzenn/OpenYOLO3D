# Outdoor open-vocab framing plan — class-aware reliability + open-vocab evaluation

Design sketch only (no implementation in this doc). Motivates and specifies the
two pieces needed to turn the detguided (LiDAR-clustering open-vocab) pipeline
into a defensible Monday narrative.

## 0. Where we are (measured this session, nuScenes v1.0-trainval val)

- **Closed-set anchor:** γ-fixed (native CenterPoint, single-sweep, YOLO bypassed)
  = **mAP 0.3407** / 150 scenes. This is the closed nuScenes-10 ceiling for our
  proposal sources — CenterPoint already covers all 10 scored classes with high
  recall + good localization.
- **detguided wiring:** verified end-to-end (YOLO-World 6-cam → frustum → pillar
  foreground → HDBSCAN → standard proposal dict → CentroidAssociator → M11/M12/
  M21/M31/M32@0.5m). Pipeline integrity PASS.
- **detguided accuracy on nuScenes-10 ≈ 0**, for two reasons gating cannot fix:
  1. **Car-recall ceiling**: ~126 car proposals over 3 scenes (~3/sample vs
     ~30 GT/sample); cluster centroids (visible-surface mean) sit off the GT box
     center → even high-confidence cars miss the center-distance match.
  2. **Closed-set redundancy**: on the 10 scored classes detguided adds nothing
     CenterPoint doesn't already have; its open-vocab boxes are either unscored
     classes or noise.
- **Negative result — global score gate fails** (the key lesson):

  | config | baseline mAP | NDS | emitted/proposals |
  |---|---|---|---|
  | detguided, 10-class prompts, **no gate** | 0.0005 | 0.0205 | 1971/1971 |
  | detguided, **global gate 0.45** | **0.0** | **0.0** | **504/1971** |

  A single global threshold culls *true* low-confidence classes together with the
  barrier false-positive flood, because the low-confidence band is shared:

  | class | n | score p10 | p50 | p90 | nature |
  |---|---|---|---|---|---|
  | car | 126 | 0.446 | **0.730** | 0.862 | high-conf, reliable |
  | barrier | 1505 | 0.130 | **0.275** | 0.525 | low-conf **FP flood** |
  | pedestrian | 107 | 0.135 | **0.285** | 0.674 | genuinely-low-conf **TP** |
  | trailer | 151 | 0.109 | **0.288** | 0.613 | genuinely-low-conf TP |
  | traffic_cone | 35 | 0.109 | **0.170** | 0.337 | genuinely-low-conf TP |
  | construction_vehicle | 21 | 0.114 | **0.150** | 0.155 | genuinely-low-conf TP |

  ⇒ **Reliability cannot be a global cutoff. It must be class-aware (different τ
  per class) and source-aware (never gate the γ recall backbone).** This is the
  empirical justification for the professor's "Reliability-aware aggregation."

---

## 1. Class-aware reliability aggregation (design + pseudocode)

**Principle:** the gate threshold for a class must reflect that class's
*precision profile*, not its raw score. A class that floods false positives at
low score (barrier) needs a HIGH τ; a class whose true detections are inherently
low-score (pedestrian, cone, construction_vehicle) needs a LOW τ or its recall
dies. And γ (CenterPoint) proposals — the recall backbone — are never gated.

```python
# Per-class score floors for the detguided / open-vocab source ONLY.
# Calibrated from the precision profile, not the raw score (see calibration).
#   high τ  -> FP-prone classes (barrier): cull the low-conf flood
#   low  τ  -> genuinely-low-conf-but-valid classes: preserve recall
TAU = {
    "car": 0.45, "bus": 0.40, "truck": 0.40, "bicycle": 0.45,
    "barrier": 0.55,                 # aggressive: kill the low-conf flood
    "pedestrian": 0.15, "trailer": 0.20,
    "traffic_cone": 0.12, "construction_vehicle": 0.12,
    # open-vocab classes get their own calibrated floors (building, tree, ...)
}
DEFAULT_TAU = 0.20

def class_aware_reliability_filter(proposals, tau=TAU, default=DEFAULT_TAU):
    kept = []
    for p in proposals:
        # source-aware: the γ recall backbone is never gated (its cars live
        # at score≈0; a gate would destroy the 0.3407 recall).
        if p.get("_source") == "gamma":
            kept.append(p); continue
        thr = tau.get(p["cls_name"], default)
        if float(p["score"]) >= thr:
            kept.append(p)
    return kept
```

**Calibration (data-driven, replaces the hand-set TAU):** on a held-out val
subset, set each class floor where it actually maximizes that class's quality —
not by eyeballing percentiles.

```python
def calibrate_tau(proposals_with_gt_match, classes, grid=np.arange(0.05, 0.90, 0.05)):
    tau = {}
    for c in classes:
        best_t, best = grid[0], -1.0
        for t in grid:
            # per-class AP (or F1 / precision@target-recall) when keeping
            # only class-c proposals with score >= t
            score = per_class_quality(proposals_with_gt_match, c, t)
            if score > best:
                best, best_t = score, t
        tau[c] = float(best_t)
    return tau
```

**Where it plugs in:** replace the failed *global* `--proposal-score-threshold`
gate with `class_aware_reliability_filter(...)` applied to the detguided portion
inside `_hybrid_union` (γ kept full). Then the union flows unchanged into the
temporal layer (M11/M12/M21/M31/M32@0.5m).

**Adaptive variant (one knob):** normalize each detection by its class median,
`z = (score - median_c) / mad_c`, then a single global gate on `z`. Equivalent
to per-class floors but with one tunable parameter — useful when per-class
calibration data is thin.

---

## 2. Open-vocabulary evaluation protocol (proposal)

**Problem:** nuScenes annotates only the 10 detection classes. The objects our
hybrid uniquely produces — `building`, `tree`, `pole`, `traffic_sign`,
`traffic_light` — have **no GT**, so standard AP is impossible. We do NOT chase
the closed-set 0.3407; we quantify the **open-vocab mapping capability** that
CenterPoint structurally has at **recall 0** (it cannot emit these classes).

Evaluate on three GT-free axes (+ one small audited proxy):

### 2.1 Capability gap (headline)
CenterPoint open-vocab recall ≡ 0 by construction. Report the count of
**temporally-confirmed** open-vocab instances the hybrid produces per scene
(confirmed = survives the M11/M12 gate, i.e. seen ≥N frames). Framing: closed-set
**0** vs hybrid **N consistent open-vocab tracks** — the contribution is the
capability, not a closed-set number.

### 2.2 Spatiotemporal consistency (GT-free — the core)
Open-vocab structures (building/tree/pole) are **static**, so consistency is
measurable without labels:
- **Static-centroid stability**: for each open-vocab track (`global_id`),
  variance of its ego-motion-compensated **global centroid** across frames.
  Low variance ⇒ a real, stably-mapped object; high variance ⇒ a flickering
  noise cluster. Report median per-track centroid std (m).
- **Track persistence ratio**: confirmed-frames / visible-frames per track.
  Real static objects persist; noise flickers. Report mean persistence.
- **Label stability (already instrumented)**: `label_switch_count` and
  `time_to_confirm` over the open-vocab tracks — does the temporal layer keep a
  "tree" labeled "tree"? Directly measures 시공간 일관성, reuses our metrics.py.

### 2.3 CLIP feature consistency (semantic correctness, GT-free)
For each open-vocab 3D instance, project its box into the cameras it appears in,
crop, and CLIP-encode per frame:
- **Intra-track cosine**: mean pairwise cosine of a track's per-frame CLIP crops
  (do views of the same instance stay semantically coherent?).
- **Prompt-alignment cosine**: mean cosine of the track's EMA-fused CLIP feature
  to its assigned text-prompt embedding (is the open-vocab label correct?).
Report both, optionally vs a random-pair baseline to show the signal is real.

### 2.4 Proxy precision (small audited set)
- **2D reprojection agreement**: project each confirmed open-vocab 3D box back to
  all 6 cameras across frames; measure 2D-IoU consistency with the YOLO-World 2D
  detections that spawned it (cross-frame agreement rate).
- **Human precision@k**: on ~10–20 scenes, spot-check the top-k confirmed
  open-vocab boxes (manual, but standard and defensible for a thesis).

### 2.5 Protocol summary
| axis | metric | GT needed |
|---|---|---|
| capability gap | # confirmed open-vocab tracks/scene (vs CenterPoint 0) | none |
| spatial consistency | global-centroid std (m); track persistence ratio | none |
| temporal label stability | lsc, ttc over open-vocab tracks | none |
| semantic consistency | intra-track + prompt-alignment CLIP cosine | none |
| proxy precision | 2D reprojection IoU agreement; human P@k (small set) | light/manual |

---

## 3. Monday narrative (defensible)

1. **Closed-set anchor:** γ-fixed CenterPoint = 0.3407 (the strong closed-set
   baseline; our temporal layer is validated on top of it).
2. **The open-vocab cost is real and quantified:** naive open-vocab labeling
   collapses closed-set mAP (γ-pipeline 0.0526; detguided ≈ 0) — the YOLO-relabel
   bottleneck — and a **global** reliability gate makes it worse (0.0005 → 0,
   measured), proving reliability must be class/source-aware.
3. **Contribution:** Hybrid proposal (γ recall backbone ∪ detguided open-vocab
   breadth) + **class-aware/source-aware** reliability aggregation + temporal
   consistency layer → open-vocab 3D detection/mapping that CenterPoint cannot do
   (recall 0), evaluated by the §2 protocol — **not** by a closed-set mAP we
   already know the hybrid can't win.

This positions γ-fixed 0.3407 as the anchor and the hybrid as the open-vocab
extension (the project's identity: real-time open-vocab 3D mapping), with the
negative results (relabel bottleneck, global-gate failure) as evidence that the
class-aware reliability + temporal machinery are necessary, not optional.
