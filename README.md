# SemWorld-3D

**Streaming Open-Vocabulary 3D Instance Mapping for Outdoor Driving Scenes**

A research project initially extending [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D) from indoor RGBD scenes to outdoor autonomous driving scenarios on the nuScenes dataset. After a 5-stage diagnostic measurement campaign (May 2026), the method direction has been revised based on data; see [Diagnosis findings](#diagnosis-findings-may-2026) below.

> ⚠️ **Status**: Method design is being revised after diagnosis. Targeting CVPR 2027 submission (Nov 2026 deadline).

---

## Motivation

Existing outdoor open-vocabulary 3D perception work (OV-Uni3DETR, OpenSight, OV-SCAN, FM-OV3D) targets **per-frame 3D detection** — they output bounding boxes, not a temporally consistent instance memory. POP-3D works at the **occupancy/voxel level**, not at instance granularity. Indoor real-time open-vocab instance segmentation methods (OpenYOLO3D, OpenMask3D, ConceptFusion) assume dense RGBD and a static, single-camera setup.

The empty slot:

> **Streaming, instance-level, open-vocabulary 3D mapping for outdoor driving scenes — with sparse LiDAR, 6-camera surround-view, dynamic objects, and ego-motion, all under a real-time constraint.**

This is the problem SemWorld-3D addresses.

## Why outdoor breaks indoor pipelines

Moving from indoor (Replica, ScanNet) to outdoor (nuScenes) is not a dataset swap. It breaks three core assumptions of indoor open-vocab 3D pipelines:

1. **Lifting assumption**: OpenYOLO3D lifts 2D detections to 3D using dense per-pixel depth from RGBD. nuScenes provides 32-beam sparse LiDAR — a 2D bounding box may contain anywhere from 0 to a few projected LiDAR points, especially at distance.
2. **Static-scene assumption**: Indoor instance association assumes scene geometry is stable between frames. nuScenes contains moving cars, pedestrians, cyclists.
3. **Single-view assumption**: Indoor scenes typically use a forward-facing camera. nuScenes uses 6 surround cameras with overlapping fields of view.

These three failure modes drive the method design.

---

## Diagnosis findings (May 2026)

A 5-stage diagnostic measurement campaign was conducted on nuScenes trainval keyframe data to quantify how indoor pipelines fail outdoors. Each stage was designed to test a specific hypothesis with explicit pass/fail thresholds.

### Setup-level findings (Tier 1 / 2 / 2-Extended)

Measured on 100 trainval samples (n=100, seed=42).

| Finding | Value | Implication |
|---|---|---|
| Per-pixel depth coverage in detection box | **0.20%** | Per-pixel lifting is meaningless; per-instance aggregation only |
| Detection-induced GT loss | **44.8%** | 2D detection recall hole is the dominant source of missing instances |
| GT distribution beyond 30m | **52%** | Operating regime is inverted vs indoor; far-range performance matters |
| Multi-view GT fraction (≥2 cams visible) | **8.6%** | Multi-view consistency is sparse on nuScenes; cannot serve as primary reliability axis |
| Detector-geometry gap (cams that see GT but detector misses) | **0.63 cams** | Robust to sample diversity; indicates systematic detector limitation, not sample bias |
| Mask3D (indoor-trained) instances per frame | **2.9** | Severe under-segmentation; cannot serve as proposal source |

### Proposal-level findings (W1 / W1.5 / Step 1 / β1 / β1.5 / Step A)

Measured on 50 samples (mini 20 + trainval 30, seed=42 fixed across all stages).

| Stage | Configuration | M-rate (1↔1 GT match) | Verdict |
|---|---|---|---|
| Mask3D (Tier 1 reference) | indoor-trained, no tuning | **0.3%** | Decision A: replaced |
| W1 — HDBSCAN baseline | default | 9.1% | over-segmentation: 31 clusters |
| W1.5 — extended sweep | 84 combos × ground filter modes | 28.5% | grid-end optimum, structural limit suspected |
| Step 1 — Mask3D vs HDBSCAN comparison | head-to-head + hybrid simulation | Mask3D 0.3% / HDBSCAN 23.2% / Hybrid +1.0pt | Mask3D replacement justified by data; hybrid yields no marginal gain |
| β1 — pillar foreground extraction | sweep, 24 combos | **36.1%** | sweet spot, +7.6pt over W1.5 baseline |
| β1.5 — verticality filter | sweep, 27 combos | 27.4% | over-engineering: −8.7pt vs β1 |
| Step A — pillar resolution sweep (final ceiling test) | 12 combos, resolution 0.2–0.5m | 36.1% | **plateau confirmed**; geometry-only family ceiling |

### What the data tells us

1. **Geometry-only proposal generation has a ceiling at ~36% on nuScenes.** Five different geometric refinements (HDBSCAN tuning, ground filtering, pillar foreground, verticality filter, pillar resolution) all converge to the same upper bound.
2. **β1.5 over-filtering analysis**: 95% of the small components removed by β1.5 were genuinely outside GT boxes (i.e., real noise). The −8.7pt loss came from larger-component aggregation, not from removing real signal. This rules out the "input refinement was too aggressive" hypothesis and points to algorithmic limits.
3. **The central design question is "where to define instances", not "how to cluster better".** This is the same axis as POP-3D's contribution: instance definition can be anchored in geometry (HDBSCAN-style), in semantics (POP-3D-style), or in a hybrid arbitration. The 5-stage diagnosis effectively rules out the geometry-only anchor.
4. **Detection recall hole is the dominant outdoor failure mode (44.8%), not sparsity per se.** Sparsity defines the *space* in which reliability is measured; detection failure defines the *target* that reliability must compensate for.

---

## Method direction (revised)

> The original direction (Decision A: HDBSCAN as proposal source + reliability-aware fusion) was data-validated for its first decision (Mask3D replacement) but the second decision (HDBSCAN as the sole geometry-first proposal) hit a 36% ceiling. The revision below preserves the geometry-first principle but moves the arbitration boundary.

### Three-tier narrative

- **(1) Problem**: Detection recall hole is the dominant source of outdoor open-vocab loss.
- **(2) Structure**: Geometry-first proposal + reliability-aware fusion.
- **(3) Role separation**:
  - **Coverage** = observability (sparse LiDAR-based, defines where instances *can* be observed)
  - **Detection** = semantics (2D open-vocab, defines what instances *are*)
  - **Reliability** = arbitration (decides which source to trust per instance)

### Action principle

> *In outdoor open-vocab mapping, do not trust the detector — verify and augment it with geometry.*

### Pipeline (3 stages)

1. **Proposal (geometry-first)**: LiDAR clustering (currently being revised; pillar foreground + 2D-detection-guided clustering under evaluation in Stage 4)
2. **Reliability**: Coverage-based score with optional multi-view anchor bootstrap; uniformity treated as ablation
3. **Fusion (3-case)**:
   - **M (matched)**: 2D detection ∩ LiDAR cluster spatial overlap → high reliability
   - **L (LiDAR-only)**: cluster without detection match → cross-camera search → unsupervised low-reliability hold
   - **D (detection-only)**: detection without cluster → 4-step recovery (multi-cam search, projection, promotion, low-reliability fallback)

### What the diagnosis ruled out

- ❌ Mask3D as proposal source (M=0.3% confirmed)
- ❌ Range as a primary reliability component (Tier 1 Fig 2: weak predictor)
- ❌ Multi-view as a primary reliability component (8.6% multi-view ratio)
- ❌ Per-pixel lifting (0.2% pixel coverage)
- ❌ Pure HDBSCAN as proposal source (36% ceiling)
- ❌ Verticality filter on pillar foreground (over-engineering, −8.7pt)

### What is preserved

- ✅ OpenYOLO3D's real-time philosophy (no CLIP recomputation, projection-only reliability, EMA per-instance accumulation)
- ✅ Open-vocabulary capability via 2D detector (YOLO-World)
- ✅ Coverage as primary reliability component (single strong-signal axis from diagnosis)
- ✅ 3-case fusion targeting the 44.8% detection-induced loss directly

---

## Expected contributions

> Contributions are being finalized as Stage 4 progresses. Current direction:

- **Diagnosis**: First systematic measurement of how indoor open-vocab 3D pipelines fail on outdoor driving data, with a 5-stage hypothesis-driven measurement protocol (released with the codebase).
- **Method**: Geometry-first proposal with reliability-aware 3-case fusion targeting detection-induced loss as the dominant outdoor failure mode.
- **Real-time on outdoor open-vocab**: A working pipeline that maintains real-time inference on nuScenes, contrasting with existing outdoor open-vocab methods that prioritize accuracy without explicit real-time guarantees.

Method module details will be added to [`docs/METHOD.md`](docs/METHOD.md) after Stage 4 (Method design refinement) is complete.

## Tech stack

- **Base**: OpenYOLO3D's real-time philosophy (core not modified; original Mask3D + indoor lifting replaced)
- **2D open-vocabulary detector**: YOLO-World
- **3D proposal**: under evaluation in Stage 4 (pillar foreground baseline + 2D-detection-guided variant)
- **Visual-language embeddings**: CLIP-family encoder for instance-level features
- **Dataset**: nuScenes (`v1.0-mini` for development, `v1.0-trainval` for full experiments)
- **Framework**: PyTorch
- **Hardware**: NVIDIA A100

## Evaluation plan

| Aspect | Metric |
|---|---|
| Detection accuracy | mAP, NDS at the instance level on nuScenes |
| Open-vocab capability | Performance on novel categories (following OV-SCAN / OpenSight protocols) |
| Distance-stratified | Per-bin mAP (0–10, 10–20, 20–30, 30–50, 50m+) — required given the 52% >30m GT distribution |
| Temporal consistency | ID switches (IDS), label stability across frames, association accuracy |
| Real-time | FPS, end-to-end latency per frame, peak GPU memory |

Comparison targets: OpenSight, OV-SCAN, FM-OV3D (outdoor open-vocab); OpenYOLO3D (indoor real-time reference); ablations of our own modules.

## Roadmap

| Stage | Period | Status | Deliverable |
|---|---|---|---|
| 1. nuScenes dataloader | May 2026 | ✅ Done | `dataloaders/nuscenes_loader.py`, calibration verified |
| 2. OpenYOLO3D integration smoke test | May 2026 | ✅ Done | Adapter, end-to-end run on nuScenes frames |
| 3. Diagnosis (Tier 1 / 2 / 2-Ext) | May 2026 | ✅ Done | Setup-level failure quantified; reports under `results/diagnosis*/` |
| 3.5. Proposal diagnosis (W1 / W1.5 / Step 1 / β1 / β1.5 / Step A) | May 2026 | ✅ Done | Geometry-only ceiling confirmed at 36% M-rate |
| 4. Method design refinement | May–Jun 2026 | 🚧 In progress | Hybrid proposal direction under evaluation; `docs/METHOD.md` to follow |
| 5. Implementation (core modules) | Jun–Jul 2026 | Pending | Working method on nuScenes |
| 6. Full experiments + ablation | Aug 2026 | Pending | Results on nuScenes trainval |
| 7. Graduation thesis presentation | Sep 2026 | Pending | Demo + thesis writeup |
| 8. Paper writing + revision | Oct 2026 | Pending | CVPR 2027 draft |
| 9. CVPR 2027 submission | Nov 1, 2026 | Pending | Final submission |

## Repository structure

```
.
├── dataloaders/
│   ├── nuscenes_loader.py          # 8-key dict interface for nuScenes
│   └── sanity_check.py             # LiDAR→camera projection visual check
├── adapters/
│   ├── nuscenes_to_openyolo3d.py   # nuScenes → OpenYOLO3D scene_dir adapter
│   └── lidar_proposals.py          # LiDARProposalGenerator (HDBSCAN + ground filter modes)
├── preprocessing/
│   ├── pillar_foreground.py        # PillarForegroundExtractor (β1)
│   └── verticality_filter.py       # VerticalityFilter (β1.5; reference, not in active path)
├── diagnosis/                      # Tier 1
├── diagnosis_tier2/                # Tier 2 + 2-Extended
├── diagnosis_w1/                   # W1: HDBSCAN baseline
├── diagnosis_w1_5/                 # W1.5: extended sweep + distance-stratified
├── diagnosis_step1/                # Step 1: Mask3D vs HDBSCAN vs Hybrid
├── diagnosis_beta1/                # β1: pillar foreground sweep
├── diagnosis_beta1_5/              # β1.5: verticality filter sweep
├── diagnosis_step_a/               # Step A: pillar resolution sweep (ceiling test)
├── configs/
│   ├── nuscenes_baseline.yaml
│   └── nuscenes_trainval.yaml
├── docs/
│   ├── CONTEXT.md
│   ├── BASELINE.md
│   ├── NUSCENES_SETUP.md
│   └── method_drafts/              # Pre-diagnosis ideas (reference only)
├── results/
│   ├── diagnosis/                  # Tier 1 outputs
│   ├── diagnosis_tier2/
│   ├── diagnosis_tier2_trainval/
│   ├── w1_clustering_check/
│   ├── w1_5_diagnostic_sweep/
│   ├── diagnosis_step1/
│   ├── diagnosis_beta1/
│   ├── diagnosis_beta1_5/
│   └── diagnosis_step_a/
└── [OpenYOLO3D core files — untouched]
```

## Setup

See [`docs/NUSCENES_SETUP.md`](docs/NUSCENES_SETUP.md) for nuScenes data layout and environment setup. The development environment `openyolo3d-dev` is a clone of OpenYOLO3D's `openyolo3d` env with `nuscenes-devkit` and `hdbscan` added.

## Branching policy

- `main`: stable, integrated stages only
- `feature/diagnosis`: Tier 1 / 2 / 2-Ext (merged)
- `feature/method-w1` … `feature/step-a-pillar-resolution`: per-stage branches for proposal diagnosis (kept for provenance)
- Stages merge to `main` only after acceptance criteria are verified end-to-end.

## License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Mask3D](https://github.com/JonasSchult/Mask3D), and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

This work is conducted as a graduation research project at [Institution].

---

**Contact**: [Author info]
