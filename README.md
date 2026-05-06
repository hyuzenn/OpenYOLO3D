# SemWorld-3D

**Streaming Open-Vocabulary 3D Instance Mapping for Outdoor Driving Scenes**

A research project initially extending [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D) from indoor RGBD scenes to outdoor autonomous driving scenarios on the nuScenes dataset. After a 7-stage diagnostic measurement campaign (May 2026), the method direction has been substantially revised; see [Diagnosis findings](#diagnosis-findings-may-2026) below.

> ⚠️ **Status**: Method direction is finalized after diagnosis; hybrid proposal simulation in progress. Targeting CVPR 2027 submission (Nov 2026 deadline).

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

A **7-stage hypothesis-driven measurement campaign** was conducted on nuScenes trainval keyframe data. Each stage tested a specific hypothesis with explicit pass/fail thresholds, isolating one variable at a time.

### Setup-level findings (Tier 1 / 2 / 2-Extended)

Measured on 100 trainval samples (n=100, seed=42).

| Finding | Value | Implication |
|---|---|---|
| Per-pixel depth coverage in detection box | **0.20%** | Per-pixel lifting is meaningless; per-instance aggregation only |
| Detection-induced GT loss | **44.8%** | 2D detection recall hole is the dominant source of missing instances |
| GT distribution beyond 30m | **52%** | Operating regime is inverted vs indoor; far-range performance matters |
| Multi-view GT fraction (≥2 cams visible) | **8.6%** | Multi-view consistency is sparse; cannot serve as primary reliability axis |
| Detector-geometry gap (cams that see GT but detector misses) | **0.63 cams** | Robust to sample diversity; systematic detector limitation |
| Mask3D (indoor-trained) instances per frame | **2.9** | Severe under-segmentation; cannot serve as proposal source |

### Proposal-level findings (7 stages)

Measured on 50 samples (mini 20 + trainval 30, seed=42 fixed across all stages).

| Stage | Method | M-rate (1↔1 GT match) | Verdict |
|---|---|---|---|
| **W1** — HDBSCAN baseline | default config | 9.1% | over-segmentation: 31 clusters/frame |
| **W1.5** — extended sweep | 84 combos × ground filter modes | 28.5% | grid-end optimum; structural limit |
| **Step 1** — Mask3D vs HDBSCAN comparison | head-to-head + hybrid simulation | Mask3D 0.3% / HDBSCAN 23.2% | Mask3D replacement justified by data |
| **β1** — pillar foreground extraction | 24-combo sweep | **36.1%** | sweet spot, +7.6pt over baseline |
| **β1.5** — verticality filter | 27-combo sweep | 27.4% | over-engineering (−8.7pt) |
| **Step A** — pillar resolution sweep (ceiling test) | 12 combos, resolution 0.2–0.5m | 36.1% | **plateau confirmed** at β1 sweet spot |
| **Option 5** — 2D detection-guided clustering | 27 combos, frustum + HDBSCAN | 15.1% | wrong axis: per-frustum HDBSCAN over-segments |
| **γ** — CenterPoint (learned 3D detector) | 8 combos, score sweep | 35.2% | **also plateaus near 36% ceiling** |

### What the data tells us

**1. The 36% M-rate is a fundamental ceiling, not an algorithmic limit.**

Seven different proposal generation strategies — unsupervised geometric clustering (HDBSCAN), pillar-based foreground extraction, verticality filtering, multi-resolution pillar sweep, 2D-detection-guided per-frustum clustering, and a fully learned 3D detector (CenterPoint) — all plateau in the same 35–36% region. This is the GT-LiDAR coverage ceiling on nuScenes: the combination of sparse 32-beam LiDAR with ~52% of GT objects beyond 30m bounds what *any* proposal source can recover from a single keyframe.

**2. Distance bias is opposite between learned and unsupervised proposals.**

| Distance bin | β1 (unsupervised geometric) M-rate | γ (CenterPoint, learned) M-rate |
|---|---|---|
| 0–10m | 34.8% | **71.5%** (+36.7pt) |
| 10–20m | 35.8% | 49.2% |
| 20–30m | 40.2% | 39.2% |
| 30–50m | 23.1% | 24.4% |
| 50m+ | 15.5% | 4.7% |

CenterPoint is dominant in dense near-range (training data distribution), while β1 holds up in sparse far-range (geometric clustering still works with few points). On per-sample paired comparison: γ wins 19/50, β1 wins 17/50, similar 14/50 — strong complementarity. **This is the key signal that motivates the hybrid direction.**

**3. Detection recall hole is the dominant outdoor failure mode (44.8%), not sparsity per se.**

Sparsity defines the *space* in which reliability is measured; detection failure defines the *target* that reliability must compensate for. The 2D detection recall by distance (74.7% at 0–10m → 9.8% at 50m+) is itself a fundamental upper bound for any method that depends on 2D detection — including the original OpenYOLO3D pipeline.

**4. The central design question is "where to define instances", not "how to cluster better".**

This is the same axis as POP-3D's contribution. The 7-stage diagnosis effectively rules out the geometry-only anchor (β1 ceiling), the detection-only anchor (Option 5 fails as wrong axis), and the learned-dense-detector anchor in isolation (γ plateaus). The data points to a hybrid arbitration: geometry and detection contribute to different distance regimes, and reliability scores arbitrate between them.

---

## Method direction

### Three-tier narrative

- **(1) Problem**: Detection recall hole is the dominant source of outdoor open-vocab loss; the geometry-only ceiling is a fundamental GT-LiDAR coverage limit.
- **(2) Structure**: Geometric proposal **augmented with learned detector proposals**, with reliability-aware fusion that arbitrates by distance regime.
- **(3) Role separation**:
  - **Coverage** = observability (sparse LiDAR-based, defines where instances *can* be observed)
  - **Detection** = semantics (2D open-vocab, defines what instances *are*)
  - **Reliability** = arbitration (distance-aware weighting between proposal sources)

### Action principle

> *In outdoor open-vocab mapping, no single proposal source breaks the 36% ceiling on nuScenes. The fix is not a better proposal generator but a reliability-aware arbitration between complementary sources.*

### Pipeline (3 stages)

1. **Proposal (hybrid)**: Pillar-based geometric clustering (β1, far-range strong) + learned 3D detector (CenterPoint, near-range strong). Class-agnostic use of the learned detector for proposal generation; open-vocab labels come from YOLO-World.
2. **Reliability**: Coverage-based score (primary) + multi-view anchor bootstrap (secondary). Distance-aware weighting between the two proposal sources.
3. **Fusion (3-case)**:
   - **M (matched)**: Both sources agree on an instance → high reliability
   - **L (one-source-only)**: Only β1 *or* only γ proposes → reliability weighted by distance regime
   - **D (detection-only)**: 2D detection without 3D match → 4-step recovery (multi-cam search, projection, promotion, low-reliability hold)

### What the diagnosis ruled out

- ❌ Mask3D as proposal source (M=0.3% confirmed)
- ❌ Range as a primary reliability component (Tier 1: weak predictor)
- ❌ Multi-view as a primary reliability component (8.6% multi-view ratio)
- ❌ Per-pixel lifting (0.2% pixel coverage)
- ❌ Pure HDBSCAN as proposal source (W1.5 ceiling)
- ❌ Verticality filter on pillar foreground (β1.5 over-engineering)
- ❌ Pure detection-guided clustering (Option 5 wrong axis)
- ❌ Single-source proposal at all (7-stage 36% ceiling)

### What is preserved

- ✅ OpenYOLO3D's real-time philosophy (no CLIP recomputation, projection-only reliability, EMA per-instance accumulation)
- ✅ Open-vocabulary capability via YOLO-World labeling (CenterPoint used class-agnostically for proposals only)
- ✅ Coverage as primary reliability component (single strong-signal axis from diagnosis)
- ✅ 3-case fusion targeting the 44.8% detection-induced loss directly
- ✅ All 7 stages of geometric experiments preserved as ablation reference

---

## Expected contributions

> Contributions are being finalized as the hybrid simulation completes.

- **Diagnosis**: First systematic 7-stage measurement of how indoor open-vocab 3D pipelines fail on outdoor driving data, quantifying a fundamental 36% M-rate ceiling under single-source proposals on nuScenes.
- **Hybrid proposal**: Geometric-clustering and learned-3D-detector proposals are shown to be complementary across distance regimes; the method arbitrates between them via a reliability score grounded in sparse LiDAR coverage.
- **Real-time on outdoor open-vocab**: A working pipeline that maintains real-time inference on nuScenes, contrasting with existing outdoor open-vocab methods that prioritize accuracy without explicit real-time guarantees.

Method module details will be added to [`docs/METHOD.md`](docs/METHOD.md) after the hybrid simulation finalizes the proposal-level architecture.

## Tech stack

- **Base**: OpenYOLO3D's real-time philosophy (core not modified; original Mask3D + indoor lifting replaced)
- **2D open-vocabulary detector**: YOLO-World (semantic labeling)
- **3D proposal — geometric branch**: Pillar foreground extraction + HDBSCAN (β1 best config, locked)
- **3D proposal — learned branch**: CenterPoint pretrained on nuScenes (mmdet3d 1.4.0; class-agnostic use)
- **Visual-language embeddings**: CLIP-family encoder for instance-level features
- **Dataset**: nuScenes (`v1.0-mini` for development, `v1.0-trainval` for full experiments)
- **Framework**: PyTorch 1.12, mmdet3d 1.4.0
- **Hardware**: NVIDIA A100

## Evaluation plan

| Aspect | Metric |
|---|---|
| Detection accuracy | mAP, NDS at the instance level on nuScenes |
| Open-vocab capability | Performance on novel categories (following OV-SCAN / OpenSight protocols) |
| Distance-stratified | Per-bin mAP (0–10, 10–20, 20–30, 30–50, 50m+) — required given the 52% >30m GT distribution |
| Source ablation | β1-only / γ-only / hybrid comparison (built into diagnosis) |
| Temporal consistency | ID switches (IDS), label stability across frames, association accuracy |
| Real-time | FPS, end-to-end latency per frame, peak GPU memory |

Comparison targets: OpenSight, OV-SCAN, FM-OV3D (outdoor open-vocab); OpenYOLO3D (indoor real-time reference); ablations of our own modules (β1-only, γ-only, hybrid w/o reliability).

## Roadmap

| Stage | Period | Status | Deliverable |
|---|---|---|---|
| 1. nuScenes dataloader | May 2026 | ✅ Done | `dataloaders/nuscenes_loader.py`, calibration verified |
| 2. OpenYOLO3D integration smoke test | May 2026 | ✅ Done | Adapter, end-to-end run on nuScenes frames |
| 3. Diagnosis (Tier 1 / 2 / 2-Ext) | May 2026 | ✅ Done | Setup-level failure quantified |
| 3.5. Proposal diagnosis (W1 → γ, 7 stages) | May 2026 | ✅ Done | 36% ceiling and complementary distance bias confirmed |
| 4. Hybrid simulation | May 2026 | 🚧 In progress | β1 ∪ γ union measurement to confirm ceiling break |
| 5. Method implementation (Coverage reliability + 3-case fusion) | Jun–Jul 2026 | Pending | Working pipeline on nuScenes |
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
│   ├── lidar_proposals.py          # LiDARProposalGenerator (HDBSCAN + ground filter)
│   └── centerpoint_proposals.py    # CenterPointProposalGenerator (mmdet3d wrapper)
├── preprocessing/
│   ├── pillar_foreground.py        # PillarForegroundExtractor (β1, locked config)
│   ├── verticality_filter.py       # VerticalityFilter (β1.5 reference)
│   └── detection_frustum.py        # FrustumExtractor (Option 5 reference)
├── diagnosis/                      # Tier 1
├── diagnosis_tier2/                # Tier 2 + 2-Extended
├── diagnosis_w1/                   # W1: HDBSCAN baseline
├── diagnosis_w1_5/                 # W1.5: extended sweep + distance-stratified
├── diagnosis_step1/                # Step 1: Mask3D vs HDBSCAN vs Hybrid sim
├── diagnosis_beta1/                # β1: pillar foreground sweep
├── diagnosis_beta1_5/              # β1.5: verticality filter sweep
├── diagnosis_step_a/               # Step A: pillar resolution sweep (β1 ceiling test)
├── diagnosis_option5/              # Option 5: 2D detection-guided clustering
├── diagnosis_gamma/                # γ: CenterPoint proposal evaluation
├── proposal/                       # (added in Option 5; reference)
├── configs/
│   ├── nuscenes_baseline.yaml
│   └── nuscenes_trainval.yaml
├── docs/
│   ├── CONTEXT.md
│   ├── BASELINE.md
│   ├── NUSCENES_SETUP.md
│   └── method_drafts/              # Pre-diagnosis ideas (reference only)
├── results/                        # Per-stage outputs (aggregate.json, report.md, figures, sweep)
└── [OpenYOLO3D core files — untouched]
```

## Setup

See [`docs/NUSCENES_SETUP.md`](docs/NUSCENES_SETUP.md) for nuScenes data layout and environment setup. The development environment `openyolo3d-dev` is built on OpenYOLO3D's `openyolo3d` env with `nuscenes-devkit`, `hdbscan`, and `mmdet3d 1.4.0` added. Pre-mmdet3d package snapshots are kept under `/home/rintern16/env_snapshots/` for reproducibility.

## Branching policy

- `main`: stable, integrated stages only
- `feature/diagnosis`: Tier 1 / 2 / 2-Ext (merged)
- `feature/method-w1` … `feature/gamma-centerpoint`: per-stage branches for proposal diagnosis (kept for provenance)
- Stages merge to `main` only after acceptance criteria are verified end-to-end.

## License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Mask3D](https://github.com/JonasSchult/Mask3D), [CenterPoint](https://github.com/tianweiy/CenterPoint) (via [mmdet3d](https://github.com/open-mmlab/mmdetection3d)), and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

This work is conducted as a graduation research project at Soongsil university.

---
<<<<<<< HEAD
=======

**Contact**: yuha@soongsil.ac.kr
>>>>>>> Pre-method: ScanNet baseline reproduction infrastructure
