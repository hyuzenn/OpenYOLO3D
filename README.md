# SemWorld-3D

**Streaming Open-Vocabulary 3D Instance Mapping for Outdoor Driving Scenes**

A research project extending [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D) from indoor RGBD scenes to outdoor autonomous driving scenarios on the nuScenes dataset.

> ⚠️ **Status**: Work in progress. Targeting CVPR 2027 submission (Nov 2026 deadline). Method is under development; see [Roadmap](#roadmap) for current stage.

---

## Motivation

Existing outdoor open-vocabulary 3D perception work (OV-Uni3DETR, OpenSight, OV-SCAN, FM-OV3D) targets **per-frame 3D detection** — they output bounding boxes, not a temporally consistent instance memory. POP-3D works at the **occupancy/voxel level**, not at instance granularity. Indoor real-time open-vocab instance segmentation methods (OpenYOLO3D, OpenMask3D, ConceptFusion) assume dense RGBD and a static, single-camera setup.

The empty slot:

> **Streaming, instance-level, open-vocabulary 3D mapping for outdoor driving scenes — with sparse LiDAR, 6-camera surround-view, dynamic objects, and ego-motion, all under a real-time constraint.**

This is the problem SemWorld-3D addresses.

## Why outdoor breaks indoor pipelines

Moving from indoor (Replica, ScanNet) to outdoor (nuScenes) is not a dataset swap. It breaks three core assumptions of indoor open-vocab 3D pipelines:

1. **Lifting assumption**: OpenYOLO3D lifts 2D detections to 3D using dense per-pixel depth from RGBD. nuScenes provides 32-beam sparse LiDAR — a 2D bounding box may contain anywhere from 0 to a few projected LiDAR points, especially at distance. Median-depth lifting becomes unreliable.
2. **Static-scene assumption**: Indoor instance association assumes scene geometry is stable between frames. nuScenes contains moving cars, pedestrians, cyclists. Naive spatial association collapses on dynamic objects.
3. **Single-view assumption**: Indoor scenes typically use a forward-facing camera. nuScenes uses 6 surround cameras with overlapping fields of view, so the same physical object appears in multiple cameras at different viewpoints, scales, and quality levels.

These three failure modes drive the method design.

## Project goals

1. **Establish a strong baseline**: Run OpenYOLO3D as-is on nuScenes via a clean integration layer; quantify exactly how each indoor assumption fails.
2. **Design outdoor-specific solutions**: Develop method modules that target the three failure modes above, while preserving OpenYOLO3D's real-time character.
3. **Demonstrate that real-time streaming open-vocab 3D instance mapping is achievable on outdoor driving data** — a setting where no prior work has shown this combination.

## Expected contributions

> Contributions will be finalized after the diagnosis stage. The current direction:

- **Task formulation**: First systematic treatment of streaming open-vocabulary 3D instance mapping in the outdoor driving setting, with an evaluation protocol on nuScenes.
- **Method**: Outdoor-specific modules addressing sparse-LiDAR lifting, dynamic-object association, and multi-view feature reliability — designed to maintain real-time inference.
- **Real-time on outdoor open-vocab**: A working pipeline that achieves real-time performance on nuScenes, in contrast to existing outdoor open-vocab methods that focus on accuracy without explicit real-time guarantees.

Method module details will be added to [`docs/METHOD.md`](docs/METHOD.md) after the diagnosis stage. Earlier method drafts under `docs/method_drafts/` are reference only and not authoritative.

## Tech stack

- **Base**: OpenYOLO3D (used as-is, core not modified)
- **2D open-vocabulary detector**: YOLO-World
- **3D mask proposal**: Mask3D (via OpenYOLO3D's pipeline)
- **Visual-language embeddings**: CLIP-family encoder for instance-level features
- **Dataset**: nuScenes (`v1.0-mini` for development, `v1.0-trainval` for full experiments)
- **Framework**: PyTorch
- **Hardware**: NVIDIA A100

## Evaluation plan

| Aspect | Metric |
|--------|--------|
| Detection accuracy | mAP, NDS at the instance level on nuScenes |
| Open-vocab capability | Performance on novel categories (following OV-SCAN / OpenSight protocols) |
| Temporal consistency | ID switches (IDS), label stability across frames, association accuracy |
| Real-time | FPS, end-to-end latency per frame, peak GPU memory |

Comparison targets: OpenSight, OV-SCAN, FM-OV3D (outdoor open-vocab); OpenYOLO3D (indoor real-time reference); ablations of our own modules.

## Roadmap

| Stage | Period | Status | Deliverable |
|-------|--------|--------|-------------|
| 1. nuScenes dataloader | May 2026 | ✅ Done | `dataloaders/nuscenes_loader.py`, calibration verified |
| 2. OpenYOLO3D integration smoke test | May 2026 | 🚧 In progress | Adapter, end-to-end run on one nuScenes frame |
| 3. Diagnosis | May–Jun 2026 | Pending | Quantitative measurement of how OpenYOLO3D fails on nuScenes; report drives method design |
| 4. Method design | Jun 2026 | Pending | `docs/METHOD.md` with finalized modules |
| 5. Implementation (core modules) | Jun–Jul 2026 | Pending | Working method on nuScenes mini |
| 6. Full experiments + ablation | Aug 2026 | Pending | Results on nuScenes trainval |
| 7. Graduation thesis presentation | Sep 2026 | Pending | Demo + thesis writeup |
| 8. Paper writing + revision | Oct 2026 | Pending | CVPR 2027 draft |
| 9. CVPR 2027 submission | Nov 1, 2026 | Pending | Final submission |

## Repository structure

```
.
├── dataloaders/
│   ├── nuscenes_loader.py      # Per-frame dict interface for nuScenes
│   └── sanity_check.py         # LiDAR→camera projection visual check
├── adapters/                   # (Stage 2) bridges dataloader to OpenYOLO3D
├── configs/
│   └── nuscenes_baseline.yaml  # Dataloader config
├── docs/
│   ├── CONTEXT.md              # Project scope and decisions
│   ├── BASELINE.md             # Baseline definition and integration spec
│   ├── NUSCENES_SETUP.md       # Setup instructions
│   └── method_drafts/          # Pre-diagnosis method ideas (reference only)
├── results/                    # Experimental outputs and reports
├── run_nuscenes.py             # nuScenes dataloader smoke test
└── [OpenYOLO3D core files — untouched]
```

## Setup

See [`docs/NUSCENES_SETUP.md`](docs/NUSCENES_SETUP.md) for nuScenes data layout and environment setup. The development environment `openyolo3d-dev` is a clone of OpenYOLO3D's `openyolo3d` env with `nuscenes-devkit` added.

## Branching policy

- `main`: stable, OpenYOLO3D base + integrated stages only
- `feature/nuscenes-dataloader`: dataloader + integration adapter (current active)
- `feature/diagnosis`: diagnosis measurements (next stage)
- `backup/initial-pipeline`: archive of pre-OpenYOLO3D experimental pipeline

Stages merge to `main` only after their full integration is verified end-to-end.

## License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Mask3D](https://github.com/JonasSchult/Mask3D), and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

This work is conducted as a graduation research project at [Institution].

---

**Contact**: [Author info]