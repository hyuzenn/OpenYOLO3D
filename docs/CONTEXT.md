\# CONTEXT.md



\## 1. Project



\*\*Working title\*\*: SemWorld-3D — Streaming Open-Vocabulary 3D Instance Mapping for Outdoor Driving Scenes



\*\*One-line goal\*\*: Build a real-time, open-vocabulary 3D instance map on nuScenes by extending OpenYOLO3D from indoor RGBD to outdoor sparse-LiDAR + multi-camera input.



\*\*Why this and not yet-another-3D-detector\*\*:

\- Existing outdoor open-vocab 3D work (OV-Uni3DETR, OpenSight, OV-SCAN, FM-OV3D) targets per-frame 3D \*\*detection\*\*, not online instance \*\*mapping\*\*. They produce boxes, not a temporally consistent instance memory.

\- POP-3D does outdoor open-vocab but at the \*\*occupancy/voxel\*\* level, not instance level.

\- OpenYOLO3D demonstrates real-time open-vocab 3D instance segmentation, but only for indoor static scenes (Replica, ScanNet200) with dense RGBD.

\- The empty slot: \*\*streaming, instance-level, open-vocab 3D mapping in outdoor driving scenes\*\* — sparse LiDAR, 6-camera surround, dynamic objects, ego-motion.



\## 2. Scope decisions (locked for this iteration)



\- \*\*Dataset\*\*: nuScenes only. Replica is \*\*not\*\* a target dataset for this work. (OpenYOLO3D already validated indoor performance; reproducing it adds nothing.)

\- \*\*Base codebase\*\*: OpenYOLO3D, used as-is. We add a nuScenes data path and method modules on top, without modifying OpenYOLO3D's core inference code.

\- \*\*Modalities\*\*: 6 surround cameras + 32-beam LiDAR + ego pose. No HD map, no extra sensors.

\- \*\*Setting\*\*: Open-vocabulary — no nuScenes class labels are used in training. Ground-truth labels are used only for evaluation.

\- \*\*Real-time constraint\*\*: Inference latency must remain comparable to OpenYOLO3D's indoor numbers. Real-time is part of the contribution, not a side note.



\## 3. Why outdoor breaks the indoor pipeline (the core technical problem)



This section is the motivation for every method module that follows. Indoor → outdoor is not a dataset swap; it breaks three assumptions in OpenYOLO3D:



1\. \*\*Lifting assumption breaks.\*\* OpenYOLO3D lifts 2D boxes to 3D using per-pixel depth from dense RGBD. nuScenes provides 32-beam sparse LiDAR — a 2D box may contain 0 to a few projected LiDAR points, especially at distance. Median-depth lifting becomes unreliable or undefined.

2\. \*\*Static-scene assumption breaks.\*\* OpenYOLO3D's instance association assumes the scene is static between frames. nuScenes scenes contain moving cars, pedestrians, cyclists. Naïve spatial association collapses on dynamic objects.

3\. \*\*Single-view assumption breaks.\*\* Indoor scenes use a single forward camera. nuScenes uses 6 cameras with overlapping FOVs. The same physical object appears in multiple cameras at different scales and viewpoints, requiring multi-view consistent instance handling.



The dataloader stage (current step) exists to confirm and quantify these breakages on real nuScenes data before designing the fix.



\## 4. Tech stack



\- \*\*Language\*\*: Python (PyTorch).

\- \*\*Base model\*\*: OpenYOLO3D (cloned/forked, untouched in core).

\- \*\*2D open-vocab detector\*\*: YOLO-World (as in OpenYOLO3D).

\- \*\*Visual-language embeddings\*\*: CLIP-family image encoder for instance-level feature aggregation. (YOLO-World text embedding alone is insufficient for fusion — image features come from a stronger VLM encoder.)

\- \*\*Data\*\*: nuScenes devkit for I/O, calibration, evaluation.

\- \*\*Environment\*\*: Linux, CUDA, conda env `openyolo3d`.



\## 5. Evaluation plan (final paper, not current step)



\- \*\*Detection accuracy\*\*: mAP, NDS on nuScenes-style metrics, evaluated at instance level using ground-truth boxes.

\- \*\*Open-vocabulary capability\*\*: Performance on novel categories not seen during any training (following OV-SCAN / OpenSight protocols).

\- \*\*Temporal consistency\*\*: ID switches (IDS), label stability over time, association accuracy across consecutive frames.

\- \*\*Real-time\*\*: FPS, end-to-end latency per frame, peak GPU memory.

\- \*\*Comparison targets\*\*: OpenSight, OV-SCAN, FM-OV3D for outdoor open-vocab; OpenYOLO3D for the real-time / indoor reference; ablations of our own method.



\## 6. Roadmap (high level)



1\. \*\*Current step — dataloader + smoke test\*\*: nuScenes data loads correctly into OpenYOLO3D's expected input format. Unit-tested calibration projection. No method changes yet.

2\. \*\*Diagnosis step\*\*: Run OpenYOLO3D's stock inference on nuScenes through the new dataloader. Quantify exactly how each of the three indoor assumptions breaks (lifting failure rate, association failure on dynamic objects, multi-view inconsistency). Output: a measurement report that drives method design.

3\. \*\*Method design step\*\*: Based on diagnosis, write a single `METHOD.md` with the proposed modules. (Drafts under `docs/method\_drafts/` exist as ideas only — not commitments.)

4\. \*\*Implementation + ablations\*\*: Implement modules, run ablations, compare against baselines on nuScenes.



\## 7. What is intentionally not in this document



\- Method module details (lifting fix, registration logic, fusion, association). These are deferred until after the diagnosis step. Existing drafts in `docs/method\_drafts/` predate this plan and are kept for reference but are not authoritative.

\- A Phase 1 / Phase 2 split. The previous "simple version first, advanced version if needed" framing is dropped — the simple versions are too weak to carry a top-tier paper and will appear only as ablation comparisons against the main method, not as standalone deliverables.

\- Indoor (Replica) experiments. Out of scope.

