# Task 2.5 — γ CenterPoint baseline audit + multi-sweep fix

Date: 2026-05-21
Branch: main @ c85ba38 (post Task 2.4 Stage C)
Env: openyolo3d-dev (mmdet3d 1.4.0, nuscenes-devkit)
Status: **Stages 1–2 complete & verified. Stage 3 BLOCKED by missing LiDAR sweep data
(see §4). Awaiting user decision before running.**

---

## Stage 1 — diagnosis: my materials vs standard CenterPoint

### 1.1 What my γ stack actually does

| Component | File | Setting |
|---|---|---|
| Checkpoint | `pretrained/centerpoint_nuscenes/centerpoint_0075voxel_..._cyclic_20e_nus_20220810-04cb3a3b.pth` | mmdet3d official voxel-0075 CenterPoint |
| Config | `centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py` | mmdet3d official |
| LiDAR input (Stage C) | `dataloaders/nuscenes_loader.py` + `configs/nuscenes_trainval.yaml` | `multi_sweep: false, num_sweeps: 1` → **single keyframe** |
| Time channel | `adapters/centerpoint_proposals.py:104` | `pc_5[:, 4] = 0.0` (single keyframe Δt) |
| Detection **label** | `nuscenes_evaluator.step_sample` | **YOLO-World** (3D box → project to 6 cams → IoU-match 2D bbox), **NOT** CenterPoint's head |
| Detection **score** | same | YOLO-World score (falls back to CP score) |
| CenterPoint's own class | — | used only for the greedy tracker's class-match; **discarded** for the detection name |

### 1.2 The mmdet3d standard this checkpoint was trained for

The checkpoint's own config test/train pipeline (verified in the `.py`):

```
LoadPointsFromFile(load_dim=5, use_dim=5)
LoadPointsFromMultiSweeps(sweeps_num=9)   # test;  sweeps_num=10 train preset
```

So the checkpoint **expects ~10-sweep aggregated point clouds** (x,y,z,intensity,Δt),
~250–300k points/sample. Standard reported result on nuScenes val: **mAP ≈ 0.565, NDS ≈ 0.653**.

### 1.3 The two confirmed root causes of mAP 0.0526

**(a) Single-sweep / 10-sweep train-test mismatch — CONFIRMED, dominant.**
Stage C fed CenterPoint a single keyframe (~25–35k points) with Δt=0. The checkpoint was
trained on ~10-sweep density. This is a severe input-distribution shift → low recall,
especially for far / small objects. Evidence from the Stage C per-class numbers:

- `car` AP_mean 0.126 but **trans_err 0.36 m** — i.e. boxes that *are* detected are
  localized well (≈ standard CenterPoint trans_err). The collapse is **recall + score
  ranking**, not localization. That is the fingerprint of a density/recall problem, not a
  broken checkpoint or a coordinate bug.
- `pedestrian` 0.035, `bicycle` 0.069, `trailer`/`construction_vehicle` ≈ 0 — exactly the
  small/sparse classes that single-sweep starves.

**(b) Detection labels & scores come from YOLO-World, not CenterPoint — CONFIRMED,
architectural.** The "γ baseline" measured by Stage C is **not** the standard CenterPoint
detector. It is `CenterPoint boxes (class discarded) + YOLO-World open-vocab label + greedy
tracker + temporal gate`. Even with a perfect CenterPoint, this pipeline's mAP is capped well
below 0.565 because every box must (i) project into a camera, (ii) IoU-match a YOLO-World 2D
detection, (iii) be one of the nuScenes-10 names, and is then scored by the YOLO confidence.
Native single-sweep CenterPoint **0.363** vs Stage-C pipeline **0.0526** = ~7× from the
relabel + gate (measured, §3).

**(c) Class-index permutation bug — CONFIRMED via the GPU run (NOT caught in static
analysis).** My Stage-1 read of the code wrongly concluded the mapping was fine. The native
mAP run's per-class breakdown exposed it: the adapter's `NUSC_10` tuple was in the canonical
nuScenes-devkit order, but the checkpoint's CenterHead emits labels in its *own* flattened
task order (`car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle,
pedestrian, traffic_cone`). 6 of 10 indices were mislabeled; only car/truck/bus/motorcycle
coincidentally aligned. pedestrian (1987 GT) and barrier (638 GT) got AP ≈ 0. Fixing the
order **doubled native mAP (0.184 → 0.363)**. **This is the real form of Gemini's
"class-agnostic mapping error" hypothesis (b) — it was a true bug, now fixed.** Lesson: the
May "sanity confirmed class order" comment was false; that sanity only checked box counts and
timing, never labels vs GT.

### 1.4 What the May γ material actually validated

`diagnosis_gamma/run_gamma.py` swept CenterPoint as a **class-agnostic proposal generator**
and scored it with **M_rate** (does a box land near a GT box), *not* mAP. CenterPoint looked
fine there (M_rate ~0.36–0.55). The mAP 0.0526 first appeared in Stage A/C through the full
detection pipeline. So the May material was never a CenterPoint *detection* baseline — it was
a *proposal-recall* check. No dishonesty, but the 0.0526 was mislabeled in spirit as a
"CenterPoint baseline" when it is really a "CenterPoint-proposals + YOLO-label pipeline" number.

---

## Stage 2 — config fix (applied, CPU-verified)

Minimal, contained to the Outdoor stack. **No** Indoor / OY3D-core / May-production changes.

| File | Change | Lines |
|---|---|---|
| `dataloaders/nuscenes_loader.py` | multi-sweep branch now preserves the per-point Δt as a 5th channel (`from_file_multisweep` already aggregated; it had been discarding `times`). Single-sweep path returns (N,4) **unchanged** → Stage C reproducibility intact. | +12 / −1 |
| `adapters/centerpoint_proposals.py` | feed the loader's Δt channel into `pc_5[:,4]` when present (≥5 cols); else 0.0 as before. | +5 / −1 |
| `configs/nuscenes_trainval_multisweep.yaml` | **new** config = trainval + `multi_sweep: true, num_sweeps: 10`. Original `nuscenes_trainval.yaml` left untouched as the Stage C reference. | new |
| `scripts/centerpoint_native_map_sanity.py` | **new** decisive diagnostic: native CenterPoint label+score → devkit eval, single-sweep vs multi-sweep (controlled), bypassing YOLO. | new |

### 2.2 Checkpoint verification
The checkpoint + config in use **are** the mmdet3d official voxel-0075 CenterPoint (the model
that scores 0.565/0.653). No checkpoint problem. The defect was the input pipeline, not weights.

### 2.3 Class-mapping fix (CRITICAL — found via GPU run, see §1.3c)
`adapters/centerpoint_proposals.py` `NUSC_10` reordered to the checkpoint's CenterHead
task order. This is the single biggest correction (+98% native mAP). The previous tuple
was the canonical devkit order, which does not match the model's output indices.

### 2.4 Code-correctness verification (CPU, no GPU needed)
Ran the patched loader on covered val **scene-0103**, sample index 15:

| nsweeps | points | cols | Δt range (s) | unique Δt |
|---:|---:|---:|---|---:|
| 1  | 34,688  | 4 | — (single) | — |
| 5  | 135,198 | 5 | 0.000–0.200 | 5 |
| 10 | 270,217 | 5 | 0.000–0.450 | 10 |

10-sweep → **7.8× density**, Δt spans 0–0.45 s in 10 discrete steps = nuScenes' ~20 Hz
sweep cadence over 9 prior sweeps. This is exactly the (x,y,z,intensity,Δt) distribution the
checkpoint was trained on. **The fix is correct.**

### 2.5 Indoor / core protection
`git diff` touches only `adapters/centerpoint_proposals.py` and
`dataloaders/nuscenes_loader.py`. Zero changes under `method_scannet/`, `utils/`, `models/`,
`adapters/lidar_proposals.py`, `diagnosis_alpha/`. Stage C results untouched.

---

## §4 — STAGE 3 BLOCKER: LiDAR sweeps are not on disk

The fix is correct but **cannot be exercised at the required 150-scene scope**: the
intermediate LiDAR sweeps were never downloaded.

- `data/nuscenes/.v1.0-trainval0X_keyframes.txt` markers read *"Thank you for downloading
  v1.0-trainval0X_**keyframes**.tgz"* → only the **keyframe-only** archives were fetched.
- `data/nuscenes/sweeps/LIDAR_TOP/` holds **3,531** files / 2.3 GB (a few logs' worth); a full
  trainval needs ~297k LiDAR sweeps.
- Per-val-scene coverage (all 150 scenes scanned): **4** scenes fully covered, **146**
  essentially empty. Mean prior-sweeps available per sample = **0.23 / 9**.
- `LidarPointCloud.from_file_multisweep` raises `FileNotFoundError` on the first missing
  sweep, so a 150-scene multi-sweep run crashes almost immediately.

A second, independent blocker: the only A100 is held by interactive PBS job **94065**
(running, `coss_agpu`, 1-GPU/user policy), so no GPU run is possible right now regardless.

**User decision (taken):** run the native single-vs-multi-sweep mAP sanity on the 4 covered
scenes (no download); user freed GPU job 94065 first. **Done** — PBS 95026/95027 Exit 0.

---

## Stage 3 / 4 result (reduced scope, see results/2026-05-21_outdoor_centerpoint_fixed_v02/report.md)

- Native CenterPoint, fixed class map, 4 covered val scenes / 162 samples:
  **single-sweep mAP 0.363 / NDS 0.330, multi-sweep mAP 0.336 / NDS 0.308.**
- Class-map fix alone: 0.184 → 0.363 (**+98%**). Checkpoint healthy (car 0.69, pedestrian
  0.65–0.80, barrier 0.69, trans_err ≈ 0.2).
- Stage-C γ pipeline 0.0526 vs native single-sweep 0.363 → YOLO relabel + gate costs ~7×.
- **Verdict: scenario (P)** — config/mapping/pipeline, not the model. Caveat: clean ≥0.5
  *ten-class* mean needs full 150-scene val with sweeps (4-scene subset has a 0-GT class +
  rare-class AP noise).
- Recommended: adopt class-map fix; cheapest clean number is full-val **native single-sweep**
  (no download needed, all keyframes present); download sweeps only for the multi-sweep delta
  / exact 0.565; treat the YOLO-World relabel as the open-vocab demo, not the headline metric.

---

## Acceptance criteria status

| # | criterion | status |
|---|---|:--:|
| 1 | Stage 1 diagnosis (my materials vs standard) | ✓ §1 (+ class-bug found at run time §1.3c) |
| 2 | Stage 2 fix (multi-sweep + ckpt + **class mapping**) | ✓ §2 + class-map bug fixed §2.3 |
| 3 | Stage 3 PBS Exit 0 + re-measurement | ✓ reduced scope (4 sweep-covered scenes; 146/150 lack data) |
| 4 | Stage 4 honest re-eval + P/Q/R verdict | ✓ scenario **P**, caveated |
| 5 | Indoor + core 변경 0건 | ✓ §2.5 (git diff: 2 Outdoor files only) |
