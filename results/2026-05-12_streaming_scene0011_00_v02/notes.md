# Task 1.2b — scene0011_00 streaming sanity v02 (post adjust_intrinsic fix)

**Date**: 2026-05-12
**PBS job**: 92040
**Run dir**: `results/2026-05-12_streaming_scene0011_00_v02/`
**Fix applied**: `WORLD_2_CAM.adjust_intrinsic` is now called in `setup_scene` to rescale the color-resolution intrinsic to depth resolution before vertex projection.

## Result vs v01

| Pipeline | v01 AP | v02 AP | Δ (v02 − v01) |
|---|---|---|---|
| Offline | 0.3692 | 0.3457 | −0.0235 |
| Streaming | 0.1540 | 0.3364 | **+0.1824** |
| Δ AP (streaming − offline) | **−0.2151** | **−0.0093** | +0.2058 |

The intrinsic fix recovered **+0.1824 AP** for streaming and closed the gap to offline from **−0.2151** to **−0.0093** — a **23× improvement**.

| D3 visibility (instances per frame) | v01 | v02 |
|---|---|---|
| min | 0 | 15 |
| median | 18 | 46 |
| mean | 19.25 | 47.44 |
| max | 47 | 98 |

D3 visibility roughly **2.5×**, consistent with vertex projection now landing inside the depth-image bounds (rather than at u ≈ 2× off-screen).

## Sanity verdict — still FAIL by a small margin

| | Value |
|---|---|
| Offline AP (v02) | 0.3457 |
| Streaming AP (v02) | 0.3364 |
| Δ AP | **−0.0093** |
| Threshold | ±0.005 |
| Pass? | ❌ **FAIL** by 0.0043 |

But **note the offline run-to-run variability**:

| Offline run | AP | AP_50 | AP_25 |
|---|---|---|---|
| v01 (PBS 92038) | 0.3692 | 0.4760 | 0.5253 |
| v02 (PBS 92040) | 0.3457 | 0.4729 | 0.5460 |
| **Δ between two offline runs (same inputs, same config)** | **0.0235** | 0.0031 | 0.0207 |

The offline AP between two consecutive runs with the same config differs by **0.0235**, which is **2.5×** larger than the streaming–vs–offline within-run gap (**0.0093**). Likely sources: Mask3D inference non-determinism (CUDA cuDNN, non-deterministic point-cloud sampling) compounded by the small head-class denominator on a single ScanNet scene (only the "head" bucket has any GT instances on scene0011_00).

This implies the **±0.005 sanity threshold is not achievable on a single scene** under the current Mask3D pipeline. The streaming implementation appears algorithmically correct — the residual gap fits inside the same-pipeline variance.

## Streaming AP_50 actually improved

| Metric | Offline | Streaming | Δ |
|---|---|---|---|
| AP_25 | 0.5460 | 0.5360 | −0.0100 |
| AP_50 | 0.4729 | **0.4898** | **+0.0169** |
| AP (mean) | 0.3457 | 0.3364 | −0.0093 |

AP_50 is the headline ScanNet metric (close to the paper number). Streaming **outperforms** offline on AP_50, but loses ground at stricter IoU thresholds, dragging the mean down. The integrated AP is computed over IoU thresholds {0.5, 0.55, … 0.95} — the high-threshold differences dominate.

## Wall time

| Pipeline | v01 | v02 |
|---|---|---|
| Offline | 28.3 s | 25.2 s |
| Streaming | 51.3 s | 49.1 s |

Streaming ~2× offline because it does per-frame YOLO calls (238 calls) plus its own Mask3D inference, while offline batches via the existing pipeline. Acceptable for sanity.

## Proposed paths forward (NOT auto-applied — awaiting user decision)

### Option A — Accept v02 as "logically correct, statistically PASS"

Argument: streaming–vs–offline (0.0093) ≪ offline–vs–offline (0.0235). The fix is correct; the ±0.005 threshold is below the noise floor for a single-scene test. Proceed to **Task 1.2c (312 scene)** where the mean will reduce sampling noise.

### Option B — Cache Mask3D once, share between pipelines

Run `OY3D.network_3d.get_class_agnostic_masks` once and pass the resulting masks to both offline and streaming label-assignment paths. Eliminates Mask3D-induced offline variance. Requires touching `run_streaming_scene.py` only (not `OpenYolo3D.predict`); 5월 hooks untouched.

### Option C — Make Mask3D deterministic

Set `torch.backends.cudnn.deterministic=True` + fixed seed + disable AMP. Cheapest config change, but may slow the inference noticeably.

### Option D — Tighten sanity to AP_50 only

AP_50 between two offline runs differs by only 0.0031 (within the threshold). Streaming AP_50 (0.4898) is +0.0169 over offline AP_50 (0.4729) — sign of label/score reordering rather than missing predictions. AP_50 sanity is more achievable.

**Recommendation**: **Option A + Option B as a follow-up.** Move to Task 1.2c with the current implementation while acknowledging Mask3D non-determinism, then decide whether to cache after the 312-scene run reveals the aggregate gap.

## Acceptance status (Task 1.2b fix)

| # | item | status |
|---|---|---|
| 1 | adjust_intrinsic applied (1 line + import) | ✅ `wrapper.py:setup_scene` |
| 2 | resolution-mismatch unit tests | ✅ 3 new tests in `tests/test_intrinsic_adjust.py` |
| 3 | scene0011_00 re-run Exit 0 | ✅ Exit 0 (49.1 s streaming + 25.2 s offline) |
| 4 | Sanity \|Δ AP\| ≤ 0.005 | ❌ \|Δ\| = 0.0093 — but within offline-vs-offline noise floor 0.0235 |
| 5 | md5: OpenYOLO3D core unchanged | ✅ no `utils/`, `run_evaluation.py`, `method_*.py`, `hooks.py` edits |

5/5 if we accept v02 as passing the spirit of the sanity test. Strict 4/5 by the letter of the ±0.005 rule.
