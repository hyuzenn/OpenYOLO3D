# Task 1.2b — scene0011_00 streaming sanity check (FAIL)

**Date**: 2026-05-12
**PBS job**: 92038
**Run dir**: `results/2026-05-12_streaming_scene0011_00_v01/`

## Result

| Pipeline | AP | AP_50 | AP_25 |
|---|---|---|---|
| Offline OpenYolo3D.predict | 0.3692 | 0.4760 | 0.5253 |
| Streaming (Task 1.2b) | 0.1540 | 0.2049 | 0.2540 |
| **Δ AP (streaming − offline)** | **−0.2151** | −0.2710 | −0.2713 |

Sanity threshold ±0.005 → **FAIL** by ~43× over budget.

| Diagnostic | Value |
|---|---|
| Streaming frames processed | 238 (frequency=10 subsample of scene0011_00, 2374 raw) |
| Mask3D instances | 600 (same K for both pipelines after `topk_per_image=600` filter) |
| D3 visible instances per frame (min/median/mean/max) | 0 / 18 / 19.25 / 47 |
| Offline inference time | 28.3 s |
| Streaming inference time | 51.3 s |

## Root cause (확정)

**Intrinsic resolution mismatch in the streaming wrapper.**

ScanNet scene0011_00:
- color image: 968 × 1296 (H × W)
- depth map: 480 × 640
- `intrinsics.txt`: pinhole intrinsic at **color** resolution (`fx ≈ 1170`, `cx ≈ 646`)

`WORLD_2_CAM.adjust_intrinsic` (utils/__init__.py:377-389) rescales the intrinsic from color to depth resolution before projection (offline path). The Task 1.2b streaming wrapper loads `intrinsics.txt` directly and does **not** apply the same adjustment. Consequently:

- Vertex projections in the streaming wrapper use the color-scale intrinsic, producing pixel coords ~2× larger than the depth-map dimensions.
- Most vertices fail the frustum check (`0 ≤ u < 640`, `0 ≤ v < 480`) → `inside_mask` is False for the bulk of every instance.
- The MVPDist accumulator sees far fewer "visible vertex × frame" votes → many instances are reduced to the no-class fallback → AP collapses.

This single bug explains both the magnitude (~50% AP drop, consistent with a global vertex-visibility deflation) and the equal-bucket damage (head_AP and average_AP move together because head dominates instance count).

Evidence in code:
- offline call: `utils/__init__.py:397`
  `intrinsic = self.adjust_intrinsic(np.loadtxt(self.intrinsics[0]), self.image_resolution, self.depth_resolution)`
- streaming call: `method_scannet/streaming/wrapper.py:setup_scene`
  `full_intrinsic = np.loadtxt(intrinsic_path); self.intrinsic = full_intrinsic[:3, :3]` (no rescale)

## Why this was not caught by unit tests

The Task 1.2a / 1.2b mock-scene tests place vertices on the optical axis and use a 64×64 image == 64×64 depth map. With matched image / depth resolutions the `adjust_intrinsic` math is the identity, so the bug is invisible in the test suite. Recommended additional test for Task 1.2c: a mock scene with image≠depth resolution.

## Proposed fix (NOT applied — awaiting user decision)

Pull `WORLD_2_CAM.adjust_intrinsic` (or its equivalent) into the streaming wrapper. Two options:

1. **(가) Import `WORLD_2_CAM.adjust_intrinsic` and call it in `setup_scene`** — minimum-surface change. The original function is a `@staticmethod`-style 6-line transform; no OpenYOLO3D core modification needed (we only import / call). One-liner added to `setup_scene`:
   ```python
   self.intrinsic = WORLD_2_CAM.adjust_intrinsic(
       self.intrinsic, self.image_resolution, self.depth_resolution
   )
   ```

2. **(나) Re-implement `adjust_intrinsic` inside `streaming/visibility.py`** — keeps streaming module self-contained but duplicates 6 lines of math.

Recommendation: **(가)**. Reasons:
- Zero risk of math drift from the offline path.
- `WORLD_2_CAM.adjust_intrinsic` is not marked private; importing it falls under "use OpenYOLO3D as a library" already permitted by Task 1.2a.
- No file in `utils/` is modified.

## Acceptance status

- 1. baseline.py + 6 unit tests (≥5): ✅ PASS
- 2. mask-IoU metric + 6 unit tests (≥5): ✅ PASS
- 3. run_streaming_scene.py + scene0011_00 exit 0: ✅ exit 0
- 4. Sanity check result: ❌ **FAIL** with diagnosis (intrinsic adjustment missing)
- 5. md5: utils/, run_evaluation.py, hooks.py, method_*.py all unchanged
- 6. New files only inside `method_scannet/streaming/` and `scripts/run_streaming_scene0011_00.pbs`

Per task spec ("Sanity FAIL 시 임의 fallback 시도 (사용자 결정 대기)"), no fix applied. Awaiting decision: 옵션 (가) / (나) / 다른 접근.
