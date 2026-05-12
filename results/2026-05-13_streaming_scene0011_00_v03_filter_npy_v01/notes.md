# Task 1.2c diagnosis — Option A: scene0011_00 v03 with filter + .npy path

**Date**: 2026-05-13
**PBS job**: 92047
**Run dir**: `results/2026-05-13_streaming_scene0011_00_v03_filter_npy_v01/`
**Purpose**: scene0011_00에 v01 path 대신 312-scene aggregate에서 쓴 path (filter=True + .npy) 적용 — H2 (filter/.npy가 streaming AP를 떨어뜨림) 직접 검증.

## v02 (filter=False, .ply) vs v03 (filter=True, .npy) — scene0011_00

| Pipeline | v02 AP | v03 AP | Δ (v03 − v02) |
|---|---|---|---|
| Offline | 0.3457 | 0.3673 | +0.0216 |
| Streaming | **0.3364** | **0.3371** | **+0.0007** |
| Δ AP (streaming − offline) | −0.0093 | −0.0302 | −0.0209 |

| Detail | v02 | v03 |
|---|---|---|
| Streaming AP_50 | 0.4898 | 0.4873 |
| Streaming AP_25 | 0.5360 | 0.5557 |
| D3 median visible / frame | 46 | 9 |
| Mask3D K (after filter) | 600+ (no filter) | filtered |
| Streaming wall time | 49.1 s | 36.0 s |

## Verdict on H2

**H2 partially BUSTED for scene0011_00**: switching to filter+.npy did **not** degrade streaming AP (0.3364 → 0.3371, statistically identical within Mask3D noise). The widened streaming–vs–offline gap (−0.0093 → −0.0302) is caused by **offline improving** (+0.0216), not streaming dropping.

But streaming AP for scene0011_00 alone (0.3371) is **much higher than the 312-scene aggregate** (0.1911). So scene0011_00 is an easy-head-dominated outlier; the 312-aggregate gap originates from **harder common/tail scenes** that scene0011_00 doesn't represent.

scene0011_00 head_AP = NaN for common/tail because the scene has no common/tail GT instances. The Task 1.2c bucket breakdown (head −0.018, common −0.066, tail −0.089) cannot be debugged via this scene alone.

## Implications for the diagnosis

- H1 (Option B): refuted at the dataset level (312/312 .npy/.ply match).
- H2 (this run): refuted at scene0011_00 level — the streaming AP is stable under filter+.npy.
- The 312-scene Δ −0.0562 must originate elsewhere — likely systematic per-frame computation differences in streaming vs offline that hurt smaller / common / tail scenes more (residual H3 territory, or some unidentified bug).

## Recommendation

scene0011_00 is no longer informative for this debug. Need a scene-level debugging signal across common/tail scenes — e.g., split the 312 set into head-only vs common-rich and compare per-bucket Δ, or pick 2-3 common/tail-heavy scenes and run them as single-scene sanity. Either way the next step is **C (full 312 re-run with v02 path: filter=False + .ply)** to see if the aggregate Δ closes when we revert the filter+.npy changes — or **D (move get_visibility_mat to GPU)**.
