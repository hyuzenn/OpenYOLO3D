# Task 1.1 — Stage 1: OpenYOLO3D ScanNet pipeline 분석

**Branch**: `feature/method-scannet-21-31`
**Date**: 2026-05-12
**Scope**: 코드 변경 0건. `streaming evaluation harness` 구축 전에 OpenYOLO3D 원본 ScanNet 추론 파이프라인이 frame 정보를 어떻게 사용하는지 정리.
**Output of this stage**: 본 문서. Stage 2(streaming spec), Stage 3(metrics) 설계의 근거.

---

## 1. 진입점 (entry point)

| Layer | 파일 | 함수/클래스 |
|---|---|---|
| CLI | `run_evaluation.py` | `__main__` → `test_pipeline_full(dataset_type=...)` |
| Per-scene | `run_evaluation.py:48` | `openyolo3d.predict(path_2_scene_data, depth_scale, datatype, processed_scene, path_to_3d_masks, is_gt)` |
| Inference core | `utils/__init__.py:96` | `OpenYolo3D` 클래스 (`predict` / `label_3d_masks_from_2d_bboxes` / `label_3d_masks_from_label_maps`) |
| Frame projection | `utils/__init__.py:325` | `WORLD_2_CAM` 클래스 (`get_mesh_projections`) |

배치/오프라인 평가는 `test_pipeline_full`이 `SCENE_NAMES_SCANNET200` 312 scene을 순회하며 scene별로 `predict` 1회 호출 후, 마지막에 `evaluate_scannet200`으로 mAP 계산. `_maybe_dump_metrics`가 `metrics.json` 직렬화.

---

## 2. ScanNet 입력 구조

Scene 디렉토리 (`data/scannet200/<scene_name>/`):

```
scene0011_00/
├── scene0011_00_vh_clean_2.ply   # 전체 reconstructed mesh (모든 vertex 포함)
├── 0011_00.npy                    # Mask3D 입력용 전처리 point cloud
├── intrinsics.txt                 # camera intrinsic (4x4, 전체 frame 공통)
├── poses/                         # camera-to-world extrinsic (frame별)
│   ├── 0.txt
│   ├── 1.txt
│   └── ... (정수 인덱스 = 시간 순서)
├── color/                         # RGB frame
│   ├── 0.jpg ... N.jpg
└── depth/                         # depth map (PNG, depth_scale 1000.0)
    ├── 0.png ... N.png
```

- **Frame 인덱스**: 파일명 정수 = ScanNet 원본 RGB-D sequence의 시간 순서 (그대로 사용; 본 task의 streaming protocol과 일치).
- **Mesh**: scene당 1개, 모든 vertex (= V) 포함. Mask3D는 이 mesh에 직접 들어가지 않고 `0011_00.npy`의 전처리 point cloud로 inference (`processed_scene` 인자).
- **Pose / intrinsic**: world-to-camera 변환에 사용. `WORLD_2_CAM`에서 한 번에 모두 stack.

ScanNet `frequency=10` (`pretrained/config_scannet200.yaml`): scene의 N개 frame 중 0, 10, 20, …만 사용 → scene당 ~237개 frame (예: scene0011_00 = 2374 frames, frequency=10 → 238개). 본 task의 streaming도 이 subsampled stream을 그대로 사용.

GT: `data/scannet200/ground_truth/<scene_name>.txt` — V줄 ASCII, 각 줄은 `label_id * 1000 + instance_id` (배경/무관 vertex는 0). 평가용 ScanNet200 instance segmentation 표준 포맷.

---

## 3. Mask3D 호출

**위치**: `OpenYolo3D.predict` → `self.network_3d.get_class_agnostic_masks(processed_scene, datatype)` (`utils/__init__.py:115`)

**Wrapper**: `Network_3D` (`utils/utils_3d.py:7`) — 단순 `mask3d` model wrapper. 한 scene당 **1회 호출**.

- **입력**: 전체 scene point cloud (`*.npy` 사전 처리 또는 원본 mesh). Frame 정보 미사용.
- **출력 (raw)**: `(masks: bool[V, K_raw], scores: float[K_raw])` — V = vertex 수, K_raw = Mask3D가 뽑은 raw proposal 수 (~수백).
- **Post-processing** (`utils/__init__.py:116-118`):
  1. score threshold (`network3d.th`)로 필터
  2. `apply_nms`로 3D mask IoU NMS (`network3d.nms`)
  3. → `self.preds_3d = (masks: bool[V, K], scores: float[K])` — K = filtered proposal 수 (보통 50-150).
- 또는 `path_to_3d_masks` 인자 지정 시 사전 저장된 `.pt` 로드 (Mask3D 재호출 회피).

**Streaming 관점**: Mask3D는 시간 정보를 보지 않고 mesh geometry만으로 instance를 추출. **scene 시작 시 1회 호출 → 결과 캐시** 가능 (옵션 (다)의 핵심). 본 task의 streaming protocol에서 Mask3D는 변경 없이 그대로 사용.

---

## 4. YOLO-World 호출

**위치**: `OpenYolo3D.predict` → `self.network_2d.get_bounding_boxes(self.world2cam.color_paths, text)` (`utils/__init__.py:127`)

**Wrapper**: `Network_2D` (`utils/utils_2d.py:36`).

- `get_bounding_boxes` (`utils_2d.py:60`): scene의 **모든 subsampled frame (color_paths 리스트)** 을 순회. 각 frame은 `inference_detector([image_path])`로 **개별 호출** (`tqdm` loop, 배치 크기 1).
- 출력: `scene_preds: dict[frame_id_str → {"bbox", "labels", "scores"}]`
  - `frame_id_str`은 `osp.basename(image_path).split(".")[0]` — 즉 `"0"`, `"10"`, `"20"`, … (정수 frame index string).
- Per-frame post-processing (`utils_2d.py:93-104`):
  - NMS (`network2d.nms` IoU threshold)
  - score threshold (`network2d.th`)
  - top-K bbox만 유지 (`network2d.topk = 100`)
  - 이미지 거의 전체를 덮는 bbox는 제거 (resolution-50px 휴리스틱)

**중요**: 호출 자체는 frame 단위지만 **루프 안에서 순차적**으로 진행되므로, **streaming wrapper에서는 frame t 처리 시점에 YOLO-World on frame t만 호출**하면 됨 — 코드 수정 없이 호출 순서만 바꾸면 됨 (per-frame 호출 인터페이스가 이미 존재).

---

## 5. Frame visibility 계산 (`WORLD_2_CAM.get_mesh_projections`)

**위치**: `utils/__init__.py:391` — `predict` 진입 시 `self.world2cam.get_mesh_projections()`로 한 번 계산하여 캐시.

**입력**:
- 전체 mesh의 V vertex × 4 (homogeneous coords)
- 각 frame의 intrinsic + extrinsic (모든 F frame stack)
- 각 frame의 depth map (모든 F frame stack, `load_depth_maps`)
- 설정: `vis_depth_threshold = 0.05m` (ScanNet)

**계산**:
1. World → camera projection (extrinsic 곱하기): `word2cam_mat[F, V, 3]` (xy 이미지 좌표 + z=projected depth).
2. **Frustum test**: `projected_points = (x/z, y/z)`, `inside_mask[f,v] = (0 ≤ u < W) ∧ (0 ≤ v < H)`.
3. **Depth consistency**: `|projected_z - depth_map[f, v_pixel, u_pixel]| ≤ vis_depth_threshold` (= 0.05m). 통과 못한 vertex는 `inside_mask` False로 덮어씀 (`utils/__init__.py:467`).

**출력** (cached on `self.mesh_projections`):
- `projected_points: int16[F, V, 2]` — 각 frame f에서 vertex v의 픽셀 좌표 (u, v). frustum 밖이어도 채워짐.
- `inside_mask: bool[F, V]` — frame f에서 vertex v가 보이는가 (frustum + depth 일치).

**Streaming 관점**: 이게 본 task의 **frame visibility 그 자체**. 옵션 (다) 구현 시:
- Mesh와 모든 pose가 사전 가용 → `get_mesh_projections`를 그대로 호출해 캐시
- 매 frame t에서 `inside_mask[t]`를 vertex-level visibility로, **Mask3D instance**별 visibility는 `inside_mask[t]`와 instance mask의 dot product
- **현재 사용 중인 정의가 곧 옵션 D1** (frustum + depth). D2 (≥N% vertex) / D3 (≥1 vertex)는 vertex-level mask 위의 instance-level aggregation 임계값 차이일 뿐, frame visibility 자체 정의는 같음.

---

## 6. MVPDist 흐름 (`label_3d_masks_from_label_maps`)

**위치**: `utils/__init__.py:158` — `label_3d_masks_from_2d_bboxes`가 호출.

**입력**:
- `prediction_3d_masks: bool[V, K]` (Mask3D 출력, vertex × instance)
- `predictions_2d_bboxes: dict[frame_id_str → {"bbox", "labels", "scores"}]` (YOLO-World 출력 전체 frame)
- `projections_mesh_to_frame: int16[F, V, 2]`, `keep_visible_points: bool[F, V]` (`WORLD_2_CAM` 출력)

**Step 1 — label map 구성** (`construct_label_maps`, `utils/__init__.py:264`):
- 매 frame f에서 `label_map[f, H, W]`를 `-1`로 초기화
- bbox를 큰 것부터 작은 것 순으로 정렬 후 bbox 사각형 영역에 `label_id`를 painting (큰 bbox 위에 작은 bbox가 덮어씀 = "inner box wins")
- 결과: `label_maps: int16[F, H, W]`

**Step 2 — visibility matrix** (`get_visibility_mat`, `utils/__init__.py:65`):
- 각 instance k가 각 frame f에서 얼마나 보이는지: `vis_score[k, f] = (mask_k ∩ inside_mask[f]).sum() / mask_k.sum()`
- 인스턴스마다 **top-K frame** (ScanNet config: `topk = 40`, GT 분기에서는 25)을 선택. 나머지 frame은 무시.
- 결과: `visibility_matrix_bool: bool[K, F]` (instance마다 ≤K개 frame True).

**Step 3 — per-instance label voting** (`utils/__init__.py:183-238`):
- 각 instance k에 대해:
  - 선택된 top-K representative frames만 본다.
  - 각 representative frame f에서: instance k의 visible vertices만 골라 그 (u, v) 픽셀 좌표에서 `label_maps[f, v, u]`를 lookup → 한 frame에서 visible vertex 수만큼의 라벨 vote
  - 모든 representative frame의 vote를 합쳐 `labels_distribution`을 만든다.
  - `class_label = mode(labels_distribution)`
  - `class_prob = (mode count / prob_normalizer) × mean_IoU_with_bbox` (IoU는 `instance_x_y_coords`의 bbox vs YOLO bbox, top-K 평균)
- `topk_per_image` (ScanNet = 600): 마지막에 (instance × class) 분포 전체에서 top-600개 instance-label pair만 남기는 후처리.

**출력**: `(prediction_3d_masks: bool[V, K_final], pred_classes: int[K_final], pred_scores: float[K_final])`

**Streaming 관점에서 비교**:

| 측면 | 현재 (offline / batch) | 본 task의 streaming (옵션 다) |
|---|---|---|
| Frame 사용 시점 | 모든 frame 수집 후 한 번에 평가 | 매 frame t에서 frame ≤ t까지만 평가 |
| Top-K frame 선택 | 전체 F frame 중 top-K | 누적 vote (frame ≤ t) 또는 sliding window |
| Label 결정 | `mode(all votes)` once | 매 frame 누적 vote로 매번 재계산, M21/22가 자리잡음 |
| Score | mode_freq × mean_IoU (모든 frame 평균) | 매 frame 재계산 (incremental smoothing 가능) |
| Spatial merging | 미적용 (M31/32가 후처리로 들어감) | 매 frame 또는 K frame마다 merge 호출 |
| Instance registration | 없음 (Mask3D 결과 전체 노출) | M11/12가 visibility-gate로 작동 |

---

## 7. 본인 method (5월 결과 재확인 + streaming 관점에서의 위치)

5월 ScanNet 측정에서 각 method axis는 offline setting에서 다음 hook으로 들어간다 (`method_scannet/hooks.py`):

| Method | Hook | Patched function | Patch 위치 |
|---|---|---|---|
| M21 (WeightedVoting) | `install_method_21` | `OpenYolo3D.label_3d_masks_from_label_maps` | MVPDist 전체 대체 |
| M22 (FeatureFusionEMA) | `install_method_22_only` | `OpenYolo3D.label_3d_masks_from_2d_bboxes` (래퍼) | MVPDist 결과 위에서 visual feature로 재라벨링 |
| M31 (IoUMerger) | `install_method_31` | `OpenYolo3D.label_3d_masks_from_2d_bboxes` (postprocess) | 최종 출력 후 IoU merge |
| M32 (HungarianMerger) | `install_method_32_only` | 최종 출력 후 Hungarian merge | |
| Phase1 | `install_phase1` | M21 + M31 동시 install | |
| Phase2 | `install_phase2` (b62ff80) | M22 + M32 동시 install | |

M11/12는 5월 ScanNet에서 skip — 정적 scene에서 "registration gate"가 의미 없음 (모든 instance가 이미 존재). **Online streaming에서 M11/12가 비로소 의미를 가진다**:
- M11 (frame-counting): instance가 K개 frame에서 연속/누적 detected되어야 "공식 instance map"에 등록
- M12 (Bayesian): 누적 detection 확률 P(instance is real | history) ≥ τ일 때 등록

→ **본인 method = "proposal-agnostic temporal consistency layer"** 의 핵심이 M11/12. ScanNet offline에서는 못 보였던 가치를 streaming setting에서 측정.

---

## 8. Pipeline flow diagram (text)

```
[Scene t=0]
  WORLD_2_CAM.__init__(scene)
    └─ enumerate poses/0.txt..N.txt (frequency=10 subsample)
    └─ load intrinsics, color_paths, depth_paths
  get_mesh_projections()
    └─ project V mesh vertices to each of F frames
    └─ inside_mask[F, V] = frustum ∧ |proj_z - depth_map[u,v]| ≤ 0.05m
  Mask3D.get_class_agnostic_masks(scene_pc)        # one shot
    └─ (masks[V,K], scores[K]) after NMS+threshold
  YOLO-World.get_bounding_boxes(all_color_paths)   # per-frame loop, but all at once
    └─ dict[frame_id_str → {bbox, labels, scores}]
  label_3d_masks_from_label_maps                   # MVPDist
    └─ construct_label_maps(F, H, W)
    └─ get_visibility_mat → top-K representative frames per instance
    └─ for each (instance, top-K frames): accumulate label votes
    └─ pred_class = mode, pred_score = (mode_freq/prob_norm) × mean_IoU
  → predictions[scene_name] = (masks, classes, scores)

[Loop over 312 scenes]
[Aggregate predictions]
evaluate_scannet200(preds, gt_dir)
  → (avgs, ar_avgs, rc_avgs, pcdc_avgs)
_maybe_dump_metrics → metrics.json
```

Streaming 변환 (옵션 (다)) 시 위 흐름이 다음으로 바뀐다 (Stage 2에서 상세 설계):

```
[Scene t=0]  Mask3D + WORLD_2_CAM init (same as offline) — pre-scene one-shot
[Frame t = 0, 10, 20, ...]
  visible_instances_t = {k : (mask_k ∩ inside_mask[t]).sum() satisfies D1/D2/D3 threshold}
  YOLO-World on color_paths[t]   ← single frame
  for k in visible_instances_t:
      accumulate label vote (M21/22 layer)
  if frame t % merge_interval == 0:
      apply M31/32 post-merge on current_instance_map
  if M11/12 enabled:
      gate visible_instances_t against accumulated detection history
  → emit current_instance_map(t)  ← incremental metrics 측정 대상
```

---

## 9. Streaming 변환을 위해 활용 가능한 자산

| 자산 | 위치 | 변경 필요? |
|---|---|---|
| 전체 mesh / pose / intrinsic / depth/ color 로딩 | `WORLD_2_CAM.__init__` | 불필요 (그대로 사용) |
| Per-frame mesh→pixel projection + visibility | `WORLD_2_CAM.get_mesh_projections` | 불필요 (출력의 frame-axis slicing만) |
| Mask3D 1회 호출 + NMS/threshold | `Network_3D` + `predict` lines 114-118 | 불필요 (그대로 사용) |
| YOLO-World per-frame inference | `Network_2D.inference_detector([image_path])` | **이미 per-frame 인터페이스 존재** |
| Label map 구성 (1 frame 단위) | `construct_label_maps` (frame loop 내부) | **frame 단위 호출 가능 — 외부에서 frame 1개씩 줘도 됨** |
| MVPDist 메인 로직 | `label_3d_masks_from_label_maps` | 대체 (M21/22 hook이 이미 함) |
| install_method_21/31, phase1, phase2 등 | `method_scannet/hooks.py` | streaming-aware 신규 install (Stage 2) |

**중요**: OpenYOLO3D core는 streaming wrapper만 추가하면 끝. 핵심 메서드는 frame-level granularity로 호출 가능한 형태로 이미 구성돼 있음 — 5월에 작업한 monkey-patch 방식을 streaming용으로 확장 가능 (코어 변경 0건 유지).

---

## 10. 핵심 발견 요약

1. **Mask3D는 scene당 1회 — 시간 정보 미사용.** 옵션 (다)와 자연 일치. Streaming wrapper는 Mask3D 출력을 캐시해 매 frame visible subset만 expose하면 됨.
2. **Frame visibility (`inside_mask`)는 이미 옵션 D1 정의를 갖는다**: frustum + depth (0.05m). D2 / D3 차이는 **instance-level threshold** (몇 % vertex 또는 1개 이상)이며 frame-visibility 정의 자체 차이는 아님. → Stage 2에서 D1/D2/D3 비교는 "instance가 frame에서 visible하다"의 판정 기준 차이로 정의.
3. **YOLO-World는 per-frame 인터페이스가 이미 존재** (`inference_detector([image_path])`). Streaming wrapper에서 frame t 도착 시 호출.
4. **MVPDist (label_3d_masks_from_label_maps)는 전체 frame 일괄 처리**. 이를 frame-incremental 누적으로 바꾸는 게 streaming의 핵심 — **M21/22가 자연스럽게 그 자리에 들어간다**.
5. **M11/12는 streaming에서 비로소 의미** — 정적 ScanNet offline에서 skip했던 instance registration axis가 본 task에서 활성화.
6. **OpenYOLO3D core 코드 변경 0건 유지 가능**. Streaming wrapper + 추가 hook으로 모두 처리.

다음 단계: Stage 2 — 옵션 (다)의 frame visibility 정의(D1/D2/D3) 비교 + streaming flow + method integration 자리.
