# Phase 2 통합 plan

METHOD_22 (FeatureFusionEMA) + METHOD_32 (HungarianMerger) 통합 시 결정해야
할 항목 정리. 코드 작업 아니라 **결정 자료**. METHOD_31 단독 ablation 결과를
기다리는 동안 평행 세션에서 작성됨.

## 1. 현재 상태 (2026-05-07 기준)

| 항목 | 상태 |
|---|---|
| Baseline (ScanNet200) | 완료 |
| Phase 1 = METHOD_21 + METHOD_31 | 완료, AP −0.0029 vs baseline (회귀) |
| METHOD_31 단독 ablation | 진행 중 (다른 터미널, GPU) |
| METHOD_21 단독 ablation | 미시작 |
| METHOD_22 코드 + 단위 테스트 | 완료 (5/5 PASS) |
| METHOD_32 코드 + 단위 테스트 | 완료 (5/5 PASS) |
| ScanNet200 prompt embeddings 캐시 | 완료 — `pretrained/scannet200_prompt_embeddings.pt` (200×512) |

다음 차단요소: METHOD_31 단독 결과. 결과 분기에 따라 통합 우선순위가 갈림.

## 2. METHOD_31 ablation 결과별 통합 분기

### Case A — METHOD_31 POSITIVE (baseline 대비 +)
- METHOD_31 자체는 working → Phase 1 회귀의 원인은 **METHOD_21 (weighted
  voting)** 일 가능성이 큼
- 우선순위:
  1. METHOD_22 단독 통합 → MVPDist 대체로 METHOD_21을 갈아끼움
  2. Mix-B (METHOD_22 + METHOD_31) 평가 → Phase 1 대체 후보
  3. Phase 2 (METHOD_22 + METHOD_32) 평가
- METHOD_21 단독 ablation은 **불필요** — Case A에서는 METHOD_21이 negative라는
  결론이 자동으로 나옴 (Phase1−METHOD_31단독 = METHOD_21의 marginal 기여)

### Case B — METHOD_31 NEUTRAL (baseline ±0)
- METHOD_31 효과 미미. METHOD_21 단독 결과를 알아야 회귀 원인 분리 가능.
- 우선순위:
  1. METHOD_21 단독 ablation 던지기
  2. METHOD_32 단독 통합 평가 — METHOD_31과 같은 axis에서 다른 알고리즘
  3. METHOD_31 vs METHOD_32 직접 비교 → 더 working 쪽 선택해서 Phase 2 진행

### Case C — METHOD_31 NEGATIVE (baseline 대비 −)
- METHOD_31 자체가 harm → METHOD_21이 살릴 가능성도 있음
- 우선순위:
  1. METHOD_21 단독 ablation 우선 (METHOD_21 working 여부 확정)
  2. METHOD_32로 METHOD_31을 대체 (Mix-A: METHOD_21 + METHOD_32)
  3. METHOD_31 IoU threshold 재검토 (현재 0.5 → 0.7 등)

## 3. METHOD_22 통합 시 본인 결정 항목

### 3.1 Visual embedding 추출 위치

METHOD_22.md: "각 프레임의 2D ROI에서 추출된 시각적 임베딩"

세 가지 옵션:

| 옵션 | 장점 | 단점 |
|---|---|---|
| (a) Cropped bbox에서 CLIP image feature | 구현 단순, 빠름 | bbox 안 background 픽셀 포함 |
| (b) Mask region pooling (SAM mask 등) | 객체 영역만 정확히 | mask 사전 추출 필요, 느림 |
| (c) **OpenYOLO3D 원본 routing 그대로 사용** | OpenYOLO3D와 일관성 보존 | 어디서 image feature가 나오는지 파악 필요 |

**조사 결과**: OpenYOLO3D는 **별도의 CLIP image encoder를 돌리지 않음**. 분류는
YOLO-World가 detection 단계에서 (text-image alignment를 통해) 직접 하고, 3D
projection 후에는 `label_3d_masks_from_label_maps` (per-pixel argmax 다수결,
METHOD_21이 이걸 대체)로 라벨이 결정됨.

따라서 METHOD_22는 visual embedding을 **새로 만들어야** 한다. 가장 일관된
경로는:

- **(a) cropped bbox에서 CLIP image encoder 호출** (`openai/clip-vit-base-patch32`,
  text encoder와 같은 variant라 임베딩 공간 정렬됨).
- 입력: per-frame 2D bbox + 이미지 → CLIP image projection → (B, 512) → 정규화.
- METHOD_22 EMA 누적은 instance ID별로 cross-frame.

**권고**: (a) 시작. 통합 후 mask region pooling으로 ablation 확장 가능.

### 3.2 EMA hyperparameter

- `ema_alpha = 0.7` (METHOD_22.md default)
- 너무 높음 (0.9): 최신 frame 반영 적음, 첫 프레임 bias
- 너무 낮음 (0.3): noise 많음
- **Ablation grid**: `[0.5, 0.7, 0.9]` — 통합 + 평가 후

### 3.3 Background class 처리

OpenYOLO3D는 inference 시 199 + 1 background로 분류. `predict_label`이
background를 골랐을 때:
- 옵션 1: 해당 instance를 prediction에서 drop
- 옵션 2: 가장 confident한 비-background class로 fallback
- ScanNet200 evaluator가 background를 어떻게 처리하는지 먼저 확인 필요

## 4. METHOD_32 통합 시 본인 결정 항목

### 4.1 Class-agnostic vs class-aware

METHOD_31은 class-aware (same-class only). METHOD_32 spec엔 명시 없음.

| 옵션 | 장점 | 단점 |
|---|---|---|
| (a) Class-agnostic | occlusion + viewpoint change 강함 | 책상 위 노트북 같은 케이스 false merge |
| (b) Class-aware (METHOD_31 일관) | 안전 | 의미적 일관성 활용 약함 |

**권고**: METHOD_32.md에 "복잡한 환경에서도 객체 단위 일관성"이라 했으니
**class-aware 시작 → ablation으로 class-agnostic 검증**. 코드는 이미 input에
label 정보가 들어있으므로 wrapper 한 줄로 분기 가능.

### 4.2 Hyperparameter

| 파라미터 | Default | Ablation grid |
|---|---|---|
| `spatial_alpha` (거리 vs 의미 가중) | 0.5 | [0.3, 0.5, 0.7] |
| `distance_threshold` (m) | 2.0 | [1.0, 2.0, 3.0] |
| `semantic_threshold` (cos sim) | 0.3 | [0.2, 0.3, 0.4] |

ablation grid 27 combos는 과함 — alpha 먼저 sweep, 베스트 alpha에서 두 threshold 잡기.

### 4.3 METHOD_31과의 stack 순서

Phase 2 = METHOD_22 + METHOD_32. METHOD_22는 분류 단계, METHOD_32는 후처리
merge 단계 → 순서 명확:
1. (Detection) YOLO-World 2D bbox
2. (METHOD_22) Per-instance visual EMA + cosine 분류 → final label
3. (METHOD_32) Hungarian merge — distance + semantic feature 사용

Mix-B (METHOD_22 + METHOD_31): 분류만 22로 갈고 merge는 31 (IoU NMS) 유지.
Mix-A (METHOD_21 + METHOD_32): 기존 METHOD_21 유지, merge만 32로.

## 5. bbox_3d 채우기 헬퍼

METHOD_32 입력 `instance_list[i]['bbox_3d']`는 optional이지만, envelope 계산엔
필요. OpenYOLO3D는 vertex-level bool mask `(V, K)` 형태로 mask를 주므로,
헬퍼 한 함수면 충분:

```python
def compute_aabb_from_vertex_mask(mask_V, vertex_coords_V3) -> np.ndarray:
    """Return (2, 3) [[x_min,y_min,z_min],[x_max,y_max,z_max]]."""
    pts = vertex_coords_V3[mask_V]
    if pts.size == 0:
        return None
    return np.stack([pts.min(axis=0), pts.max(axis=0)], axis=0)
```

위치 후보: `method_scannet/utils_bbox.py` (신규). METHOD_31 단독 ablation 끝난
후 작성. **본 평행 세션에선 만들지 않음** (보호 파일에 영향 없음).

## 6. hooks.py 통합 spec (참고)

기존: `install_phase1()` = METHOD_21 wrapper + METHOD_31 post-merge.

추가 예정:

| 함수 | 효과 |
|---|---|
| `install_method_22_only()` | METHOD_22 wrapper (MVPDist 대체) |
| `install_method_32_only()` | METHOD_32 post-merge |
| `install_phase2()` | METHOD_22 + METHOD_32 동시 |
| `install_mix_a()` | METHOD_21 + METHOD_32 |
| `install_mix_b()` | METHOD_22 + METHOD_31 |

각 install/uninstall 함수는 monkey-patch + `_method22_*`/`_method32_*` 인스턴스
attribute 패턴 일관 유지 (METHOD_31의 `_method31_merger` 참조).

prompt embeddings 로딩은 `install_phase2()` / `install_method_22_only()`에서
`pretrained/scannet200_prompt_embeddings.pt`를 한 번 로드해 `OpenYolo3D` 인스턴스
attribute로 stash. 199-class inference subset이 필요하면 `class_names`와
`openyolo3d_inference_classes`를 비교해 row-slice.

## 7. 실험 plan (5 합의 + ablation)

### 합의된 5 실험
1. baseline (완료)
2. Phase 1 = METHOD_21 + METHOD_31 (완료, AP −0.0029)
3. **Phase 2 = METHOD_22 + METHOD_32**
4. **Mix-A = METHOD_21 + METHOD_32**
5. **Mix-B = METHOD_22 + METHOD_31**

### Ablation (병행)
- METHOD_31 단독 (현재 진행 중)
- METHOD_21 단독 (METHOD_31 결과 분기에 따라)
- METHOD_22 단독, METHOD_32 단독 (Phase 2 통합 후)

## 8. 다음 즉시 작업 우선순위

METHOD_31 단독 ablation 결과 받은 직후:

1. **결과 검토** → POSITIVE / NEUTRAL / NEGATIVE 분기 (§2 참조)
2. 분기에 따라 METHOD_21 단독 ablation 던질지 결정 (Case B/C는 던짐, Case A는 skip)
3. `hooks.py`에 `install_phase2`, `install_mix_a`, `install_mix_b` 추가
4. **METHOD_22 통합 코드** — visual embedding 추출 (§3.1 옵션 (a))
5. **`utils_bbox.py` 헬퍼** 작성 (§5)
6. Phase 2 평가 → mix 평가 → 결과 비교 표

## 9. 본 평행 세션의 산출물

| 파일 | 상태 |
|---|---|
| `method_scannet/method_22_feature_fusion.py` | ✅ 작성, 단위 테스트 PASS |
| `method_scannet/method_32_hungarian_merging.py` | ✅ 작성, 단위 테스트 PASS |
| `method_scannet/tests/test_method_22.py` | ✅ 5 PASS |
| `method_scannet/tests/test_method_32.py` | ✅ 5 PASS |
| `method_scannet/extract_prompt_embeddings.py` | ✅ 작성 + 실행 |
| `pretrained/scannet200_prompt_embeddings.pt` | ✅ 생성 (200×512, 405 KB) |
| `docs/scannet200_classes_location.md` | ✅ 작성 |
| `docs/phase2_integration_plan.md` | ✅ 본 문서 |

위 어느 것도 `hooks.py` 등 ablation 작업 중인 보호 파일을 건드리지 않음.
git commit은 본인 검토 후 결정.
