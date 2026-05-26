# Task 1.1 — Stage 2: 옵션 (다) streaming evaluation 구현 spec

**Branch**: `feature/method-scannet-21-31`
**Date**: 2026-05-12
**Scope**: 코드 변경 0건. Stage 1 분석을 바탕으로 `Mask3D per-scene + frame-visible filtering` (옵션 (다)) 의 streaming evaluation 흐름을 설계.
**입력**: Stage 1 (`task_1_1_pipeline_analysis.md`).
**출력**: 본 문서. Stage 3 metric spec과 함께 Task 1.2(구현) 진입 근거.

---

## 1. 옵션 (다) 재확정

> **Mask3D per-scene (offline 한 번) + frame-visible filtering + streaming 2D label fusion**

| 단계 | 시점 | 호출 |
|---|---|---|
| scene 시작 | offline (1회) | `Mask3D` on full mesh → instance masks `M[V, K]` + scores |
| scene 시작 | offline (1회) | `WORLD_2_CAM.get_mesh_projections()` → `inside_mask[F, V]`, `projected_points[F, V, 2]` 캐시 |
| frame t = 0, 10, 20, … (streaming) | online | YOLO-World on color[t] → 2D bbox + label per frame |
| frame t | online | visible instance subset → 본인 method (M11/12/21/22) → label assign |
| frame t | online | (옵션) 누적 instance map에 M31/32 merge 적용 |
| frame t | online | incremental mAP / temporal metrics 측정 |

Stage 1 §5 정리대로 OpenYOLO3D core는 frame-단위 호출을 모두 지원 (per-frame YOLO inference, per-frame label map 구성). **Streaming wrapper만 추가**.

---

## 2. Frame visibility 옵션 D1 / D2 / D3 비교

Stage 1 §5에서 정리했듯 OpenYOLO3D의 `inside_mask[F, V]`는 **vertex-level** visibility (frustum + depth ≤ 0.05m). D1 / D2 / D3는 **이 vertex-level mask를 instance-level "이 instance를 frame t에서 처리할지" 결정 기준으로 어떻게 변환하느냐**의 차이.

### 정의 (정밀하게)

Frame t에서 Mask3D instance k의 vertex-level intersection:

```
visible_count(k, t) = |mask_k ∩ inside_mask[t]|
visible_frac(k, t)  = visible_count(k, t) / |mask_k|
```

| 옵션 | Instance-level 결정 | 의미 |
|---|---|---|
| **D1**: pure vertex-level | "instance"라는 1차 시민이 없음. 매 frame t에서 visible vertex 전부를 처리 (vertex별 label vote 누적, instance level은 후처리로 mode) | per-vertex granularity. Mask3D instance를 활용하지 않고 vertex stream으로 다룸 |
| **D2**: instance + N% threshold | `visible_frac(k, t) ≥ N%` 일 때만 instance k 처리 (strict) | 잘 보이는 instance만 frame t에 vote 추가. False positive bbox 영향 적음 |
| **D3**: instance + ≥1 vertex threshold | `visible_count(k, t) ≥ 1` 이면 instance k 처리 (lenient) | 부분적으로 보이는 instance도 vote. True positive recall 보존, false positive 노출 |

### 구현 난이도

| 옵션 | 구현 단계 | 추가 코드량 |
|---|---|---|
| D1 | vertex-level voting: frame t마다 `inside_mask[t]`의 visible vertex (=픽셀)에 대해 `label_maps[t]` lookup → vertex별 label histogram 유지 (`V × num_classes` cumulative table). Instance 결정은 매번 vertex histogram을 `mask_k`에 따라 reduce. | ~150 LoC (vertex histogram cum-table 관리, instance reduce 매번 또는 사후) |
| D2 | instance-level gate: `visible_frac(k, t) ≥ N` 필터 + 본인 method (5월 M21/M22 코드 거의 그대로 frame-incremental 호출). | ~80 LoC (gate + frame-incremental wrapper) |
| D3 | D2와 동일, threshold만 `visible_count ≥ 1`. | ~80 LoC |

### 본인 method 평가 영향

본인 method (특히 M11/12) = **"proposal-agnostic temporal consistency layer"**. 평가 목적은 이 layer가 단독 false positive / label noise를 얼마나 정화하느냐를 측정.

| 옵션 | M11/12 영향 | M21/22 영향 | M31/32 영향 |
|---|---|---|---|
| D1 | M11/12는 본질적으로 **instance-level** gate. D1은 instance가 1차 시민이 아니므로 M11/12를 자연스럽게 적용하기 어렵다 (vertex-level gate로 강제 변형 필요). 본인 method의 instance-level 가치를 측정 못 함. | vertex-level vote 누적은 가능하나 5월에 구현한 M21/M22의 인터페이스와 충돌 (instance × frame 단위) → 재작성 필요. | instance가 명시적으로 없으므로 M31/32 적용 불가 또는 사후 vertex→instance reduce 결과에 한 번만. |
| D2 | instance gate이므로 자연. **단, gate가 이미 strict하면 false positive를 미리 걸러내므로 M11/12가 처리할 노이즈가 줄어 가치를 underestimate** (방어적 비교 어려움). | M21/22 5월 인터페이스 그대로 frame-incremental 호출. | instance map이 항상 깔끔 → M31/32 매 frame 또는 K frame마다 적용 가능. |
| D3 | M11/12가 **noise를 처리할 최대 기회** — false positive도 모두 입력으로 들어옴. M11/12 contribution이 명확히 측정됨. | 5월 인터페이스 그대로. False vote도 들어오므로 M21/22의 weighting / EMA가 일을 해야 함 — contribution 측정 적절. | 부분 instance 다수 → M31/32 핵심 작업 영역. |

### 권고: **D3 (lenient: ≥1 vertex)**

**이유 (요지)**:
1. 본인 method = "proposal-agnostic **temporal** consistency layer". 평가의 핵심은 layer가 **노이즈를 정화하는 능력**. D2 strict gate는 노이즈를 사전 차단 → layer의 가치를 underestimate.
2. D1은 instance 추적 자체를 포기하므로 M11/12 (instance registration) 와 M31/32 (spatial merging) 의 의미를 잃음 — 본인 method 4축 중 3축이 평가 불가.
3. D3는 OpenYOLO3D 5월 코드 인터페이스를 그대로 활용 가능 (`label_3d_masks_from_label_maps`의 frame loop을 외부 wrapper로 옮기는 정도).
4. D2는 **추가 ablation**으로 유의미 — Task 1.2 구현 후 D3 vs D2 (e.g., N=5%) 1-2개 비교 실험만 별도로 (Task 1.5 이후).

**구체 파라미터 (D3 권고)**: `visible_count(k, t) ≥ 1` (literal). 추가 hyperparameter 없음.

→ **본 task의 사용자 결정 필요 항목 #1**. 본인 권고는 **D3**. 대안: D2 (N=5% 정도부터 시작) 도 합리적이지만 본인 method 가치 underestimate 위험 명시.

---

## 3. Streaming evaluation flow (옵션 (다) + D3)

```
INPUT: scene_name, frame_freq, method_install (e.g., baseline / M11 / M21 / ... / phase1 / phase2)

[Scene start — offline pre-compute, 1회]
  world2cam = WORLD_2_CAM(scene_path, depth_scale, config)
  projected_points, inside_mask = world2cam.get_mesh_projections()    # [F, V, 2], [F, V]
  preds_3d = Network_3D.get_class_agnostic_masks(scene_pc)            # (mask[V, K], scores[K])
  prediction_3d_masks = preds_3d[0]                                    # bool[V, K]
  scaling_params = ...                                                 # same as OpenYolo3D.predict

[Per-frame streaming loop]
  state = {
    "label_vote_table":  zeros((K, num_classes)),   # cumulative per-instance label vote (M21/22 자리)
    "iou_scores":         {k: [] for k in range(K)},# per-instance frame IoU history
    "detection_count":    zeros(K, dtype=int),       # M11/12 instance registration용 누적
    "first_seen_frame":   {-1 for k in range(K)},    # time-to-confirm 측정용
    "confirmed":          zeros(K, dtype=bool),      # M11/12 result 또는 baseline (default True)
    "current_class":      -ones(K, dtype=int),       # 매 frame 갱신되는 prediction
    "current_score":      zeros(K),
  }
  current_instance_map = []   # 후처리 후 (M31/32) 적용 결과 캐시

  for t_idx, frame_id in enumerate(frame_ids):       # frame_ids = sorted color paths의 정수 인덱스
      # === Step A: instance-level visibility (D3) ===
      vis_count_t = (prediction_3d_masks & inside_mask[t_idx].unsqueeze(-1)).sum(dim=0)  # [K]
      visible_instances_t = (vis_count_t >= 1).nonzero().flatten()    # D3 gate

      # === Step B: 2D detection on this frame ===
      frame_preds_2d = network_2d.inference_detector([color_paths[t_idx]])
      label_map_t = construct_label_maps_single(frame_preds_2d, H, W)   # [H, W]

      # === Step C: M11/12 instance registration gate ===
      if method_uses_registration:
          # M11: detection_count[k] += 1; gate: detection_count[k] >= τ_11 → confirmed
          # M12: posterior update (Bayesian); gate: P_real[k] >= τ_12 → confirmed
          state["detection_count"][visible_instances_t] += 1
          newly_confirmed = registration_gate(state, method="M11" or "M12")
          # newly_confirmed instances 시점에 first_seen 기록 (이미 기록된 건 유지)
      else:
          # baseline: 모든 visible_instances_t 즉시 confirmed
          state["confirmed"][visible_instances_t] = True

      # === Step D: label vote accumulation (M21/22 자리) ===
      for k in visible_instances_t:
          if not state["confirmed"][k]:
              continue   # M11/12 gate 통과 전이면 label 누적도 보류 (또는 누적은 하되 expose만 보류)
          visible_v_mask = prediction_3d_masks[:, k] & inside_mask[t_idx]   # [V]
          pixel_xy = projected_points[t_idx][visible_v_mask].long()         # [n_v, 2]
          labels_at_pixels = label_map_t[pixel_xy[:,1], pixel_xy[:,0]]      # [n_v]
          labels_valid = labels_at_pixels[labels_at_pixels != -1]
          # baseline: state["label_vote_table"][k] += histogram(labels_valid)
          # M21: weighted vote (distance + center 계수 — 5월 코드와 동일)
          # M22: visual feature EMA (CLIP visual embedding extract → cosine similarity)
          update_label_vote(state, k, labels_valid, method=...)
          # IoU history (score용)
          if len(pixel_xy) > 0 and len(frame_preds_2d) > 0:
              instance_pixel_bbox = aabb(pixel_xy)
              iou_to_yolo = iou(instance_pixel_bbox, frame_preds_2d_bboxes)
              state["iou_scores"][k].append(iou_to_yolo.max())

      # === Step E: per-instance current class + score ===
      for k in range(K):
          if state["confirmed"][k] and state["label_vote_table"][k].sum() > 0:
              state["current_class"][k] = state["label_vote_table"][k].argmax()
              state["current_score"][k] = (
                  state["label_vote_table"][k].max() / state["label_vote_table"][k].sum()
                  * mean(state["iou_scores"][k] or [0.0])
              )

      # === Step F: spatial merging (M31/32 자리) ===
      if method_uses_spatial_merge and (t_idx % merge_interval == 0):
          current_instance_map = apply_spatial_merge(
              prediction_3d_masks[:, state["confirmed"]],
              state["current_class"][state["confirmed"]],
              state["current_score"][state["confirmed"]],
              method="M31" or "M32",
          )
      else:
          current_instance_map = (
              prediction_3d_masks[:, state["confirmed"]],
              state["current_class"][state["confirmed"]],
              state["current_score"][state["confirmed"]],
          )

      # === Step G: incremental metric 측정 (Stage 3 metric_spec 참고) ===
      visible_gt_up_to_t = compute_visible_gt(gt, inside_mask[:t_idx+1])
      primary_mAP_t    = scannet200_AP(current_instance_map, visible_gt_up_to_t)
      secondary_mAP_t  = scannet200_AP(current_instance_map, full_scene_gt)
      ID_switches_t    = update_id_switch_log(state, gt)
      label_switches_t = update_label_switch_log(state)
      record(scene_name, t_idx, primary_mAP_t, secondary_mAP_t, ID_switches_t, label_switches_t,
             confirmed_count=int(state["confirmed"].sum()), time_to_confirm_snapshot=state["first_seen_frame"])

  return per_scene_streaming_log[scene_name]
```

---

## 4. 본인 method (M11/12/21/22/31/32) integration 자리

| Method | Streaming flow의 자리 | 변경 정도 |
|---|---|---|
| **M11** (frame-counting) | Step C: `detection_count[k] += 1`; gate `detection_count[k] ≥ τ_11`. 5월 코드는 없으므로 신규. | 신규 (~50 LoC, M21처럼 dataclass + install hook) |
| **M12** (Bayesian) | Step C: posterior 갱신 `p(real \| history) = update(p_prev, new_evidence)`; gate `p ≥ τ_12`. 신규. | 신규 (~80 LoC) |
| **M21** (WeightedVoting) | Step D `update_label_vote`. 5월 `WeightedVoting`을 frame-by-frame 호출하도록 wrapper만 추가 (vote accumulation은 이미 누적식). | 거의 그대로 (5월 `method_21_weighted_voting.py`) |
| **M22** (FeatureFusionEMA) | Step D `update_label_vote`. EMA는 frame-by-frame 자연. 5월 `FeatureFusionEMA` 그대로. | 거의 그대로 (5월 `method_22_feature_fusion.py`) |
| **M31** (IoUMerger) | Step F `apply_spatial_merge`. 매 frame 또는 K frame마다 호출. `merge_interval`은 default = end-of-scene (1회) 또는 매 frame 둘 다 spec에 포함. | 거의 그대로 (5월 `method_31_iou_merging.py`) |
| **M32** (HungarianMerger) | Step F `apply_spatial_merge`. | 거의 그대로 (5월 `method_32_hungarian_merging.py`) |

**5월 hooks.py 와의 관계**: 5월 install hook은 OpenYolo3D의 메서드를 monkey-patch한다. Streaming wrapper는 OpenYolo3D 메서드를 **호출하지 않고** 위 flow를 자체적으로 구성하므로 5월 hook을 **재사용하지 않는다**. 대신 5월 method 클래스 (`WeightedVoting`, `FeatureFusionEMA`, `IoUMerger`, `HungarianMerger`)는 streaming wrapper가 직접 instance화해서 사용.

→ **5월 hooks.py 코드 변경 0건**, streaming wrapper에서 method 클래스만 직접 import.

---

## 5. 구현 단위 (Task 1.2 / 1.3 / 1.4 분할 제안)

본 Task 1.1은 spec만. Task 1.2부터 구현. 권고 단위:

### Task 1.2 — Streaming wrapper 코어 + baseline (M11/12/21/22/31/32 미적용)

- 신규 모듈: `method_scannet/streaming/`
  - `streaming_evaluator.py`: 위 §3 flow 구현 (baseline 경로, method install 안 함)
  - `single_scene_streaming.py`: 한 scene streaming inference + per-frame log
  - `streaming_state.py`: state dataclass
  - `tests/test_streaming_baseline.py`: scene0011_00 1-scene 단위 테스트 (offline baseline 결과와 final-frame AP가 거의 일치하는지 sanity check)
- **Acceptance**: scene0011_00에서 streaming baseline의 final-frame AP가 offline baseline AP의 ±0.005 이내 (D3로 모든 visible instance 처리 시 거의 동일해야 함; 차이는 top-K representative frame selection 차이뿐).
- 코드량 추정: ~300 LoC + 단위 테스트 ~150 LoC.

### Task 1.3 — Temporal metrics 측정 모듈 (Stage 3 metric_spec 구현)

- 신규 모듈: `method_scannet/streaming/metrics_streaming.py`
  - `incremental_mAP_primary` (visible GT up to t)
  - `incremental_mAP_secondary` (full GT)
  - `id_switch_count`
  - `label_switch_count`
  - `time_to_confirm`
- 단위 테스트: 합성 데이터 (3 instance × 5 frame mock) 1-2 case.
- 코드량 추정: ~250 LoC + 단위 테스트 ~150 LoC.

### Task 1.4 — Method integration (M11/12/21/22/31/32 streaming version)

- 신규 모듈:
  - `method_scannet/streaming/method_streaming.py`: 각 method를 streaming flow에 install/uninstall 하는 wrapper (5월 hooks.py 패턴 차용; 단 monkey-patch 아닌 dependency injection)
  - `method_11_frame_counting.py` (M11, 신규)
  - `method_12_bayesian.py` (M12, 신규)
- 단위 테스트: install/uninstall + 1-scene smoke.
- 12-way ablation을 위한 entry point:
  - baseline / M11 / M12 / M21 / M22 / M31 / M32
  - phase1_streaming (M21+M31) / phase2_streaming (M22+M32) / M11+M21 / M12+M22 / M11+M21+M31 / M12+M22+M32
- 코드량 추정: ~400 LoC + 단위 테스트 ~250 LoC.

### Task 1.5 — 12-way ablation 실행 + 결과 정리

- streaming version의 각 method install로 12개 .pbs 작성 (5월 패턴)
- 결과 표는 `results/experiment_tracker.md`에 새 section으로 추가
- 본 Task에 들어가지 않음 (구현 + 측정만).

### Task 1.6 (이후) — D2 ablation (옵션)

- D3 vs D2 (N=5%) 비교 — 본인 method가 노이즈를 처리하는 정도가 visibility gate에 얼마나 의존하는지 측정.
- D3가 본 contribution. D2는 sensitivity analysis.

---

## 6. 사용자 결정 필요 항목 (출력의 일부)

### #1: Frame visibility 옵션 — D1 / D2 / D3

- **본인 권고: D3** (≥1 vertex visible, instance-level lenient gate)
- 이유:
  1. 본인 method의 "temporal consistency가 노이즈 정화"라는 contribution을 fair하게 측정.
  2. 5월 OpenYOLO3D 코드 인터페이스 활용도 최고.
  3. M11/12/31/32 모두 의미 있게 적용 가능.
- 대안:
  - **D2** (예: N=5%): sensitivity check 용으로 별도 ablation (Task 1.6).
  - **D1**: 추천 안 함 (instance 1차 시민이 사라져 M11/12/31/32 평가 어려움).

### #2: Spatial merge frequency (M31/32 호출 주기)

본 spec의 Step F는 `merge_interval` 파라미터. 옵션:
- (i) **scene 끝에 1회** (offline-equivalent; 5월 결과와 직접 비교 가능)
- (ii) **매 frame** (streaming-pure; 매 frame 새 instance map; M31/32 cost 증가)
- (iii) **K frame마다** (compromise; K = 10 정도)

본 task spec에서는 (i)를 default로 권고 (Task 1.4 baseline에서). (ii)/(iii)는 Task 1.6 ablation. **이는 Task 1.4 시점 결정으로 충분 — 본 Task 1.1에서는 결정 불필요.**

---

## 7. 구현 시 주의 사항 (기록 목적)

- OpenYOLO3D core 메서드 (`Network_2D.inference_detector`, `Network_3D.get_class_agnostic_masks`, `WORLD_2_CAM.get_mesh_projections`, `construct_label_maps`) 는 **호출만** — 수정 안 함.
- 5월 method 모듈 (`method_21_*.py`, `method_22_*.py`, `method_31_*.py`, `method_32_*.py`) 는 import만 — 수정 안 함.
- 5월 `hooks.py`는 **재사용하지 않음** (offline 평가용). Streaming은 별도 entry point.
- Mask3D scene-cache: 5월에 `path_to_3d_masks` 인자 있음 (`OpenYolo3D.predict` line 119-120). Streaming에서도 동일 캐시 활용 (scene당 Mask3D 호출 1회).
- Time-to-confirm 계산에는 GT instance ID matching이 필요 — Stage 3 metric_spec 참조.
- ScanNet GT vertex count = mesh vertex count (V) 와 일치해야 매칭 가능. `evaluate_scannet200`가 이미 V-알라인 입력 받음.

---

## 8. 다음 단계

- **Stage 3** (metric_spec): incremental mAP / ID switches / label switches / time-to-confirm 수학적 정의 + 의사 코드.
- Stage 3 끝나면 본 Task 1.1 종료. **사용자 결정 (D1/D2/D3, time-to-confirm "confirmed" 정의) 후 Task 1.2 진입**.
