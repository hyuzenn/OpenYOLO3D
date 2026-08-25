# 교수님 피드백 3-Task 설계 및 검증 계획서

> Status: **설계 문서 (구현 X)** — 기존 baseline 동작 변경 금지. 모든 신규 코드는
> CLI 플래그 또는 별도 모듈로 격리. 본 문서는 코드 수정 위치, 인터페이스, 리스크,
> 우선순위, 난이도, 실험 순서까지 포함한다.
> Date: 2026-06-03 · Author: Claude (Opus 4.7) for OpenYOLO3D project

## 0. 공통 격리 원칙

모든 Task는 아래 격리 원칙을 따른다.

1. **baseline 경로 보존**: `utils/__init__.py::OpenYolo3D` 및 ScanNet200 기본
   파이프라인은 직접 수정하지 않는다. 모든 변경은 `method_scannet/hooks.py`의
   install_* 함수가 토글하는 *patched method*를 통해서만 적용된다 (현행 M22/M31
   설치 패턴과 동일).
2. **CLI 플래그**: 각 신규 기능은 `eval_method_22_*` 또는 `nuscenes_native_evaluator`의
   argparse 옵션으로만 활성화되며, 미지정 시 현행 동작 100% 유지.
3. **새 파일 우선**: 기존 모듈 (`method_22_feature_fusion.py`, `centerpoint_proposals.py`
   등) 의 public API를 깨지 않는 대신, 신규 클래스/모듈을 추가하고 옛 코드는 옵트인.
4. **재현성**: 모든 설계안은 `results/YYYY-MM-DD_<exp>_vNN/` 규칙 준수, `config.yaml`
   스냅샷 + `metrics.json` 저장.

---

# Task 1 · Confidence-Weighted CLIP EMA

## 1.1 현재 CLIP EMA 업데이트 경로 분석

**파일**: `method_scannet/method_22_feature_fusion.py` · `method_scannet/hooks.py:434-579`

핵심 업데이트 식 (`method_22_feature_fusion.py:104-117`):

```
f_t(i) = α · f_{t-1}(i) + (1-α) · f_current(i)
```

- `α = ema_alpha = 0.7` (eval_method_22_only*.py 기본).
- `f_current` = `CLIPImageEncoder.encode_cropped_bboxes(image, matched_bbox)`
  (512-D CLIP image feature, L2 정규화는 *옵션 normalize_per_frame*).
- 업데이트 호출처: `hooks.py:578-579`

  ```python
  for prop_idx, emb in zip(matched_prop_ids, embs):
      fusion.update_instance_feature(prop_idx, emb)
  ```

- 매칭 로직 (`hooks.py:551-567`): 각 (proposal, frame) 쌍에 대해
  `compute_iou(projection_AABB, yolo_bboxes)`로 best-IoU bbox를 골라 그
  bbox를 CLIP 입력 패치로 사용. `iou_max < min_iou (=0.05~0.15)`이면 skip.
- **현 시점에서 confidence 신호는 사실상 무시됨**: M21(WeightedVoting)은
  `iou_max_val`을 `confidence`로 사용해 frame_weight에 곱하지만 (`hooks.py:161-193`),
  M22는 매칭된 bbox의 IoU/YOLO score 둘 다 EMA에 반영하지 않고 **모든 프레임
  embedding을 동일 비중**으로 누적함.

**Feature drift 가설 (교수님 가설)**:
- YOLO score가 낮은 (= occlusion, blur, partial view) bbox crop이 CLIP feature
  공간에서 큰 norm 또는 무관 카테고리 방향을 가질 수 있고, α=0.7로 누적되면
  소수의 noisy embedding이 EMA centroid를 끌어 당겨 cosine label argmax를
  drift시킨다.
- 실제로 `m22_m32_fix_ablation.py` + cosine probe (`diagnosis/cosine_distribution_probe.py`)
  결과에서 M22가 baseline 대비 AP 하락을 보였고, 본 설계는 **그 원인을
  feature drift로 검증하는 진단 + 보정** 두 단계로 분해한다.

## 1.2 Confidence-Weighted EMA 설계안 (3 variants)

세 변형 모두 `FeatureFusionEMA`의 시그니처를 *확장*만 한다 (default 인자 →
하위호환). 변경 위치: `method_scannet/method_22_feature_fusion.py`.

### A안 · Per-frame weight (가장 단순; 권장 1순위)

```
α_t(i) = clip( 1 − w_t , α_min , α_max )
f_t(i) = α_t(i) · f_{t-1}(i) + (1 − α_t(i)) · f_current(i)
```

- `w_t = conf_t · iou_t` (둘 다 [0,1]). 권장 기본 `α_min=0.5, α_max=0.95`.
- 직관: 높은 confidence frame일수록 (1−α) ↑ → 새 embedding 영향력 ↑.
- **장점**: 1줄 변경, 기존 호출 그대로, α만 동적으로 계산.
- **단점**: w 분포가 매우 좁으면 effectively constant EMA로 회귀할 수 있음.

### B안 · Sufficient-statistic weighted mean (이론적으로 가장 깔끔)

EMA를 가중 평균으로 재정의:

```
S_t = β · S_{t-1} + w_t · f_current        (모멘텀 β ∈ [0.9, 1.0])
W_t = β · W_{t-1} + w_t
f̂_t = S_t / max(W_t, ε)
```

- β=1.0이면 단순 confidence-weighted mean. β<1 은 시간적 forgetting.
- **장점**: w_t=0 frame을 자연스럽게 skip (식별 자명), 분산 추정도 가능.
- **단점**: 두 누적 상태(S,W)를 들고 다녀야 함 → instance dict shape 변경.

### C안 · Update-skip with running quality (가장 보수적)

```
if w_t < τ_skip:
    f_t(i) = f_{t-1}(i)              # skip
else:
    f_t(i) = α · f_{t-1}(i) + (1−α) · f_current
```

- `τ_skip ∈ {0.1, 0.2, 0.3}` 스윕.
- **장점**: drift 가설 검증에 직접적. EMA가 dirty frame을 절대 흡수하지 않음.
- **단점**: τ 튜닝 민감, low-conf instance 전체가 EMA를 갱신 못해 cold-start.

### D안 (보조) · A+C 결합

`(1−α_t) = w_t · 𝟙[w_t ≥ τ_skip]` — 매우 낮은 w는 hard-skip, 그 외엔 weighted.

| 안 | 추가 상태 | 계산 비용 | 구현 난이도 | drift 원인 분리력 |
|---|---|---|---|---|
| A | 없음 | O(1) | ★☆☆ | 중 |
| B | 2× 메모리 | O(1) | ★★☆ | 고 |
| C | 없음 | O(1) | ★☆☆ | 고 (가설 검증용) |
| D | 없음 | O(1) | ★★☆ | 고 |

## 1.3 Confidence Threshold 기반 update-skip 전략

권장: **C안을 진단용 plug-in으로 먼저 돌려 drift 가설을 검증한 뒤, A·B로 확장**.

- 신호 종류 (이미 코드상 사용 가능):
  - `iou_max` — `hooks.py:559` 매칭 IoU. M21에서 이미 confidence로 사용 중.
  - `yolo_score` — `preds_2d[f]["scores"]` (`utils/__init__.py:127` →
    `utils/utils_2d.py:103-106`). 현재 M22에는 미사용 (가져만 오면 됨).
  - `bbox_area_norm = (x2-x1)(y2-y1) / (W·H)` — 너무 작은 crop은 CLIP에서
    noisy. area<th_area (e.g. 0.005) 인 frame skip.
- 권장 결합: `w_t = yolo_score · iou_max · 𝟙[area > th_area]`.
- threshold 탐색: τ_skip ∈ {0.0, 0.1, 0.2, 0.3, 0.4}, single-scene smoke
  (`smoke_method_22_one_scene.py`) → 10-scene mini → full-val.

## 1.4 예상 장단점 및 Failure Case

| 상황 | A | B | C |
|---|---|---|---|
| 모든 frame에서 conf가 비슷하게 낮음 | drift 거의 동일 | mean이 robust | 전부 skip → cold-start, AP↓↓ |
| 한 instance가 occlusion-heavy | α↑ → 옛 feature 유지 (좋음) | W↑ 가 느려 stale | skip → 새 정보 못받음 |
| YOLO false-positive bbox가 IoU 우연히 ↑ | yolo_score가 낮으면 자동 보호 | 동일 | 동일 |
| Open-vocab 신규 클래스 (YOLO 미학습) | yolo_score 일률적 낮음 → 학습 약화 | 동일 | 모두 skip 위험 |
| 매우 작은 instance (멀리 있는 의자 등) | bbox_area 게이트로 보호됨 | 동일 | 동일 |

**Critical failure**: open-vocab 신규 카테고리에서 yolo_score가 구조적으로 낮음.
→ 보정: `w_t`에 prior 1.0을 floor (`w_t = max(w_t, w_floor=0.2)`) 또는
yolo_score 대신 iou_max만 사용하는 옵션 (`--m22-conf-source iou|both|score`).

## 1.5 최소 변경 구현 계획

**수정 위치**:
1. `method_scannet/method_22_feature_fusion.py`
   - `FeatureFusionEMA.__init__`: `conf_mode` ∈ {"none","weighted","skip","weighted_skip"},
     `tau_skip`, `w_floor`, `alpha_min`, `alpha_max`, `momentum_beta` 추가.
   - 신규 메서드 `update_instance_feature_weighted(iid, emb, w)` 또는
     기존 메서드에 `w: float = 1.0` 인자 추가 (default 1.0 → 기존 동작).
   - B안의 경우 `instance_features` dict에 `(S, W)` tuple로 보관 + property
     `centroid(i) = S/W` 노출.
2. `method_scannet/hooks.py:_apply_method_22`
   - `bboxes_2d_t = bbox_per_frame[f_idx]["bbox"]` 옆에서 `scores_t =
     bbox_per_frame[f_idx]["scores"]`도 같이 꺼낸다.
   - `iou_max`와 `scores_t[iou_argmax]`를 묶어 `w` 계산:

     ```python
     iou_v = float(ious[iou_argmax].item())
     yolo_v = float(scores_t[iou_argmax].item())
     area_v = (x2-x1)*(y2-y1) / (image.shape[1]*image.shape[0])
     w = iou_v * yolo_v * (1.0 if area_v >= th_area else 0.0)
     fusion.update_instance_feature(prop_idx, emb, w=w)
     ```
   - `_method22_state`에 `conf_mode`, `tau_skip`, `w_floor`, `th_area` 추가.
3. `method_scannet/eval_method_22_only_v2.py`
   - `install_method_22_only(...)`에 신규 인자 전달; CLI flag 추가:
     `--conf-mode {none,weighted,skip,weighted_skip}`,
     `--tau-skip`, `--w-floor`, `--alpha-min`, `--alpha-max`, `--th-area`.
   - **default = "none"** → 기존 결과 bit-identical.

**예상 코드 줄 수**: M22 모듈 +50줄, hooks.py +20줄, eval +15줄.

## 1.6 실험 순서 (우선순위 高 → 低)

| 순서 | 실험 | 비용 | 목적 |
|---|---|---|---|
| 1 | 진단: `--conf-mode skip --tau-skip {0.0,.1,.2,.3,.4}` smoke 1 scene | 분 | drift 가설 1차 검증 (skip이 AP↑면 drift 인정) |
| 2 | 10-scene mini sweep (skip + weighted A안) | 시간 | 안정성 확인 |
| 3 | Full scannet200 val: best-of-mini + baseline 비교 | 1 GPU·일 | 메인 결과 |
| 4 | A vs B vs C 직접 비교 (best τ 고정) | 0.5일 | ablation table |
| 5 | open-vocab 비편향 보강: yolo_score 미사용 옵션 ablation | 0.5일 | failure case 검증 |

---

# Task 2 · CenterPoint Proposal Recall Study

## 2.1 CenterPoint prediction 결과 활용 가능 여부

**현 상태 (코드 inventory)**:
- `adapters/centerpoint_proposals.py:CenterPointProposalGenerator` — mmdet3d
  CenterPoint (Voxel+SECFPN+CircleNMS, nuScenes-pretrained, 10-class) wrapper.
  Output schema (`adapters/centerpoint_proposals.py:166-177`):

  ```python
  {"proposals": [{"cls_idx", "cls_name", "score",
                  "bbox_lidar": [x,y,z,dx,dy,dz,yaw,(vx,vy)],
                  "centroid_ego": [x,y,z]}, ...],
   "n_proposals": int, "score_threshold_applied": float, "timing": {...}}
  ```
- `method_scannet/streaming/nuscenes_evaluator.py` (γ pipeline) → CenterPoint
  proposals + YOLO-World relabeling, mAP **0.0526**.
- `method_scannet/streaming/nuscenes_native_evaluator.py` (γ-fixed native) →
  CenterPoint native cls + score, mAP **0.3407** (Task 2.5 anchor).

→ **CenterPoint output은 이미 fully usable**. 별도 학습 불요, 인터페이스 안정.

**Outdoor 성능 저하 원인 후보 분해**:
1. **Proposal recall 부족** — outdoor에서 사용 중인 proposal source가 GT를
   충분히 cover하지 못함.
2. **2D ↔ 3D matching 노이즈** — CLIP/YOLO label과 LiDAR box의 IoU 매칭이 outdoor에서 깨짐.
3. **Label space mismatch** — OV vocabulary가 nuScenes-10 외 클래스에 분산.

본 Task는 **(1) 의 정량 검증**을 목적으로 한다.

## 2.2 Proposal Recall 진단 지표 설계

**대상 proposal sources** (4종 비교):
- `mask3d` — 현행 indoor 베이스, outdoor에서는 ScanNet200 pretrained
  Mask3D 적용 결과 (sanity용).
- `hdbscan` — `adapters/lidar_proposals.py:LiDARProposalGenerator` 결과.
- `centerpoint` — `adapters/centerpoint_proposals.py` (현행 γ source).
- `mask3d ∪ hdbscan ∪ centerpoint` — union (recall upper-bound).

**진단 지표 (per-scene & macro avg)**:

| 지표 | 정의 |
|---|---|
| `Recall_box@d` | GT box centroid에 d m 이내 proposal centroid 존재 비율. d ∈ {0.5, 1.0, 2.0} |
| `Recall_iou3d@τ` | GT box와 3D-IoU ≥ τ proposal 존재 비율. τ ∈ {0.25, 0.5, 0.7} |
| `Recall_pointin` | GT box 내부 점 80% 이상을 cover하는 proposal 존재 비율 |
| `Density` | n_proposals / scene (false-positive proxy) |
| `Per-class recall` | 위 지표를 nuScenes-10 클래스별로 분해 |
| `Distance-binned recall` | ego 거리 [0,15),[15,30),[30,50),[50,∞) m 분할 |
| `Recall_oracle_label` | proposal이 cover한 GT에 GT label을 *직접* 할당했을 때 mAP. → upper-bound mAP. |

**핵심 식 (`Recall_oracle_label`)**:

```
ŷ_proposal := argmax_GT IoU3D(proposal, GT)  (≥ τ_assoc)
detection_score := proposal_score
mAP_oracle := nuscenes-devkit eval over (ŷ, detection_score)
```

이 단일 수치가 **"proposal recall이 outdoor mAP의 ceiling인가"**에 대한
정량 답을 준다. 현재 γ-native가 0.3407이고 oracle이 예컨대 0.55라면
proposal recall은 ceiling이 아니다 (= 라벨링·매칭이 ceiling). 반대로
oracle이 0.40~0.42에 머무르면 proposal recall이 진짜 ceiling이다.

**구현 위치 (신규)**:
- `diagnosis/outdoor_proposal_recall_probe.py` (new) — `nuscenes_loader`로
  scene iterate, 각 source의 generator 호출, 위 지표 계산, 결과
  `results/<date>_outdoor_proposal_recall_v01/metrics.json`에 저장.
- `adapters/proposal_sources_union.py` (new, 선택) — 3-source union을
  centroid-NMS로 dedupe하는 wrapper.

## 2.3 Recall 관점에서 얻을 수 있는 이득 추정

**a-priori 예상치** (γ-native 0.3407 anchor 기반, conservative):

- CenterPoint score_threshold=0.10에서 nuScenes 공식 recall ≈ 0.55~0.65
  (per devkit).
- 그렇다면 γ-native 0.3407의 격차는 **label/score 측에 0.2+ 의 손실**이
  존재. 즉 proposal recall은 *추가로* upper bound를 끌어올릴 여지가 제한적.
- 그러나 **rare class** (construction_vehicle, traffic_cone)는 CenterPoint
  score threshold에서 missed 비율이 클 가능성이 있고, score_threshold를
  0.05까지 낮추거나 hdbscan 보강으로 +Δ recall 확보 가능. 이 Δ가
  최종 mAP에 어떻게 전사되는지 측정한다.

**결론 가설**: outdoor 저성능의 1차 원인은 **proposal recall이 아니라
label 측 (vocabulary mismatch / 2D-3D 매칭)**일 가능성이 큼. 본 Task는
이 가설을 정량적으로 기각/확정하기 위한 측정이다.

## 2.4 Geometry-guided Open-Vocabulary Bridge 구조

**목표**: CenterPoint의 geometry-rich proposal과 OV (YOLO-World/CLIP) 의
label space를 명시적으로 연결.

### 구조 (논리 흐름)

```
                LiDAR sweep ──► CenterPoint ──► (box, cls10, score, vel)
                                                       │
                                                       ▼
                              ┌────────────── Geometry-guided OV Bridge ─────────────┐
                              │                                                       │
   surround-view 6 cam ──► YOLO-World (open vocab text) ──► (cam, bbox2d, cls, score) │
                              │            ▲                                          │
                              │            │ project box3d → cam plane (AABB)         │
                              │  IoU-2D + class-prior gate                            │
                              │            │                                          │
                              │   weighted-vote label (per box) ◄── CLIP image crop   │
                              │            │                                          │
                              ▼            ▼                                          │
                  fused_class = arbiter(cp_cls, ov_cls, conf, vocab)                  │
                              │                                                       │
                              ▼                                                       │
                detection_score = α·cp_score + (1-α)·ov_conf                          │
                              │                                                       │
                              └───► nuScenes-devkit detection eval ◄──────────────────┘
```

### Arbiter 정책 (4가지 후보)

1. **CP-wins**: nuScenes-10에 속하는 카테고리는 cp_cls 채택, 아닐 때만 OV로.
   → 현재 γ-native와 같지만 OOD에 OV가 fallback (open-vocab 확장의 최소 단위).
2. **OV-overrides-CP**: ov_conf ≥ τ_ov 이면 OV label 채택. τ_ov는 calibration.
3. **Voting**: cp + N개 camera의 ov label에 대해 weighted vote (M21
   `WeightedVoting.frame_weight`을 그대로 재활용).
4. **CLIP-rerank**: CLIP image feature ↔ prompt embedding cosine으로
   재정렬 (Task 1의 weighted EMA를 이 box stream에 직접 적용).

### 인터페이스 (입력/출력 스키마)

```python
# 입력
ProposalRecord = TypedDict("ProposalRecord", {
    "sample_token": str,
    "centroid_ego": List[float],          # (3,)
    "bbox_lidar": List[float],            # (7,) or (9,)
    "size_lhw": List[float],              # (3,)
    "yaw_ego": float,
    "cp_cls_idx": int,
    "cp_cls_name": str,                   # in NUSC_10
    "cp_score": float,
})

OVDetection = TypedDict("OVDetection", {
    "cam_token": str,
    "bbox_2d": List[int],                 # x1,y1,x2,y2
    "ov_label": str,
    "ov_score": float,
    "clip_feat": Optional[List[float]],   # 512-D
})

# 출력 (devkit-compatible)
FusedRecord = {
    "sample_token": str,
    "translation": [x, y, z],
    "size": [w, l, h],
    "rotation": [w, x, y, z],
    "velocity": [vx, vy],
    "detection_name": str,                # 최종 label (nuScenes-10 or OV-mapped)
    "detection_score": float,
    "attribute_name": "",
}
```

**바인딩 코드 (계획)**:
- 신규 모듈 `method_scannet/streaming/geometry_ov_bridge.py`
- 신규 evaluator `method_scannet/streaming/nuscenes_geomov_evaluator.py`
  (`nuscenes_evaluator.py` / `nuscenes_native_evaluator.py` *복제 후 수정*,
  두 기존 파일은 보존)

## 2.5 실제 구현 전 필요한 입력/출력 정의

- **입력 가용성 확인**: ego→cam 외부 파라미터, 이미 `dataloaders/nuscenes_loader.py`에
  존재. YOLO-World 호출은 `nuscenes_evaluator.py:_yolo_label_for_proposal_iou`
  로 이미 구현되어 있음 → 재사용.
- **CLIP encoder**: `method_scannet/clip_image_encoder.py` 그대로 outdoor에서
  쓸 수 있음 (단, 6-cam batch encoding을 위해 `encode_cropped_bboxes`를
  `(image_list, bbox_per_image)` 시그니처로 thin wrapping 필요).
- **GT loader**: nuScenes-devkit `NuScenesEval`로 이미 통합되어 있음
  (`nuscenes_evaluator.py:eval_summary`).
- **출력 저장**: `results/<date>_outdoor_geomov_v01/outputs/results_nusc.json`
  (devkit format).

---

# Task 3 · GT-Free Open-Vocabulary Evaluation (OV-TCS)

## 3.1 Track Length 기반 Temporal Stability 측정

**기존 자산** (`method_scannet/streaming/metrics.py`):
- `id_switch_count` — pred id 변경 횟수 (line 66-96).
- `label_switch_count` — pred id 고정 + 라벨 변경 횟수 (line 99-116).
- `time_to_confirm(K=3)` — K consecutive frames 안정화 도달 시간 (line 119-157).
- `nuscenes_evaluator.py:CentroidAssociator` — outdoor용 cross-sample tracker.
- `nuscenes_native_evaluator.py:ClassAgnosticAssociator` — class gate 제거판.

**Track Length 정의**:

```
TL(k) = number_of_frames_with(pred_id == k)
        = | { t : k ∈ pred_history[t] } |
```

(이미 `pred_history` 형식으로 두 evaluator 모두 출력 중)

### 지표 (per-scene + macro):
- `mean_TL`, `median_TL`, `max_TL`
- `long_track_ratio = |{k : TL(k) ≥ L_min}| / |tracks|`, L_min ∈ {3,5,10}
- `coverage_TL = Σ TL(k) / (n_frames · n_unique_tracks)` — 트랙이 얼마나 *지속적으로* 살아 있나
- 분포: TL의 히스토그램 (논문 figure 후보)

**Implementation 위치**:
- `method_scannet/streaming/metrics.py`에 함수 추가:
  `track_length_stats(pred_history) -> dict`. 기존 `label_switch_count` 옆.
- 두 outdoor evaluator의 finalize 구간에서 호출 → 결과 JSON에 추가 필드.
- ScanNet 쪽도 `eval_streaming_ablation.py:155-180`에서 호출 가능 (지금
  `id_switch_count`만 부르고 있음 → 한 줄 추가).

## 3.2 Label Entropy 기반 Label Consistency 측정

**Track 단위 정의**:

```
p_k(c) = |{ t : pred_label[k,t] == c, pred_label[k,t] ≠ -1 }| / TL_valid(k)
H_k    = -Σ_c p_k(c) log p_k(c)              (낮을수록 consistent)
H̃_k   = H_k / log(|C_active|)                (스케일 정규화, [0,1])
```

여기서 `C_active`는 track k가 실제로 hit한 카테고리 set (전체 num_classes
대신 사용하면 OV vocab 크기에 robust).

### 지표:
- `mean_H̃`, `entropy_at_K_frames` (트랙 첫 K프레임 한정 H̃; M11/M12 confirm time과 매칭)
- `mode_dominance = max_c p_k(c)` (≥0.8이면 confident track 비율)
- 분포 + 히스토그램

**가설**: M11/M12/M21이 효과적으로 작동하면 H̃ ↓, mode_dominance ↑, label_switch ↓.
M22 (CLIP EMA)가 drift하면 H̃는 baseline보다 *오히려 증가* 가능 → Task 1 결과의
보조 검증.

## 3.3 OV-TCS — Open-Vocabulary Temporal Consistency Score (draft)

### 목표
Closed-set mAP가 측정 못 하는 *open-vocab 출력의 시간적 안정성·일관성*을 단일 스칼라로 요약.

### 수식 초안

```
OV-TCS(scene) =  w_S · S_stab   +   w_C · S_cons   +   w_R · S_resp
```

- **S_stab** (Stability — track length / id switch):
  ```
  S_stab = (1 − idsw_per_track) · long_track_ratio(L_min=5)
        idsw_per_track = id_switch_count / max(|tracks|,1)
  ```
- **S_cons** (Consistency — label entropy):
  ```
  S_cons = mean_k ( 1 − H̃_k )
  ```
- **S_resp** (Responsiveness — confirmation time):
  ```
  S_resp = mean_k exp( − ttc_k / τ_t )           τ_t = 5 frames
  ```

권장 default weight: `w_S = w_C = w_R = 1/3`.

### Macro 집계
`OV-TCS_macro = mean_over_scenes(OV-TCS(scene))`. Per-axis sub-score는 보고서 가독성용으로 동시 출력.

### Properties (만족)
- **GT-free**: GT 매칭 필요 없음 (단, GT가 있을 땐 stability·consistency를
  GT 매칭된 트랙으로 한정하여 *상한*도 같이 계산할 수 있음).
- **Vocab-invariant**: `C_active` 정규화 덕분에 vocabulary 크기에 unbiased.
- **Scale 합리적**: 모든 항이 [0,1]에 가깝게 위치 → 합성 시 사람 해석 가능.

### 구현 위치
- `method_scannet/streaming/metrics.py` 에 `ov_tcs_score(pred_history, tracker_state, K_long=5, tau_t=5.0) -> dict` 추가.
- `streaming/eval_streaming_ablation.py` / outdoor evaluator finalize block에서
  call → JSON에 `ov_tcs.total/S_stab/S_cons/S_resp` 4개 키 추가.

## 3.4 장점·한계·예상 Failure Case

### 장점
- GT 무관 → unannotated dataset / 신규 클래스에 직접 적용 가능.
- 본 프로젝트 기존 metric (`label_switch_count`, `time_to_confirm`) 위에서
  **모두 closed-form**으로 계산 → 추가 inference 0.
- M11~M32 ablation 비교 시, 어느 method가 *시간적으로* 일관성을 개선하는지
  closed-set mAP보다 민감하게 잡아냄.

### 한계
- *Always-predict-the-same-class* 트래커가 OV-TCS=1.0을 받음 (degenerate).
  → 대처: per-scene class 다양성 보조 지표 `entropy_over_tracks` 동시
  보고. 또는 OV-TCS 단독 사용 금지, 항상 mAP와 함께 보고.
- Tracker 자체가 단순/엄격하면 S_stab은 인공적으로 ↑ — tracker 알고리즘
  fairness가 비교 전제.
- Open-vocab vocab 자체가 동적이면 `C_active` 정의를 frozen vocabulary로 맞춰야 비교 가능.

### Failure case
| 상황 | 증상 | 보완 |
|---|---|---|
| Tracker가 모든 instance를 같은 id로 묶음 | TL ↑, H̃ → 0, OV-TCS ↑ (가짜) | mAP/recall 동시 보고 강제 |
| 모든 트랙이 1프레임만 존재 | TL=1, H̃=0(정의 회피로 가짜 ↑) | TL=1 트랙은 OV-TCS 계산에서 제외 (`TL≥2` 필터) |
| Class flip이 매우 잦지만 매번 반반 분포 | H̃ ↑ 정확히 표현 | 의도된 동작 |
| Frozen vocab vs 동적 prompt | 비교 무의미 | metadata에 vocab id 기록 |

## 3.5 논문/Supplementary 활용 가능성

- **본문**: closed-set mAP 외 보조 metric으로 한 줄 정의 + 한 표.
  - Table — "Method × {mAP, mAP_50, OV-TCS, S_stab, S_cons, S_resp}"
  - 핵심 주장: closed-set mAP에 묻힌 method 효과 (특히 M11/M21)가
    OV-TCS에서 분명히 분리되어 보임 (전제: 본문 실험으로 입증).
- **Supplementary**:
  - 정의·proof (vocabulary-invariance), per-scene 분포 그림, degenerate
    tracker 분석, ablation (w_S,w_C,w_R 가중치 robustness).
- **위험**: 새 metric은 reviewer가 "왜 필요한가"에 대해 보수적. 대안은
  본 metric을 *분석용 진단 도구*로만 활용하고 본문 metric은 mAP 유지.
  단, 본 프로젝트의 Open-Vocab 기여를 강조하려면 OV-TCS는 본문 1개
  table만이라도 등장하는 게 유효.

---

# 4. 통합: 우선순위 · 난이도 · 리스크 · 실험 순서

## 4.1 우선순위 매트릭스

| Task | 우선순위 | 구현 난이도 | 학술적 임팩트 | 리스크 |
|---|---|---|---|---|
| Task 1 (Conf-W CLIP EMA) | **A (최우선)** | ★★☆ (1~2일) | 직접적 ablation, M22 회복 → 본문 강화 | 낮음 — drift 가설이 거짓이면 보고용 negative result로도 가치 |
| Task 2 (CenterPoint Recall Study) | **B (중)** | ★★★ (3~5일) | outdoor 한계 정량화, 향후 방향 정함 | 중 — Geometry-guided OV-Bridge 전체 구현 시 비용 ↑. Recall probe만이라도 빠르게. |
| Task 3 (OV-TCS) | **B+ (중상)** | ★★☆ (1~2일) | 새 metric 제안, reviewer-divisive | 낮음 — 측정만, 기존 모델 손대지 않음 |

## 4.2 권장 실험 순서 (병렬화 포함)

**Week 1**
1. **Task 1.6 step 1+2** — confidence-skip τ-sweep 진단 (single + 10-scene). [GPU 1]
2. **Task 2.2 / 2.3** — `outdoor_proposal_recall_probe.py` (4 source × per-class × distance bin). [CPU+GPU 1, 병렬 가능]
3. **Task 3.1~3.3** — `track_length_stats`, `label_entropy_stats`, `ov_tcs_score` 추가 + 기존 ablation 결과에 후처리 적용 (재실행 X). [CPU only]

**Week 2**
4. **Task 1.6 step 3+4** — A/B/C variant ablation, full val. [GPU 1]
5. **Task 2.4** — Geometry-guided OV-Bridge prototype (CP-wins, OV-overrides-CP arbiter 2종만 먼저). [GPU 1]
6. **Task 3.5** — degenerate-tracker stress test + 가중치 robustness. [CPU]

**Week 3**
7. **Task 1.6 step 5** — failure-case ablation (open-vocab safety). [GPU 1]
8. **Task 2.4** — Voting / CLIP-rerank arbiter, full val. [GPU 1]
9. 문서화 + figure (TL 분포, OV-TCS scatter, recall probe heatmap). [CPU]

## 4.3 코드 수정 위치 요약

| Task | 파일 | 변경 종류 |
|---|---|---|
| 1 | `method_scannet/method_22_feature_fusion.py` | 인자 추가 (default 하위호환) |
| 1 | `method_scannet/hooks.py` (`_apply_method_22`) | confidence 신호 plumbing |
| 1 | `method_scannet/eval_method_22_only_v2.py` | CLI flags |
| 2 | `diagnosis/outdoor_proposal_recall_probe.py` | **신규** |
| 2 | `adapters/proposal_sources_union.py` | **신규 (선택)** |
| 2 | `method_scannet/streaming/geometry_ov_bridge.py` | **신규** |
| 2 | `method_scannet/streaming/nuscenes_geomov_evaluator.py` | **신규 (기존 두 evaluator 복제)** |
| 3 | `method_scannet/streaming/metrics.py` | 함수 3개 추가 (`track_length_stats`, `label_entropy_stats`, `ov_tcs_score`) |
| 3 | `method_scannet/streaming/eval_streaming_ablation.py` & 두 outdoor evaluator finalize | 신규 metric 호출 + JSON에 dump |

## 4.4 통합 리스크 요약

1. **Task 1 — yolo_score plumb 시 frame 동기화 버그 가능성**: `preds_2d`의 dict
   순서가 frame_id와 일치하는지 확인 필요 (이미 `frame_id_list = list(self.preds_2d.keys())` 로 사용 중 → OK).
2. **Task 2 — CenterPoint score_threshold 변경 시 NDS/devkit 영향**: 현행
   nuscenes_evaluator는 score_threshold=0.10. 하향 시 false positive 증가
   → mAP는 P/R curve 적분이므로 영향 적음. 하지만 score 분포가 dirty해질 위험.
   분리된 evaluator로 격리 필수.
3. **Task 3 — degenerate tracker로 metric inflation**: 항상 OV-TCS와 mAP를
   joint 보고하는 정책으로 차단.
4. **Cross-task — α weighting / vocabulary 정의 drift**: 본 문서가 정의한
   파라미터들은 metadata에 함께 dump (config snapshot).
5. **공유 계정 리스크** (memory: shared rintern16 account): 모든 신규 디렉토리/스크립트
   는 `results/<date>_<task>_v<NN>/` 컨벤션 준수. 기존 진행 중 실험 파일에는 손대지 않음.

---

## Appendix · 최종 결과물 체크리스트

- [x] 코드 수정 위치 (§4.3)
- [x] 예상 리스크 (§4.4 + 각 Task 내 failure case)
- [x] 우선순위 (§4.1)
- [x] 구현 난이도 (§4.1)
- [x] 예상 실험 순서 (§4.2)
- [x] baseline 보존 / 플래그 격리 (§0)
- [x] 모든 신규 모듈 인터페이스 정의 (§1.5, §2.4, §2.5, §3.3 구현 위치)
