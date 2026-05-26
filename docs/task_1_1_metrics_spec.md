# Task 1.1 — Stage 3: Temporal metrics 정의 (streaming evaluation)

**Branch**: `feature/method-scannet-21-31`
**Date**: 2026-05-12
**Scope**: 코드 변경 0건. Streaming evaluation에서 측정할 metric들의 수학적 정의 + 의사 코드 + 단위 테스트 케이스.
**입력**: Stage 1 (`task_1_1_pipeline_analysis.md`), Stage 2 (`task_1_1_streaming_design.md`).
**출력**: 본 문서. Task 1.3 (metric 구현) 진입 근거.

---

## 0. Notation

| 기호 | 의미 |
|---|---|
| `V` | 한 scene의 mesh vertex 수 |
| `K` | Mask3D가 뽑은 instance 수 (post-NMS/threshold) |
| `F` | streaming 대상 frame 수 (subsampled, frequency=10 기준) |
| `t ∈ {0, 1, ..., F-1}` | frame index (시간 순서) |
| `mask_k ∈ {0,1}^V` | instance k의 vertex membership (Mask3D 출력) |
| `inside_mask[t] ∈ {0,1}^V` | frame t에서 visible한 vertex |
| `pred_class[k, t] ∈ {0, ..., C-1, -1}` | streaming wrapper가 frame t에서 instance k에 부여한 class (`-1` = unassigned / confirmed 안 됨) |
| `pred_score[k, t] ∈ [0, 1]` | 같은 시점의 score |
| `confirmed[k, t] ∈ {0, 1}` | streaming wrapper가 frame t에서 instance k를 "confirmed"로 노출했는가 (M11/12 gate 결과 또는 baseline은 first-seen-frame부터 1) |
| `GT_inst` | scene의 GT instance 집합 (label, vertex set) |
| `gt_mask_i ∈ {0,1}^V` | GT instance i의 vertex membership |
| `gt_label[i] ∈ {0, ..., C-1}` | GT instance i의 class |

ScanNet GT는 `data/scannet200/ground_truth/<scene>.txt`의 `label_id * 1000 + instance_id` per-vertex 포맷으로 주어지므로, `gt_label[i]`, `gt_mask_i` 모두 1회 파싱으로 얻는다.

---

## 1. Incremental mAP — Primary (a) / Secondary (b)

본 task 사용자 결정: **(a)** GT visible up to t **vs (b)** full scene GT. 둘 다 측정.

### 1.1 Primary (a): visible GT up to t

#### 정의

**Visible GT up to t**: GT instance i가 frame ≤ t 중 한 번이라도 visible했으면 평가 대상.

```
gt_visible_up_to_t[i] = ∃ t' ≤ t s.t. |gt_mask_i ∩ inside_mask[t']| ≥ θ_gt
```

`θ_gt`: GT instance를 "visible"로 간주할 vertex 임계값. Stage 2의 D3와 일관성 위해 **θ_gt = 1 (≥1 vertex)** 추천. (D2를 채택 시 동일 N% 적용.)

#### Prediction up to t

`current_instance_map(t)` = streaming wrapper가 frame t에 노출한 instance map (Stage 2 §3 Step F 결과):

```
P(t) = { (mask_k, pred_class[k, t], pred_score[k, t]) : confirmed[k, t] = 1 }
```

#### mAP 계산

표준 ScanNet200 evaluator (`evaluate_scannet200`)를 사용. 즉:

```
mAP_primary(t) = evaluate_scannet200(P(t), {i : gt_visible_up_to_t[i] = 1})
```

이 함수는 5월에 이미 사용한 ScanNet evaluator를 그대로 호출 (`_maybe_dump_metrics`로 dump 가능한 형태). 차이는 "GT의 subset" 입력만.

#### 수학적 의미

- frame t = 0: 시작 시점의 정확도 (대부분 instance가 아직 visible 아님 → 작은 분모 / 큰 noise)
- frame t = F-1: scene 전체에 대한 streaming wrapper의 최종 정확도. **이 값이 offline baseline AP에 근접해야 streaming 구현이 옳음** (Task 1.2 Acceptance 기준).
- 곡선 `mAP_primary(t) vs t`: streaming wrapper가 **얼마나 빨리 정확해지는지** = 본인 method의 "temporal consistency" 가치.

### 1.2 Secondary (b): full scene GT

```
mAP_secondary(t) = evaluate_scannet200(P(t), GT_inst)   # 분모는 항상 전체 GT
```

#### 수학적 의미

- frame t = 0: 거의 0 (예측이 빈약 → recall 0).
- frame t = F-1: primary와 같음 (visible up to F-1 = 전체 GT).
- 곡선 `mAP_secondary(t) vs t`: **수렴 속도**. 가파른 곡선 = 빠른 수렴. 본인 method가 baseline 대비 더 가파른 곡선을 만들면 streaming setting에서의 가치.

### 1.3 의사 코드

```python
def incremental_mAP_primary(scene, streaming_log, gt, theta_gt=1):
    """
    streaming_log: list of dicts, one per frame t
        {'t': int, 'pred_masks': bool[V, K_t], 'pred_classes': int[K_t], 'pred_scores': float[K_t]}
    gt: parsed GT (per-vertex label*1000+inst from .txt)
    """
    inside_mask_cum = zeros(V, dtype=bool)
    visible_gt_ids = set()
    aps = []
    for entry in streaming_log:
        t = entry['t']
        inside_mask_cum |= inside_mask[t]
        for i in gt_instances(gt):
            if (gt_mask(i) & inside_mask_cum).sum() >= theta_gt:
                visible_gt_ids.add(i)
        # ScanNet evaluator는 V-aligned prediction + GT 받음
        ap_t = evaluate_scannet200_subset(
            preds={'pred_masks': entry['pred_masks'], 'pred_classes': entry['pred_classes'], 'pred_scores': entry['pred_scores']},
            gt_filtered={i: gt[i] for i in visible_gt_ids},
        )
        aps.append({'t': t, 'AP': ap_t['all_ap'], 'AP_50': ap_t['all_ap_50%'], 'AP_25': ap_t['all_ap_25%']})
    return aps

def incremental_mAP_secondary(scene, streaming_log, gt):
    aps = []
    for entry in streaming_log:
        ap_t = evaluate_scannet200_subset(
            preds={...},
            gt_filtered=gt,   # always full
        )
        aps.append({'t': entry['t'], 'AP': ap_t['all_ap'], ...})
    return aps
```

`evaluate_scannet200_subset`은 5월에 쓰던 `evaluate_scannet200`을 GT-subset 받도록 얇은 wrapper (또는 GT mask filtering 후 호출).

### 1.4 단위 테스트 케이스 (각 1-2)

**Test 1A — `incremental_mAP_primary` baseline sanity**:
- Mock 1 scene, 2 GT instance, 5 frame.
- 모든 GT instance가 frame 0에서 visible. 1개 instance만 prediction이 GT와 정확히 일치, 1개는 wrong class.
- 기대: 모든 frame에서 `AP = 0.5` (mock IoU 1.0 + 1.0/2 instance correct).

**Test 1B — `incremental_mAP_primary` progressive visibility**:
- Mock 1 scene, 3 GT instance, 3 frame.
- frame 0: GT_1 만 visible. frame 1: GT_1 + GT_2. frame 2: 모두.
- Prediction = GT (oracle).
- 기대: AP_primary가 매 frame 1.0 (분모가 visible GT subset이므로).

**Test 1C — `incremental_mAP_secondary` convergence**:
- 동일 setup. Oracle prediction에 대해 frame 0 → 2 진행하며 prediction이 GT_1, GT_1+2, GT_1+2+3 순으로 expose.
- 기대: AP_secondary가 frame 0 = 1/3 (recall 1/3), frame 1 = 2/3, frame 2 = 1.0.

---

## 2. ID switches per object

### 2.1 정의

GT instance i ↔ prediction instance k 매칭이 frame 진행하며 바뀐 횟수.

매 frame t에 GT instance i에 대해 **최적 매칭 prediction id**:

```
match(i, t) = argmax_k IoU3D(pred_mask_k(t), gt_mask_i)
                subject to: IoU3D ≥ τ_match
              else: match(i, t) = ∅
```

`τ_match`: 매칭 임계값 (ScanNet 표준 0.25 추천 — AP_25와 일관).

ID switches per object i:

```
ID_switches(i) = | { t : match(i, t) ≠ match(i, t-1) ∧ match(i, t) ≠ ∅ ∧ match(i, t-1) ≠ ∅ } |
```

`match(i, t) = ∅` 인 frame은 카운트 안 함 (visible 되었다가 안 보이는 자연 현상 제외).

Scene 단위 총합:

```
ID_switches_scene = Σ_i ID_switches(i)
```

또는 visible GT 평균:

```
ID_switches_per_visible_gt = ID_switches_scene / |{i : gt_visible_up_to_F-1[i] = 1}|
```

### 2.2 본인 method 가치

- baseline streaming: instance가 view 따라 다른 Mask3D proposal에 매칭 → ID switch 다수 발생 가능.
- M11/12: registration gate가 instance 첫 commit을 stable instance만으로 제한 → switch 감소.
- M31/32: spatial merge가 중복 instance를 같은 ID로 통합 → switch 감소.

본인 contribution이 streaming setting에서 ID switch 감소를 통해 측정될 핵심 metric.

### 2.3 의사 코드

```python
def id_switches_scene(streaming_log, gt, tau_match=0.25):
    prev_match = {}   # gt_instance_id → prev frame's pred_k (or None)
    switches = defaultdict(int)
    for entry in streaming_log:
        t = entry['t']
        pred_masks = entry['pred_masks']  # bool[V, K_t]
        for i, gt_mask_i in gt.items():
            # 가장 큰 IoU3D인 pred_k 탐색
            ious = (pred_masks & gt_mask_i.unsqueeze(-1)).sum(0) / \
                   ((pred_masks | gt_mask_i.unsqueeze(-1)).sum(0).clamp(min=1))
            best_k = ious.argmax().item()
            best_iou = ious[best_k].item()
            curr = best_k if best_iou >= tau_match else None
            if curr is not None and prev_match.get(i) is not None and prev_match[i] != curr:
                switches[i] += 1
            if curr is not None:
                prev_match[i] = curr
    return {
        'total': sum(switches.values()),
        'per_instance': dict(switches),
    }
```

**중요**: pred_k는 streaming wrapper 내부에서 **scene 전체 통틀어 안정한 ID** (Mask3D index 그대로 0..K-1)여야 한다. M31/32 merge로 새 instance id가 생기면 wrapper가 그 매핑을 명시적으로 기록해야 한다. → Task 1.4 구현 시 주의.

### 2.4 단위 테스트 케이스

**Test 2A — switch 없음**:
- 5 frame, 1 GT instance, 1 prediction. 매 frame 같은 prediction_k. → ID_switches_scene = 0.

**Test 2B — 1회 switch**:
- 5 frame, 1 GT. frame 0-1 prediction_k=0, frame 2-4 prediction_k=1 (다른 mask가 GT에 더 잘 매칭). → ID_switches_scene = 1.

---

## 3. Label switch count

### 3.1 정의

Prediction instance k의 `pred_class[k, t]`가 frame 진행 따라 바뀐 횟수.

```
LabelSwitches(k) = | { t : pred_class[k, t] ≠ pred_class[k, t-1] ∧
                          pred_class[k, t] ≠ -1 ∧ pred_class[k, t-1] ≠ -1 } |
```

`-1` (unassigned / pre-confirmed) frame은 카운트 안 함.

Scene 총합:

```
LabelSwitches_scene = Σ_k LabelSwitches(k)
```

### 3.2 본인 method 가치

- baseline streaming: per-frame label vote의 mode가 frame마다 흔들림 → label switch 다발.
- M21 (WeightedVoting): smooth weighting → 흔들림 감소.
- M22 (EMA): exponential moving → 매끄러운 변화.
- M11/12 gate가 confirmed 늦추면 label switch가 후반에 stable.

### 3.3 의사 코드

```python
def label_switches_scene(streaming_log):
    prev_class = defaultdict(lambda: -1)
    switches = defaultdict(int)
    for entry in streaming_log:
        for k, c in enumerate(entry['pred_classes']):
            if c == -1 or prev_class[k] == -1:
                prev_class[k] = c
                continue
            if c != prev_class[k]:
                switches[k] += 1
            prev_class[k] = c
    return {'total': sum(switches.values()), 'per_instance': dict(switches)}
```

### 3.4 단위 테스트

**Test 3A — switch 없음**:
- 5 frame, K=1 instance. 매 frame class=3. → label_switches_scene = 0.

**Test 3B — 2회 switch**:
- 5 frame, K=1. classes = [3, 3, 7, 7, 12]. → 2 switches (3→7, 7→12).

**Test 3C — `-1` 무시**:
- 5 frame, K=1. classes = [-1, -1, 3, 3, 7]. → 1 switch (3→7). `-1` ↔ 3 전환은 무시.

---

## 4. Time-to-confirm

### 4.1 "Confirmed" 정의 — 옵션

| 옵션 | 정의 | 장점 | 단점 |
|---|---|---|---|
| **C1: K consecutive frames same class** | instance k가 K개 **연속** frame에서 동일 `pred_class` 보였으면 confirmed. | forward-only computable. semantic stability 직접 측정. K로 noise tolerance 조절. | 일시 disappear (occlusion) → 재confirm 필요 |
| **C2: cumulative detection count ≥ N** | `Σ_t 1[confirmed[k, t] = 1] ≥ N` 만족 시 confirmed. | M11과 직접 호환 (의미 거의 같음 → metric으로서 정보 중복). | label switch와 독립 — class 잘못 바뀌어도 confirm 발동. |
| **C3: post-hoc stable label** | scene 끝 도달 후, instance k의 final `pred_class[k, F-1]` 와 일치하는 마지막 연속 streak의 시작 frame. | retroactive로 가장 의미 있는 "stable" 정의. | online metric 아님 (scene 끝나야 알 수 있음). |

### 4.2 권고: **C1 with K=3**

**이유**:
1. **Forward / online computable** — 매 frame 갱신 가능. dashboard / 시각화에 적합.
2. **M11 (frame-counting)과 의미 구분**: M11은 frame count gate (분기점), C1은 class stability gate (수렴 시점). 두 metric이 독립 정보를 제공 — confirmed by C1과 confirmed by M11이 다르면 "M11이 통과시켰지만 class가 아직 흔들림"이라는 진단 가능.
3. **K=3 권고**: ScanNet frequency=10 → 3 streaming frame = ScanNet 원본 30 frame (~1초 @ 30 FPS). Class가 1초 동안 안 바뀜 = 수렴으로 간주.
4. C2는 M11과 정보 중복 위험. C3는 online metric 안 됨.

→ **사용자 결정 필요 항목 #2**. 본인 권고: **C1, K=3**.

### 4.3 정의 (수식)

C1, K-consecutive:

```
confirmed_at[k] = min { t ≥ K-1 :  pred_class[k, t-K+1] = pred_class[k, t-K+2] = ... = pred_class[k, t]
                                  ∧ pred_class[k, t] ≠ -1 }
                or ∞ if 결코 만족 안 함
```

```
time_to_confirm[k] = confirmed_at[k] - first_seen[k]
where first_seen[k] = min { t : pred_class[k, t] ≠ -1 }
```

Scene 단위 통계:
- `TTC_mean = mean({time_to_confirm[k] : confirmed_at[k] < ∞})`
- `TTC_median`, `TTC_p90`, `unconfirmed_count = |{k : confirmed_at[k] = ∞}|`

### 4.4 의사 코드

```python
def time_to_confirm(streaming_log, K=3):
    first_seen = {}
    confirmed_at = {}
    class_history = defaultdict(list)
    for entry in streaming_log:
        t = entry['t']
        for k, c in enumerate(entry['pred_classes']):
            if c == -1:
                continue
            if k not in first_seen:
                first_seen[k] = t
            class_history[k].append((t, c))
            # 마지막 K개 entry가 모두 같은 class면 confirmed
            recent = class_history[k][-K:]
            if k not in confirmed_at and len(recent) == K and all(rc == c for _, rc in recent):
                confirmed_at[k] = t
    ttc = {k: confirmed_at[k] - first_seen[k] for k in confirmed_at}
    return {
        'per_instance': ttc,
        'mean':   mean(ttc.values()) if ttc else float('nan'),
        'median': median(ttc.values()) if ttc else float('nan'),
        'unconfirmed_count': len(first_seen) - len(confirmed_at),
    }
```

### 4.5 단위 테스트

**Test 4A — 즉시 confirm**:
- K=3, K=1 instance. classes = [3, 3, 3, 3, 3]. first_seen=0, confirmed_at=2. TTC=2.

**Test 4B — 흔들림 후 confirm**:
- K=3, K=1. classes = [3, 7, 3, 3, 3]. first_seen=0, confirmed_at=4. TTC=4.

**Test 4C — 결코 confirm 안 됨**:
- K=3, K=1. classes = [3, 7, 3, 7, 3]. → unconfirmed.

---

## 5. 통합 streaming log schema

각 frame t의 record (Stage 2 §3 Step G `record(...)`):

```python
{
    'scene_name': str,
    't': int,                                # subsampled frame index in scene
    'frame_id_original': int,                # ScanNet 원본 frame number (= t * frequency)
    'pred_masks':   bool[V, K_t_visible],    # confirmed=True instance만
    'pred_classes': int[K_t_visible],
    'pred_scores':  float[K_t_visible],
    'pred_k_ids':   int[K_t_visible],        # Mask3D 원래 idx (ID switch 계산용)
    'mAP_primary':   {'AP': ..., 'AP_50': ..., 'AP_25': ...},
    'mAP_secondary': {'AP': ..., 'AP_50': ..., 'AP_25': ...},
    'confirmed_count': int,                  # Σ confirmed
    'time_to_confirm_snapshot': dict[int, int],   # 매 frame 갱신, 최신 TTC dict
}
```

Scene 종료 후 aggregation:

```python
{
    'scene_name': str,
    'final_mAP_primary': float,              # = mAP_primary at t=F-1
    'final_mAP_secondary': float,            # = mAP_secondary at t=F-1
    'id_switches_total': int,                # §2
    'label_switches_total': int,             # §3
    'ttc_mean': float,                       # §4
    'ttc_median': float,
    'ttc_unconfirmed_count': int,
    'per_frame_mAP_primary':   list[float],  # F entries
    'per_frame_mAP_secondary': list[float],  # F entries
}
```

312 scene aggregate (Task 1.5):

```python
{
    'mean_final_mAP_primary': float,
    'mean_final_mAP_secondary': float,
    'mean_id_switches_per_visible_gt': float,
    'mean_label_switches_per_predicted_instance': float,
    'mean_ttc': float,
    'mean_per_frame_mAP_primary': list[float],   # F-aligned (scene 길이 정규화 필요)
}
```

Frame 길이가 scene마다 다르므로 `per_frame_mAP_primary` 평균은 두 옵션:
- (i) **percentile-bucket**: 각 scene의 frame을 [0, 100%]로 정규화 후 10% 간격 bucket 평균. 비교 plot에 적합.
- (ii) **absolute frame-aligned**: scene별 다른 길이 → max-len padding. 짧은 scene이 마지막 평탄해짐.

→ (i) 권고. Task 1.3 구현 시 결정.

---

## 6. Acceptance 체크 — 본 Task 1.1 Stage 3

- [x] 4 metric 수학적 정의 (§1.1, §1.2, §2.1, §3.1, §4.3)
- [x] 4 metric 의사 코드 (§1.3, §2.3, §3.3, §4.4)
- [x] 각 metric 단위 테스트 케이스 1-2개 (§1.4, §2.4, §3.4, §4.5)
- [x] Time-to-confirm "confirmed" 정의 권고 (§4.2): C1 with K=3
- [x] Frame visibility 권고 (Stage 2 §2): D3
- [x] Streaming flow + method 자리 (Stage 2 §3, §4)

---

## 7. 사용자 결정 필요 항목 (요약)

| # | 항목 | 권고 | 대안 |
|---|---|---|---|
| 1 | Frame visibility (D1/D2/D3) | **D3** (≥1 vertex, instance-level lenient) | D2 (N=5%) ablation으로 추가 |
| 2 | "Confirmed" 정의 | **C1, K=3** (3 consecutive frames same class) | C2 (cum count ≥ N) — M11과 정보 중복 / C3 (post-hoc) — online 아님 |

위 2개 결정 후 **Task 1.2 (streaming wrapper baseline 구현)** 진입 가능.
