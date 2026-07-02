# OV-TCS Paper — Storyline 설계도 (논리 구조)

> 논문 초안이 아니라 **논리 구조**. 지금까지 확정된 실험 결과만 사용한다.
> 핵심 주장: *per-frame AP는 streaming open-vocab 3D perception에서 정작 중요한
> temporal degradation을 보지 못한다. OV-TCS가 그것을 측정하며, 그 product 형태는
> 두 개의 서로 다른 failure mode(label flicker, fragmentation)에 의해 경험적으로
> 강제된다. 이 metric은 AP가 동률로 뭉개버리는 실제 method 차이를 분해한다.*

---

## 0. 확정 실험 결과 (storyline의 재료, 측정값 그대로)

- **OV-TCS 정의:** OV-TCS_C = L_norm·(1−CSR), L_norm = 1−1/L, CSR = switches/(L−1).
  native per-frame argmax label을 associator track으로 묶어 계산하는 **순수
  track-topology property** (evaluator:1000-1002).
- **AP blindness:** detection 이후 track topology만 건드리는 fragmentation 주입
  (`--frag-inject-p`)에서 **AP는 고정인데 OV-TCS만 단조 하락**. interruption
  (`--association-max-age`)을 주면 **AP·OV-TCS가 함께 하락**(positive control).
- **인도어 per-instance 예측 타당성** (ScanNet200 val312, n=6202, target=label
  correctness): partial corr(OV-TCS, correct | track_len) r=0.12, p=3.5e-21;
  nested F(length 모델에 OV-TCS 추가) **ΔR²=0.014, F=89.87, p=3.5e-21** → 통과.
- **per-scene는 실패** (n=312, target=AP_50): partial r=−0.03, p=0.61 → 신호는
  **instance granularity**에 있고 scene 집계에선 사라짐(정직히 보고).
- **Formulation ablation** (flicker/correctness target, banked 6202, CPU-only):
  설명력 순위 **stability_only(ΔR²0.023) > product(0.014) > minimum(0.012) >
  geometric(0.011) > harmonic(0.010) > length_only(0.003)**. length-only는 무용
  (partial r=−0.05). validation-tuned weighted-sum은 **λ*=0.00** 으로
  stability-only에 붕괴.
- **L_norm은 reliability weight가 아님:** 짧은 perfect-stability track이 오히려 더
  정확(정답률 80% @L=2 vs 61% @L≥5), within-bin corr도 짧은 track에서 더 강함;
  게다가 인도어는 track의 99.2%가 L≥5라 L_norm이 사실상 무반응.
- **Per-instance fragmentation decomposition** (outdoor detguided, 150 val,
  level당 ~19,000–22,000 tracks): frag 0→0.5에서 mean L 4.28→2.53, mean L_norm
  0.648→0.569, **mean (1−CSR) 0.731→0.769 (상승)**, mean OV-TCS 0.476→0.438.
  OV-TCS 하락(−0.037)을 **전적으로 L_norm이 운반**(contrib −0.058 ≥ net −0.037),
  stability 기여는 **+0.025(상승, 상쇄)** → **pure stability라면 fragmentation을
  "개선"으로 오보**.
- **Real-method money pair:** ego→global association에서 **OV-TCS 0.136→0.168
  (+24%)**, AP는 거의 평탄.
- **γ vs M32:** γ(CenterPoint, closed nuScenes-10 anchor AP 0.3407) vs M32
  Hungarian merge@1.0 — **AP는 동률**, 그러나 merge topology가 달라 Frag/CSR이 다름.
- **Outdoor open-vocab의 AP ceiling:** open-vocab proposal(detguided/hybrid)은
  closed nuScenes-10에서 **γ-fixed AP 0.3407을 넘지 못함**(car recall/localization
  ceiling), 단순 global score gate는 실패 → class/source-aware 필요.
- **Outdoor 병목은 localization** (oracle-score 분해): score calibration이 아니라
  localization이 recall→AP gap의 원인. native confidence는 이미 calibrated,
  dedup 0.548이 geometric ceiling.
- **Temporal EMA aggregation은 실패:** OFF > k=0.335 > k=1.0 (EMA가 AP/AR를 일관되게
  깎음). OV-TCS는 EMA weight를 위한 actionable 신호가 없음 → OV-TCS를 control signal이
  아니라 **metric**으로 재정의.

> 마지막 네 항목(EMA 실패, per-scene 실패, AP ceiling, localization 병목)은 **main
> contribution에서 제외**, limitation/supplementary로만 쓴다. 나머지가 main story의 척추.

---

## 1. Problem Definition

**한 문단 정의.** Streaming open-vocabulary 3D perception(SemWorld-3D: 실시간 open-vocab
3D mapping)은 같은 물리적 객체를 여러 프레임에 걸쳐 관측하면서 **online으로 open-vocab
label을 부여**한다. 그러나 이 분야는 품질을 여전히 **per-frame detection metric(AP/NDS)**
으로만 측정한다. AP/NDS는 한 프레임 안의 박스/클래스 정확도를 재는 지표이지, *같은 객체가
시간에 걸쳐 일관된 identity와 label을 유지하는가*(temporal consistency)를 재지 않는다.
결과적으로 **AP는 동일하지만 temporal 품질은 전혀 다른** 두 시스템을 구분하지 못한다.

**기존 가정.** (a) per-frame AP가 시스템 품질의 충분한 proxy다. (b) 시간 축 정보를 모으면
(temporal aggregation/EMA/voting) 품질이 단조적으로 좋아진다. (c) temporal 품질은 MOT식
track length / fragmentation으로 충분히 포착된다.

**실제로 발생하는 문제.**
- AP는 temporal degradation에 **눈이 멀어 있다**: detection을 고정한 채 track을 잘게
  쪼개는 fragmentation 주입에서 AP는 미동도 없고 OV-TCS만 하락한다.
- 단순 temporal aggregation은 **오히려 품질을 깎았다**: EMA를 켜면 OFF보다 AP/AR이
  떨어졌다(OFF > k=0.335 > k=1.0). 모으는 것이 답이 아니다.
- track length 단독은 downstream 품질의 **거의 무용한 predictor**다: 인도어 correctness
  회귀에서 length-only는 ΔR²≈0.003, partial corr이 음수(−0.05)였다.
- label이 프레임마다 튀는 flicker와 track이 깨지는 fragmentation은 **서로 다른 failure
  mode**다: flicker는 stability(1−CSR)를 움직이고, fragmentation은 L_norm을 움직이며 오히려
  stability를 끌어올린다. 단일 축 지표로는 둘을 함께 볼 수 없다.

**왜 기존 방법으로 안 풀리나.** 닫힌-어휘 MOT 지표(MOTA/IDF1)는 **고정된 class set과 GT track
ID**를 전제하므로 open-vocab·online-label 환경에 그대로 적용되지 않는다. AP/NDS는 per-frame이라
구조적으로 cross-frame identity를 못 본다. 즉 **open-vocab streaming 3D의 label/identity
안정성을 함께 재는 지표가 존재하지 않는다.**

**왜 중요한가.** 실시간 open-vocab 3D mapping은 embodied agent/로보틱스를 위한 것이고,
의미 지도가 깜빡이거나(flicker) 조각나면(fragmentation) AP가 아무리 높아도 downstream에서
쓸 수 없다. 이를 못 보는 평가 체계는 **method 순위를 잘못 매긴다.** 평가가 틀리면 연구 방향
전체가 틀어진다 — 그래서 metric이 먼저다.

---

## 2. Core Observation

우리가 실제 실험에서 발견한 사실(해석이 아니라 측정):

1. **AP blindness.** detection 이후의 track topology만 손대는 fragmentation 주입에서 AP는
   정의상 고정인데 OV-TCS만 단조 하락했다. interruption(association max-age 축소)을 주면
   AP·OV-TCS가 함께 하락한다(positive control). → AP는 per-frame detection을 재지 cross-frame
   identity 안정성을 못 잰다.

2. **Aggregation은 답이 아니다.** OV-TCS를 control signal로 EMA에 먹이는 라인은 OFF가 모든
   EMA 설정을 이겼다(OFF > k=0.335 > k=1.0, EMA가 AP/AR을 깎음). "시간 정보를 모으면 좋아진다"는
   가정이 깨졌다. → temporal 품질은 *가정할* 대상이 아니라 *측정할* 대상이다. (OV-TCS의 역할이
   method가 아닌 **metric**으로 확정된 지점.)

3. **flicker ≠ fragmentation.** 두 failure mode는 분리된다.
   - flicker는 (1−CSR)을 움직이고 L_norm은 거의 무반응: 인도어 correctness에서 stability가
     설명력을 독점(ΔR²0.023)하고 length-only는 무용(ΔR²0.003, partial r −0.05).
   - fragmentation은 L_norm을 움직이고 (1−CSR)은 **오히려 상승**: outdoor frag 0→0.5에서
     per-track (1−CSR)이 0.731→0.769로 올라갔다(track을 쪼개면 조각마다 내부 switch가 줄어
     stability가 좋아 보임). pure stability라면 fragmentation을 *개선*으로 오보한다.

4. **track length 단독은 부족하다.** OV-TCS는 length를 통제한 뒤에도 downstream correctness에
   유의한 설명력을 더한다(instance-level ΔR²=0.014, F=89.87, p=3.5e-21). 단, 신호는 instance
   granularity에 있고 per-scene 집계는 통과 못 한다(partial r=−0.03, p=0.61) — 정직히 보고.

5. **Outdoor label/method 관찰.** 같은 AP에서도 association 선택만 바꿔도 temporal 품질이
   달라진다(ego→global, OV-TCS 0.136→0.168, +24%, AP 평탄). 또 open-vocab proposal(detguided)은
   closed anchor AP 0.3407을 못 넘지만(localization ceiling) **track topology와 label을 다르게**
   만들어 OV-TCS로는 구분된다. → metric이 실제 method 차이를 분해한다.

---

## 3. Research Questions

논문 전체를 관통하는 질문 사슬:

- **RQ1.** per-frame AP/NDS는 streaming open-vocab 3D perception의 *temporal* 품질을 충분히
  평가하는가?  → (답: 아니다 — fragmentation에서 AP 무반응.)
- **RQ2.** 그렇지 않다면 temporal 품질은 무엇으로 측정되어야 하며, 그것은 단지 *track length*가
  아닌가?  (2a: length를 넘는 신호인가 — instance-level ΔR²=0.014 F=89.87. 2b: 그렇다면 **왜 그
  formulation(product)인가** — flicker는 (1−CSR)·fragmentation은 L_norm이 단독 소유.)
- **RQ3.** 이 metric은 AP가 구분 못 하는 **실제 method**를 분해하는가, 그리고 실제로 temporal
  품질을 바꾸는 method(class-aware label fusion / association 선택)는 무엇인가?
  → (ego→global +24% at flat AP; γ vs M32 AP 동률·topology 상이; label fusion.)

흐름: *AP로는 부족하다(RQ1) → 그럼 무엇으로 재나, 그게 length가 아니라 왜 product인가(RQ2) →
이 metric이 실제로 일하는가, 무엇이 품질을 바꾸나(RQ3).*

---

## 4. Proposed Solution

살아남은 contribution만 사용한다: **OV-TCS(metric)** + **Class-aware Label Fusion(method)**.

### 4.1 OV-TCS (primary)
- **왜 필요한가.** AP가 못 보는 temporal degradation(label flicker + track fragmentation)을
  단일 스칼라로 측정하기 위해.
- **무엇을 해결하나.** associator track을 따라 native open-vocab label의 **일관성**을 정량화.
  per-track 값 OV-TCS_C = L_norm·(1−CSR)을 집계.
- **기존과 무엇이 다른가.** AP/NDS는 per-frame(=cross-frame identity 못 봄). MOT(MOTA/IDF1)는
  closed-vocab + GT track ID 전제. OV-TCS는 **GT track ID 없이 native label만으로**, open-vocab에서
  동작하는 track-topology property다.
- **입력/출력.** 입력: per-frame argmax open-vocab label + associator track membership(인도어는
  Mask3D 3D mask가 track을 고정, 아웃도어는 cached LiDAR proposal에 associator 적용). 출력: per-track
  OV-TCS_C와 그 집계(+ 진단용 Track Length, CSR, Fragmentation).
- **설계의 핵심(two-axis).** L_norm = track integrity(fragmentation 축), (1−CSR) = semantic
  stability(flicker 축). 두 축을 **곱**으로 결합 — 이유는 §5.3에서 경험적으로 강제됨.

### 4.2 Class-aware Label Fusion (method)
- **왜 필요한가.** online open-vocab label은 프레임마다 튀고(flicker), 단순 global score gate로
  융합하면 실패한다(closed nuScenes-10에서 anchor 못 넘음).
- **무엇을 해결하나.** 2D open-vocab label을 3D mask/track에 **class/source-aware**로 융합해
  class label을 교정·안정화(2D→3D class correction).
- **기존과 무엇이 다른가.** naive global-score gating이 아니라 class/source 별로 다른 게이팅.
  open-vocab(GT-free) capability — closed CenterPoint anchor와는 *다른 regime*.
- **입력/출력.** 입력: 인도어 Mask3D 3D mask / 아웃도어 cached proposal + streaming 2D open-vocab
  label. 출력: fused per-track label.
- **정직한 위치.** closed nuScenes-10에서 AP로는 γ anchor(0.3407)를 못 넘는다(localization
  ceiling; oracle-score 분해상 병목은 calibration이 아니라 localization). 가치는 **open-vocab class
  correction capability**와 **temporal/label 품질**(OV-TCS로만 보이는 차이)에 있다 — AP가 아니라
  OV-TCS가 이 method의 효과를 드러낸다.

---

## 5. Main Contributions (CVPR/ICCV/ECCV 스타일)

각 contribution = 문제 → 아이디어 → 검증 → 결과.

**C1. OV-TCS: streaming open-vocab 3D perception을 위한 temporal-consistency metric.**
- 문제: AP/NDS는 per-frame이라 temporal degradation을 못 보고, MOT 지표는 open-vocab에 안 맞는다.
- 아이디어: native label을 associator track으로 묶어, track integrity와 semantic stability를
  결합한 per-track 일관성 점수.
- 검증: AP-blindness(fragmentation에서 AP 고정·OV-TCS 하락) + 예측 타당성(instance-level
  ΔR²=0.014, F=89.87) + 실제 method 분해(ego→global +24% at flat AP).
- 결과: (i) AP가 못 보는 축을 보고, (ii) track length를 넘는 정보를 담고, (iii) AP 동률 method를
  분해하는 metric. **새로움**: open-vocab·GT-ID-free temporal 품질 측정은 기존에 없었다.

**C2. metric formulation을 *주장*이 아니라 *증거*로 정당화하는 two-axis 분석.**
- 문제: "왜 L_norm·(1−CSR)인가, 왜 곱인가, 왜 다른 조합이 아닌가." (metric 논문의 급소.)
- 아이디어: 각 factor가 **서로 다른 corruption 축을 단독으로 소유**함을 보인다.
- 검증: formulation ablation(flicker는 (1−CSR)가 독점 ΔR²0.023, length-only 무용, weighted-sum
  λ*=0) + per-instance fragmentation decomposition(frag 0→0.5의 OV-TCS 하락을 L_norm이 −0.058로
  100% 운반, (1−CSR)은 0.731→0.769로 상승해 pure stability는 fragmentation을 개선으로 오보).
- 결과: single-factor는 한 축에 대해 틀리고, additive form은 한 factor를 0으로 만들 수 있어 실패;
  **product가 두 축 모두에서 올바른 최소 형태**. **새로움**: metric 설계를 두 개의 독립 corruption
  실험으로 강제 — 통상적 "ablation으로 약간 더 좋음"이 아니라 *구조적 필연성* 증명.

**C3. open-vocab streaming을 위한 Class-aware Label Fusion.**
- 문제: per-frame open-vocab label flicker + naive global-score 융합 실패(closed-10에서 anchor 미달).
- 아이디어: class/source-aware 2D→3D label correction.
- 검증: 인도어(Mask3D)·아웃도어(cached proposal) 양쪽, AP와 OV-TCS로 동시 평가.
- 결과: open-vocab class correction(GT-free) capability; AP로는 안 보이지만 OV-TCS로 드러나는
  temporal/label 품질 차이. **새로움 + 정직성**: 이 method의 효과가 *왜 AP로 측정되면 안 되는지*를
  C1/C2가 설명 — method와 metric이 서로를 정당화한다.

> (보조) cross-domain 검증: 인도어=예측 타당성(F=89.87 통과), 아웃도어=분해/판별(frag decomp,
> ego/global). 단일 데이터셋 artifact 아님. 이는 C1의 evidence로 흡수.

---

## 6. Experimental Story (실험 순서 설계)

```
5.1 Why AP fails               (RQ1)  ── fragmentation sweep(AP 고정·OV-TCS 하락) + interruption control
        ↓
5.2 Beyond Track Length        (RQ2a) ── per-instance partial corr / nested F (ΔR²=0.014,F=89.87), per-scene FAIL 정직 보고
        ↓
5.3 Why the Product form        (RQ2b) ── formulation ablation(flicker→(1−CSR)) + fragmentation decomposition(frag→L_norm, (1−CSR)↑)
        ↓
5.4 Real methods AP can't tell  (RQ3) ── ego vs global(OV-TCS 0.136→0.168 at flat AP), γ vs M32(AP 동률·topology 상이)
        ↓
5.5 Class-aware Label Fusion    (RQ3) ── indoor+outdoor, AP & OV-TCS, closed anchor 0.3407 대비 정직
        ↓
5.6 Qualitative                 ── 의미 지도 flicker/fragmentation 시각화(두 failure mode 대조)
```
- **5.1** 표: knob별 AP/NDS/OV-TCS/TrackLen/Frag/CSR. fragmentation 블록은 ΔAP≈0·ΔOV-TCS 큼,
  interruption 블록은 둘 다 하락.
- **5.2** 표: nested model {length / OV-TCS / length+OV-TCS} × {R², ΔR², F, p, LOO-RMSE}.
  per-scene FAIL(partial r=−0.03, p=0.61)은 같은 절에 명시(은폐 금지).
- **5.3** 표 2블록: (i) formulation × {R²,ΔR²,partial,LOO} on flicker(stability 0.023 > product
  0.014 > … > length 0.003, weighted-sum λ*=0), (ii) frag level별 mean L/CSR/OV-TCS + L_norm·
  stability 기여 분해(L_norm −0.058, stability +0.025). "pure stability가 fragmentation을 개선으로
  오보"가 hero line.
- **5.4** scatter: OV-TCS(x) vs temporal quality(y), AP를 marker 크기. near-AP-tie 쌍이 x축으로 벌어짐.
- **5.5** 표: method별 AP/AR/NDS/OV-TCS/TrackLen/Frag/CSR; closed anchor 0.3407과의 관계 정직 보고.
- **5.6** 정성: 동일 장면에서 flicker(색 튐) vs fragmentation(조각남) 대비.

**Supplementary / Limitations:** EMA 실패(OFF가 EMA 이김 → OV-TCS를 metric으로 재정의의 근거),
per-scene granularity(partial r=−0.03), outdoor localization ceiling(anchor 0.3407 미달, dedup
0.548 geometric ceiling), A/B/C formulation 세부, associator distance 민감도.

---

## 7. Reviewer Perspective (예상 질문 → 대응 실험)

| 리뷰어 질문 | 대응(실험 결과) | 절 |
|---|---|---|
| AP(특히 NDS)가 이미 충분하지 않나? | detection 고정·topology만 깨도 AP 무반응, OV-TCS만 하락 | 5.1 |
| 이건 그냥 MOT/IDF1 아닌가? | open-vocab·GT track ID 없음; native label을 associator track으로 — closed MOT 지표 적용 불가 | §4.1 |
| OV-TCS는 track length를 새 이름으로 부른 것 아닌가? | length 통제 후에도 ΔR²=0.014 유의(F=89.87, p=3.5e-21); flicker 축에선 length-only 무용(partial r=−0.05) | 5.2 |
| 왜 곱인가? 왜 sum/min/harmonic이 아닌가? | flicker는 (1−CSR)가, fragmentation은 L_norm이 단독 소유; additive는 factor를 0으로 붕괴(weighted-sum λ*=0) | 5.3 |
| 합성 corruption 민감성이 실제 유용성인가? | 실제 method(ego/global) AP 동률에서 OV-TCS 0.136→0.168(+24%) | 5.4 |
| Label Fusion이 anchor를 못 넘는데 왜 contribution인가? | closed CenterPoint anchor(0.3407)와 다른 open-vocab(GT-free) regime; 병목은 localization, 효과가 AP가 아니라 OV-TCS로 드러남 | 5.5 |
| per-scene은 실패했는데 신호가 진짜인가? | streaming 품질은 instance granularity(ΔR²=0.014 통과)에 있음; scene 집계 실패(p=0.61)는 정직히 보고 | 5.2 |
| 시간 정보를 모으면(EMA) 되지 않나? | EMA가 오히려 AP/AR 깎음(OFF > k=0.335 > k=1.0) → 모으기가 아니라 *측정*이 답 | supp |
| 인도어/아웃도어 cherry-picking 아닌가? | 인도어=예측 타당성(F=89.87), 아웃도어=분해/판별(frag decomp, ego/global); metric 거동 일관 | cross-domain |

---

## 8. Final Paper Message (한 문장)

**"This paper shows that per-frame AP is blind to the temporal degradation that
actually matters in streaming open-vocabulary 3D perception, introduces OV-TCS —
a track-level temporal-consistency metric whose product form is *empirically
forced* by two distinct failure modes (semantic flicker, captured by 1−CSR, and
track fragmentation, captured by L_norm) — and uses it to expose temporal-quality
differences between real methods that AP collapses to a tie."**

보조 메시지: *temporal 품질은 모아서 좋아지길 바라는(aggregation) 대상이 아니라, 먼저 올바로
측정해야 하는 대상이며, 올바른 metric은 그 자체로 method 설계를 강제한다.*
