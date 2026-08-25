# 2026-08-03 — 논문 전면 재작성 → method-paper 재프레이밍

**대상:** `paper_iccv_draft/` (canonical) · 계획 문서 `docs/identity_hygiene_restructure_plan_2026-08-01.md` · 규칙 `paper_iccv_draft/CLAUDE.md`

## 제목

> Retrospective Confirmation for Identity-Consistent Streaming 3D Perception

(17:19 판 "A Training-Free Temporal Confirmation Module for Identity-Consistent Streaming 3D Perception"에서 18:35에 교체. 제목이 module이 아니라 **메커니즘**을 지목하도록 의도적으로 바꿈.)

## 기여 재정의 — evaluation protocol → method 논문

당일 두 번의 전환이 있었다. 먼저 논문을 처음부터 다시 쓰면서 기여를 matched-control 평가로 잡았고(17:19), 이어 18:35에 **method 논문으로 재프레이밍**하여 논문의 성격 자체가 바뀌었다.

- **1차 기여 = training-free retrospective confirmation module.**
  confirmation test(N회 관측 후 emit) + **confirmed prefix의 소급 emission**.
  novelty는 **조립된 operator 전체**에 있으며, confirmation test 단독에 있지 않다 (그건 표준 tracker 관행이고, 소급 emission이 일반 online tracker가 하지 않는 부분).
- **matched control은 headline이 아니라 그 주장의 증거 기준(evidence standard)으로 강등.**
  논문을 evaluation protocol 논문으로 다시 소개하지 말 것.
- **"module"은 정의된 용어** — Method §1에서 associator + confirmation test + emission policy의 조합으로 정의.
- **code release는 reproducibility note이며 contribution bullet이 아니다.**

발견은 그대로 **명시적 trade-off**다.

- **얻는 것(동일 출력 예산):** AssA — 두 통제군 모두 대비, 양 프레임에서 bootstrap CI가 0 배제 / AMOTA 0.055→0.089, 0.166→0.203
- **잃는 것:** mAP, detection recall, DetA, aggregate HOTA
- **retro의 비용:** N−1 프레임 latency. 이것이 없으면 association 이득이 sensor frame에서는 유지되나 world frame에서는 사라진다
- **indoor:** identity-matched control 대비 lsc −27% (309/312 scene), AP 무변화, random-K는 재현 못 함

Abstract는 "왜 unfiltered baseline이 아니라 matched control인가"(대부분의 지표가 출력량만으로 반응한다)를 정면에 놓고, 마지막에 "이 프로토콜은 출력량을 바꾸는 어떤 구성요소에도 그대로 적용된다"로 확장한다.

## 집필 규칙 동결

1. 최종 기여와 그것을 뒷받침하는 실험만 남긴다.
2. **모든 수치는 canonical JSON에서 검증** — 요약·노트·생성된 md 표에서 인용 금지.
3. **본문 내부 용어 전면 금지**: OV-TCS, gate, gate sweep, Temporal Layer, Semantic Relabel, E1/E2/E2b/E2c, M11/M21/M22/M31/M32, gamma, retro, detguided. 표준 CV 용어로 매핑 — M11 → *confirmation-based track initialization*, retro → *offline (retrospective) emission*, streaming → *causal (online) emission*, E2/E2c control → *detection-budget-matched control*, E2b control → *identity-budget-matched control*, lsc → *class-label switches*. 내부 run ID는 `sec/4_supp.tex`에만.
4. 개발 히스토리·폐기된 방향·GT-free surrogate 서술 금지.
5. **CI가 0을 포함하면 승리로 쓰지 않는다** — sensor-frame IDF1은 "no detectable difference", world-frame IDF1은 소폭 손실.
6. detection quality가 좋아진다고 쓰지 않는다.
7. indoor의 zero-AP-cost를 outdoor로 일반화하지 않는다.
8. 최종 주장에 필요한 표만 유지 — **figure 없음**.

## 제거된 내용

- Fig. 1 "Same boxes, opposite verdicts" (사전등록에 의해 폐기)
- `tab:gate` 및 monotone gate-sweep 서사
- OV-TCS (본문에서 완전 제거)
- 구 figure 자산(`fig_teaser`, `fig_gate_sweep`, `fig_decomp`, `fig_mechanism`)과 생성 스크립트는 `paper_iccv_draft/retired/`로 격리

## 상태

main.tex / 0_abstract / 1_intro / 2_formatting(Method + Evaluation Protocol) / 3_finalcopy(Experiments + Discussion + Limitations + Conclusion) / 4_supp 전면 재작성 후 18:35 method-paper 재프레이밍까지 반영, 컴파일 완료(`main.pdf`).

표 구성: 본문 4개(detection-budget-matched / identity-budget-matched / causal emission / indoor), supplementary 3개(전체 metric set, causal-emission ablation, scene-bootstrap CI). **figure 없음.**

## 배경 진행

ConceptGraphs 312-scene 백그라운드 mapping 계속 진행 중 — 08-03 11:28 KST 기준 **135/312 scene 매핑 완료**(현재 scene0652_00). 지표 추출은 여전히 80-scene 체크포인트 기준이며, 완료 시 숫자만 갱신하는 방침 유지.
