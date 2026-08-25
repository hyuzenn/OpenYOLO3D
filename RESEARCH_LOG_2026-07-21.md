# 연구 진행 기록 — SemWorld-3D / Streaming Open-Vocab 3D Perception

> Notion 업데이트용. 직전 기록 6/3(OV-TCS 설계) 이후 ~ 7/21까지.
> 이번 사이클의 핵심: **"좋아 보이는 결과를 밀어붙인 게 아니라, 검증이 가설을 반박했을 때 방향을 바꿨다."**
> results 폴더 타임스탬프 기준으로 마일스톤을 날짜별로 분리해 기록.

---

## 6/12–6/13 — OV-TCS 구현 & 초기 검증 착수

6/3에 설계한 OV-TCS(= L_norm × (1−CSR))를 실제 파이프라인에 구현하고, 검증 계획의 첫 실험들을 돌리기 시작.

- **AP-blind fragmentation 실험 착수** (`ablation_ovtcs_surrogate`, `outdoor_ovtcs_fragmentation`): detection을 고정한 채 track topology만 인위적으로 조각내며 mAP가 불변인지, OV-TCS가 단조 하락하는지 확인.
- **Global associator ablation** (`ablation_global_associator`): ego vs global 연관이 mAP는 bit-identical인데 temporal 지표는 크게 달라지는 현상 최초 관측 → "mAP가 못 보는 축"의 최초 증거.
- **Metric selection / product formulation 후보 비교 시작** (`ovtcs_metric_selection`, `ablation_ovtcs_partial`): product / min / geometric / harmonic / weighted-sum 형태 비교 착수.
- **Proposal ceiling / hybrid cache** 정비 (`outdoor_proposal_ceiling`, `outdoor_hybrid_cache_build`): open-vocab proposal stream의 localization 병목 확인(closed anchor 대비 mAP 격차의 원인이 labeling이 아니라 3D localization임).

---

## 6/22–6/27 — Validation 본체 수행

검증 계획 5개 축을 본격 실행하고, 보조기여(label fusion)도 정리.

- **Product formulation 검증 완료** (6/26, `ablation_ovtcs_formulation`, `outdoor_ovtcs_fragdecomp`): 두 축(flicker/fragmentation)에서 각 factor가 배타적으로 한 실패 모드를 담당함을 확인. stability-only는 fragmentation을 "개선"으로 오판(=directionally wrong), product는 두 축 모두 방향이 맞음. weighted-sum은 tuning하면 λ*=0으로 붕괴.
- **Track-length confound 분석** (ScanNet200 val312, 6/25, `scannet_ovtcs_instance_val312`): OV-TCS가 track length를 통제한 뒤에도 label 정답성을 예측하는지(ΔR²) 측정. → 신호는 있으나 작음(ΔR²≈0.014).
- **Outdoor track metrics 정리** (6/27, `outdoor_final_trackmetrics`): nuScenes val에서 OV-TCS와 GT MOT 지표 대조용 데이터 구축.
- **보조기여: class-aware label fusion** (6/23–24, `outdoor_labelfusion_*`): 2D→3D per-class gating으로 mAP 0.3408→0.3420(track topology 불변). 단 naive global score gate는 anchor 못 넘음 → per-class 필요.
- **M22 EMA ablation** (6/23, `scannet_m22_*`): OV-TCS를 aggregation 제어신호로 되먹이는 실험. → aggregation-off가 모든 EMA 설정을 이김. OV-TCS는 method가 아니라 metric으로만 포지셔닝하기로 결정.

---

## 7/15–7/18 — Paper audit → E1 사전등록 GT validation → **가설 반박**

이번 사이클의 전환점 (1): 리뷰어 관점에서 스스로를 감사하고, 사전등록 방식으로 OV-TCS를 대규모 검증한 결과가 **가설을 지지하지 않음**.

- **Reviewer-관점 headline audit** (7/15, `m1m3_headline_audit`): 3대 리젝 사유 도출 —
  (1) beyond-length 효과가 약함(ΔR²≈0.014), (2) nuScenes GT ID를 가진 상태에서 HOTA/IDF1/AMOTA 직접 비교가 누락(치명적), (3) open-vocab을 표방하는데 flagship이 closed 라벨.
- **E1: 21-variant 사전등록 GT validation** (7/17–18, `e1_prereg`, `e1_smoke`, `e1_grid`): "OV-TCS가 GT tracking quality의 proxy로 쓸 만한가"를 사전등록 그리드로 검증.
  - **GT tracking metric(TrackEval HOTA/AssA/IDF1 + AMOTA devkit) 대조 수행** *(주의: 이는 GT 지표와의 대조이지, third-party framework 검증이 아님 — framework 검증은 아래 7/21 feasibility의 미완 항목)*.
  - **Synthetic fragmentation validation** 및 **21-variant preregistered evaluation** 수행.
  - **결과: CONCERNING.** OV-TCS는 AssA와 거의 무상관(ρ≈+0.011), 오히려 detection quality proxy처럼 거동(최대 상관은 DetA). class-aware association 하에서 (1−CSR) factor가 fragmentation이 최악인 지점에서 상수로 고정됨.
- **결론: OV-TCS를 main contribution에서 내린다.** temporal consistency monitoring 용도로는 의미가 있으나 GT tracking quality proxy로는 부족. **Negative result를 숨기지 않고 exploratory result로만 유지하기로 결정.** (유일한 잔존 유효신호: L_norm 단독이 AssA의 GT-free proxy로 ρ≈0.751 — diagnostic로만.)

---

## 7/20 — E2 Controlled Gate Isolation → **같은 데이터에서 대안 발견**

이번 사이클의 전환점 (2): OV-TCS 검증 과정에서 **오히려 기존 Temporal Layer의 효과를 GT tracking metric으로 처음 제대로 측정**하게 됨.

- **배경 재평가**: 예전엔 mAP만으로 "Temporal Layer 효과 거의 없음"으로 판단했었음. GT tracking metric으로 다시 재니 HOTA↑·AssA↑·IDF1↑·AMOTA↑가 일관되게 나타남.
- **E2 실험** (`2026-07-20_e2_gate_sweep`, PBS 105088): gamma_global에서 **gate만 분리**, 나머지 전부 동결(detector/associator/거리/max_age/score-th/evaluator/proposal cache). N={1,2,3,5} 스윕.
  - **Validity anchor**: N=1은 baseline과 **byte-identical**(md5 동일, 0개 suppressed) → gate 경로가 gating 외엔 아무것도 안 함을 증명. N≥2의 모든 델타는 gate 단독 효과.
  - **결과(핵심 수치)**: mAP 0.3408→0.2082, NDS 0.3150→0.2825 (단조 감소) / HOTA 0.2011→0.2573, IDF1 0.1764→0.2797, DetA 0.0995→0.1615 (단조 증가). 모두 **동일 box geometry** 위에서.
  - **메커니즘**: 같은 box-set 위에서 mAP↓와 DetA↑가 동시에 — mAP는 strict-recall 적분이라 gate가 지운 짧은 spurious track에 벌점, DetA/HOTA는 그 제거가 사는 precision+continuity에 보상.
  - **최적점**: AssA는 N=3, AMOTA는 N=2에서 정점(단조 아님). (mAP, AMOTA)에서 N=5는 N=3에 dominated. 권장 operating point N=2(또는 continuity 우선 시 N=3).
  - **결론**: Streaming perception에는 **Detection metric ↔ Tracking metric trade-off가 인과적으로 존재**.

---

## 7/21 — 방향 전환 & 논문 전면 재작성

### (a) Main contribution 교체
- 기존 **OV-TCS metric** → **Training-free Temporal Layer + Streaming evaluation protocol**.
- 핵심 분석: **"mAP가 temporal improvement를 체계적으로 과소평가한다"**. OV-TCS는 negative result 및 discussion으로 이동.

### (b) 귀속 정정 (honesty 핵심 순간) — **[신규 명시]**
- 예전엔 "+32% 상대 AMOTA"를 통째로 "temporal layer 효과"로 귀속했었음.
- E2 controlled 실험으로 **분해·정정**:
  - **gate(M11)** = class-agnostic 축 담당 → HOTA/AssA/IDF1 반전을 소유. AMOTA엔 +0.009만 기여.
  - **semantic relabel(M21)** = class-aware 축 담당 → AMOTA에 추가 **+0.042** 기여(라벨만 수정, geometry/identity 불변).
- 즉 **continuity(gate) vs semantics(relabel)**는 직교하며 각각 별도로 측정 가능 → 이 분해 자체를 **보조기여**로 채택.

### (c) Protocol 기여 — AMOTA floor 발견 — **[신규 명시]**
- uncalibrated open-vocab score stream에선 **official AMOTA가 정확히 0으로 붕괴**(YOLO-score가 devkit recall-threshold sweep 아래로 떨어짐).
- 따라서 class-agnostic HOTA leg는 "스타일 선택"이 아니라 **open-vocab stream을 측정하기 위한 필수 요건** → protocol contribution으로 기록.

### (d) 논문 전면 재작성 완료
- `paper_iccv_draft/` LaTeX 전체 재작성(Abstract/Intro/Related/Method/Protocol/Experiments/Discussion/Limitations/Conclusion). 7페이지, undefined 참조 0 컴파일 확인.
- Figure 4종 재구성: teaser / gate sweep(+Pareto) / gate-relabel decomposition / mechanism.
- **제목 확정**: *"What mAP Misses: A Training-Free Temporal Layer for Streaming Open-Vocabulary 3D Perception."*
- **옛 OV-TCS 자산 정리**: 미참조 figure 5개 삭제, `figure_specs.md`·`CLAUDE.md`를 새 story로 동기화.

### (e) 외부검증 feasibility study — **[신규 명시]**
- 남은 최대 약점 = generality("우리 파이프라인에서만 되는 것 아니냐").
- 후보 조사(ConceptGraphs/HOV-SG/OpenMask3D-online 등) 결과 **ConceptGraphs 추천**: MIT, actively maintained, per-frame streaming, training-free, 객체가 detection 이력을 누적 → **gate 삽입 = guard 한 줄**(재학습·재설계 불필요). 예상 3–5 engineer-day + A100 몇 시간.
- **미결 결정(중요)**: 외부검증으로 무엇을 살 것인가 —
  (A) **generality 보강**(ConceptGraphs, indoor·저비용, 단 tracking GT 없어 fragmentation/label-switch 증거만) vs
  (B) **두 번째 GT-tracking leg**(driving domain, Waymo/Argoverse, 고비용·고가치). → 착수 전 이 결정 먼저.

---

## 현재 상태

**완료**
- E1(사전등록 GT validation) / E2(gate isolation)
- GT tracking evaluation, gate↔relabel 분해
- Reviewer-관점 audit
- 논문 전면 재작성 + Figure 재구성 + 제목 확정 + 옛 자산 정리
- 외부검증 feasibility study

**진행 예정**
- 외부검증 방향 결정(A generality vs B 2nd GT-tracking leg)
- ConceptGraphs 환경 구축 → third-party framework validation
- (option) open-vocab AMOTA를 floor에서 끌어올릴 **score calibration**
- (option) multi-dataset GT tracking (Waymo/Argoverse)
- Qualitative figure(scene-level flicker vs hold)
- Figure polishing / 교수님 피드백 반영 / ICCV(또는 3DV·WACV) 최종 제출

---

## 이번 사이클의 의미 (서사)

1. OV-TCS를 main contribution으로 준비
2. preregistration 기반으로 검증 수행
3. 결과가 가설을 지지하지 않음(AssA 무상관, detection proxy 거동)
4. 이를 숨기지 않고 인정(exploratory/negative로 강등)
5. **같은 실험에서 드러난 Temporal Layer의 강점을 GT tracking metric으로 재측정 → 연구를 재구성**

가장 큰 전환점은 "가설이 반박됐을 때 방향을 바꾼 것", 그리고 "잘못된 귀속(+32% AMOTA)을 스스로 잡아내 gate/relabel로 정직하게 분해한 것".
