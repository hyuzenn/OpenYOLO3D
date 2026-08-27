# Related Work — pending fixes (post-CBMOT-global)

Status: staging draft in `sec/1_intro.tex:106-213` + 9 new `main.bib` entries.
Build verified: 0 undefined, 0 errors, 0 overfull, **11 pages** (was 10).
Superiority-claim scan across `sec/*.tex`: clean.
Bib/citation bijection: 35 entries / 35 distinct cited, no unused, no duplicates.

**Do not edit the manuscript until the global CBMOT result is in.**

## 1. Factual defect to fix (introduced during staging)

`sec/1_intro.tex` ~line 138: currently claims AB3DMOT **and CenterPoint**
"carry the same minimum-hits rule onto 3D boxes."

**CenterPoint has no min-hits confirmation.** Verified against the paper's
tracking section: *"Following SORT, we keep unmatched tracks up to T=3 frames
before deleting them"* — that is max-age (death), not confirmation (birth).
New detections initialize tracks immediately with age a=0 and are output at
once; there is no N-consecutive-frames requirement.

Fix: attribute min-hits to AB3DMOT only; describe CenterPoint as greedy
velocity-based centre-point association with a max-age deletion rule.
Extra care warranted: CenterPoint is our own detector.

## 2. Deferred work (blocked on global CBMOT)

- Finalize the CBMOT paragraph (`sec/1_intro.tex` ~lines 145-160).
- Choose positioning among P1/P2/P3 (below).
- Trim Related Work; bring manuscript 11 pages -> 10.
- Only afterwards, revise Abstract/Contributions if the positioning demands it.

## 3. Positioning candidates (ALL PROVISIONAL)

- **P1 distinct trade-off axes.** Persistence-based selection and confidence
  accumulation pay at different points; visible only at equal emission budget.
  No superiority claim; survives any experimental outcome.
- **P2 matched-emission measurement gap.** The whole lifecycle line suppresses
  output but was never compared at fixed budget. Literature-only, experiment
  independent.
- **P3 indoor transfer to label stability.** Only candidate fully independent
  of CBMOT. Strengthened by AutoSeg3D reporting no tracking metric despite a
  tracking framing.

## 4. Verified bibliographic facts (do not re-derive)

| key | verified |
|---|---|
| `cbmot` | Benbarka, Nuri et al., IROS 2021, arXiv:2107.04327. Refined score gates an `active` flag: *"When a tracklet is not active, the tracking module does not consider it an output yet keeps it in memory."* -> CBMOT changes **which boxes are emitted**, not only scores. Explicitly replaces count-based min-hits. |
| `simpletrack` | Pang, Ziqi et al., **ECCV 2022 Workshops** pp.680-696 (not arXiv-only). Names *output* as a lifecycle decision distinct from birth/death. |
| `polymot` | Li, Xiaoyu et al., IROS 2023. Learning-free. Count-based birth (hit_min), max-age death. nuScenes AMOTA 75.4. |
| `fastpoly` | Li, Xiaoyu et al., **RA-L 2024** (not IROS). Adopts CBMOT's score-fusion equation directly. AMOTA 75.8. Title: "Algorithm" in RA-L/author BibTeX vs "Framework" on arXiv — using "Algorithm". |
| `blackmanpopoli` | Blackman & Popoli, *Design and Analysis of Modern Tracking Systems*, Artech House, 1999. Canonical source for BOTH M-of-N history logic and SPRT score logic. |
| `retrodiction` | Koch, Wolfgang, IEEE T-AES 36(1):2-14, 2000. **fixed-INTERVAL** retrodiction, not fixed-lag — never attribute "fixed-lag" to this title. |
| `embodiedsam` | Xu, Xiuwei et al., ICLR 2025 (Oral). Learned (3D U-Net, query lifting, dual-level decoder). |
| `onlineanyseg` | Tang, Yijie et al., **CVPR 2025** (not arXiv-only). Genuinely **training-free** — so "training-free" must NOT be used as our differentiator. |
| `autoseg3d` | Wang, Hanshi et al., **NeurIPS 2025**, arXiv:2512.07599 (id resolves exactly). Reframes online 3D seg as instance tracking, optimizes identity coherence, yet reports **no tracking metric** — AP only. |

| `vidstability` | Zhang, Hong & Wang, Naiyan, arXiv:1611.06467, 2016. Abstract verified: *"we demonstrate that the stability metric has low correlation with accuracy metric."* Stability decomposed into fragment / center-position / scale-ratio error. Scores box trajectories against GT tracks. |
| `stability3d` | Wang, Jiabao et al., **ECCV 2024**, arXiv:2407.04305. Stability Index (SI) over confidence, localization, extent, heading. Abstract verified: claims stability *"cannot be accessed by existing metrics such as mAP and MOTA"* — it does **NOT** claim a weak *correlation* with mAP; do not paraphrase it that way. |

Unresolved / not cited: Blackman 1986 (no trustworthy TOC found);
Bar-Shalom & Fortmann 1988 (no evidence it is a canonical M/N source — do not
cite for M/N).

## 5. Claim-support check

All other Related Work claims trace to the audited primary sources above.
The only unsupported one is the CenterPoint min-hits attribution in §1.
