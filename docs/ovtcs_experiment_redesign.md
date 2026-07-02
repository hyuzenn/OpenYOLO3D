# OV-TCS Experiment Section — Outline around the final claim

## Final contribution (the one sentence everything serves)

**OV-TCS_C is a temporal-consistency metric for streaming open-vocabulary 3D
perception that (i) captures degradation AP is blind to, (ii) carries
predictive information beyond track length, and (iii) separates real methods
that AP cannot distinguish.**

Metric: **OV-TCS_C = (1 − 1/L)·(1 − CSR)**, CSR = switches/(L−1), computed on
native labels grouped by associator track (`nuscenes_native_evaluator.py:1000-1002`).
Pure **track-topology property** — so it only separates methods that change
*track grouping / labels*, never per-frame voting. Report A/B/C, headline C.

The paper walks a reviewer down three questions **in order**. Each question =
one section = one figure + one table + one objection killed. Each experiment
has exactly one job; nothing is decorative.

Shared setup: nuScenes val 150 (primary), native evaluator with
`--collect-track-metrics`. Caches on disk → **GPU 0** throughout.

---

## §1 — Why is AP insufficient?  *(claim i)*

**Experiment.** Controlled corruption that acts on track topology *after*
detection, so the nuScenes AP-scored proposal set is held fixed → AP flat by
construction, OV-TCS the only thing that moves.
- **Fragmentation sweep** `--frag-inject-p {0,.1,.2,.3,.4,.5}` (wired): breaks
  one track into many → L↓, frag↑, C↓, AP flat.
- **Interruption (positive control)** `--association-max-age {10,5,2,1}`:
  degrades real association so *both* AP and OV-TCS fall. Calibrates the OV-TCS
  scale and proves it is not inert to everything.

**Figure 1** (hero) — twin-y line plot, x = corruption strength: AP (left axis,
flat across fragmentation) vs OV-TCS_C (right axis, monotone drop). Inset/3rd
panel = interruption, where both fall together.

**Table 1** — per knob value: `strength, AP, NDS, OV-TCS_C, TrackLen, Frag, CSR`.
Fragmentation block shows ΔAP≈0 with ΔOV-TCS large; interruption block shows the
positive control (both move).

**Key message.** A model can degrade in temporal consistency with *zero* AP
change. AP measures per-frame detection, not cross-frame identity stability.

**Reviewer objection killed.** *"AP already measures everything that matters —
why a new metric?"* → Here is a controlled axis of degradation AP literally
cannot see, and a positive control proving OV-TCS still tracks real failure.

---

## §2 — Why is Track Length alone insufficient?  *(claim ii — the crux)*

This is the sharpest objection: OV-TCS_C **contains L explicitly**
((1−1/L) factor), so a reviewer will say it is track length in disguise. Two
independent rebuttals, one constructed, one statistical.

**2a. Constructed (causal).** Semantic-flicker sweep
`--sem-flicker-p {0,.1,.2,.3,.4,.5}`, applied **only to the OV-TCS label
sequence, never the AP-scored class** (diagnostic-only). Flicker flips labels
*within an unchanged track* → **L held constant, CSR rises, C drops**. The clean
"identical Track Length, different semantic stability" example by construction.

**2b. Statistical (observational).** Partial correlation + nested F-test on real
per-instance data (indoor, n=6202, already banked
`results/2026-06-25_scannet_ovtcs_instance_val312_v03`):
- PC(OV-TCS_C, downstream | TrackLen) significant,
- nested models mAP~len vs mAP~len+OV-TCS: **ΔR² F=89.87, p=3.5e-21**,
- LOO-CV RMSE improves when OV-TCS added on top of length.

**Figure 2** — two panels. (left) flicker sweep: TrackLen flat line, OV-TCS_C
dropping, AP flat = three lines, one moves. (right) added-variable / partial-
residual plot: downstream-quality residual vs OV-TCS_C residual after
regressing out TrackLen, with fit + CI showing nonzero slope.

**Table 2** — nested model comparison: rows = {length-only, OV-TCS-only,
length+OV-TCS}; cols = {R², ΔR², F, p, LOO-RMSE}. The length+OV-TCS row wins.

**2c. Why *this* formulation? (the two-axis argument).** OV-TCS_C =
L_norm·(1−CSR) is a *product of two factors*, and the justification is that the
metric must register **two independent failure modes**, with each factor owning
one — neither factor alone covers both. Established post-hoc on banked data
(`results/2026-06-26_ablation_ovtcs_formulation_v01`, zero GPU):

- **Semantic-flicker axis → carried by (1−CSR).** Formulation ablation on the
  per-instance correctness target (n=6202): among parameter-free forms
  `(1−CSR)` carries the flicker signal (ΔR² over length 0.0232); **L_norm is
  inert/dilutes** (length-only ΔR² 0.0028, partial-corr negative). Honest and
  important: *stability, not the product, owns this axis* — which is exactly
  where the flicker corruption lives.
- **Fragmentation axis → carried by L_norm.** Per-instance cache-replay
  `--frag-inject-p` sweep on nuScenes detguided (150 val, ~19k–22k tracks/level,
  `results/2026-06-26_outdoor_ovtcs_fragdecomp_v01`): frag 0→0.5 drops mean
  OV-TCS 0.476→0.438, **carried entirely by L_norm** (monotone contribution
  −0.058, ≥ the net −0.037). Per-track **(1−CSR) does not fall — it RISES**
  (0.731→0.769): breaking a track leaves fewer switches inside each piece. So a
  pure-stability metric would *mis-report fragmentation as improvement* — **L_norm
  is necessary for fragmentation sensitivity**, not dead weight. (Supersedes the
  earlier scene-aggregate backout, which was directionally right but approximate.)

The two factors are complementary, not redundant: fragmentation moves L_norm and
not stability (§1), flicker moves stability and not L_norm (§2a). **The product
is the minimal form responsive to both axes.** Single factors are each blind to
one corruption (eliminated). Among two-axis forms (product/harmonic/geometric/
min) the product best preserves the flicker signal (ΔR² 0.0142 vs ≤0.0122) and
is simplest. The weighted sum is *additive* → its validation-tuned λ collapsed
to λ*=0.00 (i.e. stability-only → fragmentation-blind); additive forms don't
*enforce* two-axis sensitivity and the free parameter buys nothing.

**Table 2b** (main) — two blocks. (i) Formulation × {R², ΔR², partial-corr,
LOO-RMSE} on the flicker target, parameter-free forms + tuned-λ control. (ii)
Per-instance fragmentation decomposition (`--frag-inject-p` sweep): per corruption
level, mean L / CSR / OV-TCS and the L_norm-vs-stability contribution split —
showing L_norm carries the OV-TCS drop while (1−CSR) rises. Cost: **flicker block
zero-GPU post-hoc; fragmentation block one cache-replay sweep (no new inference).**

**Key message.** OV-TCS carries temporal-quality information that survives
controlling for track length — it is not a reparameterization of L — and the
product form is the right way to combine L with semantic stability.

**Reviewer objection killed.** *"This is just track length / track count with
extra steps."* → Fixed-length flicker moves the metric (causal), and OV-TCS adds
significant explanatory power over length on real data (statistical). *And* the
formulation ablation (2c) answers the follow-up *"why multiply these two terms?"*

*Scope note, reported honestly:* per-scene aggregation GATE-FAILs; the signal
lives at instance granularity. Stated in §2, not hidden.

---

## §3 — Why is OV-TCS practically useful?  *(claim iii)*

**Experiment.** Real-method comparison, all existing pipeline flags, shared
caches. Report **AP, AR, NDS, OV-TCS_C, TrackLen, Frag, CSR** per method:

| Method | Flag |
|---|---|
| Baseline (γ, ego) | `--axes baseline --proposal-source gamma` |
| Global association | `--association-frame global` |
| Class-agnostic association | `--association-class-agnostic` |
| Class-aware label fusion *(thesis main)* | `--proposal-source hybrid --fuse-allow …` |
| M31 IoU / M32 Hungarian merge | `--axes m31` / `--axes m32 --m32-distance 1.0` |

**Money rows (near-identical AP, OV-TCS separates):**
- **baseline-ego vs global**: OV-TCS_C 0.136 → 0.168 (**+24%**) at near-flat AP
  (`results/2026-06-12_outdoor_ovtcs_assoc_compare_v01`) — the headline.
- **γ vs M32@1.0**: M32 recovers AP to the 0.3407 anchor (AP tie) via different
  merge topology → different Frag/CSR.

**Figure 3** — scatter OV-TCS_C (x) vs downstream temporal quality (y), one point
per method, AP encoded as marker size/color. The near-AP-tie pairs sit at the
same size but spread along x → "AP can't tell them apart, OV-TCS can," in one
picture.

**Table 3** — the method comparison above, near-AP-tie rows highlighted, with
ΔAP and ΔOV-TCS vs baseline columns.

**Key message.** On real, already-published methods, OV-TCS resolves a quality
ordering that AP collapses to a tie.

**Reviewer objection killed.** *"Sensitivity to synthetic corruptions doesn't
mean it's useful for real systems."* → Two real method pairs with matched AP are
cleanly separated by OV-TCS, and the better-OV-TCS method is the temporally
better one.

---

## How the three sections chain (the spine)

§1 establishes the *need* (AP blind). §2 defends the metric against its single
most damaging objection (it's not just L). §3 delivers the *payoff* (separates
real systems). Drop any one and the chain breaks: without §1 there's no
motivation, without §2 the metric is dismissed as L-renamed before §3 is read,
without §3 it's a synthetic curiosity. Each experiment feeds exactly one rung.

---

## Figures & tables — final placement

**Main paper (3 figs, 3 tables — one per question):**
- Fig 1 / Table 1 → §1 corruption (AP-blind + positive control).
- Fig 2 / Table 2 → §2 beyond-track-length (flicker construction + partial-corr/F-test).
- Table 2b → §2c formulation (two-axis: flicker→(1−CSR), fragmentation→L_norm; product = minimal both-axis form).
- Fig 3 / Table 3 → §3 real-method separation (scatter + comparison table).

**Supplementary:**
- Table S1 — full A/B/C formulation breakdown (p90 frag, singleton fraction).
- Fig S1 — per-knob monotonicity / Spearman robustness for all sweeps.
- Table S2 — indoor ScanNet cross-domain replication (instance PASS / scene FAIL).
- Fig S2 — sensitivity to associator distance threshold (M32 1.0 vs 0.5).

---

## Implementation status (no code yet — next step)

Reuse: native evaluator + `--collect-track-metrics`, `--frag-inject-p`,
`--association-frame/--association-class-agnostic`, `--fuse-allow`, m31/m32 axes,
existing caches; analysis via `results/2026-06-13_ablation_ovtcs_partial_v01/analyze.py`.

Net-new surface (deferred until narrative approved):
1. `--sem-flicker-p` knob, diagnostic-only on the OV-TCS label sequence (~10 lines) — feeds §2a.
2. Surface **AR** in the summary dict (`evaluator:945-957`, ~5 lines) — feeds Table 3.
3. One cache-replay run to fill paired AP/NDS for the ego-vs-global row of Table 3.
4. Formulation ablation → Table 2b. **DONE.**
   - Flicker block (0 GPU): `results/2026-06-26_ablation_ovtcs_formulation_v01/`
     (formulation + reliability on banked instances).
   - Fragmentation block (cache-replay, no new inference):
     `results/2026-06-26_outdoor_ovtcs_fragdecomp_v01/` — per-instance `--frag-inject-p`
     sweep (added 1-line passthrough to `eval_ovtcs_track_outdoor.py`).
   Result = two-axis justification, instance-level: flicker→(1−CSR), fragmentation→L_norm
   (and pure (1−CSR) mis-reports fragmentation as improvement).
