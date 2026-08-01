# Pre-registration — E2b: track-count-matched control

**Date frozen:** 2026-07-31 (before any E2b computation)
**Parent:** `preregistration_2026-07-28.md` §3 row 2, which makes the surviving
"class-aware identity hygiene" conclusion **contingent on addressing the
track-count confound**. This document freezes that test.

## 1. Hypothesis under test

**H0 (confound):** the retro-gate's AssA/AMOTA advantage over the box-count-matched
threshold (E2c) is explained by emitting *fewer distinct track IDs*, not by *which*
tracks temporal confirmation selects. AssA and AMOTA are identity-count sensitive,
so any procedure that prunes track IDs could inherit the win.

E2b falsifies H0 by matching the control on **track-ID count** instead of box count.

## 2. Arms (2-arm minimal comparison)

- **Arm A (reused, no recompute):** retro-gate N=3, exactly the E2c gate cell
  (`2026-07-30_e2c_retro_thrmatch_v01`).
- **Arm B (new control):** from the ungated Baseline track set, keep exactly
  K_s track IDs **per scene s**, where K_s = number of distinct track IDs the
  retro-gate emits in scene s (per frame-type). Selection rule is deliberately
  **non-temporal**: top-K_s by mean detection score. All boxes of a kept track
  are emitted; no other filtering.
- **Secondary sanity (not decision-bearing):** random-K_s over 5 seeds
  (20260718+i), reported as a range.

Matching target: distinct emitted track-ID count, exact per scene per frame.
Box budgets will **not** match between arms; both budgets are reported openly.

Everything else identical to E2c: detector cache, GT, associator config
(`max_age` 5, score threshold 0), evaluator, stats path (paired Wilcoxon
one-sided, rank-biserial, 10k scene-bootstrap of the combined Δ, seed 20260718),
both ego and global frames. Same execution environment as E2c (`coss_a6gpu`,
CPU eval) so E2b–E2c numbers are directly comparable.

## 3. Decision rule (pre-committed)

Win criterion identical to E2c: Δ whose 95 % CI excludes zero; dataset-level
metrics by consistent sign; pattern required in **both** frames; ties and mixed
frames resolve downward.

| Outcome (gate vs. top-K control) | Pre-committed conclusion |
|---|---|
| Gate wins AssA in **both** frames | Track-count confound **rejected**. Row-2 identity-hygiene claim becomes unconditional; the temporal layer may be framed as an identity/semantics layer (method-first framing admissible). |
| Control matches or beats gate on AssA in **either** frame | Confound **confirmed**. Identity-hygiene claim falls; the paper's surviving contribution is the evaluation protocol + negative result only. |

AMOTA is read the same way as a secondary (class-aware) axis; it cannot rescue
a lost AssA verdict on its own.

## 4. Stop conditions / scope

- One operating point only (N=3, matching E2c). No extension, no additional
  selection rules, no threshold re-tuning without a new pre-registration.
- If per-scene exact K-matching is infeasible for some scene (K_s exceeds
  available baseline tracks — impossible by construction, since gate tracks are
  a subset of baseline tracks; recorded here as an assertion to be checked),
  the run aborts and the failure is reported, not patched silently.

## 5. Artifacts

`results/2026-07-31_e2b_trackmatch_v01/`: `e2b_report.json`, `e2b_table.md`,
figure, one-page summary, `run.log`. Same artifacts-before-interpretation
workflow as E2c: table → figure → summary → separate review; no manuscript
change from within the run.
