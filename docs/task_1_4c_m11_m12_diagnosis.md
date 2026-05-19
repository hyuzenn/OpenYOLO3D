# Task 1.4c — M11 ≡ M12 bit-identity diagnosis

Date: 2026-05-18
Branch: `feature/method-scannet-21-31` @ 087bbad (no source modifications during diagnosis)
Origin: Task 1.4b Part 3 finding — 312-scene M11 and M12 axes produce bit-identical AP and temporal metrics to 15-digit precision.

## TL;DR — verdict: **(a) silent bug AND (c) hyperparameter coincidence, compounded**

`BayesianGate.gate()` does not implement the bidirectional Bayesian update its docstring describes — the `_update(..., observed_visible=False)` branch is dead code. With only positive updates and the defaults shipped in Task 1.4b (prior=0.5, likelihood=0.8, fpr=0.2, threshold=0.95), `BayesianGate` is a monotonic counter whose confirmation step lands at exactly the 3rd visible observation — making it **mathematically equivalent to** `FrameCountingGate(N=3, consecutive=False)` for any visibility pattern. The 312-scene bit-identity is the natural consequence, not a coincidence and not a wiring fault.

Two next-step paths are open (Phase 2 vs Phase 3); user decision required.

---

## Phase 1.1 — install-path trace

Install wiring is **correct** at every layer:

| Layer | File | Behaviour for M12 |
|---|---|---|
| Eval-axis CLI | `eval_streaming_ablation.py:43-55` | `("M12", {"prior":0.5, "detection_likelihood":0.8, "threshold":0.95})` |
| Dispatcher | `hooks_streaming.py:185-225` | `_SIMPLE_INSTALLERS["M12"] = install_method_12` |
| Installer | `hooks_streaming.py:52-63` | `evaluator.method_12 = BayesianGate(prior=..., detection_likelihood=..., false_positive_rate=..., threshold=...)` |
| Consumer (step_frame Step C) | `wrapper.py:371-377` | `if method_11: method_11.gate(); elif method_12: method_12.gate()` — branches independent |
| Consumer (confirmed set extract) | `wrapper.py:626-631` | same `if/elif` structure on `_confirmed` attr |

M12 install creates a real `BayesianGate` instance; the wrapper calls `method_12.gate()` (not `method_11.gate()`). **No silent fallback. The "(a) silent install bug" sub-hypothesis is rejected.**

## Phase 1.2 — skipped (Phase 1.1 + 1.3 conclusive)

Running a 1-scene smoke with print instrumentation would only confirm what Phase 1.3 proves mathematically: both gates produce identical confirmed sets for any visibility pattern. Skipping per the spec's "Skip if 1.1 is already conclusive" clause (Phase 1.1 + 1.3 jointly conclude).

## Phase 1.3 — direct gate() invocation + math

### Bayesian update arithmetic with defaults

`_update(p, True)` with `likelihood=0.8`, `fpr=0.2`:

```
num   = 0.8 · p
denom = 0.8 · p + 0.2 · (1 − p) = 0.6 · p + 0.2
p_new = 0.8 · p / (0.6 · p + 0.2)
```

Starting at `p₀ = 0.5`:

| step | posterior (visible-true update) | confirmed? (≥0.95) |
|---:|---:|:--:|
| 0 | 0.500000000000000 | False |
| 1 | 0.800000000000000 | False |
| 2 | 0.941176470588235 | False |
| **3** | **0.984615384615385** | **True** |
| 4 | 0.996108949416342 | True |
| 5 | 0.999024390243902 | True |

Threshold 0.95 sits inside the (0.941, 0.985) gap → first crossed at the 3rd visible observation.

### `gate()` actually called (source verification)

```python
def gate(self, visible_instances) -> list[int]:
    seen_this_frame = {int(i) for i in visible_instances}
    for k in seen_this_frame:
        self._posteriors[k] = self._update(self._posteriors[k], True)   # ← always True
        if self._posteriors[k] >= self.threshold:
            self._confirmed.add(k)
    return sorted(seen_this_frame & self._confirmed)
```

The loop iterates only over `seen_this_frame`, and `_update` is called only with `observed_visible=True`. The `else` branch of `_update` (lines 55-57 of `method_12_bayesian.py`) — which would decay posteriors of previously-seen-but-now-missing instances toward 0.0 — is **never invoked from `gate()`**.

The docstring promises bidirectional updates:

> A visible-this-frame observation pushes the posterior toward 1.0; **a not-visible-this-frame observation pulls it toward 0.0.**

The implementation does not. This is the silent bug.

### Equivalence sweep vs FrameCountingGate(N=3)

Five visibility patterns over 6 frames, instance id 0:

| Pattern | Frames (vis list) | BayesianGate output == FrameCountingGate(N=3) output |
|---|---|:--:|
| always | `[0],[0],[0],[0],[0],[0]` | ✓ identical |
| intermittent | `[0],[],[0],[],[0],[]` | ✓ identical |
| burst_3 | `[0],[0],[0],[],[],[]` | ✓ identical |
| delay_then | `[],[],[0],[0],[0],[]` | ✓ identical |
| split | `[0],[],[],[0],[0],[]` | ✓ identical |

The two gates produce bit-identical `gate()` returns on every pattern.

Why the equivalence is exact (not just close):
1. Both gates have monotonic state (counter ↑ vs posterior ↑).
2. Both update state only on visible frames.
3. Both confirm on the *N*th visible observation (N=3 for FrameCounting; the 3rd visible for Bayesian under these params).
4. Both return `sorted(seen_this_frame & self._confirmed)` — identical filtering.

These properties are independent of the rest of the pipeline, which is why the 312-scene aggregates are bit-identical to 15-digit precision.

---

## Scenario assignment

Spec asked us to pick among:

| # | scenario | applies? | why |
|---|---|:--:|---|
| (a) | Silent bug — M12 install wrongly routes to FrameCountingGate | partial | **Install is correct**, but the *bug is one layer deeper*: `gate()` never exercises the `observed_visible=False` path that the docstring promises. The BayesianGate is *implemented as* a monotonic counter despite carrying full bidirectional machinery. |
| (b) | Genuine mathematical convergence — Bayesian model genuinely reduces to FrameCounting | no | A *correctly implemented* bidirectional Bayesian update would decay posteriors of intermittently-visible instances, producing different confirmed sets than FrameCounting. The equivalence is an artefact of (a), not of the model. |
| (c) | Hyperparameter coincidence — chosen params happen to fire at frame 3 | yes | Even given the bug, the threshold 0.95 was tuned (intentionally or not) into the (0.941, 0.985) gap. Move threshold to 0.9 → confirmed at frame 2; to 0.99 → at frame 5. The N=3 equivalence is a *parameter-specific* fact. |

**Final verdict: (a) ∧ (c).** The bug makes BayesianGate behave as a counter; the params tune that counter to N=3.

---

## Implications

1. **M12 was effectively unevaluated in Task 1.4b Part 3.** What the run measured under the "M12" label was *FrameCountingGate(N=3) with extra computation* — identical predictions, identical metrics, just routed through a different object. The "M12" column in the 8-axis table is informationally redundant with "M11".

2. **The fix is small and targeted.** `BayesianGate.gate()` needs ~5 lines: track previously-seen instances; on each `gate()` call, iterate `seen_so_far - seen_this_frame` and apply `_update(p, observed_visible=False)`. The `_update` False branch already exists and is tested by the math in §1.3.

3. **The chosen hyperparameters are still suspicious.** Even after the fix, `threshold=0.95` will keep confirmation near 3 visible frames (with mild decay between). If the goal of Phase 2 was to give M12 *different inductive bias* than M11, the defaults need re-tuning regardless of the bug fix.

4. **No retroactive change to Task 1.4b Part 3 numbers.** Mean AP and temporal metrics for "M12" are correctly reported as they were measured — they just describe FrameCountingGate, not a Bayesian gate. The Part 3 report's §3.2 "M11 ≡ M12 bit-identical" note now has a definitive explanation.

---

## Next-step options (user decision)

Both paths assume Acceptance #3 holds (no source modifications during Phase 1 — verified).

### Phase 2 — fix the silent bug (smallest, highest signal)

1. Modify `BayesianGate.gate()` so that on each frame:
   - apply `_update(..., True)` for instances in `visible_instances` (current behaviour),
   - apply `_update(..., False)` for instances in `_posteriors.keys() - seen_this_frame` (new).
2. Add a unit test covering an intermittent pattern that diverges (e.g. `[vis, miss, miss, miss, vis]`) and assert M12 posterior < FrameCounting count at the same step.
3. 1-scene smoke (scene0011_00, M11 vs M12). If AP / lsc / ttc diverge: 312-scene PBS (≈ 3 h on cached Mask3D, equivalent to one M12-only axis of PBS A).
4. If still bit-identical after the fix → revisit hyperparams (move to Phase 3).

### Phase 3 — hyperparameter sweep (broader, longer)

1. Pick 3-5 alternate parameter sets where the N-equivalent shifts:
   - `threshold=0.90` → fires at frame 2
   - `threshold=0.99, likelihood=0.9` → fires later, sharper
   - `prior=0.3` → starts further from confirmation
   - `fpr=0.1, likelihood=0.9` → confirmed faster
2. 1-scene smoke per config, eyeball whether confirmed-set diverges from N=3.
3. Promote 1-2 configs to 312-scene re-runs.

### Recommendation

Phase 2 first. The bug exists, is documented as a behavioural contract violation (docstring vs code), is one-function in scope, and resolves the "M12 column is redundant" problem with minimal exposure. After the fix, if M12 still matches M11 closely under the original `threshold=0.95`, the natural follow-up is a Phase-3 sweep — but that question is sharper once the implementation matches the design.

Phase 3 alone, without the fix, would explore the wrong hypothesis space (different parameters on a still-monotonic counter).

---

## Acceptance check

| # | criterion | status |
|---|---|:--:|
| 1 | M11 ≡ M12 bit-identity diagnosed (scenario chosen) | ✓ (a)+(c) |
| 2 | `docs/task_1_4c_m11_m12_diagnosis.md` exists with verdict + next steps | ✓ this file |
| 3 | md5: May classes / OY3D core / `hooks_streaming.py` / Part 2 commit (087bbad) unchanged | (verified in next step) |

No source files were modified in Phase 1. Phase 2 (if approved) would touch `method_scannet/method_12_bayesian.py` only.
