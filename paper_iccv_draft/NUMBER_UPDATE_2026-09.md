# Thesis draft — number update map (old → corrected pipeline)

Draft: `paper_iccv_draft/` on `thesis/graduation-2026` (frozen at 3725c5a).
Corrections behind the new numbers: custom evaluator fix (ad94732), official
pred-attribute rule (438f121), 10-sweep CenterPoint input + adapter fix
(cbdff03). New detector output: 718,786 boxes (was 1,029,380).

Sources (all already adopted by the conference manuscript on `main`):
- Tab 1: `audit/table1_regen_c1fix_2026-08-29/TABLE1_REGENERATION_REPORT.md`, `paper_stats.json`
- Tab 2: `audit/tables23s1fig3_regen_2026-09-07/t2_trackmatch/e2b_report.json`
- Tab 3: `audit/tables23s1fig3_regen_2026-09-07/t3_causal/e2_report.json`
- Tab 4: `audit/tables23s1fig3_regen_2026-09-07/nsweep_rows.json` (cross-table audit: 0 failures)
- Tab 5: unchanged (indoor is not touched by any nuScenes fix; `main` keeps identical values)

## Tab 1 — detection-budget-matched (`tab:detmatch`)

| Frame | Arm | Boxes | mAP | NDS | HOTA | AssA | DetA | DetRe | IDF1 | AMOTA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sensor | Baseline | 1,029,380 → **718,786** | .3408 → **.5613** | .3150 → **.6480** | .1372 → **.1793** | .1896 → **.2007** | .1007 → **.1620** | .5904 → **.6690** | .0759 → **.1112** | .0501 → **.0858** |
| Sensor | Control | 360,309 → **257,259** | .3324 → **.5504** | .3110 → **.6450** | .2098 → **.2735** | .2044 → **.2119** | .2184 → **.3570** | .5182 → **.6160** | .1517 → **.2082** | .0547 → **.0942** |
| Sensor | Confirm. | 360,309 → **257,259** | .2023 → **.3009** | .2518 → **.5229** | .1960 → **.2462** | .2535 → **.2744** | .1537 → **.2232** | .3859 → **.4287** | .1531 → **.2054** | .0887 → **.1338** |
| World | Baseline | 1,029,380 → **718,786** | .3408 → **.5613** | .3150 → **.6480** | .2011 → **.2837** | .4089 → **.5014** | .0995 → **.1613** | .5838 → **.6663** | .1764 → **.2715** | .1580 → **.2896** |
| World | Control | 754,500 → **517,566** | .3396 → **.5598** | .3143 → **.6479** | .2307 → **.3305** | .4175 → **.5146** | .1282 → **.2133** | .5668 → **.6550** | .2190 → **.3391** | .1655 → **.3137** |
| World | Confirm. | 754,500 → **517,566** | .2900 → **.4921** | .3096 → **.6200** | .2263 → **.3224** | .4314 → **.5293** | .1196 → **.1973** | .5326 → **.6141** | .2121 → **.3288** | .2033 → **.3864** |

Bold pattern unchanged (sensor IDF1 CI still spans zero → no bold).
Caption thresholds: 0.1875 → **0.2214** (sensor), 0.1211 → **0.1272** (world).

Text (§detmatch), combined-delta 95% CIs (control → confirmation):
- AssA: sensor [+0.0370,+0.0615] → **[+0.0471,+0.0785]**; world [+0.0086,+0.0195] → **[+0.0098,+0.0203]**
- Per-scene sensor AssA: mean +0.0340 → **+0.0460**; aggregate diff 0.0491 → **0.0625**; 110/150 → **113/150**; p 4.6e-13 → **2.6e-15**; r 0.68 → **0.74**
- AMOTA: +0.034 / +0.038 → **+0.040 / +0.073** (0.0942→0.1338, 0.3137→0.3864)
- DetA: [−.0700,−.0593] → **[−.1408,−.1267]**; [−.0101,−.0072] → **[−.0183,−.0138]**
- DetRe: [−.1483,−.1170] → **[−.2087,−.1665]**; [−.0404,−.0286] → **[−.0477,−.0350]**
- HOTA: [−.0180,−.0103] → **[−.0323,−.0231]**; [−.0059,−.0027] → **[−.0100,−.0063]**
- IDF1 sensor: [−.0046,+.0064] → **[−.0108,+.0037]** (still spans 0; per-scene p 0.45 → **0.007**, 77/150 → **65/150** — per-scene test now significant, aggregate CI is not: reword); world [−.0091,−.0047] → **[−.0135,−.0070]**

## Tab 2 — identity-budget-matched (`tab:idmatch`)

| Frame | Arm | Boxes | HOTA | AssA | DetA | IDF1 | AMOTA |
|---|---|---:|---:|---:|---:|---:|---:|
| Sensor | Top-K | 190,708 → **138,053** | .2461 → **.2959** | .2346 → **.2607** | .2607 → **.3383** | .2081 → **.2542** | .0547 → **.1004** |
| Sensor | Random-K (AssA) | | | .2321–.2460 → **.2595–.2765** | | | |
| Sensor | Confirm. | 360,309 → **257,259** | .1960 → **.2462** | .2535 → **.2744** | .1537 → **.2232** | .1531 → **.2054** | .0887 → **.1338** |
| World | Top-K | 608,168 → **412,203** | .2538 → **.3630** | .4173 → **.5074** | .1553 → **.2610** | .2596 → **.3969** | .1587 → **.2911** |
| World | Random-K (AssA) | | | .3878–.3934 → **.4705–.4776** | | | |
| World | Confirm. | 754,500 → **517,566** | .2263 → **.3224** | .4314 → **.5293** | .1196 → **.1973** | .2121 → **.3288** | .2033 → **.3864** |

Caption K: 64,428 → **41,207** (sensor), 108,225 → **68,222** (world); also in `2_formatting.tex:171-172`.
Delete the caption's "0.054713 vs 0.054715" sentence (the coincidence is gone: 0.1004 vs 0.0942).
Text: AssA CIs [+.0129,+.0258] → **[+.0071,+.0216]**; [+.0101,+.0185] → **[+.0177,+.0267]**; scenes 117 → **107**, 111 → **116**; p 2.2e-16 → **3.9e-15**, 4.1e-11 → **2.9e-16**.

⚠ **Claim change:** sensor random-K max **0.2765 > confirmation 0.2744**. "Random selection reproduces none of it" is false in the sensor frame; true only in world (.4776 < .5293). `main` rewords this as "in the sensor frame the random arms are *not* separated from it".

## Tab 3 — causal emission (`tab:emission`)

| Frame | Arm | mAP | HOTA | AssA | AMOTA |
|---|---|---:|---:|---:|---:|
| Sensor | Control | .3213 → **.5287** | .2417 → **.3044** | .2105 → **.2147** | .0574 → **.0997** |
| Sensor | Causal | .1360 → **.2047** | .2056 → **.2484** | .2861 → **.3088** | .0800 → **.1290** |
| World | Control | .3367 → **.5562** | .2729 → **.3864** | .4315 → **.5308** | .1825 → **.3417** |
| World | Causal | .2569 → **.4480** | .2456 → **.3451** | .4199 → **.5184** | .2089 → **.4050** |

Caption: 221,297 @ 0.2519 → **168,193 @ 0.3320** (sensor); 503,619 @ 0.1539 → **361,426 @ 0.1673** (world).
Text: sensor AssA CI [+.0577,+.0922] → **[+.0722,+.1162]** (vs retro **[+.0471,+.0785]**), 110/150 → **112/150**;
world CI [−.0183,−.0047] → **[−.0195,−.0049]**, p 0.008 → **0.036**, r −0.25 → **−0.20**, 53/150 → **59/150**.
Direction unchanged (world AssA still reverses).

## Tab 4 — confirmation-window sweep (`tab:nsweep`)

| Frame | N | Boxes | ΔAssA | 95% CI | ΔAMOTA | ΔDetRe |
|---|---|---:|---:|---|---:|---:|
| Sensor | 2 | 551,452 → **375,982** | +.0219 → **+.0296** | **[+.0217,+.0383]** | +.0177 → **+.0176** | −.0817 → **−.1210** |
| Sensor | 3 | 360,309 → **257,259** | +.0491 → **+.0625** | **[+.0471,+.0785]** | +.0339 → **+.0396** | −.1323 → **−.1873** |
| Sensor | 4 | 269,330 → **203,019** | +.0767 → **+.0933** | **[+.0720,+.1149]** | +.0393 → **+.0478** | −.1606 → **−.2217** |
| Sensor | 5 | 217,295 → **171,395** | +.1038 → **+.1221** | **[+.0960,+.1480]** | +.0414 → **+.0529** | −.1794 → **−.2408** |
| World | 2 | 869,354 → **594,547** | +.0058 → **+.0103** | **[+.0068,+.0140]** | +.0228 → **+.0614** | −.0219 → **−.0278** |
| World | 3 | 754,500 → **517,566** | +.0139 → **+.0148** | **[+.0098,+.0203]** | +.0378 → **+.0726** | −.0342 → **−.0409** |
| World | 4 | 658,099 → **457,121** | +.0184 → **+.0181** | **[+.0120,+.0250]** | +.0424 → **+.0736** | −.0412 → **−.0492** |
| World | 5 | 576,711 → **409,463** | +.0220 → **+.0199** | **[+.0130,+.0276]** | +.0381 → **+.0682** | −.0470 → **−.0556** |

⚠ **Text change:** sensor AMOTA no longer "flattens" (+.0478 → +.0529 still rising). World AMOTA still peaks at N=4 (+.0736) and falls at N=5 (+.0682).
Control AssA N=2→5: .1980→.2107 → **.2080→.2146** (sensor), .4131→.4262 → **.5061→.5260** (world).

## Tab 5 — indoor (`tab:indoor`)

No change.

## Other body text

- `0_abstract.tex:20` AMOTA: 0.055→0.089 → **0.094→0.134**; 0.166→0.203 → **0.314→0.386**
- `2_formatting.tex:139-140` detector: 1,029,380 boxes / mAP .3408 / NDS .3150 → **718,786 / .5613 / .6480**; mention 10-sweep input
- `2_formatting.tex:159-161,171-172,178-179` budgets/thresholds/K: see Tabs 1–2 above
- `26.9%` (indoor): unchanged
