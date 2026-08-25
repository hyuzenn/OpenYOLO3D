#!/usr/bin/env python3
"""Confirmation-window sensitivity table, straight from the run reports.

N=3 is READ FROM THE FROZEN E2c RUN, not recomputed, so the sweep's N=3 row is
by construction the same number the manuscript's Table 1 reports.

Usage: python scripts/aggregate_n_sweep.py [--date 2026-08-04]
"""
import argparse
import glob
import json
import sys

FROZEN_N3 = 'results/2026-07-30_e2c_retro_thrmatch_v01/e2_report.json'
FRAMES = [('ego', 'Sensor'), ('global', 'World')]


def load(n, date):
    if n == 3:
        return FROZEN_N3
    hits = sorted(glob.glob(f'results/{date}_nsweep_N{n}_v*/e2_report.json'))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default='2026-08-04')
    args = ap.parse_args()

    rows, missing = [], []
    for n in (2, 3, 4, 5):
        path = load(n, args.date)
        if path is None:
            missing.append(n)
            continue
        rep = json.load(open(path))
        for key, label in FRAMES:
            arm = rep['arms'].get(key)
            if arm is None:
                missing.append(f'{n}/{key}')
                continue
            g, c = arm['rows']['Gate'], arm['rows']['Control']
            ci = arm['bootstrap_ci_combined_delta']
            rec = arm['recall_DetRe']
            rows.append(dict(
                N=n, delay=n - 1, frame=label, boxes=g['boxes'],
                d_assa=g['AssA'] - c['AssA'], ci_assa=ci['AssA'],
                d_amota=g['AMOTA'] - c['AMOTA'],
                d_map=g['mAP'] - c['mAP'],
                d_recall=rec['Gate'] - rec['Control'], ci_recall=ci['DetRe'],
                src=path))

    if not rows:
        sys.exit(f'no reports found (missing N={missing})')

    def iv(c):
        return f'[{c[0]:+.4f}, {c[1]:+.4f}]'

    print('\n### Confirmation-window sensitivity '
          '(retrospective emission vs exact detection-budget-matched control)\n')
    print('| N | Delay (frames) | Frame | Emitted boxes | dAssA | 95% CI '
          '| dAMOTA | dmAP | dDetRe | 95% CI |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f'| {r["N"]} | {r["delay"]} | {r["frame"]} | {r["boxes"]:,} '
              f'| {r["d_assa"]:+.4f} | {iv(r["ci_assa"])} '
              f'| {r["d_amota"]:+.4f} | {r["d_map"]:+.4f} '
              f'| {r["d_recall"]:+.4f} | {iv(r["ci_recall"])} |')

    print('\nDelay is N-1 frames = (N-1)/2 s at the 2 Hz annotated-frame rate.')
    print('AMOTA and mAP are whole-split estimators: point estimates, no interval.')
    print('N=3 rows are read from the frozen E2c run, not recomputed.')
    if missing:
        print(f'\nSTILL MISSING: {missing}')
    print('\nSources:')
    for p in dict.fromkeys(r['src'] for r in rows):
        print(f'  {p}')


if __name__ == '__main__':
    main()
