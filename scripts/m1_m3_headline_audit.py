"""M1 + M3 headline audit — population-controlled ego/global OV-TCS comparison
and aggregation reconciliation. Uses only existing cached outputs:
E1 tracking submissions (per-frame boxes with tracking_id + per-frame label)
and nuScenes GT metadata. No inference, no re-association.

Outputs: results/2026-07-15_m1m3_headline_audit_v01/{audit.json, audit.md}
"""
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

E1 = '/home/rintern16/OpenYOLO3D/results/e1_outdoor_mot_compare_v05/outputs'
NUSC = '/home/rintern16/OpenYOLO3D/data/nuscenes/v1.0-trainval'
OUT = 'results/2026-07-15_m1m3_headline_audit_v01'
os.makedirs(OUT, exist_ok=True)

print('loading sample.json ...', flush=True)
samples = {s['token']: (s['scene_token'], s['timestamp'])
           for s in json.load(open(f'{NUSC}/sample.json'))}


def load_arm(name):
    sub = json.load(open(f'{E1}/tracking_submission_{name}.json'))['results']
    tracks = defaultdict(list)   # (scene, tid) -> [(ts, label, x, y, sample_token)]
    n_boxes = Counter()          # sample_token -> box count
    for st, boxes in sub.items():
        scene, ts = samples[st]
        n_boxes[st] = len(boxes)
        for b in boxes:
            tracks[(scene, b['tracking_id'])].append(
                (ts, b['tracking_name'], b['translation'][0], b['translation'][1], st))
    for v in tracks.values():
        v.sort(key=lambda r: r[0])
    return tracks, n_boxes


def track_stats(obs):
    L = len(obs)
    labels = [o[1] for o in obs]
    if L < 2:
        return L, None, None, None
    sw = sum(labels[i] != labels[i + 1] for i in range(L - 1))
    csr = sw / (L - 1)
    lnorm = 1 - 1 / L
    return L, csr, lnorm, lnorm * (1 - csr)


arms = {}
for name in ('ego', 'global'):
    print(f'building tracks: {name}', flush=True)
    tracks, n_boxes = load_arm(name)
    per_track = {}
    for key, obs in tracks.items():
        L, csr, lnorm, c = track_stats(obs)
        per_track[key] = dict(L=L, csr=csr, lnorm=lnorm, ovtcs=c,
                              scene=key[0], obs=obs)
    arms[name] = dict(per_track=per_track, n_boxes=n_boxes)

report = {}

# ---------- sanity + box conservation ----------
for name, a in arms.items():
    scored = [t for t in a['per_track'].values() if t['L'] >= 2]
    report[f'{name}_n_tracks_all'] = len(a['per_track'])
    report[f'{name}_n_tracks_scored'] = len(scored)
    report[f'{name}_pooled_ovtcs'] = float(np.mean([t['ovtcs'] for t in scored]))
    report[f'{name}_pooled_lnorm'] = float(np.mean([t['lnorm'] for t in scored]))
    report[f'{name}_pooled_stability'] = float(np.mean([1 - t['csr'] for t in scored]))
    report[f'{name}_total_boxes'] = sum(a['n_boxes'].values())
    report[f'{name}_boxes_in_scored'] = sum(t['L'] for t in scored)

eq = sum(arms['ego']['n_boxes'][k] == arms['global']['n_boxes'][k]
         for k in arms['ego']['n_boxes'])
report['samples_with_equal_box_count'] = eq
report['n_samples'] = len(arms['ego']['n_boxes'])

# ---------- M3: aggregation reconciliation ----------
def scene_table(a):
    by = defaultdict(list)
    for t in a['per_track'].values():
        if t['L'] >= 2:
            by[t['scene']].append(t['ovtcs'])
    return {s: (float(np.mean(v)), len(v), float(np.sum(v))) for s, v in by.items()}

se, sg = scene_table(arms['ego']), scene_table(arms['global'])
common = sorted(set(se) & set(sg))
report['n_scenes'] = len(common)
scene_deltas = [sg[s][0] - se[s][0] for s in common]
report['scene_weighted_delta'] = float(np.mean(scene_deltas))
report['pooled_delta'] = report['global_pooled_ovtcs'] - report['ego_pooled_ovtcs']
# correlation of per-scene delta with scene track counts (is the gap driven by small scenes?)
ne = np.array([se[s][1] for s in common], float)
d = np.array(scene_deltas)
report['corr_scene_delta_vs_ego_ntracks'] = float(np.corrcoef(d, ne)[0, 1])

# cluster (scene-level) bootstrap of the POOLED (track-weighted) delta
rng = random.Random(0)
sums_e = np.array([se[s][2] for s in common]); cnt_e = np.array([se[s][1] for s in common], float)
sums_g = np.array([sg[s][2] for s in common]); cnt_g = np.array([sg[s][1] for s in common], float)
boot = []
idx_all = np.arange(len(common))
for _ in range(10000):
    idx = np.array([rng.randrange(len(common)) for _ in idx_all])
    boot.append(sums_g[idx].sum() / cnt_g[idx].sum() - sums_e[idx].sum() / cnt_e[idx].sum())
boot = np.sort(boot)
report['pooled_delta_ci95'] = [float(boot[249]), float(boot[9749])]
report['pooled_delta_frac_positive'] = float((boot > 0).mean())
# scene-weighted bootstrap too, for comparison with the old +0.049
boot2 = []
for _ in range(10000):
    idx = np.array([rng.randrange(len(common)) for _ in idx_all])
    boot2.append(float(d[idx].mean()))
boot2 = np.sort(boot2)
report['scene_weighted_delta_ci95'] = [float(boot2[249]), float(boot2[9749])]

# ---------- M1a: length-stratified deltas ----------
bins = [(2, 2), (3, 3), (4, 4), (5, 9), (10, 19), (20, 10 ** 9)]
strata = []
for lo, hi in bins:
    row = dict(bin=f'{lo}' if lo == hi else f'{lo}-{hi if hi < 10**9 else "+"}')
    for name in ('ego', 'global'):
        ts = [t for t in arms[name]['per_track'].values() if lo <= t['L'] <= hi]
        row[f'{name}_n'] = len(ts)
        row[f'{name}_ovtcs'] = float(np.mean([t['ovtcs'] for t in ts])) if ts else None
        row[f'{name}_stability'] = float(np.mean([1 - t['csr'] for t in ts])) if ts else None
    strata.append(row)
report['length_strata'] = strata

# ---------- M1b: distribution standardization ----------
# global's per-length mean ovtcs, reweighted to ego's length distribution
def by_len(a):
    m = defaultdict(list)
    for t in a['per_track'].values():
        if t['L'] >= 2:
            m[t['L']].append(t['ovtcs'])
    return {L: (np.mean(v), len(v)) for L, v in m.items()}

be, bg = by_len(arms['ego']), by_len(arms['global'])
tot_e = sum(n for _, n in be.values())
num = den = 0.0
for L, (mean_e, n) in be.items():
    if L in bg:
        num += bg[L][0] * n
        den += n
report['global_ovtcs_under_ego_length_dist'] = float(num / den)
report['ego_length_mass_covered_by_global'] = float(den / tot_e)

# ---------- M1c: GT-instance-paired comparison ----------
print('loading sample_annotation.json ...', flush=True)
anns = json.load(open(f'{NUSC}/sample_annotation.json'))
eval_samples = set(arms['ego']['n_boxes'])
gt_by_sample = defaultdict(list)   # sample -> (x, y, instance_token)
for a in anns:
    if a['sample_token'] in eval_samples:
        gt_by_sample[a['sample_token']].append(
            (a['translation'][0], a['translation'][1], a['instance_token']))
del anns
print('matching pred boxes to GT instances ...', flush=True)

def gt_match(arm):
    # per-frame nearest-GT within 2.0 m (BEV), then majority vote per track
    track_votes = defaultdict(Counter)
    per_sample_pred = defaultdict(list)  # sample -> (x, y, trackkey)
    for key, t in arm['per_track'].items():
        for (ts, lab, x, y, st) in t['obs']:
            per_sample_pred[st].append((x, y, key))
    for st, preds in per_sample_pred.items():
        gts = gt_by_sample.get(st)
        if not gts:
            continue
        g = np.array([(x, y) for x, y, _ in gts])
        for (x, y, key) in preds:
            dd = np.hypot(g[:, 0] - x, g[:, 1] - y)
            i = int(dd.argmin())
            if dd[i] <= 2.0:
                track_votes[key][gts[i][2]] += 1
    return {key: c.most_common(1)[0][0] for key, c in track_votes.items()}

inst = {}
for name in ('ego', 'global'):
    m = gt_match(arms[name])
    per_inst = defaultdict(list)
    for key, itok in m.items():
        t = arms[name]['per_track'][key]
        if t['L'] >= 2:
            per_inst[itok].append(t)
    inst[name] = per_inst
    report[f'{name}_matched_scored_tracks'] = sum(len(v) for v in per_inst.values())
    report[f'{name}_matched_instances'] = len(per_inst)

common_inst = sorted(set(inst['ego']) & set(inst['global']))
report['n_common_instances'] = len(common_inst)
pe, pg, frag_e, frag_g = [], [], [], []
for it in common_inst:
    te, tg = inst['ego'][it], inst['global'][it]
    # frame-weighted mean over the instance's tracks
    pe.append(sum(t['ovtcs'] * t['L'] for t in te) / sum(t['L'] for t in te))
    pg.append(sum(t['ovtcs'] * t['L'] for t in tg) / sum(t['L'] for t in tg))
    frag_e.append(len(te)); frag_g.append(len(tg))
pe, pg = np.array(pe), np.array(pg)
report['inst_paired_ego_mean'] = float(pe.mean())
report['inst_paired_global_mean'] = float(pg.mean())
report['inst_paired_delta'] = float((pg - pe).mean())
report['inst_paired_frac_improved'] = float((pg > pe).mean())
report['inst_paired_frac_tied'] = float((pg == pe).mean())
report['inst_mean_frag_ego'] = float(np.mean(frag_e))
report['inst_mean_frag_global'] = float(np.mean(frag_g))
# instance-level bootstrap CI on the paired delta
boot3 = []
n = len(pe)
for _ in range(10000):
    idx = np.array([rng.randrange(n) for _ in range(n)])
    boot3.append(float((pg[idx] - pe[idx]).mean()))
boot3 = np.sort(boot3)
report['inst_paired_delta_ci95'] = [float(boot3[249]), float(boot3[9749])]

with open(f'{OUT}/audit.json', 'w') as f:
    json.dump(report, f, indent=1)
print(json.dumps({k: v for k, v in report.items() if k != 'length_strata'}, indent=1))
print('\nlength strata:')
for r in report['length_strata']:
    print(r)
print('DONE_AUDIT')
