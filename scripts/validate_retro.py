"""Retro-emission patch validation: the 5 pre-launch checks, as a report.

Compares the arms produced by scripts/run_retro_validation.pbs against the
frozen cells of results/2026-07-20_e2_gate_sweep. Exits 1 if any check fails,
so the PBS job's own exit code carries the verdict.

Usage: python scripts/validate_retro.py --run-dir results/2026-07-30_retro_validation_v01
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FROZEN = Path("results/2026-07-20_e2_gate_sweep/cells")
# md5 of the frozen tracks.json artifacts (control == m11_N1, verified)
MD5_BASELINE = "191bf5ccc98d625b182f38727db5215c"
MD5_N3 = "f63b52448b56f981f76cacc3218f384e"
# frozen reference numbers for the N=3 streaming gate
REF_N3 = {"mAP": 0.2601, "n_pred_boxes_total": 513938}
REF_N3_GT = {"HOTA": 0.2465, "AssA": 0.4290, "DetA": 0.1427, "IDF1": 0.2445,
             "amota": 0.1670, "n_tracks": 108225}

rows: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    rows.append((name, bool(ok), detail))


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(run_dir: Path, name: str) -> Path:
    hits = sorted((run_dir / "cells" / name).glob("axis_*"))
    if len(hits) != 1:
        raise SystemExit(f"{name}: expected 1 axis_* dir, got {hits}")
    return hits[0]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    rd = args.run_dir

    a, b, c = (cell(rd, n) for n in ("v_baseline", "v_n1_retro", "v_n3_streaming"))
    d, e = (cell(rd, n) for n in ("v_p1_retro_s10", "v_p1_stream_s10"))

    # ---- 1. baseline arm bit-identical --------------------------------
    h_a = md5(a / "tracks.json")
    check("1. baseline arm bit-identical to frozen control",
          h_a == MD5_BASELINE, f"md5 {h_a} vs {MD5_BASELINE}")
    m_a = load(a / "metrics.json")
    m_ref = load(FROZEN / "control/axis_baseline/metrics.json")
    diffs = [k for k, v in m_ref.items()
             if isinstance(v, (int, float, str, bool)) and m_a.get(k) != v
             and k != "axis_walltime_s"]
    check("1b. baseline metrics.json values unchanged", not diffs,
          "identical (walltime excluded)" if not diffs else f"differ: {diffs}")

    # ---- 2. retro N=1 bit-identical to baseline -----------------------
    h_b = md5(b / "tracks.json")
    check("2. retro N=1 bit-identical to baseline",
          h_b == MD5_BASELINE, f"md5 {h_b} vs {MD5_BASELINE}")
    aud_b = load(b / "fire_audit.json")
    check("2b. retro N=1 dropped nothing (confirms on sight)",
          aud_b["n_pending_dropped"] == 0, f"dropped={aud_b['n_pending_dropped']}")

    # ---- 3. streaming gate reproduces the frozen numbers --------------
    h_c = md5(c / "tracks.json")
    check("3. streaming N=3 tracks.json bit-identical to frozen m11_N3",
          h_c == MD5_N3, f"md5 {h_c} vs {MD5_N3}")
    m_c = load(c / "metrics.json")
    bad = {k: (m_c.get(k), v) for k, v in REF_N3.items()
           if not (isinstance(m_c.get(k), (int, float))
                   and abs(m_c[k] - v) < (1e-4 if isinstance(v, float) else 0.5))}
    check("3b. streaming N=3 mAP / box count match frozen", not bad,
          f"mAP={m_c.get('mAP')} boxes={m_c.get('n_pred_boxes_total')}"
          + ("" if not bad else f" MISMATCH {bad}"))
    e1 = c / "e1_metrics.json"
    if e1.exists():
        g = load(e1)
        got = dict(g["gt_based"]); got["n_tracks"] = g["n_tracks"]
        bad2 = {k: (got.get(k), v) for k, v in REF_N3_GT.items()
                if not (isinstance(got.get(k), (int, float))
                        and abs(got[k] - v) < (1e-3 if isinstance(v, float) else 0.5))}
        check("3c. streaming N=3 HOTA/AssA/DetA/IDF1/AMOTA match frozen", not bad2,
              f"HOTA={got.get('HOTA'):.4f} AssA={got.get('AssA'):.4f} "
              f"DetA={got.get('DetA'):.4f} IDF1={got.get('IDF1'):.4f} "
              f"amota={got.get('amota'):.4f} n_tracks={got.get('n_tracks')}"
              + ("" if not bad2 else f" MISMATCH {bad2}"))
    else:
        check("3c. streaming N=3 GT/MOT metrics match frozen", False,
              "e1_metrics.json missing")

    # ---- 4. no pending-buffer leak across scenes ----------------------
    # invariant: buffered == flushed + dropped_at_scene_end + left_at_axis_end
    for label, cd in (("N=1 retro", b), ("phase1 retro smoke", d)):
        au = load(cd / "fire_audit.json")
        lhs = au["n_retro_buffered"]
        rhs = (au["n_retro_flushed"] + au["n_pending_dropped"]
               + au["n_pending_at_axis_end"])
        check(f"4. no pending leak ({label}): buffered == flushed+dropped+leftover",
              lhs == rhs,
              f"{lhs} == {au['n_retro_flushed']}+{au['n_pending_dropped']}"
              f"+{au['n_pending_at_axis_end']} = {rhs}")
    au_d = load(d / "fire_audit.json")
    check("4b. retro actually fired on the smoke arm (with M21/M31 live)",
          au_d["n_retro_flushed"] > 0 and au_d["n_pending_dropped"] > 0,
          f"emitted={au_d['n_retro_flushed']} dropped={au_d['n_pending_dropped']}")
    check("4c. parking lot drained every scene (no cross-scene carry)",
          au_d["n_pending_at_axis_end"] == 0,
          f"left at axis end={au_d['n_pending_at_axis_end']}")

    # ---- 5. per-sample ordering + export schema unchanged -------------
    td, te = load(d / "tracks.json"), load(e / "tracks.json")
    pd_, pe = td["pred"], te["pred"]
    check("5. token key order identical (retro vs streaming, same scenes)",
          list(pd_) == list(pe), f"{len(pd_)} tokens, order match={list(pd_) == list(pe)}")
    kd = {k for v in pd_.values() for box in v for k in box}
    ke = {k for v in pe.values() for box in v for k in box}
    check("5b. per-box key schema identical", kd == ke,
          f"{sorted(kd)}" if kd == ke else f"retro-only={kd - ke} stream-only={ke - kd}")
    check("5c. tracks.json top-level schema identical",
          set(td) == set(te), f"{sorted(td)}")
    # Retro recovers the pre-confirmation frames, so it emits strictly more in
    # aggregate. NOT asserted per token: with M31 live, a retro frame's larger
    # pre-merge set can legitimately suppress a box the streaming arm kept.
    only_confirmed = set()
    for t in pe:
        only_confirmed |= {b["tracking_id"] for b in pd_[t]} - {b["tracking_id"] for b in pe[t]}
    conf_ids = {b["tracking_id"] for v in pe.values() for b in v}
    n_ret_, n_str_ = sum(map(len, pd_.values())), sum(map(len, pe.values()))
    check("5d. retro emits more boxes than streaming in aggregate", n_ret_ > n_str_,
          f"n_boxes retro={n_ret_} stream={n_str_} (+{n_ret_ - n_str_})")
    # 5e as first written asserted retro's id set is a SUBSET of streaming's. That
    # is invalid once M21/M31 are live (E2c's phase1): a track whose only streaming
    # box was suppressed by the merge can survive in retro, which has more boxes to
    # merge. Adjudicated by results/2026-07-30_retro_m11_isolation_v01 -- with the
    # merge off, the two id sets are exactly equal (8179 == 8179, 0 either way) and
    # every retro track spans >= N frames. The invariant that actually holds:
    retro_ids = {b["tracking_id"] for v in pd_.values() for b in v}
    lost = conf_ids - retro_ids
    check("5e. retro loses no track that streaming emitted", not lost,
          f"lost={len(lost)}; retro-only={len(retro_ids - conf_ids)} "
          f"(bounded by n_merged_by_m31, see isolation run)")

    # informational: prefix-recovery accounting (not a pass/fail gate)
    n_ret, n_str = sum(map(len, pd_.values())), sum(map(len, pe.values()))
    print(f"\n[info] smoke arm: retro {n_ret} boxes vs streaming {n_str} "
          f"(+{n_ret - n_str}); confirmed tracks={len(conf_ids)}; "
          f"expected +2/track = +{2 * len(conf_ids)}\n")

    print("| # | check | verdict | detail |")
    print("|---|---|---|---|")
    for name, ok, detail in rows:
        print(f"| {name.split('.')[0]} | {name.split('. ', 1)[1]} | "
              f"{'PASS' if ok else 'FAIL'} | {detail} |")
    n_fail = sum(1 for _, ok, _ in rows if not ok)
    print(f"\n{len(rows) - n_fail}/{len(rows)} checks passed"
          + ("" if n_fail else " -- cleared to launch E2c"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
