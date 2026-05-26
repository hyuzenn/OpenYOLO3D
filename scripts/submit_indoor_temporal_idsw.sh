#!/bin/bash
# Task 3.2 — submit the 5 re-evaluation jobs for the missing temporal metrics
# (ID Sw / Obj for all axes + TTC for the After axes).
#
# Run this AT HOME when ready:   bash scripts/submit_indoor_temporal_idsw.sh
# It only calls qsub; the cluster runs the jobs independently of your laptop.
# Each job self-guards with a 2-scene SMOKE gate, so a broken run dies in
# minutes instead of wasting GPU hours.
set -euo pipefail
cd /home/rintern16/OpenYOLO3D

echo ">> base axes A (baseline, M11)"
qsub -N idsw_base_a   -v AXES="baseline M11",OUTTAG=base_a scripts/run_indoor_temporal_idsw_base.pbs

echo ">> base axes B (M21, M31)"
qsub -N idsw_base_b   -v AXES="M21 M31",OUTTAG=base_b      scripts/run_indoor_temporal_idsw_base.pbs

echo ">> after M22"
qsub -N idsw_m22after -v CONFIG="M22_after"                scripts/run_indoor_temporal_idsw_after.pbs

echo ">> after M32"
qsub -N idsw_m32after -v CONFIG="M32_after"                scripts/run_indoor_temporal_idsw_after.pbs

echo ">> after M22+M32"
qsub -N idsw_m22m32after -v CONFIG="M22+M32_after"         scripts/run_indoor_temporal_idsw_after.pbs

echo
echo "submitted. check with:  qstat -u rintern16"
