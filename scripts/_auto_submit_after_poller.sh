#!/bin/bash
# Detached poller: wait for the PBS scheduler to recover, then submit the 3
# "after" jobs exactly once. Survives shell/session exit (run via setsid+nohup).
# Idempotent: will not resubmit if the jobs are already in the queue.
export PATH=/opt/pbs/bin:$PATH
cd /home/rintern16/OpenYOLO3D || exit 1
LOG=results/m22_m32_fix/_auto_submit_poller.log
mkdir -p results/m22_m32_fix
echo "[$(date)] poller started (pid $$, ppid $PPID)" >>"$LOG"

already_queued() { qstat -u rintern16 2>/dev/null | grep -Eq 'idsw_m22after|idsw_m32after|idsw_m22m'; }

MAXMIN="${MAXMIN:-1440}"              # poll window in minutes (env override)
echo "[$(date)] poll window = ${MAXMIN} min" >>"$LOG"
for i in $(seq 1 "$MAXMIN"); do       # at 60s cadence
  if qstat -B >/dev/null 2>&1; then
    if already_queued; then
      echo "[$(date)] scheduler up but jobs already in queue — not resubmitting" >>"$LOG"; exit 0
    fi
    echo "[$(date)] scheduler UP after $i checks — submitting 3 after jobs" >>"$LOG"
    qsub -N idsw_m22after    -v CONFIG="M22_after"     scripts/run_indoor_temporal_idsw_after.pbs >>"$LOG" 2>&1
    qsub -N idsw_m32after    -v CONFIG="M32_after"     scripts/run_indoor_temporal_idsw_after.pbs >>"$LOG" 2>&1
    qsub -N idsw_m22m32after -v CONFIG="M22+M32_after" scripts/run_indoor_temporal_idsw_after.pbs >>"$LOG" 2>&1
    echo "[$(date)] submitted; queue snapshot:" >>"$LOG"
    qstat -u rintern16 >>"$LOG" 2>&1
    exit 0
  fi
  sleep 60
done
echo "[$(date)] scheduler STILL DOWN after ~${MAXMIN}min — gave up; resubmit manually" >>"$LOG"
