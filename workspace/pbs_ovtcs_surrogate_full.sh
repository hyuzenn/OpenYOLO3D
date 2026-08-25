#!/bin/bash
LOG=/home/rintern16/OpenYOLO3D/workspace/pbs_ovtcs_surrogate_full.log
{
  echo "=== full start $(date) HOST=$(hostname) ==="
  cd /home/rintern16/OpenYOLO3D || exit 2
  export CUDA_HOME=/tools/cuda/cuda11.4
  export PATH=/tools/cuda/cuda11.4/bin:$PATH
  export LD_LIBRARY_PATH=/tools/cuda/cuda11.4/lib64:$LD_LIBRARY_PATH
  source /home/rintern16/miniconda3/etc/profile.d/conda.sh
  conda activate openyolo3d
  echo "PYTHON=$(which python)"
  python -u -m method_scannet.streaming.eval_ovtcs_surrogate \
    --cp-cache-dir results/outdoor_native_temporal_cpcache_thr000_single_gravity \
    --output results/2026-06-12_ablation_ovtcs_surrogate_v01
  echo "=== full EXIT=$? DONE=$(date) ==="
} > "$LOG" 2>&1
