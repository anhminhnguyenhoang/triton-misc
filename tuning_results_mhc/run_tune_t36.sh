#!/usr/bin/env bash
# Full re-tune of MHC fused-sinkhorn on Triton 3.6.0, MI300X (gfx942).
# Uses 7 GPUs (0..6 inside the mhc-xdit container) in parallel per C bucket.
# Logs land in tuning_results_mhc/logs/t36/tune_C${C}.log.
#
# Run from inside mhc-xdit:
#   setsid nohup bash tuning_results_mhc/run_tune_t36.sh \
#     > tuning_results_mhc/logs/t36/run_all.log 2>&1 &

set -euo pipefail
cd /workspace/aiter
mkdir -p tuning_results_mhc/logs/t36

M_LIST="1024 2048 4096 8192 16384"
C_LIST="128 512 1024 4096 32768"
WORKERS="0 1 2 3 4 5 6"
EARLY_STOP=600

for C in $C_LIST; do
  echo "[$(date)] === Tuning C=$C ==="
  python tuning_results_mhc/tune_mhc.py --kernel fused_mhc \
    -M $M_LIST -n 4 -C $C \
    --hres-mode sinkhorn --workers $WORKERS \
    --config-file tuning_results_mhc/tuning_space_mhc.json \
    --max-configs-without-improvement $EARLY_STOP \
    2>&1 | tee tuning_results_mhc/logs/t36/tune_C${C}.log
done

echo "[$(date)] === All C buckets done ==="
