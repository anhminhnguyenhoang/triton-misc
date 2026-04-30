#!/usr/bin/env bash
# Drives a full re-tune of the fused MHC kernel across the production C buckets,
# pinned to GPUs 4-7 (last 4 of 8). Logs land in tuning_results_mhc/logs/tune_C${C}.log.
#
# Run from inside the dev container (mhc-dev) under tmux/setsid:
#   setsid nohup bash tuning_results_mhc/run_tune_all.sh \
#     > tuning_results_mhc/logs/run_all.log 2>&1 &
#
# Monitor via:
#   docker exec mhc-dev bash -c "tail -n 50 -f /workspace/aiter/tuning_results_mhc/logs/tune_C4096.log"

set -euo pipefail
cd /workspace/aiter
mkdir -p tuning_results_mhc/logs

M_LIST="1024 2048 4096 8192 16384"
C_LIST="128 512 1024 4096 32768"

for C in $C_LIST; do
  echo "[$(date)] === Tuning C=$C ==="
  python tuning_results_mhc/tune_mhc.py --kernel fused_mhc \
    -M $M_LIST -n 4 -C $C \
    --hres-mode sinkhorn --workers 4 5 6 7 \
    --config-file tuning_results_mhc/tuning_space_mhc.json \
    --max-configs-without-improvement 200 \
    2>&1 | tee tuning_results_mhc/logs/tune_C${C}.log
done

echo "[$(date)] === All C buckets done ==="
