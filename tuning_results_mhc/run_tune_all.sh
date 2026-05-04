#!/usr/bin/env bash
# Drives a full re-tune of the fused MHC kernel across the production C buckets,
# pinned to GPUs 4-7 (last 4 of 8). Logs land in tuning_results/logs/tune_C${C}.log.
#
# Run from inside the dev container (anguyenh-dev-2) under a tmux session:
#   tmux new-session -d -s mhc-tune \
#     'cd /home/anguyenh/aiter && bash tuning_results/run_tune_all.sh; \
#      echo DONE; sleep 86400'
#
# Monitor via:
#   docker exec anguyenh-dev-2 bash -c "tmux capture-pane -t mhc-tune -pS -200"
#   docker exec anguyenh-dev-2 bash -c "tail -n 50 -f /home/anguyenh/aiter/tuning_results/logs/tune_C4096.log"

set -euo pipefail
cd /home/anguyenh/aiter
mkdir -p tuning_results/logs

M_LIST="1024 2048 4096 8192 16384"
C_LIST="128 512 1024 4096 32768"

for C in $C_LIST; do
  echo "[$(date)] === Tuning C=$C ==="
  python tuning_results/tune_mhc.py --kernel fused_mhc \
    -M $M_LIST -n 4 -C $C \
    --workers 4 5 6 7 \
    --max-configs-without-improvement ${PATIENCE:-500} \
    2>&1 | tee tuning_results/logs/tune_C${C}.log
done

echo "[$(date)] === All C buckets done ==="
