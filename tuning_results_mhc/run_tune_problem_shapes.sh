#!/usr/bin/env bash
# Focused re-tune of the (M, C) shapes where Triton currently regresses or loses
# to HIP, using a high patience setting (5000) to give the random walk a fair
# chance of reaching the better region of the search space.
#
# Targets:
#   - C=512 : M=1024, 2048 (lose to HIP), M=8192 (regressed vs 3.5.1)
#   - C=4096: M=1024, 2048 (lose to HIP), M=16384 (regressed vs 3.5.1)
#   - C=32768: M=1024, 2048 (lose to HIP)
#
# Run from inside the dev container under setsid nohup:
#   setsid nohup bash tuning_results/run_tune_problem_shapes.sh \
#     > tuning_results/logs/run_tune_problem_shapes.log 2>&1 &

set -euo pipefail
cd /home/anguyenh/aiter
mkdir -p tuning_results/logs

PATIENCE="${PATIENCE:-5000}"

run_tune() {
  local C=$1
  local M_LIST=$2
  echo "[$(date)] === C=$C M='$M_LIST' patience=$PATIENCE ==="
  python tuning_results/tune_mhc.py --kernel fused_mhc \
    -M $M_LIST -n 4 -C $C \
    --workers 4 5 6 7 \
    --max-configs-without-improvement $PATIENCE \
    2>&1 | tee tuning_results/logs/tune_C${C}_problem.log
}

run_tune 512   "1024 2048 8192"
run_tune 4096  "1024 2048 16384"
run_tune 32768 "1024 2048"

echo "[$(date)] === All problem shapes done ==="
