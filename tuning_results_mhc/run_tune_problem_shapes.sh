#!/usr/bin/env bash
# Focused re-tune of the (M, C) shapes where Triton still regresses or loses to
# HIP after the latest full sweep, using a higher patience setting to give the
# random walk a fair chance of reaching the better region of the search space.
#
# IMPORTANT: clear the corresponding `tested_configs_mhc_fused_mhc_sk20_M*_n*_C*.json`
# files before running, otherwise the cache will skip the previously-tested space
# and the search will mostly re-confirm the existing winner.
#
# Round-2 (post-refactor) targets:
#   - C=512: M=1024 (HIP 1.57x), M=2048 (HIP 1.03x, near-tie)
#
# Run from inside the dev container under setsid nohup:
#   setsid nohup bash tuning_results/run_tune_problem_shapes.sh \
#     > tuning_results/logs/run_tune_problem_shapes.log 2>&1 &

set -euo pipefail
cd /home/anguyenh/aiter
mkdir -p tuning_results/logs

PATIENCE="${PATIENCE:-1000}"

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

run_tune 512 "1024 2048"

echo "[$(date)] === All problem shapes done ==="
