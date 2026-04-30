#!/usr/bin/env bash
# Re-tune the small-C buckets (128, 512, 1024) with USE_REDUCE_SPLITC=False
# only. The default tuning space biases 94% of configs toward
# USE_REDUCE_SPLITC=True, which starves the inline path that is optimal for
# C < 4096. Search space here is 576 configs (vs 9792); with a high
# early-stop budget we can effectively explore it all.
#
# Pinned to GPUs 4-7. Logs in tuning_results_mhc/logs/retune_inline_C${C}.log.

set -euo pipefail
cd /workspace/aiter
mkdir -p tuning_results_mhc/logs

M_LIST="1024 2048 4096 8192 16384"
C_LIST="128 512 1024"

for C in $C_LIST; do
  echo "[$(date)] === Retune inline-only C=$C ==="
  python tuning_results_mhc/tune_mhc.py --kernel fused_mhc \
    -M $M_LIST -n 4 -C $C \
    --hres-mode sinkhorn --workers 4 5 6 7 \
    --config-file tuning_results_mhc/tuning_space_mhc_inline.json \
    --max-configs-without-improvement 400 \
    2>&1 | tee tuning_results_mhc/logs/retune_inline_C${C}.log
done

echo "[$(date)] === All inline-only buckets done ==="
