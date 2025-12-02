#!/bin/bash

BATCH_SIZES=(1 1 1 2)
NUM_HEADS=(5 24 3 2)
SEQ_LENS=(75600 16452 118808 29760)
# OUTPUT_JSON="attn_qk_int8_benchmarks.json"

# # Remove old results if they exist
# rm -f ${OUTPUT_JSON}

# Run benchmarks for all configurations
for i in ${!BATCH_SIZES[@]}; do
    echo "Running benchmark $((i+1))/4: batch_size=${BATCH_SIZES[i]}, num_heads=${NUM_HEADS[i]}, seq_len=${SEQ_LENS[i]}"
    TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 AMDGCN_USE_BUFFER_OPS=0 TRITON_HIP_USE_BLOCK_PINGPONG=1 python op_tests/op_benchmarks/triton/bench_attn_qk_int8_per_block.py \
        --batch_size ${BATCH_SIZES[i]} \
        --num_heads ${NUM_HEADS[i]} \
        --seq_len ${SEQ_LENS[i]} \
        -o
done

# echo "Benchmarks complete. Results saved to ${OUTPUT_JSON}"
echo "Benchmarks complete."