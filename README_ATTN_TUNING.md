# Attention QK INT8 Per Block Tuning

This directory contains a multi-threaded tuning script for the attention QK INT8 per-block kernel.

## Files

- `tune_attn_qk_int8_multithreading.py` - Main tuning script with multi-processing support
- `tuning_space_attn_qk_int8.json` - Full tuning space configuration
- `tuning_space_attn_qk_int8_test.json` - Small tuning space for testing

## Usage

### Basic Usage

```bash
python tune_attn_qk_int8_multithreading.py \
  --seq-len 16452 29760 75600 118808 \
  --batch-size 1 \
  --num-heads 2 \
  --head-dim 128 \
  --config-file tuning_space_attn_qk_int8.json \
  --num-workers 32 \
  --max-configs-without-improvement 10000
```

### Quick Test with Small Config Space

```bash
python tune_attn_qk_int8_multithreading.py \
  --seq-len 8192 \
  --batch-size 1 \
  --num-heads 5 \
  --head-dim 128 \
  --config-file tuning_space_attn_qk_int8_test.json \
  --num-workers 2 \
  --max-configs-without-improvement 50
```

## Command-Line Arguments

- `--seq-len`: List of sequence length values to tune (required)
  - Example: `--seq-len 8192 16384 32768`
  
- `--batch-size`: Batch size (default: 1)

- `--num-heads`: Number of attention heads (default: 5)

- `--head-dim`: Head dimension (default: 128)

- `--num-workers`: Number of parallel workers (default: auto-detect based on available GPUs)

- `--config-file`: Path to tuning space JSON file (default: `tuning_space_attn_qk_int8.json`)

- `--max-configs-without-improvement`: Early stopping threshold (default: 1000)
  - Stops tuning if no improvement found after this many configs

## Output Files

The script generates two types of output files:

1. **Best Configs File**: `best_configs_B{B}_H{H}_S{S1}_{S2}_..._D{D}_YYYYMMDD_HHMMSS.json`
   - Contains the best configuration for each sequence length
   - Format:
     ```json
     {
       "8192": {
         "config": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, ...},
         "time_ms": 1.234,
         "batch_size": 1,
         "num_heads": 5,
         "head_dim": 128
       },
       ...
     }
     ```

2. **Tested Configs Cache**: `tested_configs_B{B}_H{H}_S{seq_len}_D{D}.json`
   - Tracks all tested configurations (one file per sequence length)
   - Enables resume capability if tuning is interrupted
   - Contains successful and failed configs separately

## Features

- **Multi-processing**: Parallel workers with round-robin GPU assignment
- **Resume capability**: Automatically skips already-tested configs
- **Early stopping**: Stops after N configs without improvement
- **Error handling**: Catches and logs GPU errors, continues tuning
- **Real-time monitoring**: Separate monitor process for immediate result saving
- **File locking**: Thread-safe config caching

## Tuning Space Configuration

The tuning space JSON file defines the parameter search space:

```json
{
  "block_size_m": [64, 128, 256],
  "block_size_n": [16, 32, 64, 128],
  "num_warps": [1, 2, 4, 8],
  "num_stages": [1, 2, 3, 4],
  "waves_per_eu": [0, 1, 2, 4],
  "stage": [1, 2, 3]
}
```

The script will test all combinations (Cartesian product) of these parameters.

## Example Workflow

1. Start with a quick test using the small config space:
   ```bash
   python tune_attn_qk_int8_multithreading.py \
     --seq-len 8192 \
     --config-file tuning_space_attn_qk_int8_test.json \
     --max-configs-without-improvement 50
   ```

2. If successful, run full tuning for your target sequence lengths:
   ```bash
   python tune_attn_qk_int8_multithreading.py \
     --seq-len 8192 16384 32768 65536 \
     --config-file tuning_space_attn_qk_int8.json
   ```

3. Results are saved to `best_configs_B*_H*_S*_D*_*.json`

4. If interrupted, simply re-run the same command - it will resume from where it left off.

## Notes

- The script uses random shuffling of configs for better exploration
- GPU core dumps are automatically cleaned up
- Failed configs are tracked separately from successful ones
- Monitor process shows real-time progress and best configs

