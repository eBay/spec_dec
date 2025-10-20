#!/bin/bash

# Overhead scaling experiment for batch speculative decoding
# This script runs the unified benchmark with different batch sizes
# and enables profiling to capture overhead metrics

# Configuration
GPU_DEVICE=1
INPUT_FILE="data/spec_bench/question.jsonl"
NUM_PROMPTS=100
MAX_NEW_TOKENS=128
N_DRAFT_TOKENS=5
LOG_DIR="logs/overhead_scaling"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Overhead Scaling Experiment"
echo "======================================"
echo "GPU Device: $GPU_DEVICE"
echo "Input file: $INPUT_FILE"
echo "Num prompts: $NUM_PROMPTS"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Draft tokens: $N_DRAFT_TOKENS"
echo "Log directory: $LOG_DIR"
echo "======================================"
echo

# Batch sizes to test
BATCH_SIZES=(1 2 4 8 16 32)

# Run experiments for each batch size
for BS in "${BATCH_SIZES[@]}"; do
    LOG_FILE="$LOG_DIR/overhead_bs${BS}.log"

    echo "Running batch size $BS..."
    echo "  Log file: $LOG_FILE"

    # Run the benchmark with profiling enabled
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python unified_benchmark.py \
        --methods Ours-Batch-Cache \
        --input_file "$INPUT_FILE" \
        --num_prompts $NUM_PROMPTS \
        --max_new_tokens $MAX_NEW_TOKENS \
        --n_draft_tokens $N_DRAFT_TOKENS \
        --batch_size $BS \
        --enable_profiling > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"

        # Extract key metrics for quick preview
        TPS=$(grep "Ours-Batch-Cache" "$LOG_FILE" | grep -oP 'Tokens/s\s+\K[\d.]+' | head -1)
        VERIF_CALLS=$(grep "Total verification calls:" "$LOG_FILE" | grep -oP '\d+' | tail -1)
        echo "  Tokens/s: $TPS"
        echo "  Verification calls: $VERIF_CALLS"
    else
        echo "  ✗ Failed (check log for details)"
    fi

    echo
done

echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo
echo "To generate the LaTeX table, run:"
echo "  python c_overhead_b.py"
echo
echo "Log files saved in: $LOG_DIR"
echo "======================================"