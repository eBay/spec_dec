#!/bin/bash

# Group rate experiment for batch speculative decoding
# This script tests the impact of grouping (same-length sequences) on overhead
# Compares Random vs All-Mean configurations across different batch sizes

# Configuration
GPU_DEVICE=1
NUM_PROMPTS=100
MAX_NEW_TOKENS=50
N_DRAFT_TOKENS=5
WINDOW_SIZE=50
LOG_DIR="logs/group_rate"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Group Rate Experiment"
echo "======================================"
echo "GPU Device: $GPU_DEVICE"
echo "Num prompts: $NUM_PROMPTS"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Draft tokens: $N_DRAFT_TOKENS"
echo "Window size: $WINDOW_SIZE"
echo "Log directory: $LOG_DIR"
echo "======================================"
echo

# Batch sizes to test
BATCH_SIZES=(2 4 8 16 32)

# Configurations: Random and All-Mean
declare -A CONFIGS
CONFIGS["random"]="data/multi30k_random_100.jsonl"
CONFIGS["all-mean"]="data/multi30k_16tokens_100.jsonl"

# Run experiments for each batch size and configuration
for BS in "${BATCH_SIZES[@]}"; do
    echo "Testing batch size $BS..."
    echo "--------------------------------------"

    for CONFIG_NAME in "random" "all-mean"; do
        INPUT_FILE="${CONFIGS[$CONFIG_NAME]}"
        LOG_FILE="$LOG_DIR/group_rate_bs${BS}_${CONFIG_NAME}.log"

        echo "  Running $CONFIG_NAME configuration..."
        echo "    Input file: $INPUT_FILE"
        echo "    Log file: $LOG_FILE"

        # Run the benchmark with profiling enabled
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python unified_benchmark.py \
            --methods Ours-XBatch \
            --input_file "$INPUT_FILE" \
            --num_prompts $NUM_PROMPTS \
            --max_new_tokens $MAX_NEW_TOKENS \
            --n_draft_tokens $N_DRAFT_TOKENS \
            --batch_size $BS \
            --window_size $WINDOW_SIZE \
            --scheduling_strategy cross_batch \
            --sort_by_length \
            --enable_profiling > "$LOG_FILE" 2>&1

        if [ $? -eq 0 ]; then
            echo "    ✓ Completed successfully"

            # Extract key metrics for quick preview
            GROUP_RATE=$(grep "Cross-batch (same length):" "$LOG_FILE" | grep -oP '\(\s*\K[\d.]+(?=%\))')
            TPS=$(grep "Ours-XBatch" "$LOG_FILE" | grep -oP '\d+\.\d+\s+\K[\d.]+(?=\s)' | head -1)
            echo "    Group rate: ${GROUP_RATE}%"
            echo "    Tokens/s: $TPS"
        else
            echo "    ✗ Failed (check log for details)"
        fi
        echo
    done
    echo
done

echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo
echo "To generate the LaTeX table, run:"
echo "  python scripts/analyze_group_rate.py"
echo
echo "Log files saved in: $LOG_DIR"
echo "======================================"