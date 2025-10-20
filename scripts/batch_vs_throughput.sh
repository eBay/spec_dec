
#!/bin/bash
# Throughput vs Batch Size Benchmark
# Generates table of token throughput for different methods and batch sizes

# Configuration
BATCH_SIZES=(1 2 4 8 16 32)
# BATCH_SIZES=(8)
# BATCH_SIZES=(32)
MAX_NEW_TOKENS=128
N_DRAFT_TOKENS=5
NUM_PROMPTS=480
INPUT_FILE="data/spec_bench/question.jsonl"
WINDOW_SIZE=48
GPU_DEVICE=2
MAX_INPUT_LEN=1024

# Methods to benchmark
# METHODS=(
#     "RD-HF"
#     "RD-HF-Batch"
#     "SP-HF"
#     "BSP-Adaptive"
#     "Ours-Batch-Cache"
#     "Ours-XBatch"
# )

# METHODS=(
#     # "RD-HF"
#     # "RD-HF-Batch"
#     # "SP-HF"
#     # "BSP-Adaptive"
#     # "Ours-Batch-Cache"
#     # "Ours-XBatch"
#     "DSD"
# )

METHODS=(
    # "RD-HF"
    "RD-HF-Batch"
    "SP-HF"
    # "BSP-Adaptive"
    "Ours-Batch-Cache"
    "Ours-XBatch"
    # "DSD"
)

# Model configurations - arrays for bash
# declare -a MODEL_NAMES=("qwen" "vicuna" "glm4")
declare -a MODEL_NAMES=("qwen" "glm4")
declare -A TARGET_MODELS=(
    ["qwen"]="Qwen/Qwen3-8B"
    ["vicuna"]="lmsys/vicuna-7b-v1.3"
    ["glm4"]="zai-org/GLM-4-9B-0414"
)
declare -A DRAFT_MODELS=(
    ["qwen"]="Qwen/Qwen3-0.6B"
    ["vicuna"]="double7/vicuna-68m"
    ["glm4"]="jukofyork/GLM-4.5-DRAFT-0.6B-v3.0"
)

# Output directory
OUTPUT_DIR="benchmark_results/batch_vs_throughput"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Throughput vs Batch Size Benchmark"
echo "=========================================="
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Methods: ${METHODS[@]}"
echo "Model configs: ${MODEL_NAMES[@]}"
echo "Input: $INPUT_FILE"
echo ""

# Run benchmarks for each model configuration
for model_name in "${MODEL_NAMES[@]}"; do
    echo "######################################"
    echo "Running with model: $model_name"
    echo "Target: ${TARGET_MODELS[$model_name]}"
    echo "Draft: ${DRAFT_MODELS[$model_name]}"
    echo "######################################"

    # CSV file for results with model suffix
    RESULTS_CSV="$OUTPUT_DIR/throughput_table_${model_name}.csv"
    echo "Method,Batch_Size,Throughput_TPS,Pure_Decoding_Time,Total_Wall_Time,Success,Error" > $RESULTS_CSV
    echo "Output: $RESULTS_CSV"
    echo ""

    for batch_size in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------"
    echo "Running batch_size=$batch_size"
    echo "----------------------------------------"
    
    for method in "${METHODS[@]}"; do
        # Skip unsupported configurations
        if [[ "$method" == "SP-HF" && $batch_size -gt 1 ]]; then
            echo "  Skipping $method (only supports batch_size=1)"
            echo "$method,$batch_size,0,0,0,SKIPPED,None" >> $RESULTS_CSV
            continue
        fi
        
        # RD-HF only needs batch_size=1 (sequential processing)
        if [[ "$method" == "RD-HF" && $batch_size -gt 1 ]]; then
            echo "  Skipping $method (only run with batch_size=1)"
            echo "$method,$batch_size,0,0,0,SKIPPED,None" >> $RESULTS_CSV
            continue
        fi
        
        echo "  Running $method"

        # Build command based on method
        CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICE python unified_benchmark.py"
        CMD="$CMD --methods $method"
        CMD="$CMD --input_file $INPUT_FILE"
        CMD="$CMD --num_prompts $NUM_PROMPTS"
        CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
        CMD="$CMD --n_draft_tokens $N_DRAFT_TOKENS"
        CMD="$CMD --batch_size $batch_size"
        CMD="$CMD --max_input_len $MAX_INPUT_LEN"
        CMD="$CMD --target_model ${TARGET_MODELS[$model_name]}"
        CMD="$CMD --draft_model ${DRAFT_MODELS[$model_name]}"
        CMD="$CMD --output_dir $OUTPUT_DIR/${model_name}_${method}_b${batch_size}"
        
        # Add method-specific parameters
        if [[ "$method" == "Ours-XBatch" ]]; then
            CMD="$CMD --window_size $WINDOW_SIZE"
            CMD="$CMD --scheduling_strategy cross_batch"
            CMD="$CMD --sort_by_length"
        fi
        
        # Execute benchmark
        eval $CMD > "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}.log" 2>&1

        # Extract results from CSV output
        if [ -f "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" ]; then
            # Parse the CSV to extract metrics (based on fieldnames order in unified_benchmark.py)
            # Fields: method, batch_size, num_prompts, max_new_tokens, n_draft_tokens,
            #         pure_decoding_time(6), tokens_per_second(7), tar(8), latency(9),
            #         tokenization_time(10), post_processing_time(11), total_wall_time(12),
            #         draft_calls(13), verification_calls(14), success(15), error(16)
            throughput=$(tail -n 1 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" | cut -d',' -f7)
            pure_time=$(tail -n 1 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" | cut -d',' -f6)
            wall_time=$(tail -n 1 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" | cut -d',' -f12)
            success=$(tail -n 1 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" | cut -d',' -f15)
            error=$(tail -n 1 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}/benchmark_results.csv" | cut -d',' -f16)
            
            # Clean up error message (remove quotes and newlines)
            error=$(echo "$error" | tr '\n' ' ' | sed 's/"/\\"/g')
            
            echo "$method,$batch_size,$throughput,$pure_time,$wall_time,$success,\"$error\"" >> $RESULTS_CSV
            echo "    ✓ $method: ${throughput} tokens/s"
        else
            # Try to extract error from log file
            error_msg="No output file generated"
            if [ -f "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}.log" ]; then
                # Get last few lines that might contain error
                last_error=$(tail -n 20 "$OUTPUT_DIR/${model_name}_${method}_b${batch_size}.log" | grep -E "(Error|error|Exception|Traceback)" | tail -n 1)
                if [ ! -z "$last_error" ]; then
                    error_msg=$(echo "$last_error" | tr '\n' ' ' | sed 's/"/\\"/g' | cut -c1-200)
                fi
            fi
            
            echo "$method,$batch_size,0,0,0,FAILED,\"$error_msg\"" >> $RESULTS_CSV
            echo "    ✗ $method: FAILED"
        fi
    done  # End of method loop
    done  # End of batch_size loop

    echo ""
    echo "Completed benchmarks for $model_name"
    echo "Results saved to: $RESULTS_CSV"
    echo ""
done  # End of model loop

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to:"
for model_name in "${MODEL_NAMES[@]}"; do
    echo "  - $OUTPUT_DIR/throughput_table_${model_name}.csv"
done
echo ""
echo "Generating summary tables..."

# Run the aggregation script for each model
for model_name in "${MODEL_NAMES[@]}"; do
    echo "Processing results for $model_name..."
    # You may need to update batch_throughput_agg.py to accept model suffix
    python3 scripts/batch_throughput_agg.py --model $model_name 2>/dev/null || \
        echo "Note: batch_throughput_agg.py may need updating to handle model suffix"
done

echo ""
echo "To view detailed logs:"
echo "  ls $OUTPUT_DIR/*.log"