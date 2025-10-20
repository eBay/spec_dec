import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set GPU device

import argparse
import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
from transformers import AutoTokenizer

# Import vLLM inference functions
from src.baselines.vllm_inference import (
    run_vllm_standard_inference,
    run_vllm_speculative_inference
)
from src.utils.data_loader import json_loader


MODEL_CONFIGS = {

    "vicuna": {
        "target": "lmsys/vicuna-7b-v1.3",
        "draft": "double7/vicuna-68m",
        "name": "Vicuna"
    },

}

def load_prompts(input_file: str, num_prompts: Optional[int] = None) -> List[str]:
    """Load prompts from JSON file."""
    data = json_loader(input_file)

    prompts = []
    for item in data:
        if isinstance(item, dict):
            if 'turns' in item and item['turns']:
                # Use first turn as the prompt
                prompts.append(item['turns'][0])
            elif 'question' in item:
                prompts.append(item['question'])
            elif 'prompt' in item:
                prompts.append(item['prompt'])
        else:
            prompts.append(str(item))

    if num_prompts is not None:
        prompts = prompts[:num_prompts]

    print(f"Loaded {len(prompts)} prompts from {input_file}")
    return prompts

def run_vllm_scaling(
    model_pair: str,
    prompts: List[str],
    max_new_tokens: int,
    n_draft_tokens: int,
    output_dir: str,
    max_num_seqs: int
) -> None:
    """Run vLLM benchmark for a specific model pair."""

    config = MODEL_CONFIGS[model_pair]
    print(f"\n{'='*60}")
    print(f"Processing {config['name']} models")
    print(f"{'='*60}")
    print(f"Target: {config['target']}")
    print(f"Draft: {config['draft']}")

    # Load tokenizer for comparison
    tokenizer = AutoTokenizer.from_pretrained(config['target'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run vLLM WITHOUT speculative decoding (ground truth)
    print(f"\n1. Running vLLM standard inference (ground truth) with max_num_seqs={max_num_seqs}...")
    outputs_standard, time_standard, tps_standard, _ = run_vllm_standard_inference(
        model_name=config['target'],
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        max_num_seqs=max_num_seqs
    )
    print(f"   Time: {time_standard:.2f}s ({tps_standard:.2f} tokens/sec)")

    # Run vLLM WITH speculative decoding
    print(f"\n2. Running vLLM with speculative decoding with max_num_seqs={max_num_seqs}...")
    outputs_speculative, time_speculative, tps_speculative, _ = run_vllm_speculative_inference(
        target_model=config['target'],
        draft_model=config['draft'],
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        n_draft_tokens=n_draft_tokens,
        temperature=0.0,
        max_num_seqs=max_num_seqs
    )
    print(f"   Time: {time_speculative:.2f}s ({tps_speculative:.2f} tokens/sec)")

    # Store results
    results = {
        'model_pair': model_pair,
        'model_name': config['name'],
        'max_num_seqs': max_num_seqs,
        'standard': {
            'time': time_standard,
            'tokens_per_sec': tps_standard
        },
        'speculative': {
            'time': time_speculative,
            'tokens_per_sec': tps_speculative
        },
        'speedup': time_standard / time_speculative if time_speculative > 0 else 0
    }



    # Print comparison
    print(f"\n{'='*50}")
    print("CORRECTNESS COMPARISON")
    print(f"{'='*50}")

    # Performance comparison
    print(f"\n{'='*50}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*50}")
    print(f"vLLM-Standard:    {time_standard:.2f}s ({tps_standard:.2f} tokens/sec)")
    print(f"vLLM-Speculative: {time_speculative:.2f}s ({tps_speculative:.2f} tokens/sec)")

    if time_speculative < time_standard:
        speedup = time_standard / time_speculative
        print(f"✓ Speculative decoding is {speedup:.2f}x faster")
    else:
        slowdown = time_speculative / time_standard
        print(f"⚠ Speculative decoding is {slowdown:.2f}x slower")


    return results

def main():
    parser = argparse.ArgumentParser(description="vLLM verification benchmark")

    # Data configuration
    parser.add_argument("--input_file", type=str, default="data/spec_bench/question.jsonl",
                        help="Path to input JSON file with prompts")
    parser.add_argument("--num_prompts", type=int, default=480,
                        help="Number of prompts to use (None for all)")

    # Model configuration
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                        # default=["vicuna", "qwen", "glm4"],
                        default=["vicuna"],
                        help="Model pairs to evaluate")
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--n_draft_tokens", type=int, default=5,
                        help="Number of draft tokens for speculative decoding")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="vllm_scaling_results",
                        help="Directory for output files")

    args = parser.parse_args()

    print("=" * 80)
    print("vLLM Scaling BENCHMARK")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Draft tokens: {args.n_draft_tokens}")

    # Load prompts
    prompts = load_prompts(args.input_file, args.num_prompts)

    # Store all results
    all_results = []

    # Process each model pair
    for model_pair in args.models:
        for max_num_seqs in [1, 2, 4, 8, 16, 32]:
            result = run_vllm_scaling(
                model_pair=model_pair,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                n_draft_tokens=args.n_draft_tokens,
                output_dir=args.output_dir,
                max_num_seqs=max_num_seqs
            )
            all_results.append(result)

    # Generate summary
    print("\n" + "=" * 80)
    print("SCALING SUMMARY")
    print("=" * 80)

    for model_pair in args.models:
        model_name = MODEL_CONFIGS[model_pair]['name']
        print(f"\n{model_name} Results:")
        print("-" * 60)
        print(f"{'Batch Size':<12} {'Standard (s)':<15} {'Speculative (s)':<18} {'Speedup':<10}")
        print("-" * 60)

        for result in all_results:
            if result['model_pair'] == model_pair:
                print(f"{result['max_num_seqs']:<12} "
                      f"{result['standard']['time']:<15.2f} "
                      f"{result['speculative']['time']:<18.2f} "
                      f"{result['speedup']:<10.2f}x")

        print("\nTokens per second:")
        print(f"{'Batch Size':<12} {'Standard':<15} {'Speculative':<18} {'Improvement':<10}")
        print("-" * 60)
        for result in all_results:
            if result['model_pair'] == model_pair:
                improvement = ((result['speculative']['tokens_per_sec'] - result['standard']['tokens_per_sec'])
                              / result['standard']['tokens_per_sec'] * 100)
                print(f"{result['max_num_seqs']:<12} "
                      f"{result['standard']['tokens_per_sec']:<15.2f} "
                      f"{result['speculative']['tokens_per_sec']:<18.2f} "
                      f"{improvement:+.1f}%")

    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "vllm_scaling_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "num_prompts": len(prompts),
            "results": all_results
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("SCALING BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()