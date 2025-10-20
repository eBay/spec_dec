#!/usr/bin/env python3
"""
SGLANG benchmark comparing standard inference vs speculative decoding.
Uses same evaluation metrics as verification_benchmark.py.
"""

import os

os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
import argparse
import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import sglang
#print(sglang.__version__)
import torch
from transformers import AutoTokenizer

from src.baselines.sglang_inference import (
    run_sglang_standard_inference,
    run_sglang_speculative_inference
)
from src.utils.data_loader import json_loader


@dataclass
class VerificationResult:
    """Result of comparing outputs against ground truth."""
    method: str
    model_pair: str
    batch_size: int
    exact_matches: int
    partial_match_score: float  # Average fraction of tokens matching up to first difference
    total_prompts: int

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total_prompts if self.total_prompts > 0 else 0.0

    @property
    def formatted_result(self) -> str:
        """Format as 'exact_count / partial_percentage' for table."""
        return f"{self.exact_matches} / {self.partial_match_score:.1%}"


MODEL_CONFIGS = {
    "qwen": {
         "target": "Qwen/Qwen3-8B",
         "draft": "Tengyunw/qwen3_8b_eagle3",
         "name": "Qwen"
     },
    "vicuna": {
        "target": "lmsys/vicuna-7b-v1.3",
        "draft": "yuhuili/EAGLE-Vicuna-7B-v1.3",
        "name": "Vicuna"
    },
    "glm4": {
         "target": "zai-org/GLM-4-9B-0414",
         "draft": "jukofyork/GLM-4.5-DRAFT-0.6B-v3.0",
         "name": "GLM4"
    }
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


def compare_outputs(
    outputs: List[str],
    ground_truth: List[str],
    prompts: List[str],
    tokenizer: Any
) -> Tuple[int, float]:
    """
    Compare outputs against ground truth.
    Returns (exact_matches, partial_match_score).
    partial_match_score is the average fraction of matching tokens up to first difference.
    """
    exact_matches = 0
    partial_match_scores = []

    for i, (output, gt_output, prompt) in enumerate(zip(outputs, ground_truth, prompts)):
        # Check exact match
        if output == gt_output:
            exact_matches += 1
            partial_match_scores.append(1.0)  # Exact match is 100% partial match
        else:
            # Check partial match (up to first differing token)
            # Tokenize both outputs
            output_tokens = tokenizer.encode(output, add_special_tokens=True)
            gt_tokens = tokenizer.encode(gt_output, add_special_tokens=True)

            # Find first difference
            min_len = min(len(output_tokens), len(gt_tokens))
            matches_up_to = 0

            for j in range(min_len):
                if output_tokens[j] == gt_tokens[j]:
                    matches_up_to += 1
                else:
                    break

            # Calculate partial match score as fraction of ground truth tokens matched
            if len(gt_tokens) > 0:
                partial_score = matches_up_to / len(gt_tokens)
            else:
                partial_score = 0.0
            if partial_score<=1:
                print("partial score: ", partial_score)
                print("prompt: ", prompt)
                print("Normal sequence: ", gt_output)
                print("EAGLE: ", output)
            partial_match_scores.append(partial_score)

    # Calculate average partial match score
    avg_partial_score = sum(partial_match_scores) / len(partial_match_scores) if partial_match_scores else 0.0

    return exact_matches, avg_partial_score


def run_sglang_benchmark(
    model_pair: str,
    prompts: List[str],
    max_new_tokens: int,
    n_draft_tokens: int,
    output_dir: str
) -> List[VerificationResult]:
    """Run SGLANG benchmark for a specific model pair."""

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

    results = []

    # Run SGLang WITHOUT speculative decoding (ground truth)
    print(f"\n1. Running SGLang standard inference (ground truth)...")
    outputs_standard, time_standard, tps_standard, _ = run_sglang_standard_inference(
        model_name=config['target'],
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        tokenizer=tokenizer
    )
    print(f"   Time: {time_standard:.2f}s ({tps_standard:.2f} tokens/sec)")

    # Run SGLang WITH speculative decoding
    print(f"\n2. Running SGLang with speculative decoding...")
    outputs_speculative, time_speculative, tps_speculative, _ = run_sglang_speculative_inference(
        target_model=config['target'],
        draft_model=config['draft'],
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        n_draft_tokens=n_draft_tokens,
        temperature=0.0,
        tokenizer=tokenizer
    )
    print(f"   Time: {time_speculative:.2f}s ({tps_speculative:.2f} tokens/sec)")

    # Compare outputs (speculative vs standard as ground truth)
    exact_matches, partial_match_score = compare_outputs(
        outputs_speculative,
        outputs_standard,
        prompts,
        tokenizer
    )

    # Create verification result for speculative decoding
    spec_result = VerificationResult(
        method="SGLang-Spec",
        model_pair=model_pair,
        batch_size=len(prompts),  # SGLANG handles batching internally
        exact_matches=exact_matches,
        partial_match_score=partial_match_score,
        total_prompts=len(prompts)
    )
    results.append(spec_result)

    # Create baseline result (standard SGLANG is the ground truth, so perfect match)
    standard_result = VerificationResult(
        method="SGLang-Standard",
        model_pair=model_pair,
        batch_size=len(prompts),
        exact_matches=len(prompts),
        partial_match_score=1.0,
        total_prompts=len(prompts)
    )
    results.append(standard_result)

    # Print comparison
    print(f"\n{'='*50}")
    print("CORRECTNESS COMPARISON")
    print(f"{'='*50}")
    print(f"SGLANG-Standard (baseline): {standard_result.formatted_result}")
    print(f"SGLANG-Speculative:        {spec_result.formatted_result}")

    # Performance comparison
    print(f"\n{'='*50}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*50}")
    print(f"SGLANG-Standard:    {time_standard:.2f}s ({tps_standard:.2f} tokens/sec)")
    print(f"SGLANG-Speculative: {time_speculative:.2f}s ({tps_speculative:.2f} tokens/sec)")

    if time_speculative < time_standard:
        speedup = time_standard / time_speculative
        print(f"✓ Speculative decoding is {speedup:.2f}x faster")
    else:
        slowdown = time_speculative / time_standard
        print(f"⚠ Speculative decoding is {slowdown:.2f}x slower")

    # Save outputs for analysis
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs_file = output_path / f"sglang_outputs_{model_pair}.json"
    with open(outputs_file, 'w') as f:
        json.dump({
            "model_config": config,
            "num_prompts": len(prompts),
            "max_new_tokens": max_new_tokens,
            "n_draft_tokens": n_draft_tokens,
            "outputs": {
                "prompts": prompts,
                "standard": outputs_standard,
                "speculative": outputs_speculative
            },
            "performance": {
                "standard": {"time": time_standard, "tokens_per_sec": tps_standard},
                "speculative": {"time": time_speculative, "tokens_per_sec": tps_speculative},
                "speedup": time_standard / time_speculative if time_speculative > 0 else 0
            },
            "correctness": {
                "exact_matches": exact_matches,
                "partial_match_score": partial_match_score,
                "total_prompts": len(prompts)
            }
        }, f, indent=2)

    print(f"\nOutputs saved to: {outputs_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SGLang verification benchmark")

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
    parser.add_argument("--output_dir", type=str, default="sglang_verification_results",
                        help="Directory for output files")

    args = parser.parse_args()

    print("=" * 80)
    print("SGLANG VERIFICATION BENCHMARK")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Draft tokens: {args.n_draft_tokens}")

    # Load prompts
    prompts = load_prompts(args.input_file, args.num_prompts)
    #prompts = [
        #"The weather today is",
        #"Machine learning is a field of",
        #"The capital of France is",
        #"In the year 2024, technology will",
        #"Artificial intelligence has revolutionized",
        #"The stock market performance shows",
        #"Climate change impacts include",
        #"The history of human civilization"
    #]

    # Store all results
    all_results = []

    # Process each model pair
    for model_pair in args.models:
        results = run_sglang_benchmark(
            model_pair=model_pair,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            n_draft_tokens=args.n_draft_tokens,
            output_dir=args.output_dir
        )
        all_results.extend(results)

    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_pair in args.models:
        model_name = MODEL_CONFIGS[model_pair]['name']
        print(f"\n{model_name} Results:")
        print("-" * 40)

        for result in all_results:
            if result.model_pair == model_pair:
                print(f"  {result.method:15s}: {result.formatted_result:10s} "
                      f"(exact: {result.exact_match_rate:6.1%}, "
                      f"partial: {result.partial_match_score:6.1%})")

    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "sglang_verification_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "num_prompts": len(prompts),
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)

    print(f"\nDetailed results saved to: {results_file}")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()