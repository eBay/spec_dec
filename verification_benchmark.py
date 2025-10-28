#!/usr/bin/env python3
"""
Output verification benchmark for speculative decoding methods.
Compares different implementations against ground truth (Non-Spec-1).
Produces LaTeX table showing sequence-level exact match and match until first false token.

python verification_benchmark.py --input_file data/spec_bench/question.jsonl --num_prompts 480 --models glm4 --batch_sizes 1 4 --max_new_tokens 50 --output_dir test_verification
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU devices to use
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import baseline and custom methods
from src.baselines.inference import (
    run_standard_inference,
    run_standard_batch_inference,
    run_hf_batch_1_spec_decoding
)
from src.baselines.bsp import run_adaptive_speculative_inference
from src.baselines.dsd import run_dsd_inference
from src.custom.batch_speculative import run_speculative_inference
from src.custom.cross_batch_speculative import run_cross_batch_inference
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
        "draft": "Qwen/Qwen3-0.6B",
        "name": "Qwen"
    },
    "vicuna": {
        "target": "lmsys/vicuna-7b-v1.3",
        "draft": "double7/vicuna-68m",
        "name": "Vicuna"
    },
    "glm4": {
        "target": "zai-org/GLM-4-9B-0414",
        "draft": "jukofyork/GLM-4.5-DRAFT-0.6B-v3.0",
        "name": "GLM4"
    },

}


def setup_models(model_pair: str, device: str = "cuda:0") -> Tuple[Any, Any, Any, torch.device]:
    """Load and setup models for a specific model pair."""
    config = MODEL_CONFIGS[model_pair]
    device_obj = torch.device(device)
    
    print(f"\nLoading {config['name']} models...")
    print(f"  Target: {config['target']}")
    print(f"  Draft: {config['draft']}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['target'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load target model
    target_model = AutoModelForCausalLM.from_pretrained(
        config['target'],
        torch_dtype=torch.float16,
        device_map=device_obj
    )
    
    # Load draft model
    draft_model = AutoModelForCausalLM.from_pretrained(
        config['draft'],
        torch_dtype=torch.float16,
        device_map=device_obj
    )
    
    return target_model, draft_model, tokenizer, device_obj


def load_prompts(input_file: str, num_prompts: Optional[int] = None) -> List[str]:
    """Load prompts from JSON file."""
    data = json_loader(input_file)
    
    # Extract prompts from the data structure
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
                # Try to use string representation
                prompts.append(str(item))
        else:
            prompts.append(str(item))
    
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    
    print(f"Loaded {len(prompts)} prompts from {input_file}")
    return prompts


def generate_ground_truth(
    prompts: List[str],
    model_pair: str,
    max_new_tokens: int,
    device: str,
    output_dir: str
) -> Dict[str, str]:
    """Generate ground truth using Non-Spec-1 (standard inference with batch_size=1)."""
    print(f"\nGenerating ground truth for {MODEL_CONFIGS[model_pair]['name']}...")
    
    # Setup models
    target_model, _, tokenizer, device_obj = setup_models(model_pair, device)
    
    # Run standard inference with batch_size=1
    print(f"Running standard inference on {len(prompts)} prompts...")
    outputs, time_taken, tokens_per_sec, _ = run_standard_inference(
        model=target_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        device=str(device_obj),
        temperature=0.0  # Ensure deterministic generation
    )
    
    print(f"  Completed in {time_taken:.2f}s ({tokens_per_sec:.2f} tokens/s)")
    
    # Create ground truth dictionary
    ground_truth = {
        "model_pair": model_pair,
        "model_config": MODEL_CONFIGS[model_pair],
        "num_prompts": len(prompts),
        "max_new_tokens": max_new_tokens,
        "generation_time": time_taken,
        "outputs": {prompt: output for prompt, output in zip(prompts, outputs)}
    }
    
    # Save ground truth
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ground_truth_file = output_path / f"ground_truth_{model_pair}.json"
    with open(ground_truth_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"  Saved ground truth to {ground_truth_file}")
    
    return ground_truth["outputs"]


def load_ground_truth(model_pair: str, output_dir: str) -> Optional[Dict[str, str]]:
    """Load previously generated ground truth if it exists."""
    ground_truth_file = Path(output_dir) / f"ground_truth_{model_pair}.json"
    
    if ground_truth_file.exists():
        print(f"Loading existing ground truth from {ground_truth_file}")
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
        return data["outputs"]
    
    return None


def compare_outputs(
    outputs: List[str],
    ground_truth: Dict[str, str],
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
    
    for prompt, output in zip(prompts, outputs):
        if prompt not in ground_truth:
            print(f"Warning: Prompt not in ground truth: {prompt[:50]}...")
            raise ValueError(f"Prompt not in ground truth: {prompt}")
        
        gt_output = ground_truth[prompt]
        
        # Check exact match
        if output == gt_output:
            exact_matches += 1
            partial_match_scores.append(1.0)  # Exact match is 100% partial match
        else:
            # Check partial match (up to first differing token)
            # Tokenize both outputs
            output_tokens = tokenizer.encode(output, add_special_tokens=False)
            gt_tokens = tokenizer.encode(gt_output, add_special_tokens=False)
            
            # Find first difference
            min_len = min(len(output_tokens), len(gt_tokens))
            matches_up_to = 0
            
            for i in range(min_len):
                if output_tokens[i] == gt_tokens[i]:
                    matches_up_to += 1
                else:
                    break
            
            # Calculate partial match score as fraction of ground truth tokens matched
            if len(gt_tokens) > 0:
                partial_score = matches_up_to / len(gt_tokens)
            else:
                partial_score = 0.0
            partial_match_scores.append(partial_score)
    
    # Calculate average partial match score
    avg_partial_score = sum(partial_match_scores) / len(partial_match_scores) if partial_match_scores else 0.0
    
    return exact_matches, avg_partial_score


def run_verification_method(
    method_name: str,
    prompts: List[str],
    model_pair: str,
    batch_size: int,
    max_new_tokens: int,
    n_draft_tokens: int,
    device: str
) -> List[str]:
    """Run a specific verification method and return outputs."""
    print(f"\n  Running {method_name} (batch_size={batch_size})...")
    
    # Setup models
    target_model, draft_model, tokenizer, device_obj = setup_models(model_pair, device)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run the appropriate method
    if method_name == "Non-Spec-Batch":
        outputs, _, _, _ = run_standard_batch_inference(
            model=target_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=str(device_obj),
            temperature=0.0
        )
    
    elif method_name == "Ours-Batch":
        result = run_speculative_inference(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            n_draft_tokens=n_draft_tokens,
            device=str(device_obj),
            use_cache=True
        )
        # Handle variable return formats
        outputs = result[0]
    
    elif method_name == "Ours-XBatch":
        result = run_cross_batch_inference(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            n_draft_tokens=n_draft_tokens,
            device=str(device_obj),
            use_cache=True,
            verbose_acceptance=False,
            enable_profiling=False
        )
        outputs = result[0]
    
    elif method_name == "DSD":
        result = run_dsd_inference(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            n_draft_tokens=n_draft_tokens,
            device=str(device_obj),
            batch_size=batch_size
        )
        outputs = result[0]
    
    elif method_name == "BSP":
        result = run_adaptive_speculative_inference(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            n_draft_tokens=n_draft_tokens,
            device=str(device_obj),
            batch_size=batch_size
        )
        outputs = result[0]

    elif method_name == "HF-Spec-1":
        outputs, _, _, _ = run_hf_batch_1_spec_decoding(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            batch_size=1,  # Always batch_size=1 for HF speculative decoding
            device=str(device_obj),
            temperature=0.0  # Deterministic generation
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    return outputs



def main():
    parser = argparse.ArgumentParser(description="Output verification benchmark for speculative decoding")
    
    # Data configuration
    parser.add_argument("--input_file", type=str, default="data/spec_bench/question.jsonl",
                        help="Path to input JSON file with prompts")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to use (None for all)")
    
    # Model configuration

    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()),
                        default=["qwen", "vicuna", "glm4"],
                        help="Model pairs to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum new tokens to generate")
    parser.add_argument("--n_draft_tokens", type=int, default=5,
                        help="Number of draft tokens for speculative methods")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4],
                        help="Batch sizes to test")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="verification_results",
                        help="Directory for output files")
    parser.add_argument("--regenerate_ground_truth", action="store_false",
                        help="Force regeneration of ground truth even if it exists")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OUTPUT VERIFICATION BENCHMARK")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Max new tokens: {args.max_new_tokens}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.input_file, args.num_prompts)
    
    # Store all results
    all_results = []
    
    # Process each model pair
    for model_pair in args.models:
        print(f"\n{'='*60}")
        print(f"Processing {MODEL_CONFIGS[model_pair]['name']} models")
        print(f"{'='*60}")
        
        # Get or generate ground truth
        ground_truth = None
        # if not args.regenerate_ground_truth:
        #     ground_truth = load_ground_truth(model_pair, args.output_dir)
        
        if ground_truth is None:
            ground_truth = generate_ground_truth(
                prompts, model_pair, args.max_new_tokens, 
                args.device, args.output_dir
            )
        
        # Setup tokenizer for comparison
        _, _, tokenizer, _ = setup_models(model_pair, args.device)
        
        # Define methods to test
        # methods_batch1 = ["HF-Spec-1", "Ours-Batch", "Ours-XBatch", "DSD", "BSP"]
        # methods_batch1 = ["Ours-Batch", "Ours-XBatch"]
        # methods_batch1 = ["DSD", "Non-Spec-Batch", "Ours-Batch", "Ours-XBatch", "BSP"]
        # methods_batch1 = ["DSD", "HF-Spec-1"]
        # methods_batch_n = ["Ours-XBatch", "DSD", "BSP"]  # Methods that support variable batch
        # methods_batch_n = ["Non-Spec-Batch", "DSD", "Ours-Batch", "Ours-XBatch", "BSP"]  # Methods that support variable batch
        # methods_batch_n = ["DSD", "Non-Spec-Batch"]  # Methods that support variable batch
        methods_batch_n = ["Ours-Batch", "Ours-XBatch"]
        
        # Test each batch size
        for batch_size in args.batch_sizes:
            print(f"\n--- Batch Size {batch_size} ---")
            
            # Determine which methods to test for this batch size
            if batch_size == 1:
                methods = methods_batch1
            else:
                methods = methods_batch_n
            
            for method in methods:
                try:
                    # Run the method
                    outputs = run_verification_method(
                        method, prompts, model_pair, batch_size,
                        args.max_new_tokens, args.n_draft_tokens, args.device
                    )
                    
                    # Compare outputs
                    exact_matches, partial_match_score = compare_outputs(
                        outputs, ground_truth, prompts, tokenizer
                    )
                    
                    # Store result
                    result = VerificationResult(
                        method=method,
                        model_pair=model_pair,
                        batch_size=batch_size,
                        exact_matches=exact_matches,
                        partial_match_score=partial_match_score,
                        total_prompts=len(prompts)
                    )
                    all_results.append(result)
                    
                    print(f"    {method}: {result.formatted_result} "
                          f"(exact: {result.exact_match_rate:.1%}, "
                          f"partial: {result.partial_match_score:.1%})")
                    
                except Exception as e:
                    print(f"    {method}: ERROR - {str(e)}")
                    # Add failed result
                    result = VerificationResult(
                        method=method,
                        model_pair=model_pair,
                        batch_size=batch_size,
                        exact_matches=0,
                        partial_match_score=0.0,
                        total_prompts=len(prompts)
                    )
                    all_results.append(result)
    
    # Generate output files
    print("\n" + "=" * 80)
    print("GENERATING OUTPUT FILES")
    print("=" * 80)
    
    # Save detailed results to JSON
    results_json = output_path / "verification_results.json"
    with open(results_json, 'w') as f:
        json.dump({
            "args": vars(args),
            "num_prompts": len(prompts),
            "results": [asdict(r) for r in all_results]
        }, f, indent=2)
    print(f"Saved detailed results to {results_json}")
    
    # Generate LaTeX table
    # latex_file = output_path / "verification_results.tex"
    # generate_latex_table(all_results, str(latex_file))
    
    # Generate summary
    summary_file = output_path / "verification_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("OUTPUT VERIFICATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for model_pair in args.models:
            f.write(f"\n{MODEL_CONFIGS[model_pair]['name']} Results:\n")
            f.write("-" * 40 + "\n")
            
            for batch_size in args.batch_sizes:
                f.write(f"\nBatch Size {batch_size}:\n")
                
                for result in all_results:
                    if result.model_pair == model_pair and result.batch_size == batch_size:
                        f.write(f"  {result.method:15s}: {result.formatted_result:10s} "
                               f"(exact: {result.exact_match_rate:6.1%}, "
                               f"partial: {result.partial_match_score:6.1%})\n")
    
    print(f"Saved summary to {summary_file}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - Ground truth: ground_truth_*.json")
    print(f"  - LaTeX table: verification_results.tex")
    print(f"  - Detailed results: verification_results.json")
    print(f"  - Summary: verification_summary.txt")


if __name__ == "__main__":
    main()