#!/usr/bin/env python3
"""
Unified CLI benchmark script for speculative decoding experiments.
Based on test_spec_decode_correctness.py patterns with minimal invasive changes.
"""

import os
import argparse
import json
import time
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import all baseline methods
from src.baselines.inference import (
    run_standard_inference, 
    run_hf_batch_1_spec_decoding, 
    run_standard_batch_inference
)
from src.baselines.vllm_inference import (
    run_vllm_standard_inference,
    run_vllm_speculative_inference
)
from src.baselines.bsp import run_adaptive_speculative_inference
from src.baselines.dsd import run_dsd_inference
from src.custom.batch_speculative import run_speculative_inference
from src.custom.cross_batch_speculative import run_cross_batch_inference
from src.utils.data_loader import json_loader


def setup_models(target_model_name: str, draft_model_name: str, device: str):
    """
    Load and setup the models for benchmarking.
    Reused from test_spec_decode_correctness.py with minimal changes.
    """
    device_obj = torch.device(device)
    
    print(f"Loading target model: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map=device_obj
    )
    
    print(f"Loading draft model: {draft_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=torch.float16,
        device_map=device_obj
    )
    
    return target_model, draft_model, tokenizer, device_obj


def load_prompts(dataset: str, num_prompts: int, input_file: Optional[str] = None) -> List[str]:
    """Load prompts from various sources."""
    if input_file and Path(input_file).exists():
        # Load from JSON file
        data = json_loader(input_file)
        prompts = [item['turns'][0] if 'turns' in item and item['turns'] else str(item) for item in data]
        return prompts[:num_prompts]
    elif dataset == "test":
        # Test prompts from test_spec_decode_correctness.py + additional ones
        all_prompts = [
            "The weather today is",
            "Machine learning is a field of",
            "The capital of France is",
            "In the year 2024, technology will",
            "Artificial intelligence has revolutionized",
            "The stock market performance shows",
            "Climate change impacts include",
            "The history of human civilization",
            "Deep learning models require",
            "The future of computing involves",
            "Natural language processing enables",
            "Computer vision systems can",
            "Quantum computing promises to",
            "Data science has transformed",
            "Neural networks learn by",
            "Blockchain technology offers",
            "Cybersecurity measures protect",
            "Cloud computing provides",
            "Internet of Things connects",
            "Virtual reality creates",
            "Augmented reality enhances",
            "Robotics automation improves",
            "Renewable energy sources include",
            "Space exploration reveals",
            "Genetic engineering allows",
            "Medical breakthroughs help",
            "Environmental conservation requires",
            "Global warming affects",
            "Ocean pollution threatens",
            "Sustainable development promotes",
            "Education technology transforms",
            "Social media influences",
            "Economic policies impact",
            "Political systems vary",
            "Cultural diversity enriches",
            "Human psychology explains",
            "Scientific research advances",
            "Mathematical concepts underlie",
            "Historical events shape",
            "Philosophical questions explore"
        ]
        return all_prompts[:num_prompts]
    else:
        raise ValueError(f"Dataset '{dataset}' not supported. Use 'test' or provide --input_file")


# Method registry with unified signatures
BENCHMARK_METHODS = {
    "RD-HF": {
        "func": run_standard_inference,
        "requires_models": ["target"],
        "supports_batch_size": True,
        "supports_vllm": False
    },
    "SP-HF": {
        "func": run_hf_batch_1_spec_decoding, 
        "requires_models": ["target", "draft"],
        "supports_batch_size": False,  # HF spec decode only supports batch_size=1
        "supports_vllm": False
    },
    "RD-HF-Batch": {
        "func": run_standard_batch_inference,
        "requires_models": ["target"],
        "supports_batch_size": True,
        "supports_vllm": False
    },
    "RD-vLLM-Batch": {
        "func": run_vllm_standard_inference,
        "requires_models": ["target"],
        "supports_batch_size": False,  # vLLM handles batching automatically
        "supports_vllm": True
    },
    "SP-vLLM-Batch": {
        "func": run_vllm_speculative_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": False,  # vLLM handles batching automatically
        "supports_vllm": True
    },
    "Ours-Batch-Cache": {
        "func": run_speculative_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": True,
        "supports_vllm": False,
        "use_cache": True
    },
    "Ours-Batch-NoCache": {
        "func": run_speculative_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": True,
        "supports_vllm": False,
        "use_cache": False
    },
    "Ours-XBatch": {
        "func": run_cross_batch_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": True,
        "supports_vllm": False,
        "use_cache": True
    },
    "BSP-Adaptive": {
        "func": run_adaptive_speculative_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": True,  # BSP processes all prompts together
        "supports_vllm": False
    },
    "DSD": {
        "func": run_dsd_inference,
        "requires_models": ["target", "draft"],
        "supports_batch_size": True,
        "supports_vllm": False
    }
}


def run_benchmark_method(method_name: str, method_config: Dict, models: Dict, 
                        prompts: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Unified interface for running any benchmark method.
    Handles different method signatures automatically.
    """
    method_func = method_config["func"]
    
    # Prepare common arguments
    common_args = {
        "prompts": prompts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,  # Deterministic for fair comparison
        "max_input_len": args.max_input_len
    }
    
    # Method-specific argument handling
    if method_config["supports_vllm"]:
        # vLLM methods use model names, not loaded models
        if "target" in method_config["requires_models"] and "draft" not in method_config["requires_models"]:
            # Standard vLLM methods (only target model)
            common_args["model_name"] = args.target_model
        if "draft" in method_config["requires_models"]:
            # Speculative vLLM methods (both target and draft)
            common_args["target_model"] = args.target_model
            common_args["draft_model"] = args.draft_model
            common_args["n_draft_tokens"] = args.n_draft_tokens
    else:
        # HuggingFace methods use loaded models
        common_args["tokenizer"] = models["tokenizer"]
        common_args["device"] = str(models["device"])
        
        if "draft" in method_config["requires_models"]:
            # Methods that need both target and draft models
            common_args["target_model"] = models["target"]
            common_args["draft_model"] = models["draft"]
            if method_name.startswith("Ours-") or method_name == "BSP-Adaptive" or method_name == "DSD":
                if method_name.startswith("Ours-"):
                    common_args["n_draft_tokens"] = args.n_draft_tokens
                    # Add use_cache parameter for Ours variants
                    common_args["use_cache"] = method_config.get("use_cache", True)
                    # Add profiling parameter if enabled
                    common_args["enable_profiling"] = args.enable_profiling
                elif method_name == "DSD":
                    common_args["n_draft_tokens"] = args.n_draft_tokens
                # Remove temperature for Ours methods, BSP-Adaptive, and DSD as they don't support it
                del common_args["temperature"]
                
                # Add cross-batch specific parameters
                if method_name == "Ours-XBatch":
                    # Use CLI argument for batch size
                    common_args["batch_size"] = args.batch_size
                    common_args["scheduling_strategy"] = args.scheduling_strategy
                    common_args["sort_by_length"] = args.sort_by_length
                    common_args["window_size"] = args.window_size
                    common_args["verbose_acceptance"] = args.verbose
        else:
            # Methods that only need target model
            common_args["model"] = models["target"]
    
    # Add batch size only for methods that support it
    if method_config["supports_batch_size"]:
        common_args["batch_size"] = args.batch_size
    elif method_name == "SP-HF":
        # SP-HF needs batch_size=1 even though it doesn't "support" variable batch sizes
        common_args["batch_size"] = 1
    # Don't pass batch_size to vLLM methods or other methods that don't support it
    
    print(f"\nRunning {method_name}...")
    
    try:
        # Call the method with appropriate arguments
        result = method_func(**common_args)
        
        # Handle different return signatures
        if len(result) == 3:
            # DSD method: (outputs, total_time, tokens_per_second)
            outputs, total_time, tokens_per_second = result
            tar = None
            latency = None
            draft_calls = None
            verification_calls = None
            timing_breakdown = None
        elif len(result) == 4:
            # Baseline methods: (outputs, pure_decoding_time, tokens_per_second_pure, timing_breakdown)
            outputs, total_time, tokens_per_second, timing_breakdown = result
            tar = None
            latency = None
            draft_calls = None
            verification_calls = None
        elif len(result) == 6:
            # Custom methods (old format): (outputs, pure_decoding_time, tokens_per_second_pure, tar, latency, timing_breakdown)
            outputs, total_time, tokens_per_second, tar, latency, timing_breakdown = result
            draft_calls = None
            verification_calls = None
        elif len(result) == 8:
            # Custom methods (new format with call counts): (outputs, pure_decoding_time, tokens_per_second_pure, tar, latency, timing_breakdown, draft_calls, verification_calls)
            outputs, total_time, tokens_per_second, tar, latency, timing_breakdown, draft_calls, verification_calls = result
        else:
            raise ValueError(f"Unexpected return format from {method_name}: got {len(result)} values")
            
        # Extract timing breakdown details if available
        tokenization_time = timing_breakdown.tokenization_time if timing_breakdown else None
        post_processing_time = timing_breakdown.post_processing_time if timing_breakdown else None
        total_wall_time = timing_breakdown.total_time if timing_breakdown else total_time
            
        return {
            "method": method_name,
            "outputs": outputs,
            "pure_decoding_time": total_time,  # This is now pure decoding time
            "tokens_per_second": tokens_per_second,  # Now based on pure decoding
            "tar": tar,
            "latency": latency,
            "tokenization_time": tokenization_time,
            "post_processing_time": post_processing_time,
            "total_wall_time": total_wall_time,
            "num_prompts": len(prompts),
            "draft_calls": draft_calls,
            "verification_calls": verification_calls,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        print(f"Error running {method_name}: {e}")
        return {
            "method": method_name,
            "outputs": [],
            "pure_decoding_time": 0.0,
            "tokens_per_second": 0.0,
            "tar": None,
            "latency": None,
            "tokenization_time": None,
            "post_processing_time": None,
            "total_wall_time": 0.0,
            "num_prompts": len(prompts),
            "draft_calls": None,
            "verification_calls": None,
            "success": False,
            "error": str(e)
        }


def save_results(results: List[Dict], output_dir: str, args: argparse.Namespace):
    """Save benchmark results to CSV and JSON files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save to CSV for easy analysis
    csv_file = output_path / "benchmark_results.csv"
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            "method", "batch_size", "num_prompts", "max_new_tokens", "n_draft_tokens",
            "pure_decoding_time", "tokens_per_second", "tar", "latency", 
            "tokenization_time", "post_processing_time", "total_wall_time",
            "draft_calls", "verification_calls", "success", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                "method": result["method"],
                "batch_size": args.batch_size,
                "num_prompts": result["num_prompts"],
                "max_new_tokens": args.max_new_tokens,
                "n_draft_tokens": args.n_draft_tokens,
                "pure_decoding_time": result["pure_decoding_time"],
                "tokens_per_second": result["tokens_per_second"],
                "tar": result["tar"],
                "latency": result["latency"],
                "tokenization_time": result["tokenization_time"],
                "post_processing_time": result["post_processing_time"],
                "total_wall_time": result["total_wall_time"],
                "draft_calls": result.get("draft_calls", ""),
                "verification_calls": result.get("verification_calls", ""),
                "success": result["success"],
                "error": result["error"]
            })
    
    # Save detailed results to JSON
    json_file = output_path / "benchmark_results_detailed.json"
    with open(json_file, 'w') as f:
        # Remove outputs to keep file size manageable
        results_for_json = []
        for result in results:
            result_copy = result.copy()
            result_copy["outputs"] = result_copy["outputs"][:3]  # Keep only first 3 outputs as samples
            results_for_json.append(result_copy)
        
        json.dump({
            "args": vars(args),
            "results": results_for_json
        }, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")


def print_results_summary(results: List[Dict]):
    """Print a summary table of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    # Table header
    print(f"{'Method':<15} {'Pure Time(s)':<12} {'Tokens/s':<10} {'TAR':<6} {'Latency(ms)':<12} {'Draft':<8} {'Verify':<8} {'Status':<8}")
    print("-" * 100)
    
    last_status = "N/A"  # Default status if no results
    for result in results:
        tar_str = f"{result['tar']:.3f}" if result['tar'] is not None else "N/A"
        latency_str = f"{result['latency']:.1f}" if result['latency'] is not None else "N/A"
        status = "✓ OK" if result['success'] else "✗ FAIL"
        last_status = status
        
        draft_str = str(result.get('draft_calls', 'N/A'))
        verify_str = str(result.get('verification_calls', 'N/A'))
        print(f"{result['method']:<15} {result['pure_decoding_time']:<12.2f} "
              f"{result['tokens_per_second']:<10.1f} {tar_str:<6} {latency_str:<12} "
              f"{draft_str:<8} {verify_str:<8} {status:<8}")
        
        # Show timing breakdown if available
        if result.get('tokenization_time') is not None:
            tok_time = result['tokenization_time']
            proc_time = result['post_processing_time'] or 0.0
            wall_time = result['total_wall_time']
            overhead = ((wall_time - result['pure_decoding_time']) / wall_time * 100) if wall_time > 0 else 0
            print(f"{'':>15} └─ Timing: {tok_time:.2f}s tokenization, {proc_time:.2f}s post-proc, {overhead:.1f}% overhead")
    
    print("-" * 80)
    return last_status

def main():
    parser = argparse.ArgumentParser(description="Unified benchmark for speculative decoding methods")
    
    # Model configuration
    parser.add_argument("--target_model", type=str, default="lmsys/vicuna-7b-v1.3",
                        help="Target model name or path")
    parser.add_argument("--draft_model", type=str, default="double7/vicuna-68m",
                        help="Draft model name or path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    
    # Benchmark configuration
    parser.add_argument("--methods", nargs="+", choices=list(BENCHMARK_METHODS.keys()),
                        default=["RD-HF", "SP-HF", "Ours-Batch-Cache", "Ours-Batch-NoCache"],
                        help="Methods to benchmark")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for methods that support it")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--n_draft_tokens", type=int, default=5,
                        help="Number of draft tokens for speculative methods")
    
    # Cross-batch scheduling configuration
    parser.add_argument("--scheduling_strategy", type=str, choices=["cross_batch", "batch"], default="cross_batch",
                        help="Scheduling strategy for Ours-XBatch: 'cross_batch' (same-length grouping) or 'batch' (fallback mode)")
    parser.add_argument("--sort_by_length", action="store_true",
                        help="Sort prompts by token length in ascending order for better cross-batch grouping (Ours-XBatch)")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Maximum number of sequences to consider at once for cross-batch scheduling (prevents OOM)")
    
    # Data configuration
    parser.add_argument("--dataset", type=str, choices=["test"], default="test",
                        help="Dataset to use for benchmarking")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to input JSON file (overrides --dataset)")
    parser.add_argument("--num_prompts", type=int, default=6,
                        help="Number of prompts to benchmark")
    parser.add_argument("--max_input_len", type=int, default=1024,
                        help="Maximum input length in tokens (prompts will be truncated if longer)")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--enable_profiling", action="store_true",
                        help="Enable detailed profiling for supported methods")
    
    args = parser.parse_args()
    
    print("Unified Speculative Decoding Benchmark")
    print("=" * 50)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Requested prompts: {args.num_prompts}")
    
    # Load prompts
    prompts = load_prompts(args.dataset, args.num_prompts, args.input_file)
    print(f"Loaded prompts: {len(prompts)}")
    
    if len(prompts) < args.num_prompts:
        print(f"⚠️  Warning: Only {len(prompts)} prompts available in '{args.dataset}' dataset (requested {args.num_prompts})")
    
    # Setup models (only for non-vLLM methods)
    needs_hf_models = any(not BENCHMARK_METHODS[method]["supports_vllm"] for method in args.methods)
    if needs_hf_models:
        target_model, draft_model, tokenizer, device = setup_models(
            args.target_model, args.draft_model, args.device
        )
        models = {
            "target": target_model,
            "draft": draft_model,
            "tokenizer": tokenizer,
            "device": device
        }
    else:
        models = {}
    
    # Run benchmarks
    results = []
    for method_name in args.methods:
        method_config = BENCHMARK_METHODS[method_name]
        
        result = run_benchmark_method(method_name, method_config, models, prompts, args)
        results.append(result)
    
    # Print and save results
    print_results_summary(results)
    save_results(results, args.output_dir, args)
    
    print(f"\nBenchmark completed! Check {args.output_dir} for detailed results.")


if __name__ == "__main__":
    main()