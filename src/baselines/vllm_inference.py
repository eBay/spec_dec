"""
vLLM-based inference implementations for benchmarking
"""

import time
from time import perf_counter
from typing import List, Tuple, NamedTuple

import torch
from vllm import LLM, SamplingParams

MAX_MODEL_LEN = 2048


class TimingBreakdown(NamedTuple):
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float

def run_vllm_standard_inference(model_name: str, prompts: List[str], 
                               max_new_tokens: int, temperature: float = 0.0, max_num_seqs: int = 128) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run standard inference using vLLM with automatic batching.
    
    Args:
        model_name: Name or path of the model
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling (0.0 for deterministic)
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    print(f"Running vLLM standard inference with model: {model_name}")
    
    # Time model initialization (considered as setup overhead, not pure decoding)
    torch.cuda.synchronize()
    setup_start = perf_counter()
    
    # Initialize the model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN, 
        # max_num_seqs=128,
        max_num_seqs=max_num_seqs,
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    
    torch.cuda.synchronize()
    setup_time = perf_counter() - setup_start
    
    # Time tokenization (vLLM handles this internally, but we can approximate as minimal)
    tokenization_time = 0.0  # vLLM does this internally, hard to separate
    
    # Time pure decoding
    torch.cuda.synchronize()
    pure_decoding_start = perf_counter()
    
    # Process all prompts at once (vLLM handles batching automatically)
    outputs = llm.generate(prompts, sampling_params)
    
    torch.cuda.synchronize()
    pure_decoding_time = perf_counter() - pure_decoding_start
    
    # Time post-processing
    torch.cuda.synchronize()
    post_processing_start = perf_counter()
    
    all_outputs = [output.outputs[0].text for output in outputs]
    
    torch.cuda.synchronize()
    post_processing_time = perf_counter() - post_processing_start
    
    # Calculate total tokens generated (more accurate using vLLM's token counts)
    total_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second_pure = total_tokens_generated / pure_decoding_time if pure_decoding_time > 0 else 0.0
    
    # Create timing breakdown
    total_time = tokenization_time + pure_decoding_time + post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=tokenization_time,
        pure_decoding_time=pure_decoding_time,
        post_processing_time=post_processing_time,
        total_time=total_time
    )
    
    return all_outputs, pure_decoding_time, tokens_per_second_pure, timing_breakdown


def run_vllm_speculative_inference(target_model: str, draft_model: str, 
                                  prompts: List[str], max_new_tokens: int, 
                                  n_draft_tokens: int, temperature: float = 0.0, max_num_seqs: int = 128) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run inference with speculative decoding using vLLM with automatic batching.
    
    Args:
        target_model: Name or path of the target model
        draft_model: Name or path of the draft model
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        n_draft_tokens: Number of tokens for the draft model to generate
        temperature: Temperature for sampling (0.0 for deterministic)
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    print(f"Running vLLM speculative inference with target: {target_model}, draft: {draft_model}")
    
    # Time model initialization (setup overhead)
    torch.cuda.synchronize()
    setup_start = perf_counter()
    
    # Initialize the model with speculative decoding
    llm = LLM(
        model=target_model,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN, 
        # max_num_seqs=128,
        max_num_seqs=max_num_seqs,
        speculative_config={
            "model": draft_model,
            "num_speculative_tokens": n_draft_tokens,
        }
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
    )
    
    torch.cuda.synchronize()
    setup_time = perf_counter() - setup_start
    
    # Time tokenization (vLLM handles this internally)
    tokenization_time = 0.0  # vLLM does this internally, hard to separate
    
    # Time pure decoding (speculative decoding)
    torch.cuda.synchronize()
    pure_decoding_start = perf_counter()
    
    # Process all prompts at once (vLLM handles batching automatically)
    outputs = llm.generate(prompts, sampling_params)
    
    torch.cuda.synchronize()
    pure_decoding_time = perf_counter() - pure_decoding_start
    
    # Time post-processing
    torch.cuda.synchronize()
    post_processing_start = perf_counter()
    
    all_outputs = [output.outputs[0].text for output in outputs]
    
    torch.cuda.synchronize()
    post_processing_time = perf_counter() - post_processing_start
    
    # Calculate total tokens generated (more accurate using vLLM's token counts)
    total_tokens_generated = sum(len(output.outputs[0].token_ids) for output in outputs)
    tokens_per_second_pure = total_tokens_generated / pure_decoding_time if pure_decoding_time > 0 else 0.0
    
    # Create timing breakdown
    total_time = tokenization_time + pure_decoding_time + post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=tokenization_time,
        pure_decoding_time=pure_decoding_time,
        post_processing_time=post_processing_time,
        total_time=total_time
    )
    
    return all_outputs, pure_decoding_time, tokens_per_second_pure, timing_breakdown