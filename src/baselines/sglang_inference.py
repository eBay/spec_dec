"""
SGLang-based inference implementations for benchmarking
"""
import os
os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

import time
from time import perf_counter
from typing import List, Tuple, NamedTuple

import torch
# from vllm import LLM, SamplingParams

MAX_MODEL_LEN = 2048




# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import nest_asyncio
import argparse

nest_asyncio.apply()
import sglang as sgl
from sglang.srt.server_args import ServerArgs
# https://github.com/sgl-project/sglang/issues/8230


class TimingBreakdown(NamedTuple):
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float

def run_sglang_standard_inference(model_name: str, prompts: List[str], 
                               max_new_tokens: int, tokenizer, temperature: float = 0.0, max_num_seqs: int = 64) -> Tuple[List[str], float, float, TimingBreakdown]:
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
    global torch
    print(f"Running vLLM standard inference with model: {model_name}")
    
    # Time model initialization (considered as setup overhead, not pure decoding)
    torch.cuda.synchronize()
    setup_start = perf_counter()
    
    llm = sgl.Engine(
        # model_path="lmsys/vicuna-7b-v1.3",
        model_path=model_name,
        skip_tokenizer_init=True,
        # model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # Speculative decoding (EAGLE-2/3 style):
        # speculative_algorithm="EAGLE3",
        # speculative_algorithm="EAGLE3",
        # speculative_draft_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",  # example draft
        # speculative_draft_model_path=draft_model,
        # speculative_draft_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        # speculative_draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        # speculative_num_steps=3,
        # speculative_eagle_topk=4,
        # speculative_num_draft_tokens=5,
        disable_cuda_graph=True,
        dtype="bfloat16",
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
        # Throughput polish (optional; tune per GPU):
        cuda_graph_max_bs=8,               # capture bigger decode batches
        mem_fraction_static=0.7,           # avoid OOM if tight on KV cache
        max_running_requests=max_num_seqs,
        trust_remote_code=True,  # Add this flag
    )
    # patch
    base = getattr(llm, "model", None) or getattr(llm, "_model", None)
    if base is not None and not hasattr(base, "hot_token_id"):
        # Prefer EOS as the stop/hot id
        eos = getattr(getattr(base, "config", None), "eos_token_id", None)
        if eos is not None:
            import torch
            base.hot_token_id = torch.as_tensor([eos], device="cuda" if torch.cuda.is_available() else "cpu")
            


    sampling = {"temperature": 0, "max_new_tokens": 128}
    
    torch.cuda.synchronize()
    setup_time = perf_counter() - setup_start
    
    # Time tokenization (vLLM handles this internally, but we can approximate as minimal)
    tokenization_time = 0.0  # vLLM does this internally, hard to separate
    
    # Time pure decoding
    torch.cuda.synchronize()
    pure_decoding_start = perf_counter()

    input_ids = tokenizer(prompts, return_tensors=None, padding=False, truncation=True).input_ids
    outputs = llm.generate(input_ids=input_ids, sampling_params=sampling)
    
    torch.cuda.synchronize()
    pure_decoding_time = perf_counter() - pure_decoding_start
    
    # Time post-processing
    torch.cuda.synchronize()
    post_processing_start = perf_counter()

    print(outputs[0])
    all_outputs = [tokenizer.decode(output['output_ids'], skip_special_tokens=False) for output in outputs]
    
    torch.cuda.synchronize()
    post_processing_time = perf_counter() - post_processing_start
    
    # Calculate total tokens generated (more accurate using vLLM's token counts)
    total_tokens_generated = sum(len(output['output_ids']) for output in outputs)
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


def run_sglang_speculative_inference(target_model: str, draft_model: str, 
                                  prompts: List[str], max_new_tokens: int, 
                                  n_draft_tokens: int, tokenizer, temperature: float = 0.0, max_num_seqs: int = 64) -> Tuple[List[str], float, float, TimingBreakdown]:
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
    global torch
    # Time model initialization (setup overhead)
    torch.cuda.synchronize()
    setup_start = perf_counter()
    
    # Initialize the model with speculative decoding
    # llm = LLM(
    #     model=target_model,
    #     tensor_parallel_size=torch.cuda.device_count(),
    #     gpu_memory_utilization=0.85,
    #     max_model_len=MAX_MODEL_LEN, 
    #     max_num_seqs=128,
    #     speculative_config={
    #         "model": draft_model,
    #         "num_speculative_tokens": n_draft_tokens,
    #     }
    # )
    
    # # Create sampling parameters
    # sampling_params = SamplingParams(
    #     max_tokens=max_new_tokens,
    #     temperature=temperature,
    #     top_p=0.95 if temperature > 0 else 1.0,
    # )
    llm = sgl.Engine(
        # model_path="lmsys/vicuna-7b-v1.3",
        model_path=target_model,
        skip_tokenizer_init=True,
        # model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # Speculative decoding (EAGLE-2/3 style):
        # speculative_algorithm="EAGLE3",
        speculative_algorithm="EAGLE",
        # speculative_draft_model_path="yuhuili/EAGLE-Vicuna-7B-v1.3",  # example draft
        speculative_draft_model_path=draft_model,
        # speculative_draft_model_path="yuhuili/EAGLE-LLaMA3-Instruct-8B",
        # speculative_draft_model_path="yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=5,
        disable_cuda_graph=True,
        dtype="bfloat16",
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
        #max_running_requests=64,
        # Throughput polish (optional; tune per GPU):
        cuda_graph_max_bs=8,               # capture bigger decode batches
        mem_fraction_static=0.7,           # avoid OOM if tight on KV cache
        max_running_requests=max_num_seqs,
        trust_remote_code=True,  # Add this flag
    )
    # patch
    base = getattr(llm, "model", None) or getattr(llm, "_model", None)
    if base is not None and not hasattr(base, "hot_token_id"):
        # Prefer EOS as the stop/hot id
        eos = getattr(getattr(base, "config", None), "eos_token_id", None)
        if eos is not None:
            import torch
            base.hot_token_id = torch.as_tensor([eos], device="cuda" if torch.cuda.is_available() else "cpu")
            


    sampling = {"temperature": 0, "max_new_tokens": 128}
    
    torch.cuda.synchronize()
    setup_time = perf_counter() - setup_start
    
    # Time tokenization (vLLM handles this internally)
    tokenization_time = 0.0  # vLLM does this internally, hard to separate
    
    # Time pure decoding (speculative decoding)
    torch.cuda.synchronize()
    pure_decoding_start = perf_counter()
    
    # Process all prompts at once (vLLM handles batching automatically)
    # outputs = llm.generate(prompts, sampling_params)
    input_ids = tokenizer(prompts, return_tensors=None, padding=False, truncation=True).input_ids
    outputs = llm.generate(input_ids=input_ids, sampling_params=sampling)

    
    torch.cuda.synchronize()
    pure_decoding_time = perf_counter() - pure_decoding_start
    
    # Time post-processing
    torch.cuda.synchronize()
    post_processing_start = perf_counter()
    
    all_outputs = [tokenizer.decode(output['output_ids'], skip_special_tokens=False) for output in outputs]
    
    torch.cuda.synchronize()
    post_processing_time = perf_counter() - post_processing_start
    
    # Calculate total tokens generated (more accurate using vLLM's token counts)
    total_tokens_generated = sum(len(output['output_ids']) for output in outputs)
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