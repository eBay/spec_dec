"""
Baseline inference implementations including standard and HuggingFace speculative decoding
"""

import time
from time import perf_counter
from typing import List, Dict, Any, Tuple, NamedTuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class TimingBreakdown(NamedTuple):
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float


def run_standard_inference(model, tokenizer, prompts: List[str], 
                          max_new_tokens: int, batch_size: int, device: str, temperature: float = 0.7, max_input_len: int = 1024) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run standard batch inference using HuggingFace Transformers.
    
    Args:
        model: The target model
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    print(f"Running standard inference with batch size: {batch_size}")
    
    all_outputs = []
    total_tokens_generated = 0
    
    # Timing accumulators
    total_tokenization_time = 0.0
    total_pure_decoding_time = 0.0
    total_post_processing_time = 0.0
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Standard Inference Batches"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Time tokenization (without padding)
        torch.cuda.synchronize()
        tokenization_start = perf_counter()
        # batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=False)
        batch_inputs = tokenizer(batch_prompts, return_tensors=None, padding=False, truncation=True, max_length=max_input_len)
        # batch_inputs = [torch.as_tensor(x, dtype=torch.long) for x in batch_inputs["input_ids"]]  # <<< key change
        torch.cuda.synchronize()
        batch_tokenization_time = perf_counter() - tokenization_start
        total_tokenization_time += batch_tokenization_time
        
        # Pad sequences and move to device (counted as part of decoding)
        pure_decoding_start = perf_counter()

        batch_inputs = tokenizer.pad(batch_inputs, padding_side='left', padding=True, return_tensors="pt").to(device)
        batch_inputs.attention_mask = (batch_inputs.input_ids != tokenizer.pad_token_id).long().to(device)
        input_length = batch_inputs.input_ids.shape[1]
        
        # Time pure decoding
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=0.95,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            
        torch.cuda.synchronize()  # Ensure all GPU operations complete before measuring time
        batch_pure_decoding_time = perf_counter() - pure_decoding_start
        total_pure_decoding_time += batch_pure_decoding_time
        
        # Count tokens generated (stop at EOS)
        for j in range(len(outputs)):
            generated_portion = outputs[j][input_length:]
            # Find EOS token if present
            eos_positions = (generated_portion == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                # Count tokens up to and including first EOS
                tokens_count = eos_positions[0].item() + 1
            else:
                # Count all generated tokens if no EOS
                tokens_count = len(generated_portion)
            total_tokens_generated += tokens_count
        
        # Time post-processing (decoding)
        torch.cuda.synchronize()
        post_processing_start = perf_counter()
        decoded_outputs = [tokenizer.decode(output[input_length:], skip_special_tokens=False) 
                          for output in outputs]
        torch.cuda.synchronize()
        batch_post_processing_time = perf_counter() - post_processing_start
        total_post_processing_time += batch_post_processing_time
        
        all_outputs.extend(decoded_outputs)
    
    # Calculate metrics based on pure decoding time
    print(f"TPS breakdown: {total_pure_decoding_time} / {total_tokens_generated} = {total_tokens_generated / total_pure_decoding_time}")
    
    tokens_per_second_pure = total_tokens_generated / total_pure_decoding_time if total_pure_decoding_time > 0 else 0.0
    
    # Create timing breakdown
    total_time = total_tokenization_time + total_pure_decoding_time + total_post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=total_tokenization_time,
        pure_decoding_time=total_pure_decoding_time,
        post_processing_time=total_post_processing_time,
        total_time=total_time
    )
    
    return all_outputs, total_pure_decoding_time, tokens_per_second_pure, timing_breakdown


def run_standard_batch_inference(model, tokenizer, prompts: List[str], 
                          max_new_tokens: int, batch_size: int, device: str, temperature: float = 0.0, max_input_len: int = 1024) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run standard batch inference with specified batch size.
    This is a wrapper around run_standard_inference with explicit batch processing.
    
    Args:
        model: The target model
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference
        device: Device to run inference on
        temperature: Temperature for sampling (0.0 for deterministic)
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    return run_standard_inference(model, tokenizer, prompts, max_new_tokens, batch_size, device, temperature, max_input_len)


def run_hf_batch_1_spec_decoding(target_model, draft_model, tokenizer, prompts: List[str], 
                                  max_new_tokens: int, batch_size: int, device: str, temperature: float = 0.7, max_input_len: int = 1024) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run HuggingFace's built-in speculative decoding with batch size 1.
    
    This function uses HuggingFace Transformers' native speculative decoding implementation
    by passing the draft model as the assistant_model parameter to generate().
    
    Args:
        target_model: The target (larger) model
        draft_model: The draft (smaller) model used for speculation
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference (must be 1)
        device: Device to run inference on
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    # Assert batch size is 1 as HF speculative decoding only supports batch size 1
    assert batch_size == 1, "HuggingFace speculative decoding only supports batch_size=1"
    
    print(f"Running HuggingFace native speculative decoding with batch size: {batch_size}")
    
    all_outputs = []
    total_tokens_generated = 0
    
    # Timing accumulators
    total_tokenization_time = 0.0
    total_pure_decoding_time = 0.0
    total_post_processing_time = 0.0
    
    # Process prompts one by one (batch size = 1)
    for prompt in tqdm(prompts, desc="HF Speculative Decoding"):
        # Time tokenization (without padding)
        torch.cuda.synchronize()
        tokenization_start = perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_input_len)

        torch.cuda.synchronize()
        prompt_tokenization_time = perf_counter() - tokenization_start
        total_tokenization_time += prompt_tokenization_time
        
        # Move to device (counted as part of decoding since batch_size=1 doesn't need padding)
        inputs = inputs.to(device)
        input_length = inputs.input_ids.shape[1]
        
        # Time pure decoding (speculative decoding)
        with torch.no_grad():
            torch.cuda.synchronize()
            pure_decoding_start = perf_counter()
            
            outputs = target_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                # temperature=temperature,
                # top_p=0.95,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                assistant_model=draft_model,  # This enables speculative decoding
                # Optional: can adjust num_assistant_tokens (default is 5)
                # num_assistant_tokens=5,
            )
            
            torch.cuda.synchronize()
            prompt_pure_decoding_time = perf_counter() - pure_decoding_start
            total_pure_decoding_time += prompt_pure_decoding_time
        
        # Count tokens generated (stop at EOS)
        generated_portion = outputs[0, input_length:]
        # Find EOS token if present
        eos_positions = (generated_portion == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # Count tokens up to and including first EOS
            tokens_generated_for_prompt = eos_positions[0].item() + 1
        else:
            # Count all generated tokens if no EOS
            tokens_generated_for_prompt = len(generated_portion)
        total_tokens_generated += tokens_generated_for_prompt
        
        # Time post-processing (decoding)
        torch.cuda.synchronize()
        post_processing_start = perf_counter()
        generated_text = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=False)
        torch.cuda.synchronize()
        prompt_post_processing_time = perf_counter() - post_processing_start
        total_post_processing_time += prompt_post_processing_time
        
        all_outputs.append(generated_text)
    
    # Calculate metrics based on pure decoding time
    tokens_per_second_pure = total_tokens_generated / total_pure_decoding_time if total_pure_decoding_time > 0 else 0.0
    
    # Create timing breakdown
    total_time = total_tokenization_time + total_pure_decoding_time + total_post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=total_tokenization_time,
        pure_decoding_time=total_pure_decoding_time,
        post_processing_time=total_post_processing_time,
        total_time=total_time
    )
    
    print(f"\nHF Speculative Decoding completed:")
    print(f"  - Processed {len(prompts)} prompts")
    print(f"  - Total tokens generated: {total_tokens_generated}")
    print(f"  - Pure decoding time: {total_pure_decoding_time:.2f} seconds")
    print(f"  - Tokens per second (pure): {tokens_per_second_pure:.2f}")
    print(f"  - Timing breakdown: {timing_breakdown.tokenization_time:.2f}s tokenization, {timing_breakdown.pure_decoding_time:.2f}s decoding, {timing_breakdown.post_processing_time:.2f}s post-processing")
    
    return all_outputs, total_pure_decoding_time, tokens_per_second_pure, timing_breakdown