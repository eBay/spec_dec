# TODO: 
# KV-cache
# acceptance rate 


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # Set the GPU devices to use
import argparse
import json
import time
from time import perf_counter
from typing import List, Dict, Any, Tuple, NamedTuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from tqdm import tqdm
# from pytorch_memlab import MemReporter

# def debug_oom():


class TimingBreakdown(NamedTuple):
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float




def batch_oracle_verification(model, input_tensors, draft_tokens_tensors, past_key_values, device):
    """
    Verifies the predictions of the draft model against the oracle (target) model for a batch of sequences.
    
    Args:
        model: The target/oracle model
        input_tensors: Tensor of input token sequences [batch_size, input_seq_len]
        draft_tokens_tensors: Tensor of draft generated tokens [batch_size, draft_seq_len]
        device: Device to run verification on
        
    Returns:
        Tuple of first false positions for each sequence in batch and accepted draft tokens
    # Process each sequence in the batch
    """
    batch_size = input_tensors.shape[0]
    

    # Concatenate all sequences with their draft tokens at once
    max_input_len = input_tensors.shape[1]
    combined_tokens = torch.cat([input_tensors, draft_tokens_tensors], dim=1)
    
    # Single forward pass for all sequences
    with torch.no_grad():
        # outputs = model(combined_tokens, past_key_values=past_key_values, use_cache=True)
        outputs = model(combined_tokens, use_cache=True)
        # outputs = model(draft_tokens_tensors, past_key_values=past_key_values, use_cache=True)
        # Extract logits for positions after the input
        next_token_logits = outputs.logits[:, max_input_len-1:-1, :]
        # next_token_logits = outputs.logits[:, max_input_len-1:, :]
        next_key_values = outputs.past_key_values

    
    # Get predictions for all sequences at once
    predicted_tokens = torch.argmax(next_token_logits, dim=-1)
    
    # Compare with draft tokens in parallel
    matches = (predicted_tokens == draft_tokens_tensors)
    
    # Find first mismatch for each sequence
    first_false_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    for i in range(batch_size):
        mismatches = torch.where(~matches[i])[0]
        if len(mismatches) > 0:
            # In their original implementation, the add 1 more token from the draft speculation
            first_false_positions[i] = mismatches[0].item() + 1
        else:
            first_false_positions[i] = matches.shape[1]
    
    accepted_tokens_list = [draft_tokens_tensors[i][:first_false_positions[i]] for i in range(batch_size)]
    
    # Truncate KV cache to only keep values for accepted tokens
    min_accepted = first_false_positions.min().item()
    if min_accepted < draft_tokens_tensors.shape[1]:
        # Truncate KV cache: remove entries for rejected tokens
        truncated_kv = []
        for layer_kv in next_key_values:
            # Each layer_kv is a tuple of (key, value) tensors
            # Shape is typically [batch_size, num_heads, seq_len, head_dim]
            key, value = layer_kv
            # Keep only up to input_len + min_accepted tokens
            truncated_key = key[:, :, :-(draft_tokens_tensors.shape[1] - min_accepted), :]
            truncated_value = value[:, :, :-(draft_tokens_tensors.shape[1] - min_accepted), :]
            truncated_kv.append((truncated_key, truncated_value))
        next_key_values = tuple(truncated_kv)
        next_key_values = DynamicCache.from_legacy_cache(next_key_values)
    
    return first_false_positions, accepted_tokens_list, next_key_values


def run_dsd_inference(target_model, draft_model, tokenizer, prompts: List[str],
                             max_new_tokens: int, batch_size: int,
                             n_draft_tokens: int, device: str, max_input_len: int = 1024) -> Tuple[List[str], float, float, TimingBreakdown]:
    """
    Run batch inference with speculative decoding, using proper batching and left padding.

    Args:
        target_model: Target (oracle) model
        draft_model: Draft (smaller) model
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        batch_size: Batch size for inference
        n_draft_tokens: Number of tokens for draft model to generate at once
        device: Device to run inference on
        max_input_len: Maximum input length for truncation

    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, timing breakdown)
    """
    print(f"Running speculative inference with batch size: {batch_size}, draft tokens: {n_draft_tokens}")

    # max_new_tokens = 64

    all_outputs = []
    total_tokens_generated = 0

    # Timing accumulators
    total_tokenization_time = 0.0
    total_pure_decoding_time = 0.0
    total_post_processing_time = 0.0
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Speculative Inference Batches"):
        # Get a batch of prompts
        batch_prompts = prompts[i:i+batch_size]
        actual_batch_size = len(batch_prompts)

        # Time tokenization
        torch.cuda.synchronize()
        tokenization_start = perf_counter()
        encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                                  truncation=True, max_length=max_input_len)
        torch.cuda.synchronize()
        batch_tokenization_time = perf_counter() - tokenization_start
        total_tokenization_time += batch_tokenization_time

        # Time pure decoding (includes moving to device and all generation)
        torch.cuda.synchronize()
        pure_decoding_start = perf_counter()

        input_ids = encoded_inputs.input_ids.to(device)
        attention_mask = encoded_inputs.attention_mask.to(device)
        
        # Track input lengths to extract outputs later
        input_lengths = attention_mask.sum(dim=1).tolist()
        
        # Initialize generated sequences with the input
        generated_ids = input_ids
        
        # Initialize token counters
        tokens_generated = torch.zeros(actual_batch_size, dtype=torch.int).to(device)
        past_key_values = None

        # Simple loop: generate until max_new_tokens for all sequences
        while tokens_generated.min() < max_new_tokens:
            # Calculate remaining tokens needed
            remaining_tokens = max_new_tokens - tokens_generated
            max_draft_this_iter = min(n_draft_tokens, remaining_tokens.min().item())

            if max_draft_this_iter <= 0:
                break

            # Generate draft tokens with the draft model
            draft_outputs = draft_model.generate(
                input_ids=generated_ids,
                # attention_mask=torch.ones_like(generated_ids).to(device),
                max_new_tokens=max_draft_this_iter,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            
            # Extract draft tokens (excluding input)
            draft_tokens = []
            for seq_idx in range(actual_batch_size):
                seq_len = generated_ids[seq_idx].shape[0]
                seq_draft = draft_outputs.sequences[seq_idx][seq_len:]
                draft_tokens.append(seq_draft)
            
            # Create a tensor of draft tokens
            max_draft_len = max(t.shape[0] for t in draft_tokens)
            if max_draft_len == 0:  # No draft tokens generated
                break
            
            # Pad draft tokens to same length
            padded_draft_tokens = []
            for seq_draft in draft_tokens:
                if len(seq_draft) < max_draft_len:
                    padding = torch.full((max_draft_len - len(seq_draft),), 
                                        tokenizer.pad_token_id, 
                                        device=device)
                    seq_draft = torch.cat([seq_draft, padding])
                padded_draft_tokens.append(seq_draft)
            
            draft_tokens_tensor = torch.stack(padded_draft_tokens)
            
            # Verify draft tokens against target model predictions
            first_false_positions, accepted_tokens, past_key_values = batch_oracle_verification(
                target_model, generated_ids, draft_tokens_tensor, past_key_values, device
            )
            
            # Process accepted tokens - this is the key part from spec_decoding_deployment.py
            matched_tokens = first_false_positions
            
            # Create new input tensors with left padding to align sequences
            input_list = []
            max_matched = torch.max(matched_tokens)
            
            for seq_idx, matched in enumerate(matched_tokens):
                # Add left padding to ensure all sequences have the same length
                padding_tokens = torch.full((max_matched - matched,), 
                                           tokenizer.pad_token_id, 
                                           device=device)
                
                # Concatenate: padding + original input + accepted tokens
                new_seq = torch.cat([
                    padding_tokens,
                    generated_ids[seq_idx],
                    accepted_tokens[seq_idx]
                ])
                
                input_list.append(new_seq)
            
            # Stack the new inputs to form the new batch
            generated_ids = torch.stack(input_list)
            
            # Update token counters
            tokens_generated += matched_tokens

            
            # Update attention mask for new sequence lengths
            attention_mask = torch.ones_like(generated_ids)

        # End of pure decoding
        torch.cuda.synchronize()
        batch_pure_decoding_time = perf_counter() - pure_decoding_start
        total_pure_decoding_time += batch_pure_decoding_time

        # reporter = MemReporter()
        # reporter.report()
        # exit()

        # Time post-processing (decoding)
        torch.cuda.synchronize()
        post_processing_start = perf_counter()

        # Extract and decode the generated outputs (excluding input)
        for seq_idx in range(actual_batch_size):
            # Handle left padding when extracting original input length
            # Count non-padding tokens from the beginning
            pad_token_id = tokenizer.pad_token_id
            seq = generated_ids[seq_idx]
            pad_mask = seq == pad_token_id
            
            # Find first non-padding token
            non_pad_mask = ~pad_mask  # Invert to get non-padding positions
            if non_pad_mask.any():
                first_non_pad = non_pad_mask.nonzero()[0].item()
            else:
                first_non_pad = 0
                
            # Extract original input length (may include padding)
            orig_input_len = input_lengths[seq_idx]
            
            # The start index should be: first_non_pad + orig_input_len
            start_idx = first_non_pad + orig_input_len
            
            # Extract generated tokens (excluding input and left padding)
            generated_seq = seq[start_idx:]
            
            # Count tokens generated for this prompt
            total_tokens_generated += len(generated_seq)
            
            # Decode output
            output_text = tokenizer.decode(generated_seq, skip_special_tokens=False)
            all_outputs.append(output_text)

        # End post-processing
        torch.cuda.synchronize()
        batch_post_processing_time = perf_counter() - post_processing_start
        total_post_processing_time += batch_post_processing_time

    # Calculate metrics based on pure decoding time
    print(f"TPS breakdown: {total_pure_decoding_time} / {total_tokens_generated} = {total_tokens_generated / total_pure_decoding_time if total_pure_decoding_time > 0 else 0}")

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
