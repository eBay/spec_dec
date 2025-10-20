"""
Custom batch speculative decoding implementation
"""

import time
from time import perf_counter
from typing import Any, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
# from transformers.generation.candidate_generator import _crop_past_key_values

from tqdm import tqdm

from .verification import batch_oracle_verification, from_legacy_cache, TimingBreakdown, realign_kv_cache, pad_sequences_for_alignment_fixed




def run_speculative_inference(target_model, draft_model, tokenizer, prompts: List[str], 
                             max_new_tokens: int, batch_size: int, 
                             n_draft_tokens: int, device: str, use_cache: bool,
                             verbose_acceptance: bool = False, enable_profiling: bool = False, max_input_len: int = 1024) -> Tuple[List[str], float, float, float, float, TimingBreakdown, int, int]:
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
        verbose_acceptance: Enable detailed step-by-step acceptance length logging
        
    Returns:
        Tuple of (generated outputs, pure decoding time, tokens per second based on pure decoding, TAR, latency per iteration, timing breakdown)
    """
    print(f"Running speculative inference with batch size: {batch_size}, draft tokens: {n_draft_tokens}")
    
    all_outputs = []
    total_tokens_generated = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_draft_calls = 0
    total_verification_calls = 0
    iteration_times = []
    
    # Timing accumulators
    total_tokenization_time = 0.0
    total_pure_decoding_time = 0.0
    total_post_processing_time = 0.0
    
    # Initialize profiling timers (accumulated across all batches)
    total_stage1_draft_generate_time = 0.0
    total_stage2_verification_time = 0.0
    total_stage3_update_alignment_time = 0.0
    
    # Initialize acceptance logging
    step_counter = 0
    if verbose_acceptance:
        print("Step-by-step acceptance logging enabled")
        print("Format: Step X: [seq0_accepted, seq1_accepted, ...] (only active sequences)")
    
    
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Speculative Inference Batches"):
        # Get a batch of prompts
        batch_prompts = prompts[i:i+batch_size]
        actual_batch_size = len(batch_prompts)
        
        # Time tokenization (without padding)
        torch.cuda.synchronize()
        tokenization_start = perf_counter()
        # encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=False)
        encoded_inputs = tokenizer(batch_prompts, return_tensors=None, padding=False, truncation=True, max_length=max_input_len)
        # input_ids = [torch.as_tensor(x, dtype=torch.long) for x in encoded_inputs["input_ids"]]  # <<< key change
        torch.cuda.synchronize()
        batch_tokenization_time = perf_counter() - tokenization_start
        total_tokenization_time += batch_tokenization_time
        
        # Pad sequences and move to device (counted as part of decoding)
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        # ).to(device)
        encoded_inputs = tokenizer.pad(encoded_inputs, padding_side='left', padding=True, return_tensors="pt").to(device)
        attention_mask = (encoded_inputs.input_ids != tokenizer.pad_token_id).long().to(device)
        
        # Track input lengths to extract outputs later
        input_lengths = attention_mask.sum(dim=1).tolist()
        
        # Initialize generated sequences with the input
        generated_ids = encoded_inputs.input_ids.clone()
        
        # Initialize token counters
        tokens_generated = torch.zeros(actual_batch_size, dtype=torch.int).to(device)
        
        # Continue generating until all sequences reach max_new_tokens
        # FIXED: DEBUG!! This is problematic!!!
        # The while loop continues until ALL sequences reach max_new_tokens, 
        # This means faster sequences must wait for slower ones, wasting computation cycles and memory on already-completed sequences.
        
        
        # while torch.min(tokens_generated) < max_new_tokens:
        #     # How many more tokens we need to generate for each sequence
        #     remaining_tokens = max_new_tokens - tokens_generated
            
        #     # Generate draft tokens with the draft model
        #     with torch.no_grad():
        #         # Calculate max draft tokens for this iteration
        #         max_draft_this_iter = torch.min(torch.tensor([n_draft_tokens, torch.min(remaining_tokens)])).item()
        #         if max_draft_this_iter <= 0:
        #             break
        completed_mask = torch.zeros(actual_batch_size, dtype=torch.bool, device=device)
        # TODO: max_cache_length
        # target_past_key_values = StaticCache(target_model.config, max_batch_size=actual_batch_size, device=device, dtype=target_model.dtype)
        target_past_key_values = DynamicCache()
        
        # Start timing pure decoding (speculative decoding loop)
        torch.cuda.synchronize()
        batch_pure_decoding_start = perf_counter()
        
        # Initialize batch-level profiling timers
        batch_stage1_draft_generate_time = 0.0
        batch_stage2_verification_time = 0.0
        batch_stage3_update_alignment_time = 0.0
        
        # Modify the while loop condition
        # while not completed_mask.all() and tokens_generated.max() < max_new_tokens:
        while not completed_mask.all():
            torch.cuda.synchronize()  # Ensure all GPU operations complete before timing
            iteration_start_time = perf_counter()
            # Update completed mask
            eos_mask = (generated_ids == tokenizer.eos_token_id).any(dim=1)
            # completed_mask = (tokens_generated >= max_new_tokens)
            completed_mask = (tokens_generated >= max_new_tokens) | eos_mask
            active_mask = ~completed_mask
            
            # Skip processing for completed sequences
            if not active_mask.any():
                break
            
            # Only process active sequences
            active_indices = torch.where(active_mask)[0]
            active_generated_ids = generated_ids[active_indices]
            
            remaining_tokens = max_new_tokens - tokens_generated[active_indices]
            max_draft_this_iter = min(n_draft_tokens, torch.min(remaining_tokens).item())
            
            # ============ STAGE 1: DRAFT MODEL GENERATE - BEGIN ============
            torch.cuda.synchronize()
            stage1_start = perf_counter()
            
            # TODO: DEBUG: Did not use KV-cache!!!
            # Done: Need to fix attention mask!!!
            draft_attention_mask = (active_generated_ids != tokenizer.pad_token_id).long()

            
            draft_outputs = draft_model.generate(
                # input_ids=generated_ids,
                # attention_mask=torch.ones_like(generated_ids).to(device),
                # max_new_tokens=max_draft_this_iter,
                input_ids=active_generated_ids,
                # attention_mask=torch.ones_like(active_generated_ids).to(device),
                attention_mask=draft_attention_mask.to(device),
                max_new_tokens=max_draft_this_iter,
                # temperature=0,
                # top_p=0.95,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            
            # print("1: =================================")
            # print(draft_outputs)

            # Extract draft tokens, correctly mapping from the active-only output
            draft_tokens = []
            active_seq_iterator = 0  # Iterator for the smaller draft_outputs tensor
            for seq_idx in range(actual_batch_size):
                # Only process draft tokens for sequences that are still active
                if active_mask[seq_idx]:
                    
                    # Get the length of the original sequence we are working on
                    seq_len = generated_ids[seq_idx].shape[0]
                    
                    # Get the corresponding output from the draft model using our separate iterator
                    full_draft_output_seq = draft_outputs.sequences[active_seq_iterator]
                    
                    # Slice it to get only the newly generated tokens
                    seq_draft = full_draft_output_seq[seq_len:]
                    draft_tokens.append(seq_draft)
                    
                    # Track total draft tokens and advance our iterator
                    total_draft_tokens += len(seq_draft)
                    active_seq_iterator += 1
                else:
                    # draft_tokens.append(None)
                    # If the sequence is not active, just append an empty tensor to keep list indices aligned
                    draft_tokens.append(torch.tensor([tokenizer.pad_token_id], dtype=torch.long, device=device))
                    # padding_tensor = torch.full((max_draft_this_iter,), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    # inactive_draft = torch.zeros_like(last_active_seq_draft, dtype=torch.long, device=device)
                    # draft_tokens.append(inactive_draft)
                    
            # ragged batch speculation requires right padding
            draft_tokens_tensor = torch.nn.utils.rnn.pad_sequence(
                draft_tokens,
                batch_first=True,
                padding_value=tokenizer.pad_token_id
            )

            # draft_tokens_tensor = torch.stack(draft_tokens)
            
            # WARNING: There's no need for draft madel attention mask!!!!
            # FIX: Create an attention mask for the draft tokens part
            draft_tokens_attention_mask = (draft_tokens_tensor != tokenizer.pad_token_id).long()
            draft_tokens_attention_mask = torch.ones_like(draft_tokens_attention_mask)

            # FIX: Combine the base mask and the draft mask
            combined_attention_mask = torch.cat([attention_mask, draft_tokens_attention_mask], dim=1)
            
            torch.cuda.synchronize()
            batch_stage1_draft_generate_time += perf_counter() - stage1_start
            total_draft_calls += 1
            # ============ STAGE 1: DRAFT MODEL GENERATE - END ============

            # ============ STAGE 2: VERIFICATION - BEGIN ============
            torch.cuda.synchronize()
            stage2_start = perf_counter()
            
            # Verify draft tokens against target model predictions
            verification_result = batch_oracle_verification(
                target_model, generated_ids, draft_tokens_tensor, combined_attention_mask, target_past_key_values, device, tokenizer, use_cache
            )
            torch.cuda.synchronize()
            batch_stage2_verification_time += perf_counter() - stage2_start
            total_verification_calls += 1
            # ============ STAGE 2: VERIFICATION - END ============
            # ============ STAGE 3: SEQUENCE UPDATE/ALIGNMENT - BEGIN ============
            torch.cuda.synchronize()
            stage3_start = perf_counter()
            # Unpack results
            first_false_positions, accepted_tokens, next_token_predictions, target_past_key_values = verification_result
            
            # Process accepted tokens - this is the key part from spec_decoding_deployment.py
            matched_tokens = first_false_positions
            
            # Track total accepted tokens (vectorized)
            total_accepted_tokens += matched_tokens[active_mask].sum().item()
            
            # Log step-by-step acceptance if verbose mode is enabled
            if verbose_acceptance:
                step_acceptances = []
                for seq_idx in range(actual_batch_size):
                    if active_mask[seq_idx]:
                        step_acceptances.append(matched_tokens[seq_idx].item())
                step_counter += 1
                print(f"  Step {step_counter}: Accepted lengths = {step_acceptances}")
            
            # Use the next token predictions from the target model
            # This eliminates the duplicate computation
            # FIXED: Only process active sequences to prevent completed sequences from exceeding max_new_tokens
            for seq_idx in range(actual_batch_size):
                # Only process active sequences
                if active_mask[seq_idx]:
                    # Append the next token prediction to accepted tokens
                    next_token = next_token_predictions[seq_idx].unsqueeze(0)
                    
                    # Check if we need to generate this token (rare edge case)
                    if next_token_predictions[seq_idx] == -1:
                        raise ValueError(f"!!!Should not happen!!!: next_token_predictions[seq_idx] == -1 for seq_idx {seq_idx}")
                    
                    # Append the next token to accepted tokens
                    accepted_tokens[seq_idx] = torch.cat([accepted_tokens[seq_idx], next_token])
                    
                    # Increment matched tokens by 1 to account for the extra token from target
                    matched_tokens[seq_idx] += 1
            

            

            
            # Create new input tensors with left padding to align sequences
            # FIXED: Clear accepted tokens for completed sequences to prevent them from being modified
            # Vectorized: Clear matched tokens for completed sequences
            matched_tokens[~active_mask] = 0
            
            # Clear accepted tokens for completed sequences (list operation)
            for seq_idx in range(actual_batch_size):
                if not active_mask[seq_idx]:
                    accepted_tokens[seq_idx] = torch.tensor([], dtype=torch.long, device=device)
            
            
            generated_ids, original_content_lengths, new_padding_lengths, old_padding_lengths = pad_sequences_for_alignment_fixed(generated_ids, accepted_tokens, matched_tokens, tokenizer, device)

            # Realign AND trim the KV cache
            target_past_key_values = realign_kv_cache(
            target_model,  # Pass the model
            target_past_key_values, 
            original_content_lengths, 
            new_padding_lengths, 
            old_padding_lengths,
            matched_tokens  # Number of accepted tokens per sequence
            )
            # Replace the old call with the new in-place version
            # target_past_key_values = realign_kv_cache_inplace(
            #     target_model,
            #     target_past_key_values, 
            #     original_content_lengths, 
            #     new_padding_lengths, 
            #     old_padding_lengths,
            #     matched_tokens
            # )
            # 2. Realign the KV cache using the map from the step above
            # target_past_key_values = realign_kv_cache(
            #     target_past_key_values, original_content_lengths, new_padding_lengths, old_padding_lengths
            # )

            # Update token counters
            # tokens_generated += matched_tokens
            tokens_generated[active_indices] += matched_tokens[active_indices]

            
            # Update attention mask for new sequence lengths
            # Since completed sequences don't change their generated_ids anymore,
            # their attention masks will naturally remain the same when recalculated
            attention_mask = (generated_ids != tokenizer.pad_token_id).long()
            
            torch.cuda.synchronize()
            batch_stage3_update_alignment_time += perf_counter() - stage3_start
            # ============ STAGE 3: SEQUENCE UPDATE/ALIGNMENT - END ============
 
            # Track iteration time
            torch.cuda.synchronize()  # Ensure all GPU operations complete before measuring time
            iteration_end_time = perf_counter()
            iteration_times.append((iteration_end_time - iteration_start_time) * 1000)  # Convert to milliseconds
        
        # End timing pure decoding
        torch.cuda.synchronize()
        batch_pure_decoding_time = perf_counter() - batch_pure_decoding_start
        total_pure_decoding_time += batch_pure_decoding_time
        
        # Accumulate batch-level profiling timers to totals
        total_stage1_draft_generate_time += batch_stage1_draft_generate_time
        total_stage2_verification_time += batch_stage2_verification_time
        total_stage3_update_alignment_time += batch_stage3_update_alignment_time
        
        # Start timing post-processing (output extraction and decoding)
        torch.cuda.synchronize()
        batch_post_processing_start = perf_counter()
        
        # Extract and decode the generated outputs (excluding input)
        for seq_idx in range(actual_batch_size):
            # Handle left padding when extracting original input length
            # Count non-padding tokens from the beginning
            pad_token_id = tokenizer.pad_token_id
            seq = generated_ids[seq_idx]
            pad_mask = seq != pad_token_id
            
            # Find first non-padding token
            if pad_mask.any():
                first_non_pad = pad_mask.nonzero()[0].item() if pad_mask.nonzero().numel() > 0 else 0
            else:
                first_non_pad = 0
                
            # Extract original input length (may include padding)
            orig_input_len = input_lengths[seq_idx]
            
            # The start index should be: first_non_pad + orig_input_len
            start_idx = first_non_pad + orig_input_len
            
            # Extract generated tokens (excluding input and left padding)
            generated_seq = seq[start_idx:]
            
            # Count tokens generated for this prompt (stop at EOS)
            # Find EOS token if present
            eos_positions = (generated_seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                # Count tokens up to and including first EOS
                tokens_count = eos_positions[0].item() + 1
            else:
                # Count all generated tokens if no EOS
                tokens_count = len(generated_seq)
            total_tokens_generated += tokens_count
            
            # Decode output
            output_text = tokenizer.decode(generated_seq, skip_special_tokens=False)
            all_outputs.append(output_text)
        
        # End timing post-processing
        torch.cuda.synchronize()
        batch_post_processing_time = perf_counter() - batch_post_processing_start
        total_post_processing_time += batch_post_processing_time
    
    # Calculate metrics based on pure decoding time
    tokens_per_second_pure = total_tokens_generated / total_pure_decoding_time if total_pure_decoding_time > 0 else 0.0
    tar = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0.0
    avg_latency_per_iteration = sum(iteration_times) / len(iteration_times) if iteration_times else 0.0
    
    # Create timing breakdown
    total_time = total_tokenization_time + total_pure_decoding_time + total_post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=total_tokenization_time,
        pure_decoding_time=total_pure_decoding_time,
        post_processing_time=total_post_processing_time,
        total_time=total_time
    )
    
    # Print profiling results if enabled
    if enable_profiling:
        print("\n" + "="*60)
        print("PROFILING RESULTS - Stage Time Breakdown:")
        print("="*60)
        
        total_stage_time = total_stage1_draft_generate_time + total_stage2_verification_time + total_stage3_update_alignment_time
        
        print(f"Stage 1 (Draft Generate):    {total_stage1_draft_generate_time:8.3f}s ({total_stage1_draft_generate_time/total_stage_time*100:5.1f}%)")
        print(f"Stage 2 (Verification):      {total_stage2_verification_time:8.3f}s ({total_stage2_verification_time/total_stage_time*100:5.1f}%)")
        print(f"Stage 3 (Update/Alignment):  {total_stage3_update_alignment_time:8.3f}s ({total_stage3_update_alignment_time/total_stage_time*100:5.1f}%)")
        print("-"*60)
        print(f"Total Stage Time:            {total_stage_time:8.3f}s")
        print(f"Total Pure Decoding Time:    {total_pure_decoding_time:8.3f}s")
        print(f"Overhead (non-stage time):   {total_pure_decoding_time - total_stage_time:8.3f}s")
        print("="*60)
    
    print(f"Total draft calls: {total_draft_calls}, Total verification calls: {total_verification_calls}")
    
    return all_outputs, total_pure_decoding_time, tokens_per_second_pure, tar, avg_latency_per_iteration, timing_breakdown, total_draft_calls, total_verification_calls