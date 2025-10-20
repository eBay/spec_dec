"""
Batch oracle verification for speculative decoding

Check https://huggingface.co/docs/transformers/en/cache_explanation
"""

from typing import Any, List, Tuple, Optional, NamedTuple
from time import perf_counter

import torch
from transformers import DynamicCache, StaticCache
from transformers.generation.candidate_generator import _crop_past_key_values

def from_legacy_cache(
    cache: DynamicCache, past_key_values: Optional[tuple[tuple[torch.FloatTensor, torch.FloatTensor]]] = None
) -> "DynamicCache":
    """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
    backward compatibility."""
    # cache = cls()
    if past_key_values is not None:
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
    return cache



class TimingBreakdown(NamedTuple):
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float






def realign_kv_cache(model, past_key_values, original_content_lengths, new_padding_lengths, old_padding_lengths, accepted_token_lengths):
    """
    Correctly realigns the DynamicCache by shifting content to match new padding
    AND trimming to only keep KV values for accepted tokens.
    
    Args:
        model: The model (needed for _crop_past_key_values)
        past_key_values: The current KV cache (DynamicCache)
        original_content_lengths: Content lengths before accepting new tokens
        new_padding_lengths: New padding required for alignment
        old_padding_lengths: Previous padding lengths
        accepted_token_lengths: Number of tokens accepted for each sequence
    """
    if past_key_values is None:
        return None
    
    # Calculate final content lengths after accepting tokens
    final_content_lengths = original_content_lengths + accepted_token_lengths
    
    
    # I think this is not necessary, because max_final_length = final_content_lengths.max()
    # First, find the maximum length we need to keep across all sequences
    # max_final_length = (new_padding_lengths + final_content_lengths).max().item()
    
    
    # we accept original_content_lengths + accepted_token_lengths (including the bonus token)
    # But the bonus token's kv is not generated yet (it's from the sampling on the last step's logits), so we need to subtract 1
    
    max_final_length = final_content_lengths.max().item() - 1
    # Create a new list to store realigned key-value pairs
    realigned_past = []
    
    # Loop over layers
    for layer_idx, (key_cache, value_cache) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
        batch_size, num_heads, current_seq_len, head_dim = key_cache.shape
        
        # Create new tensors for this layer with the required size
        # DynamicCache will grow as needed, so we size it to max_final_length
        new_key = torch.zeros(batch_size, num_heads, max_final_length, head_dim,
                            device=key_cache.device, dtype=key_cache.dtype)
        new_value = torch.zeros(batch_size, num_heads, max_final_length, head_dim,
                              device=value_cache.device, dtype=value_cache.dtype)
        
        # Semi-vectorized KV cache realignment - vectorize validation, keep simple loop for copying
        # Calculate all source and destination ranges (vectorized)
        source_starts = old_padding_lengths  
        source_ends = old_padding_lengths + original_content_lengths + accepted_token_lengths - 1
        dest_starts = new_padding_lengths
        
        dest_ends = new_padding_lengths + original_content_lengths + accepted_token_lengths - 1
        
        # Simple loop for copying (complex vectorization is error-prone for this use case)
        for i in range(batch_size):
            source_start, source_end = source_starts[i].item(), source_ends[i].item()
            dest_start, dest_end = dest_starts[i].item(), dest_ends[i].item()
            new_key[i, :, dest_start:max_final_length, :] = key_cache[i, :, source_start:source_end, :]
            new_value[i, :, dest_start:max_final_length, :] = value_cache[i, :, source_start:source_end, :]

        realigned_past.append((new_key, new_value))

        
    # Create a new DynamicCache from the realigned past
    realigned_cache = DynamicCache.from_legacy_cache(realigned_past)

    realigned_cache.crop(max_final_length)

    
    return realigned_cache



def pad_sequences_for_alignment_fixed(generated_ids, accepted_tokens, matched_tokens, tokenizer, device):
    """
    Fixed version of pad_sequences_for_alignment that properly handles padding accumulation.
    
    This implementation fixes the compounding left-padding bug by:
    1. Stripping existing left padding from each sequence
    2. Appending the newly accepted tokens to the unpadded sequences
    3. Re-padding the entire batch with fresh left padding to ensure alignment
    
    Args:
        generated_ids: Current generated sequences (may contain left padding)
        accepted_tokens: List of accepted tokens for each sequence (already included the bonus token)
        matched_tokens: Number of matched tokens for each sequence (kept for compatibility)
        tokenizer: The tokenizer (used for pad_token_id)
        device: Device to create tensors on
        
    Returns:
        torch.Tensor: New batch of sequences with proper left padding for alignment
    
    Note: The matched_tokens parameter is kept for drop-in compatibility with the original
          function signature, but is not used in this fixed implementation since all sequences
          are re-aligned based on their actual lengths after appending accepted tokens.
    """
    batch_size = generated_ids.size(0)
    pad_token_id = tokenizer.pad_token_id
    
    # Step 1: Unpad each sequence and append accepted tokens
    unpadded_sequences = []
    original_content_lengths = [] # <-- METADATA 1
    old_padding_lengths = [] # <<< ADDED: To store the padding from the *previous* state

    # Vectorized padding detection
    non_pad_mask = generated_ids != pad_token_id
    
    # Find first non-padding token for each sequence (vectorized)
    # Use argmax to find first True position (first non-padding token)
    first_non_pad_indices = torch.argmax(non_pad_mask.int(), dim=1)
    
    # Check for edge case: sequences that are all padding
    # has_non_pad = non_pad_mask.any(dim=1)
    # if not has_non_pad.all():
    #     invalid_seqs = torch.where(~has_non_pad)[0]
    #     raise ValueError(f"!!!Should not happen!!!: entire sequence is padding for seq_idx {invalid_seqs.tolist()}")
    
    # Convert to lists for compatibility with existing code
    old_padding_lengths = first_non_pad_indices.tolist()
    # old_padding_lengths = [0] * batch_size
    
    # Extract unpadded sequences and calculate lengths (still need loop for tensor slicing)
    for seq_idx in range(batch_size):
        first_non_pad_idx = first_non_pad_indices[seq_idx].item()
        seq = generated_ids[seq_idx]
        
        # Extract the unpadded sequence (everything after the padding)
        unpadded_seq = seq[first_non_pad_idx:]
        # unpadded_seq = seq[:]
        
        # Save the length of the original content (from the old cache)
        original_content_lengths.append(unpadded_seq.size(0))
        
        # Append the newly accepted tokens for this sequence
        updated_seq = torch.cat([unpadded_seq, accepted_tokens[seq_idx]])
        unpadded_sequences.append(updated_seq)
    
    # Step 2: Find the maximum length among all updated sequences
    max_length = max(seq.size(0) for seq in unpadded_sequences)
    
    # Step 3: Re-pad all sequences with fresh left padding using torch.nn.utils.rnn.pad_sequence
    # Calculate padding lengths before using pad_sequence
    new_padding_lengths = [max_length - seq.size(0) for seq in unpadded_sequences]
    
    # Use torch.nn.utils.rnn.pad_sequence for efficient left padding
    
    padded_sequences_tensor = torch.nn.utils.rnn.pad_sequence(
        unpadded_sequences, 
        batch_first=True, 
        padding_value=pad_token_id,
        padding_side='left'
    )
    
    return padded_sequences_tensor, torch.tensor(original_content_lengths, device=device), torch.tensor(new_padding_lengths, device=device), torch.tensor(old_padding_lengths, device=device)

def batch_oracle_verification(model, input_tensors, draft_tokens_tensors, attention_mask, target_past_key_values, device, tokenizer, use_cache):
    """
    Verifies the predictions of the draft model against the oracle (target) model for a batch of sequences.
    
    Args:
        model: The target/oracle model
        input_tensors: Tensor of input token sequences [batch_size, input_seq_len]
        draft_tokens_tensors: Tensor of draft generated tokens [batch_size, draft_seq_len]
        device: Device to run verification on
        
    Returns:
        Tuple of (first false positions, accepted draft tokens, next token predictions)
        The next token predictions are what the target model thinks should come after the accepted tokens
    # Process each sequence in the batch
    # FIXED: DEBUG: Sequential Oracle Verification. Should be parallelized.
    """
    batch_size = input_tensors.shape[0]
    
    
    # Concatenate all sequences with their draft tokens at once
    max_input_len = input_tensors.shape[1]
    combined_tokens = torch.cat([input_tensors, draft_tokens_tensors], dim=1)
    batch_size, draft_seq_len = draft_tokens_tensors.shape
    # Single forward pass for all sequences
    with torch.no_grad():
            

        if not use_cache:
            outputs: Any = model(combined_tokens, attention_mask=attention_mask)
        else:

            
            # ===== Prefill =====
            
            # First time call, prefill the empty kv cache
            needs_prefill = (target_past_key_values is None or 
                           (hasattr(target_past_key_values, 'key_cache') and not target_past_key_values.key_cache))
            
            if needs_prefill:
                position_ids = torch.clamp(torch.cumsum(attention_mask[:,:input_tensors.shape[1]], dim=-1) - 1, min=0)
                prefil_outputs: Any = model(input_tensors, attention_mask=attention_mask[:,:input_tensors.shape[1]], past_key_values=target_past_key_values, position_ids=position_ids, device=device)
                prefil_past_key_values = prefil_outputs.past_key_values
            else:
                prefil_past_key_values = target_past_key_values
            
            # Crop if it's a DynamicCache (not a tuple)
            if hasattr(prefil_past_key_values, 'crop'):
                prefil_past_key_values.crop(max_length=input_tensors.shape[1] - 1)

            # ===== DEBUG =====
            # correct_cache = 100 * prefil_past_key_values.key_cache[0][:, 0, :, 5]
            
            # print("correct_cache", correct_cache)
            # print("buggy_cache", buggy_cache)
            # if buggy_cache is not None:
            #     assert (correct_cache[:, -1] == buggy_cache[:, -1]).all()
            #     print("assert passed!!!")
            # print("=" * 100)
            
            cache_position = torch.arange(input_tensors.shape[1] - 1, input_tensors.shape[1] + draft_tokens_tensors.shape[1], device=device)
            
            # ===== ReFill =====
            
            full_position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
            start_pos = input_tensors.shape[1] - 1
            end_pos = input_tensors.shape[1] + draft_tokens_tensors.shape[1]
            position_ids = full_position_ids[:, start_pos:end_pos]
            outputs = model(torch.cat([input_tensors[:, -1:], draft_tokens_tensors], dim=1), attention_mask=attention_mask, past_key_values=prefil_past_key_values, cache_position=cache_position, position_ids=position_ids)
            
            
            
            target_past_key_values = outputs.past_key_values
            # =================================== FIX END ===================================


    # Extract logits for positions after the input (we want to predict the next token after each draft token)
    # Remove :-1 to include prediction for position after all draft tokens
    if not use_cache:
        next_token_logits = outputs.logits[:, max_input_len-1:, :]

    else:

        
        next_token_logits = outputs.logits[:, :-1, :]
            


    # Get predictions for all sequences at once
    predicted_tokens = torch.argmax(next_token_logits, dim=-1)
    
    # Compare with draft tokens in parallel (only compare the first N predictions with N draft tokens)
    # if not use_cache:
    #     matches = (predicted_tokens[:, :draft_tokens_tensors.shape[1]] == draft_tokens_tensors)
    # else:
        # matches = (predicted_tokens[:, :] == draft_tokens_tensors)
    matches = (predicted_tokens[:, :draft_tokens_tensors.shape[1]] == draft_tokens_tensors)

    # Find first mismatch for each sequence using vectorized operations
    # For sequences with all matches, we'll use the last position
    default_position = matches.shape[1] - 1
    
    # Create a mask for mismatches and find first False position per sequence
    # torch.argmax on boolean tensor returns first True position, so we use ~matches
    mismatch_positions = torch.argmax((~matches).int(), dim=1)
    
    # Check if there are any mismatches per sequence (if all True, argmax returns 0)
    has_mismatch = ~matches.all(dim=1)
    
    # Use mismatch position if there's a mismatch, otherwise use default position
    first_false_positions = torch.where(has_mismatch, mismatch_positions, default_position)
    
    # Get next token predictions using advanced indexing
    batch_indices = torch.arange(batch_size, device=device)
    next_token_predictions = predicted_tokens[batch_indices, first_false_positions]
            


    accepted_tokens_list = [draft_tokens_tensors[i][:first_false_positions[i]] for i in range(batch_size)]

    return first_false_positions, accepted_tokens_list, next_token_predictions, target_past_key_values


