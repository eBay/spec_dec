"""
Custom batch speculative decoding implementation with cross-batch scheduling
"""

import time
from time import perf_counter
from typing import Any, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
# from transformers.generation.candidate_generator import _crop_past_key_values
from tqdm import tqdm
from .verification import batch_oracle_verification, from_legacy_cache, TimingBreakdown
from collections import defaultdict

# DEBUGGING = True
DEBUGGING = False

class SequencePool:
    """Single pool managing all sequences directly"""
    def __init__(self, device, pad_token_id):
        # Flat lists - no batch grouping
        self.sequences: List[torch.Tensor] = []  # just save the sequences, no matter padded or unpadded
        self.kv_caches: List[Optional[DynamicCache]] = []
        self.prompt_lengths: List[int] = []
        self.tokens_generated: List[int] = []
        self.is_active: List[bool] = []
        self.device = device
        self.pad_token_id = pad_token_id
        self.active_window_indices = []  # Indices of sequences currently in the window
        self.next_window_start = 0       # Next sequence index to add to window
        self.pbar = None  # Progress bar for tracking completed sequences
        self.total_sequences = 0  # Total number of sequences to process
        # Statistics for branch tracking
        self.cross_batch_count = 0  # Counter for same length group branch
        self.fallback_count = 0  # Counter for fallback realignment branch
        
    def tokenize_whole_dataset_without_padding(self, prompts, tokenizer, sort_by_length=False, max_input_len=1024):
        """Tokenize all prompts without padding, keeping them as individual sequences
        
        Args:
            prompts: List of text prompts
            tokenizer: Tokenizer to use
            sort_by_length: If True, sort prompts by token length in ascending order
            max_input_len: Maximum input length in tokens (prompts will be truncated if longer)
        """
        # First tokenize all prompts
        tokenized_data = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_input_len)
            input_ids = encoded.input_ids[0].to(self.device)  # Remove batch dimension
            tokenized_data.append((input_ids, len(input_ids)))
        
        # Sort by length if requested
        if sort_by_length:
            tokenized_data.sort(key=lambda x: x[1])  # Sort by token length
        
        # Store in sequence pool
        for input_ids, length in tokenized_data:
            self.sequences.append(input_ids)
            self.prompt_lengths.append(length)
            self.tokens_generated.append(0)
            self.is_active.append(True)
        
        self.kv_caches = [None] * len(prompts)
        
        # Initialize progress bar
        self.total_sequences = len(prompts)
        self.pbar = tqdm(total=self.total_sequences, desc=f"Generating sequences {self.total_sequences}", leave=True)
    
    def refill_window(self, window_size):
        """Maintain window at window_size by replacing completed sequences"""
        # Remove completed sequences
        self.active_window_indices = [idx for idx in self.active_window_indices if self.is_active[idx]]
        
        # Fill up to window_size
        while len(self.active_window_indices) < window_size and self.next_window_start < len(self.sequences):
            if self.is_active[self.next_window_start]:
                self.active_window_indices.append(self.next_window_start)
            self.next_window_start += 1
        

    def _find_same_length_group(self, batch_size):
        """Find indices of active sequences with the same length and cache state, within active window"""
        # same length grouping branch

        # Group by (length, has_cache) tuple
        groups = defaultdict(list)
        
        # Only consider sequences within the active window
        for idx in self.active_window_indices:
            if self.is_active[idx]:
                seq_len = len(self.sequences[idx])
                has_cache = self.kv_caches[idx] is not None
                group_key = (seq_len, has_cache)
                groups[group_key].append(idx)
                
                # Early return if we have enough sequences with same length AND cache state
                if len(groups[group_key]) >= batch_size:
                    return groups[group_key][:batch_size]
        # Return the best group if no group reaches batch_size
        # if groups:
        #     # Prioritize groups WITHOUT cache (fresh sequences) for better cohort formation
        #     fresh_groups = {k: v for k, v in groups.items() if k[1] == False}
        #     if fresh_groups:
        #         # Among fresh groups, prefer shorter sequences (less padding)
        #         # Sort by: 1) group size (larger better), 2) sequence length (shorter better)
        #         best_key = min(fresh_groups.keys(), key=lambda k: (-len(fresh_groups[k]), k[0]))
        #         return fresh_groups[best_key][:batch_size]
            
        #     # Fall back to any group, still preferring shorter sequences
        #     best_key = min(groups.keys(), key=lambda k: (-len(groups[k]), k[0]))
        #     return groups[best_key][:batch_size]
        return []
    
    def _get_active_sequences(self, batch_size):
        """Get any active sequences up to batch_size from active window"""
        # fallback branch
        active_indices = [idx for idx in self.active_window_indices if self.is_active[idx]]
        return active_indices[:batch_size]
    
    def get_batch(self, strategy='cross_batch', batch_size=8):
        """Returns indices and prepared batch data from active window
        
        Args:
            strategy: 'cross_batch' for same-length AND same-cache-state grouping, 'batch' for fallback
            batch_size: Number of sequences to batch together
            
        Returns:
            Tuple of (indices, batch_seqs, attention_mask, kvs, mode_used)
            where mode_used is 'cross_batch' or 'fallback'
        """
        assert strategy in ['cross_batch', 'batch']
        # DEBUGGING
        # strategy = 'batch'
        if strategy == 'cross_batch': # same length grouping
            indices = self._find_same_length_group(batch_size)
            
            # Use same-length group if we found any
            if indices:
                # Same length - just stack, no padding needed!
                seqs = [self.sequences[i] for i in indices]
                kvs = [self.kv_caches[i] for i in indices]
                # Create attention mask (all ones since no padding)
                batch_seqs = torch.stack(seqs)
                # attention_mask = torch.ones_like(batch_seqs)
                attention_mask = batch_seqs != self.pad_token_id
                
                if None not in kvs:
                    # print(f"seq len:", [len(seq) for seq in seqs])
                    # print(f"kv seqlen:", [kv.key_cache[0].shape[2] for kv in kvs])
                    kvs = DynamicCache.from_batch_splits(kvs)
                else:
                    # print(f"seq len:", [len(seq) for seq in seqs])
                    # print(f"kv none")
                    kvs = None
                self.cross_batch_count += 1
                return indices, batch_seqs, attention_mask, kvs, 'cross_batch'
            else:
                pass # TO Fallback
            # else:
            #     raise ValueError("Get batch should not happen here")
        # else:
        # Fallback: take any active from window, apply lazy_align (unpad-repad)
        # print("fallback")
        indices = self._get_active_sequences(batch_size)
        if not indices:
            return [], None, None, None, 'none'
            
        seqs = [self.sequences[i] for i in indices]
        kvs = [self.kv_caches[i] for i in indices]
        
        # This is where padding happens for the first time!
        aligned_seqs, mask, aligned_kv = lazy_align_sequences_and_kv_cache(
            seqs, kvs, self.pad_token_id, self.device
        )
        
        # update seq in sequences pool
        for i, idx in enumerate(indices):
            self.sequences[idx] = aligned_seqs[i]
        
        self.fallback_count += 1
        return indices, aligned_seqs, mask, aligned_kv, 'fallback'

    def write_back(self, indices, accepted, bonus, new_kvs, eos_token_id, max_new_tokens, window_size):
        """Simple direct write-back with completion tracking and window refill"""
        completed_count = 0  # Track how many sequences complete in this batch
        
        for i, idx in enumerate(indices):
            # Direct append
            parts = [self.sequences[idx], accepted[i]]
            if bonus[i] != -1:
                parts.append(bonus[i].unsqueeze(0))
            self.sequences[idx] = torch.cat(parts)

            # Update KV and metadata
            # self.kv_caches[idx] = new_kvs[i] if i < len(new_kvs) else self.kv_caches[idx]
            assert bonus[i] != -1
            self.tokens_generated[idx] += len(accepted[i]) + 1
            self.kv_caches[idx] = new_kvs[i]
            self.kv_caches[idx].crop(len(self.sequences[idx]) - 1)
            
            # Check for completion
            if (self.sequences[idx] == eos_token_id).any() or self.tokens_generated[idx] >= max_new_tokens:
                if self.is_active[idx]:  # Only count if it was active before
                    completed_count += 1
                self.is_active[idx] = False
                # Free KV cache memory for completed sequences
                self.kv_caches[idx] = None
        

        
        # Update progress bar if sequences completed
        if completed_count > 0 and self.pbar:
            self.pbar.update(completed_count)
            self.pbar.set_postfix({"Active kv cache": sum((1 if kv is not None else 0) for kv in self.kv_caches)})

            

def lazy_align_sequences_and_kv_cache(
    sequences: List[torch.Tensor],
    individual_kv_caches: List[Optional[DynamicCache]],
    pad_token_id: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, Optional[DynamicCache]]:
    """
    Align sequences and KV caches for batch verification.
    Properly realigns KV caches to match the padding changes when batching sequences.
    
    CRITICAL: This function implements the unpad-repad pattern to prevent padding accumulation.
    Sequences come in WITH padding, we unpad them, then repad with fresh padding.
    
    Args:
        sequences: List of individual sequence tensors (already contain accepted tokens, WITH padding)
        individual_kv_caches: List of individual KV caches (aligned with sequence padding)
        pad_token_id: Token ID to use for padding
        device: Device to use
        
    Returns:
        Tuple of (batched_sequences, attention_mask, batch_kv_cache)
    """
    num_sequences = len(sequences)
    
    # Step 1: Unpad sequences and track original padding
    # This is CRITICAL to prevent padding accumulation
    unpadded_sequences = []
    old_padding_lengths = []
    content_lengths = []


    # non_pad_mask = generated_ids != pad_token_id
    # first_non_pad_indices = torch.argmax(non_pad_mask.int(), dim=1)
    # old_padding_lengths = first_non_pad_indices.tolist()
    
    for idx, seq in enumerate(sequences):
        non_pad_mask = seq != pad_token_id
        # if DEBUGGING: print(seq)
        # if DEBUGGING: print(non_pad_mask)
        # first_non_pad = torch.argmax(non_pad_mask.int(), dim=0)
        first_non_pad = non_pad_mask.nonzero()[0].item()
        old_padding_lengths.append(first_non_pad)
        unpadded_seq = seq[first_non_pad:]
        unpadded_sequences.append(unpadded_seq)
        content_lengths.append(len(unpadded_seq))
    # if DEBUGGING: print("="*100)
    
    # Step 2: Repad sequences with FRESH left padding (prevents accumulation)
    # Use pad_sequence to batch and pad
    batched_sequences = torch.nn.utils.rnn.pad_sequence(
        unpadded_sequences,
        batch_first=True,
        padding_value=pad_token_id,
        padding_side='left'
    )    

    # Step 3: Find max length and calculate new padding needed
    # max_length = max(len(seq) for seq in unpadded_sequences) if unpadded_sequences else 0
    max_length = batched_sequences.shape[1]
    new_padding_lengths = [max_length - len(seq) for seq in unpadded_sequences]
    
    # Create attention mask
    attention_mask = (batched_sequences != pad_token_id).long()
    
    
    # Step 4: Realign KV caches with new padding (simplified like realign_kv_cache)
    
    # NEED to DOUBLE CHECK THIS. Lazy realignment, first iteration? 
    
    if individual_kv_caches and all(cache is not None for cache in individual_kv_caches):
        # Calculate the max KV cache length (sequences are padded to max_length, KV is -1)
        max_kv_length = max_length - 1 if max_length > 0 else 0
        
        if DEBUGGING:
            print(f"Realigning KV caches: old_padding={old_padding_lengths}, new_padding={new_padding_lengths}")
            print(f"Content lengths: {content_lengths}, max_kv_length: {max_kv_length}")
        
        # Get first valid cache as template for layer count and dimensions
        # template_cache = next((cache for cache in individual_kv_caches if cache is not None), None)
        cache_device = individual_kv_caches[0].key_cache[0].device
        layer_count = len(individual_kv_caches[0].key_cache)
        _, num_heads, _, head_dim = individual_kv_caches[0].key_cache[0].shape
        dtype = individual_kv_caches[0].key_cache[0].dtype

        # Create realigned caches for all sequences
        all_realigned_layers = []
        
        for layer_idx in range(layer_count):
            
            # Create batch tensors for this layer
            batch_key = torch.zeros(num_sequences, num_heads, max_kv_length, head_dim,
                                    device=cache_device, dtype=dtype)
            batch_value = torch.zeros(num_sequences, num_heads, max_kv_length, head_dim,
                                        device=cache_device, dtype=dtype)
            
            # Copy content for each sequence at the right position
            for i in range(num_sequences):
                # if i < len(individual_kv_caches) and individual_kv_caches[i] is not None:
                kv_cache = individual_kv_caches[i]
                # if layer_idx < len(kv_cache.key_cache):
                key_cache = kv_cache.key_cache[layer_idx]
                value_cache = kv_cache.value_cache[layer_idx]
                
                old_start = old_padding_lengths[i]
                old_end = old_start + content_lengths[i] - 1
                new_start = new_padding_lengths[i]
                # new_end = max_kv_length
                
                # Direct copy from old position to new position

                batch_key[i, :, new_start:max_kv_length, :] = key_cache[0, :, old_start:old_end, :]
                batch_value[i, :, new_start:max_kv_length, :] = value_cache[0, :, old_start:old_end, :]

            all_realigned_layers.append((batch_key, batch_value))
        
        # Create batch KV cache from realigned layers
        batch_kv_cache = DynamicCache.from_legacy_cache(all_realigned_layers)
    else:
        # First time inference, no KV cache yet - trigger fresh computation
        batch_kv_cache = None
    
    return batched_sequences, attention_mask, batch_kv_cache


def run_cross_batch_inference(target_model, draft_model, tokenizer, prompts: List[str],
                             max_new_tokens: int, batch_size: int, 
                             n_draft_tokens: int, device: str, use_cache: bool,
                             verbose_acceptance: bool, enable_profiling: bool,
                             scheduling_strategy: str = 'cross_batch',
                             sort_by_length: bool = False,
                             window_size: int = 32, max_input_len: int = 1024) -> Tuple[List[str], float, float, float, float, TimingBreakdown, int, int]:
    """
    Run cross-batch speculative inference with same-length grouping optimization.
    """
    print(f"Running cross-batch speculative inference with batch size: {batch_size}, draft tokens: {n_draft_tokens}, scheduling: {scheduling_strategy}, sort_by_length: {sort_by_length}, window_size: {window_size}")
    
    # Initialize timing
    total_tokenization_time = 0.0
    total_pure_decoding_time = 0.0
    total_post_processing_time = 0.0
    total_tokens_generated = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_draft_calls = 0
    total_verification_calls = 0
    
    
    # Tokenization
    torch.cuda.synchronize()
    tokenization_start = perf_counter()
    
    pool = SequencePool(device, tokenizer.pad_token_id)
    pool.tokenize_whole_dataset_without_padding(prompts, tokenizer, sort_by_length, max_input_len)
    
    torch.cuda.synchronize()
    total_tokenization_time = perf_counter() - tokenization_start
    
    # # Track per-sequence KV caches
    # sequence_kv_caches = [None] * len(prompts)
    
    # Initialize window
    pool.refill_window(window_size)
    
    # Main decoding loop
    torch.cuda.synchronize()
    pure_decoding_start = perf_counter()
    
    # Initialize profiling timers
    stage1_get_batch_time = 0.0
    stage2_draft_generate_time = 0.0
    stage3_verification_time = 0.0
    stage4_write_back_time = 0.0
    
    step_counter = 0
    if verbose_acceptance:
        print("Step-by-step acceptance logging enabled")
    
    while any(pool.is_active):
        # ============ STAGE 1: GET BATCH - BEGIN ============
        torch.cuda.synchronize()
        stage1_start = perf_counter()
        
        # 1. Get batch (use scheduling strategy parameter)
        batch_result = pool.get_batch(
            strategy=scheduling_strategy, batch_size=batch_size
        )
        
        indices, aligned_seqs, attention_mask, kv_list, mode_used = batch_result
        
        # If no active sequences in window, we're done
        if not indices:
            break
        
        actual_batch_size = len(indices)
        
        batch_kv_cache = kv_list
        
        torch.cuda.synchronize()
        stage1_get_batch_time += perf_counter() - stage1_start
        # ============ STAGE 1: GET BATCH - END ============
        
        # ============ STAGE 2: DRAFT MODEL GENERATE - BEGIN ============
        torch.cuda.synchronize()
        stage2_start = perf_counter()
        
        # 2. Generate draft tokens
        
        # Calculate max draft tokens for this iteration
        remaining_tokens = [max_new_tokens - pool.tokens_generated[idx] for idx in indices]
        max_draft_this_iter = min(n_draft_tokens, min(remaining_tokens))
        
        if max_draft_this_iter <= 0:
            # Mark sequences as complete if they've reached max tokens
            for idx in indices:
                pool.is_active[idx] = False
            continue
        
        draft_tokens_tensor = draft_model.generate(
            input_ids=aligned_seqs,
            attention_mask=attention_mask,
            max_new_tokens=max_draft_this_iter,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False,
            output_scores=False,
        )

        # Extract only the newly generated tokens (exclude the input)
        draft_tokens_tensor = draft_tokens_tensor[:, aligned_seqs.shape[1]:]
        total_draft_tokens += draft_tokens_tensor.shape[1] * actual_batch_size
        total_draft_calls += 1
        
        # Create combined attention mask
        draft_tokens_attention_mask = torch.ones_like(draft_tokens_tensor)
        combined_attention_mask = torch.cat([attention_mask, draft_tokens_attention_mask], dim=1)
        
        torch.cuda.synchronize()
        stage2_draft_generate_time += perf_counter() - stage2_start
        # ============ STAGE 2: DRAFT MODEL GENERATE - END ============
        
        
        # # contiguous memory
        # draft_tokens_tensor = draft_tokens_tensor.contiguous()
        # combined_attention_mask = combined_attention_mask.contiguous()
        # aligned_seqs = aligned_seqs.contiguous()
        # attention_mask = attention_mask.contiguous()
        # if batch_kv_cache is not None:
        #     for i in range(len(batch_kv_cache.key_cache)):
        #         batch_kv_cache.key_cache[i]  = batch_kv_cache.key_cache[i].contiguous()
        #         batch_kv_cache.value_cache[i] = batch_kv_cache.value_cache[i].contiguous()
        
        # ============ STAGE 3: VERIFICATION - BEGIN ============
        torch.cuda.synchronize()
        stage3_start = perf_counter()
        
        
        
        # 3. Verify with oracle model
        verification_result = batch_oracle_verification(
            target_model, aligned_seqs, draft_tokens_tensor, 
            combined_attention_mask, batch_kv_cache, device, 
            tokenizer, use_cache
        )
        torch.cuda.synchronize()
        stage3_verification_time += perf_counter() - stage3_start
        # ============ STAGE 3: VERIFICATION - END ============
        
        # ============ STAGE 4: WRITE BACK - BEGIN ============
        torch.cuda.synchronize()
        stage4_start = perf_counter()
        
        # Unpack results
        first_false_positions, accepted_tokens, next_token_predictions, new_batch_kv_cache = verification_result
        
        # Track accepted tokens
        total_accepted_tokens += first_false_positions.sum().item()
        total_verification_calls += 1
        
        if verbose_acceptance:
            step_counter += 1
            step_acceptances = [first_false_positions[i].item() for i in range(actual_batch_size)]
            print(f"  Step {step_counter}: Accepted lengths = {step_acceptances}")
        
        # Process bonus tokens (next token predictions)
        bonus_tokens = []
        for i in range(actual_batch_size):
            if next_token_predictions[i] != -1:
                bonus_tokens.append(next_token_predictions[i])
            else:
                bonus_tokens.append(torch.tensor(-1, device=device))
        

        
        # 4. Write back to pool with individual KV caches
        # For same-length batches, split the batch KV cache back to individual caches
        # if new_batch_kv_cache is not None and isinstance(new_batch_kv_cache, DynamicCache):
        # Split batch KV cache into individual caches using the built-in method
        individual_kv_caches = new_batch_kv_cache.batch_split(
            full_batch_size=actual_batch_size, 
            split_size=1
        )
        
        pool.write_back(indices, accepted_tokens, bonus_tokens, individual_kv_caches, 
                       tokenizer.eos_token_id, max_new_tokens, window_size)
        # Refill window to maintain size
        pool.refill_window(window_size)
        
        torch.cuda.synchronize()
        stage4_write_back_time += perf_counter() - stage4_start
        # ============ STAGE 4: WRITE BACK - END ============
        
    torch.cuda.synchronize()
    total_pure_decoding_time = perf_counter() - pure_decoding_start
    
    # Close progress bar
    if pool.pbar:
        pool.pbar.close()
    
    # 5. Detokenization
    torch.cuda.synchronize()
    post_processing_start = perf_counter()
    
    all_outputs = []
    # Correctly calculate total tokens generated by summing across all sequences
    # Use the tracked tokens_generated counter which correctly accounts for actual generation
    total_tokens_generated = sum(pool.tokens_generated)
    
    for i, seq in enumerate(pool.sequences):
        # Extract generated portion (skip prompt and any padding)
        # First, remove left padding from the sequence
        non_pad_mask = seq != tokenizer.pad_token_id
        # if non_pad_mask.any():
        first_non_pad = non_pad_mask.nonzero()[0].item()
        unpadded_seq = seq[first_non_pad:]
        # Now extract the generated portion using the original prompt length
        prompt_len = pool.prompt_lengths[i]
        generated_portion = unpadded_seq[prompt_len:]
        
        decoded_output = tokenizer.decode(generated_portion, skip_special_tokens=False)
        all_outputs.append(decoded_output)
    
    torch.cuda.synchronize()
    total_post_processing_time = perf_counter() - post_processing_start
    
    # Calculate metrics
    print(f"TPS breakdown: {total_pure_decoding_time} / {total_tokens_generated} = {total_tokens_generated / total_pure_decoding_time}")
    tokens_per_second = total_tokens_generated / total_pure_decoding_time if total_pure_decoding_time > 0 else 0
    TAR = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
    
    # Calculate average latency
    avg_latency = (total_pure_decoding_time / step_counter * 1000) if step_counter > 0 else 0
    
    # Create timing breakdown
    timing_breakdown = TimingBreakdown(
        tokenization_time=total_tokenization_time,
        pure_decoding_time=total_pure_decoding_time,
        post_processing_time=total_post_processing_time,
        total_time=total_tokenization_time + total_pure_decoding_time + total_post_processing_time
    )
    
    
    # Print profiling results
    print("\n" + "="*60)
    print("PROFILING RESULTS - Stage Time Breakdown:")
    print("="*60)
    
    total_stage_time = stage1_get_batch_time + stage2_draft_generate_time + stage3_verification_time + stage4_write_back_time
    
    print(f"Stage 1 (Get Batch):        {stage1_get_batch_time:8.3f}s ({stage1_get_batch_time/total_stage_time*100:5.1f}%)")
    print(f"Stage 2 (Draft Generate):   {stage2_draft_generate_time:8.3f}s ({stage2_draft_generate_time/total_stage_time*100:5.1f}%)")
    print(f"Stage 3 (Verification):     {stage3_verification_time:8.3f}s ({stage3_verification_time/total_stage_time*100:5.1f}%)")
    print(f"Stage 4 (Write Back):       {stage4_write_back_time:8.3f}s ({stage4_write_back_time/total_stage_time*100:5.1f}%)")
    print("-"*60)
    print(f"Total Stage Time:           {total_stage_time:8.3f}s")
    print(f"Total Pure Decoding Time:   {total_pure_decoding_time:8.3f}s")
    print(f"Overhead (non-stage time):  {total_pure_decoding_time - total_stage_time:8.3f}s")
    print("="*60)
    
    # Print branch statistics
    print("\n" + "="*60)
    print("GET_BATCH BRANCH STATISTICS:")
    print("="*60)
    total_get_batch_calls = pool.cross_batch_count + pool.fallback_count
    if total_get_batch_calls > 0:
        cross_batch_percent = (pool.cross_batch_count / total_get_batch_calls) * 100
        fallback_percent = (pool.fallback_count / total_get_batch_calls) * 100
        print(f"Total get_batch calls:       {total_get_batch_calls}")
        print(f"Cross-batch (same length):   {pool.cross_batch_count:5} ({cross_batch_percent:5.1f}%)")
        print(f"Fallback (realignment):      {pool.fallback_count:5} ({fallback_percent:5.1f}%)")
    else:
        print("No get_batch calls were made.")
    print("="*60)
    
    print(f"\nCross-batch inference completed. Tokens/s: {tokens_per_second:.2f}, TAR: {TAR:.3f}")
    print(f"Total draft calls: {total_draft_calls}, Total verification calls: {total_verification_calls}")
    
    return all_outputs, total_pure_decoding_time, tokens_per_second, TAR, avg_latency, timing_breakdown, total_draft_calls, total_verification_calls
     