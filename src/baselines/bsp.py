from typing import List, NamedTuple
import torch
import time
from time import perf_counter

def _run_and_timing(fn):
    torch.cuda.synchronize()
    start_t = time.time()
    ret = fn()
    torch.cuda.synchronize()
    dur = time.time() - start_t
    return ret, dur

class AdaptiveSpeculativeDecoding:
    def __init__(self, model, assist_model, tokenizer, speculative_step=1, device='cuda'):
        self.model = model.to(device)
        self.assist_model = assist_model.to(device)
        self.tokenizer = tokenizer
        self.device=device

        self.speculative_step = 1 if speculative_step is None else speculative_step

        # stats
        self.pos_correct = torch.zeros([self.speculative_step], device=device)
        self.pos_cnt = 0

        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0

    def _speculative(self, input_ids, attention_mask, kv_cache, speculate_step):
        batch_size = input_ids.shape[0]
        generated_tokens = [[] for _ in range(batch_size)]
        for i in range(speculate_step):
            ret = self.assist_model(input_ids,
                                    attention_mask=attention_mask, 
                                    use_cache=True, 
                                    past_key_values=kv_cache)
            input_ids = torch.argmax(ret.logits[:, -1:], axis=2)

            for b in range(batch_size):
                generated_tokens[b].append(input_ids[b, 0])

            attention_mask = self._extend_mask(attention_mask) 
            kv_cache = ret.past_key_values
        return generated_tokens, attention_mask, kv_cache
    
    def _last_pos_logits(self, logits, mask):
        last_pos = torch.sum(mask, axis=1) - 1
        return logits[torch.arange(logits.shape[0]), last_pos]
    
    def _extend_mask(self, mask):
        return torch.cat([mask, torch.ones([mask.shape[0], 1], device=self.device, dtype=torch.int32)], axis=1)

    @torch.inference_mode()
    def generate(self, prompts:List[str], num_out:int, collect_stats=False, speculative_step=None, max_input_len=1024):
        speculative_step = self.speculative_step if speculative_step is None else speculative_step
        self.tokenizer.padding_side='right'
        token_seqs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len)
        batch_size = len(prompts)
        assist_kv_cache = None
        input_ids = token_seqs['input_ids'].to(self.device)
        attention_mask = input_attention_mask = token_seqs['attention_mask'].to(self.device)
        prompt_len = attention_mask.sum(axis=1)

        # prefill
        ret, t_prefill = _run_and_timing(lambda: self.model(input_ids, attention_mask=input_attention_mask, use_cache=True))
        self.time_verify += t_prefill
        self.verify_calls += 1
        first_token = torch.argmax(self._last_pos_logits(ret.logits, attention_mask), axis=1).unsqueeze(1) 
        attention_mask = self._extend_mask(attention_mask)
        input_ids = torch.cat([input_ids, first_token], axis=1)
        kv_cache = ret.past_key_values
        generated_tokens = input_ids
        valid_lens = torch.ones(batch_size, device=self.device) 

        # stats
        while True:
            (speculated_tokens, attention_mask, assist_kv_cache), t_spec = _run_and_timing(lambda: self._speculative(input_ids, attention_mask, assist_kv_cache, speculative_step))
            self.time_speculate += t_spec
            # verify
            speculated_tokens = torch.tensor(speculated_tokens, device=self.device, dtype=torch.int64)
            verify_inputs = torch.cat([first_token, speculated_tokens], axis=1)
            ret, t_verify = _run_and_timing(lambda: self.model(verify_inputs, attention_mask=attention_mask, use_cache=True, past_key_values=kv_cache))
            self.time_verify += t_verify
            self.verify_calls += 1
            logits = ret.logits
            kv_cache = ret.past_key_values
            correct = logits[:, :-1].argmax(dim=2)

            # mask wrong predictions
            check_mask = torch.cumsum(correct == speculated_tokens, 1) == torch.arange(1, speculative_step + 1, device=self.device)

            correct_len = torch.sum(check_mask, axis=1)
            first_token = torch.argmax(logits[torch.arange(logits.shape[0]), correct_len], axis=1).unsqueeze(1)
            input_ids = torch.concat([speculated_tokens[:, -1:], first_token], axis=1)
            attention_mask[:, -speculative_step:] = check_mask
            attention_mask = self._extend_mask(attention_mask)
            generated_tokens = torch.cat([generated_tokens, speculated_tokens, first_token], axis=1)

            # update stats
            if collect_stats: 
                not_ended = (valid_lens < num_out).unsqueeze(1)
                self.pos_correct += (check_mask * not_ended).sum(dim=0)
                self.pos_cnt += not_ended.sum() 

            valid_lens += correct_len + 1
            if torch.all(valid_lens >= num_out):
                break
        ret = []
        for b in range(batch_size):
            valid_token = torch.nonzero(attention_mask[b], as_tuple=True)[0]
            tokens = generated_tokens[b][valid_token][:prompt_len[b] + num_out]
            # Only decode the newly generated tokens (skip the prompt)
            generated_only = tokens[prompt_len[b]:]
            ret.append(self.tokenizer.decode(generated_only, skip_special_tokens=False))
        
        return ret

    def get_stats(self):
        return self.pos_correct / self.pos_cnt, self.time_speculate, self.time_verify, self.verify_calls

    def reset_stats(self):
        self.pos_correct = 0
        self.pos_cnt = 0
        self.time_speculate = 0
        self.time_verify = 0
        self.verify_calls = 0


# Wrapper function for unified benchmark compatibility
class TimingBreakdown(NamedTuple):
    """Timing breakdown class to match expected format."""
    tokenization_time: float
    pure_decoding_time: float
    post_processing_time: float
    total_time: float


def run_adaptive_speculative_inference(target_model, draft_model, tokenizer, prompts: List[str],
                                       max_new_tokens: int, device: str, n_draft_tokens: int = 5,
                                       batch_size: int = 4, max_input_len: int = 1024):
    """
    Run adaptive speculative decoding (BSP) inference.
    
    Args:
        target_model: The target model
        draft_model: The draft/assist model  
        tokenizer: The tokenizer
        prompts: List of prompts to process
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to run inference on
        n_draft_tokens: Number of draft tokens (speculative steps)
        batch_size: Batch size for processing prompts in chunks
        max_input_len: Maximum input length in tokens
        
    Returns:
        Tuple of (outputs, pure_decoding_time, tokens_per_second, tar, latency, timing_breakdown, draft_calls, verification_calls)
    """
    
    # Initialize BSP decoder
    bsp_decoder = AdaptiveSpeculativeDecoding(
        model=target_model,
        assist_model=draft_model, 
        tokenizer=tokenizer,
        speculative_step=n_draft_tokens,
        device=device
    )
    
    # Measure tokenization time for all prompts (do this once for all batches)
    torch.cuda.synchronize()
    tokenization_start = perf_counter()
    # Pre-tokenize all prompts to measure total tokenization time
    test_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len)
    torch.cuda.synchronize()
    tokenization_time = perf_counter() - tokenization_start
    
    # Process prompts in batches
    all_outputs = []
    total_pure_decoding_time = 0.0
    total_verify_calls = 0
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Reset stats for this batch
        bsp_decoder.reset_stats()
        
        # Measure pure decoding time for this batch
        torch.cuda.synchronize()
        batch_decoding_start = perf_counter()
        
        # Run generation with stats collection for this batch
        batch_outputs = bsp_decoder.generate(
            prompts=batch_prompts,
            num_out=max_new_tokens,
            collect_stats=True,
            speculative_step=n_draft_tokens,
            max_input_len=max_input_len
        )
        
        torch.cuda.synchronize()
        batch_decoding_time = perf_counter() - batch_decoding_start
        total_pure_decoding_time += batch_decoding_time
        
        # Collect outputs
        all_outputs.extend(batch_outputs)
        
        # Accumulate stats
        _, _, _, batch_verify_calls = bsp_decoder.get_stats()
        total_verify_calls += batch_verify_calls
        
        # Clean up GPU memory between batches to prevent OOM
        if i + batch_size < len(prompts):  # Don't clean on last batch
            torch.cuda.empty_cache()
    
    # Use total accumulated pure decoding time
    pure_decoding_time = total_pure_decoding_time
    outputs = all_outputs
    
    # Post-processing time is already included in generate() output
    # BSP's generate() returns already decoded text, so post-processing is minimal
    post_processing_time = 0.0
    
    # Calculate overall TAR (Token Acceptance Rate) - BSP doesn't track this properly
    tar = 0.0
    
    # Count actual tokens generated (up to EOS for each sequence)
    total_tokens_generated = 0
    for output_text in outputs:
        # Count tokens in the generated text
        # Since BSP's generate() already returns decoded text, we need to re-tokenize to count
        output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
        # Find EOS token if present
        if tokenizer.eos_token_id in output_tokens:
            eos_idx = output_tokens.index(tokenizer.eos_token_id)
            tokens_count = eos_idx + 1  # Include the EOS token
        else:
            tokens_count = len(output_tokens)
        total_tokens_generated += tokens_count
    
    # Calculate tokens per second based on pure decoding time
    tokens_per_second = total_tokens_generated / pure_decoding_time if pure_decoding_time > 0 else 0.0
    
    # Calculate average latency per token in ms
    latency = (pure_decoding_time / total_tokens_generated * 1000) if total_tokens_generated > 0 else 0.0
    
    # Create timing breakdown
    total_time = tokenization_time + pure_decoding_time + post_processing_time
    timing_breakdown = TimingBreakdown(
        tokenization_time=tokenization_time,
        pure_decoding_time=pure_decoding_time,
        post_processing_time=post_processing_time,
        total_time=total_time
    )
    
    # Calculate draft calls (number of speculative generation rounds)
    # Each verify call corresponds to one round of speculation
    draft_calls = total_verify_calls
    verification_calls = total_verify_calls
    
    return outputs, pure_decoding_time, tokens_per_second, tar, latency, timing_breakdown, draft_calls, verification_calls