

# Batch Speculative Decoding

For the paper: Batch Speculative Decoding Done Right

## Environment Setup
```bash
pip install -r requirements.txt
# make sure you use transformers==4.51.3. They later redesign the KV cache API and it won't work.
# Also for sglang use the requirements_sglang.txt"

```

## Key Files

**Core Implementation**
- `src/custom/batch_speculative.py` - Main batch speculative decoding with unpad-update-repad pattern
- `src/custom/cross_batch_speculative.py` - Cross-batch scheduling implementation
- `src/custom/verification.py` - Prefill-refill oracle verification system

**Testing & Validation**
- `test_spec_decode_correctness.py` - Comprehensive correctness testing suite. There are many tests, find ones you need, turn them on, to compare different spec with non-spec outputs.


---


## How To Run

Inference: 
```bash
# EXSPEC 
CUDA_VISIBLE_DEVICES=0 python unified_benchmark.py --methods Ours-XBatch --input_file data/spec_bench/question.jsonl --num_prompts 100 --max_new_tokens 128 --n_draft_tokens 5 --batch_size 16 --window_size 48 --scheduling_strategy cross_batch --sort_by_length


# EQSPEC 
CUDA_VISIBLE_DEVICES=0 python unified_benchmark.py --methods Ours-Batch-Cache --input_file data/spec_bench/question.jsonl --num_prompts 100 --max_new_tokens 128 --n_draft_tokens 5 --batch_size 16 --enable_profiling

```
verification experiment:
```bash
# you may edit methods_batch_n and methods_batch_1 in verification_benchmark.py to add more methods to compare
python verification_benchmark.py --input_file data/spec_bench/question.jsonl --num_prompts 480 --models glm4 --batch_sizes 4 8 --max_new_tokens 50 --output_dir test_verification
```

---
## Bibtex

```bibtex
@misc{zhang2025batchspeculativedecodingright,
      title={Batch Speculative Decoding Done Right}, 
      author={Ranran Haoran Zhang and Soumik Dey and Ashirbad Mishra and Hansi Wu and Binbin Li and Rui Zhang},
      year={2025},
      eprint={2510.22876},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.22876}, 
}
```



## License

This project is licensed under the terms of the [Apache 2.0 License](LICENSE).