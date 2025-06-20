# LMCache MMLU Testing Suite

## Overview
Tests LMCache KV transfer correctness vs vLLM baseline using MMLU benchmark.
Compares dense (Llama 3.1 8B) and MLA (DeepSeek V2 Lite) architectures.

## Models Tested
- **Llama 3.1 8B**: Dense attention architecture
- **DeepSeek V2 Lite**: Multi-head Latent Attention (MLA)

## Quick Start
```bash
# Test single model
./deploy-1-vllm.sh "meta-llama/Llama-3.1-8B"
python3 1-mmlu.py --model "meta-llama/Llama-3.1-8B" --number-of-subjects 15

./deploy-2-lmcache.sh "meta-llama/Llama-3.1-8B"  
python3 2-mmlu.py --model "meta-llama/Llama-3.1-8B" --number-of-subjects 15

# Summarize all results
python3 summarize_scores.py
```

## Buildkite Pipeline
```bash
buildkite-agent pipeline upload .buildkite/correctness/pipeline.mmlu.yml
```

Pipeline tests both models with vLLM baseline and LMCache KV transfer (4 total tests).

## Files
- **`deploy-1-vllm.sh`**: Single vLLM engine (port 8000) - for baseline
- **`deploy-2-lmcache.sh`**: Dual LMCache engines (ports 8000/8001) + Redis - for KV transfer
- **`1-mmlu.py`**: MMLU test on single vLLM engine (baseline)
- **`2-mmlu.py`**: MMLU test on dual LMCache engines (KV transfer)
- **`summarize_scores.py`**: Results comparison and analysis

## Architecture
- **Baseline**: Single vLLM → port 8000
- **KV Transfer**: vLLM producer (port 8000) → Redis (port 6379) ← vLLM consumer (port 8001)

## Requirements
- Docker with nvidia runtime
- Redis server  
- HuggingFace token (set `HF_TOKEN` env var)