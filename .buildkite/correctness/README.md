# LMCache MMLU Testing Suite

## Overview
Tests LMCache KV transfer correctness using dual-engine setup with MMLU benchmark.

## Quick Start
```bash
# Single vLLM baseline
./deploy-1-vllm.sh "deepseek-ai/DeepSeek-V2-Lite"

# Dual LMCache KV transfer setup  
./deploy-2-lmcache.sh "deepseek-ai/DeepSeek-V2-Lite"

# Run tests
python3 baseline-mmlu.py --model MODEL --number-of-subjects 12 --result-file baseline.txt
python3 1-mmlu.py --model MODEL --number-of-subjects 12
python3 2-mmlu.py --model MODEL --number-of-subjects 12

# Summarize results
python3 summarize_scores.py
```

## Files
- **`deploy-1-vllm.sh`**: Single vLLM engine (port 8000)
- **`deploy-2-lmcache.sh`**: Dual LMCache engines (ports 8000/8001) + Redis
- **`baseline-mmlu.py`**: Standard MMLU test (single engine)
- **`1-mmlu.py`**: KV transfer test variant 1
- **`2-mmlu.py`**: KV transfer test variant 2
- **`summarize_scores.py`**: Results aggregation (handles .txt + .jsonl)
- **`pipeline.mmlu.yml`**: Buildkite CI pipeline

## Architecture
- **Single Engine**: vLLM → port 8000
- **Dual Engine**: vLLM producer (port 8000) → Redis (port 6379) ← vLLM consumer (port 8001)

## Requirements
- Docker with nvidia runtime
- Redis server
- HuggingFace token (set `HF_TOKEN` env var)