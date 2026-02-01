# Kimi-K2-Thinking Usage Guide

## Installing vLLM and LMCache 

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install lmcache vllm
```

## Launching Kimi-K2-Thinking on 8xH200

```bash
PYTHONHASHSEED=0 \
vllm serve moonshotai/Kimi-K2-Instruct \
  --tensor-parallel-size 8 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2  \
  --trust-remote-code
```

Note: The smallest deployment unit for Kimi-K2 FP8 weights with 128k seqlen on mainstream H200 or H20 platform is a cluster with 16 GPUs with either Tensor Parallel (TP) or "data parallel + expert parallel" (DP+EP).  