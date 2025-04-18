#!/bin/bash

PORT=8000
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
CONFIG_FILE=/home/ubuntu/ShaotingFS2/LMCache/config/qmsum.yaml

export TOKENIZERS_PARALLELISM=false
export LMCACHE_USE_EXPERIMENTAL=True

LMCACHE_CONFIG_FILE=$CONFIG_FILE vllm serve $MODEL_NAME --port $PORT --gpu-memory-utilization 0.95 --tensor-parallel-size 1  --trust-remote-code --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}' --disable-log-stats --enable-chunked-prefill=False
