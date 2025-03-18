#!/bin/bash

PORT=8000
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
CONFIG_FILE=/home/ubuntu/shaotingf/LMCache/config/qmsum.yaml

export TOKENIZERS_PARALLELISM=false
export LMCACHE_USE_EXPERIMENTAL=True

LMCACHE_CONFIG_FILE=$CONFIG_FILE vllm serve $MODEL_NAME --port $PORT --gpu-memory-utilization 0.8 --tensor-parallel-size 1  --trust-remote-code --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}' --disable-log-stats
