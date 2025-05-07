#!/bin/bash

# Default values
PORT=8000
CONFIG_FILE=../config/qmsum.yaml
LOG_FILE=10.log

# Parse -p|--port, -c|--config and -l|--log arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -l|--log)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

export TOKENIZERS_PARALLELISM=false
export LMCACHE_USE_EXPERIMENTAL=True

LMCACHE_CONFIG_FILE=$CONFIG_FILE vllm serve $MODEL_NAME --port $PORT --max-model-len 50000 --tensor-parallel-size 1  --trust-remote-code --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}' --disable-log-stats --enable-chunked-prefill=False 2>&1 | tee "$LOG_FILE"
