#!/bin/bash
set -euxo pipefail

eval "$(conda shell.bash hook)"
conda activate buildkite

MODEL=deepseek-ai/DeepSeek-V2-Lite
PORT=8000

LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=1 \
VLLM_USE_V1=0 \
LMCACHE_CONFIG_FILE=.buildkite/mmlu_scripts/lmc-cpu.yaml \
python3 -m vllm.entrypoints.api_server \
  --model $MODEL \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}' &
SERVER_PID=$!

# Wait until the vLLM server is ready
until curl --fail http://localhost:8000/health; do
  if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ vLLM server process exited prematurely"
    exit 1
  fi
  echo "Waiting for vLLM server to become ready..."
  sleep 2
done

mkdir mmlu-results
python3 .buildkite/mmlu_scripts/mmlu_bench.py \
  --nsub 6 \
  --parallel 16 \
  > mmlu-results/v0_lmcache_deepseek2_no_mla.txt || true

kill $SERVER_PID || true
