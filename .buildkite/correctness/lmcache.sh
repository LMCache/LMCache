#!/bin/bash

# Step 1: Deploy vllm.entrypoints.api_server on port 8000 with deepseek-v2-lite
export MODEL=deepseek-ai/DeepSeek-V2-Lite
export PORT=8000
export VLLM_MLA_DISABLE=0 # enable the MLA optimization of deepseek (compressing KV cache into latent space)

# HF_TOKEN and IMAGE should be set in the environment before running this script
CONTAINER_ID=$(sudo docker run -d --runtime=nvidia --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_LOCAL_CPU=True" \
    --env "LMCACHE_MAX_LOCAL_CPU_SIZE=40" \
    --env "LMCACHE_REMOTE_SERDE=naive" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    lmcache/vllm-openai:latest \
    $MODEL \
    --max-model-len 6000 \
    --port 8000 \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')

echo "Started container: $CONTAINER_ID"

# Start 10-minute self-destruct
(sleep 600 && echo "Timeout reached, force killing container..." && sudo docker kill $CONTAINER_ID) &
TIMER_PID=$!

sleep 60
# Wait until the vLLM server is ready
until curl --fail http://localhost:8000/health; do
  if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ vLLM server container exited prematurely"
    exit 1
  fi
  echo "Waiting for vLLM server to become ready..."
  sleep 5
done

echo "✅ LMCache server is ready"

# Step 2: Run mmlu_bench.py

mkdir -p mmlu-results

python3 .buildkite/correctness/mmlu_bench.py \
  --nsub 6 \
  --parallel 16 \
  > mmlu-results/lmcache_mla.txt || true

# Step 3: Kill the vLLM server
echo "🛑 Stopping container..."
sudo docker kill $CONTAINER_ID || true
kill $TIMER_PID 2>/dev/null || true

echo "✅ LMCache with MLA test completed"

