#!/bin/bash

set -euo pipefail

echo "🧪 Testing LMCache (no MLA) with minimal configuration..."

# Export your HF_TOKEN
export HF_TOKEN=<YOUR_HF_TOKEN>
export MODEL=deepseek-ai/DeepSeek-V2-Lite
export VLLM_MLA_DISABLE=1

# Kill any existing containers
sudo docker ps -q | xargs -r sudo docker kill || true

echo "🚀 Starting LMCache container..."
CONTAINER_ID=$(sudo docker run -d --runtime=nvidia --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "VLLM_MLA_DISABLE=1" \
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

# Wait for container to start
sleep 10

# Check if container is running
if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ Container exited prematurely. Logs:"
    sudo docker logs $CONTAINER_ID
    exit 1
fi

echo "✅ Container is running. Waiting for server to be ready..."

# Wait for server to be ready (with timeout)
timeout=300
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ LMCache server is ready!"

        # Test a simple request
        echo "🧪 Testing server with a simple request..."
        curl -s -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "deepseek-ai/DeepSeek-V2-Lite",
                "prompt": "What is 2+2?",
                "max_tokens": 10,
                "temperature": 0
            }' | jq .

        echo "✅ LMCache test completed successfully!"
        sudo docker kill $CONTAINER_ID
        exit 0
    fi

    if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
        echo "❌ Container exited during startup. Logs:"
        sudo docker logs $CONTAINER_ID
        exit 1
    fi

    echo "Waiting for server... ($elapsed/$timeout seconds)"
    sleep 10
    elapsed=$((elapsed + 10))
done

echo "❌ Timeout waiting for server to be ready"
sudo docker logs $CONTAINER_ID
sudo docker kill $CONTAINER_ID
exit 1