#!/bin/bash
set -e

MODEL="/models/Qwen3.5-9B"
CONTAINER_NAME="lmcache-e2e-test"
PORT=8199

echo "=== LMCache CacheBlend ROCm E2E Test ==="
echo "Model: $MODEL"
echo "Port: $PORT"

# Stop any existing test container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Launch vLLM container with our patched LMCache installed
docker run -d \
  --name $CONTAINER_NAME \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=16g \
  -v /mnt/nvme1n1p1/models:/models \
  -v /home/hotaisle/LMCache:/opt/LMCache \
  -p ${PORT}:8000 \
  --entrypoint bash \
  vllm/vllm-openai-rocm:v0.18.0 \
  -c "
    # Install our patched LMCache
    cd /opt/LMCache
    pip install -e '.[vllm]' --no-build-isolation 2>&1 | tail -5
    echo '=== LMCache installed ==='

    # Launch vLLM with LMCache CacheBlend
    python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.9 \
      --port 8000 \
      --host 0.0.0.0 \
      --kv-transfer-config '{\"kv_connector\": \"LMCacheConnector\", \"kv_role\": \"kv_both\"}' \
      --lmcache-config-file /opt/LMCache/lmcache_config.yaml
  "

echo "Container started: $CONTAINER_NAME"
echo "Waiting for server..."
