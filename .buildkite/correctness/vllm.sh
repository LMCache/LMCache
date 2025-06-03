#!/bin/bash

# Step 1: Deploy vllm.entrypoints.api_server on port 8000 with deepseek-v2-lite
export MODEL=deepseek-ai/DeepSeek-V2-Lite
export PORT=8000
export VLLM_MLA_DISABLE=0 # enable the MLA optimization of deepseek (compressing KV cache into latent space)

# HF_TOKEN and IMAGE should be set in the environment before running this script
CONTAINER_ID=$(sudo docker run -d --runtime=nvidia --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    lmcache/vllm-openai:latest \
    $MODEL \
    --max-model-len 6000 \
    --port 8000 \
    --trust-remote-code)

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

echo "✅ vLLM server is ready"

# Step 2: Run mmlu_bench.py

mkdir -p mmlu-results

# Verify MMLU data exists
if [ ! -d "data/test" ]; then
    echo "❌ ERROR: MMLU data not found. Expected data/test directory."
    echo "🔄 Attempting to download MMLU data as fallback..."
    bash .buildkite/correctness/download-data.sh

    if [ ! -d "data/test" ]; then
        echo "❌ FATAL: Failed to download MMLU data"
        echo "Current directory: $(pwd)"
        echo "Contents:"
        ls -la
        exit 1
    fi
fi

echo "✅ MMLU data found. Test subjects: $(ls data/test/*.csv | wc -l)"

python3 .buildkite/correctness/mmlu_bench.py \
  --nsub 6 \
  --parallel 16 \
  > mmlu-results/vllm_baseline.txt || true

# Verify result file was created
if [ -f "mmlu-results/vllm_baseline.txt" ]; then
    echo "✅ Result file created: $(wc -l < mmlu-results/vllm_baseline.txt) lines"
    echo "📊 Last few lines of results:"
    tail -3 mmlu-results/vllm_baseline.txt
else
    echo "❌ WARNING: Result file not created"
    echo "📁 Contents of mmlu-results/:"
    ls -la mmlu-results/ || echo "mmlu-results directory not found"
fi

# Step 3: Kill the vLLM server
echo "🛑 Stopping container..."
sudo docker kill $CONTAINER_ID || true
kill $TIMER_PID 2>/dev/null || true

echo "✅ vLLM baseline test completed"

