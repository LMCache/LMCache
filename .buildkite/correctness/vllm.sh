#!/bin/bash

# Generalized vLLM baseline script
# Usage: ./vllm.sh <model> <output_file> [max_model_len] [mla_disable]
# Example: ./vllm.sh "deepseek-ai/DeepSeek-V2-Lite" "vllm_baseline_mla.txt" 6000 0
# Example: ./vllm.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "vllm_baseline_dense.txt" 12000 1

MODEL=${1:-"deepseek-ai/DeepSeek-V2-Lite"}
OUTPUT_FILE=${2:-"vllm_baseline.txt"}
MAX_MODEL_LEN=${3:-6000}
VLLM_MLA_DISABLE=${4:-0}

export PORT=8000

echo "🚀 Starting vLLM baseline test with:"
echo "   Model: $MODEL"
echo "   Output: $OUTPUT_FILE"
echo "   Max model length: $MAX_MODEL_LEN"
echo "   MLA disabled: $VLLM_MLA_DISABLE"

# Clean up any existing containers on port 8000
echo "🧹 Cleaning up any existing containers on port 8000..."
sudo docker ps -q --filter "publish=8000" | xargs -r sudo docker kill
sudo docker ps -aq --filter "publish=8000" | xargs -r sudo docker rm
sleep 5

# HF_TOKEN and IMAGE should be set in the environment before running this script
CONTAINER_ID=$(sudo docker run -d --runtime=nvidia --gpus all \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_MLA_DISABLE=$VLLM_MLA_DISABLE" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    lmcache/vllm-openai:latest \
    $MODEL \
    --max-model-len $MAX_MODEL_LEN \
    --port 8000 \
    --trust-remote-code)

echo "Started container: $CONTAINER_ID"

# Check if container started successfully
sleep 5
if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ Container failed to start. Checking logs..."
    sudo docker logs $CONTAINER_ID
    exit 1
fi

# Start 10-minute self-destruct
(sleep 600 && echo "Timeout reached, force killing container..." && sudo docker kill $CONTAINER_ID) &
TIMER_PID=$!

# Wait longer for model loading
echo "⏳ Waiting for model to load (this may take a few minutes)..."
sleep 120

# Wait until the vLLM server is ready AND the model is loaded
echo "🔍 Checking server health..."
until curl --fail http://localhost:8000/health; do
  if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ vLLM server container exited prematurely"
    exit 1
  fi
  echo "Waiting for vLLM server to become ready..."
  sleep 10
done

echo "🔍 Checking if model is loaded..."
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL"; do
  if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ vLLM server container exited prematurely"
    exit 1
  fi
  echo "Waiting for model $MODEL to be loaded..."
  sleep 10
done

echo "✅ vLLM server is ready and model is loaded"

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
  --nsub 12 \
  --parallel 1 \
  --debug \
  --model "$MODEL" \
  > mmlu-results/$OUTPUT_FILE || true

# Verify result file was created
if [ -f "mmlu-results/$OUTPUT_FILE" ]; then
    echo "✅ Result file created: $(wc -l < mmlu-results/$OUTPUT_FILE) lines"
    echo "📊 Last few lines of results:"
    tail -3 mmlu-results/$OUTPUT_FILE
else
    echo "❌ WARNING: Result file not created"
    echo "📁 Contents of mmlu-results/:"
    ls -la mmlu-results/ || echo "mmlu-results directory not found"
fi

# Step 3: Kill the vLLM server
echo "🛑 Stopping container..."
sudo docker kill $CONTAINER_ID || true
kill $TIMER_PID 2>/dev/null || true

echo "✅ vLLM baseline test completed for $MODEL"

