#!/bin/bash

# ASSUMPTION: lmcache/vllm-openai:latest-nightly Docker image is available

# Overview:
# This script is used to deploy a single vLLM serving engine on port 8000

# Arguments:
MODEL_URL=$1

if [ -z "$MODEL_URL" ]; then
    echo "Usage: $0 <MODEL_URL>"
    echo "Example: $0 deepseek-ai/DeepSeek-V2-Lite"
    exit 1
fi

echo "🚀 Starting single vLLM setup with:"
echo "   Model: $MODEL_URL"
echo "   Port: 8000"

# Utility:
free_port() {
    if [ -z "$1" ]; then
        echo "Usage: free_port <port>"
        return 1
    fi

    local port=$1
    
    echo "🧹 Cleaning up any existing containers on port $port..."
    sudo docker ps -q --filter "publish=$port" | xargs -r sudo docker kill
    sudo docker ps -aq --filter "publish=$port" | xargs -r sudo docker rm
    sleep 2
}

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Clean up port
free_port 8000

# Deploy the vLLM serving engine (without LMCache)
echo "🔧 Starting vLLM serving engine on port 8000..."
CONTAINER_ID=$(sudo docker run -d --runtime=nvidia --gpus all \
    --name vllm-server \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_MLA_DISABLE=0" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    lmcache/vllm-openai:latest-nightly \
    $MODEL_URL \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192)

echo "Started vLLM container: $CONTAINER_ID"

# Check if container started successfully
sleep 10
if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ Container failed to start. Checking logs..."
    sudo docker logs $CONTAINER_ID
    exit 1
fi

# Start 10-minute self-destruct
(sleep 600 && echo "Timeout reached, force killing container..." && sudo docker kill $CONTAINER_ID 2>/dev/null) &
TIMER_PID=$!

# Wait longer for model loading
echo "⏳ Waiting for model to load (this may take a few minutes)..."
sleep 120

# Wait until the vLLM server is ready AND the model is loaded
echo "🔍 Checking server health..."
total_time_elapsed=0
until curl --fail -s http://localhost:8000/health; do
    if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
        echo "❌ vLLM server container exited prematurely"
        sudo docker logs $CONTAINER_ID
        exit 1
    fi
    echo "Waiting for vLLM server to become ready..."
    sleep 10
    total_time_elapsed=$((total_time_elapsed + 10))
done

echo "🔍 Checking if model is loaded..."
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL_URL"; do
    if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
        echo "❌ vLLM server container exited prematurely"
        exit 1
    fi
    echo "Waiting for model $MODEL_URL to be loaded..."
    sleep 10
    echo "--------------------------------"
    echo "Most recent serving engine logs:"
    echo "--------------------------------"
    sudo docker logs --tail 10 $CONTAINER_ID
    echo "--------------------------------"
    total_time_elapsed=$((total_time_elapsed + 10))
done

echo "✅ vLLM serving engine is ready and model is loaded"
echo "🔧 Server: http://localhost:8000"

# Store container ID for cleanup scripts
echo "$CONTAINER_ID" > .vllm-server.pid
echo "$TIMER_PID" > .vllm-timer.pid

echo "✅ Single vLLM setup completed successfully!"
echo "ℹ️  Use 'sudo docker kill $CONTAINER_ID' to stop the container" 