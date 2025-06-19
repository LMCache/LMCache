#!/bin/bash

# ASSUMPTION: lmcache/vllm-openai:latest-nightly Docker image is available

# Overview:
# This script is used to deploy 2 vLLM + LMCache serving engines on port 8000 and 8001
# They will have a peer to peer connection through a Redis server on port 6379
# The purpose is to send requests to the first serving engine to store KV Caches and then send requests to the second serving engine to retrieve KV Caches
# This way the responses returned by the second serving engine can be used to test the correctness of LMCache KV Transfer

# Arguments:
MODEL_URL=$1

if [ -z "$MODEL_URL" ]; then
    echo "Usage: $0 <MODEL_URL>"
    echo "Example: $0 deepseek-ai/DeepSeek-V2-Lite"
    exit 1
fi

echo "🚀 Starting dual LMCache setup with:"
echo "   Model: $MODEL_URL"
echo "   Producer port: 8000"
echo "   Consumer port: 8001"

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

# Clean up ports
free_port 8000
free_port 8001

# Install and start Redis server
echo "🔧 Installing Redis server..."
sudo apt-get update -qq
sudo apt-get install -y redis-server

echo "🔧 Starting Redis server on port 6379..."
sudo systemctl stop redis-server 2>/dev/null || true
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Wait for Redis server to be ready
echo "⏳ Waiting for Redis server to be ready..."
sleep 5

# Test Redis connection
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis server failed to start"
    exit 1
fi
echo "✅ Redis server is running and responding to ping"

# Deploy the first vLLM + LMCache serving engine on port 8000 (KV producer)
echo "🔧 Starting KV producer on port 8000..."
PRODUCER_ID=$(sudo docker run -d --runtime=nvidia --gpus '"device=0"' \
    --name lmcache-producer \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_REMOTE_URL=redis://host.docker.internal:6379" \
    --env "LMCACHE_REMOTE_SERDE=naive" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_MLA_DISABLE=0" \
    --add-host=host.docker.internal:host-gateway \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    lmcache/vllm-openai:latest-nightly \
    $MODEL_URL \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}')

echo "Started KV producer container: $PRODUCER_ID"

# Deploy the second vLLM + LMCache serving engine on port 8001 (KV consumer)
echo "🔧 Starting KV consumer on port 8001..."
CONSUMER_ID=$(sudo docker run -d --runtime=nvidia --gpus '"device=1"' \
    --name lmcache-consumer \
    --env "HF_TOKEN=$HF_TOKEN" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_REMOTE_URL=redis://host.docker.internal:6379" \
    --env "LMCACHE_REMOTE_SERDE=naive" \
    --env "CUDA_VISIBLE_DEVICES=0" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "VLLM_MLA_DISABLE=0" \
    --add-host=host.docker.internal:host-gateway \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8001 \
    lmcache/vllm-openai:latest-nightly \
    $MODEL_URL \
    --port 8001 \
    --trust-remote-code \
    --max-model-len 8192 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}')

echo "Started KV consumer container: $CONSUMER_ID"

# Check if containers started successfully
sleep 10
if ! sudo docker ps -q --filter "id=$PRODUCER_ID" | grep -q .; then
    echo "❌ Producer container failed to start. Checking logs..."
    sudo docker logs $PRODUCER_ID
    exit 1
fi

if ! sudo docker ps -q --filter "id=$CONSUMER_ID" | grep -q .; then
    echo "❌ Consumer container failed to start. Checking logs..."
    sudo docker logs $CONSUMER_ID
    exit 1
fi

# Start 20-minute self-destruct for containers
(sleep 1200 && echo "Timeout reached, force killing containers..." && \
    sudo docker kill $PRODUCER_ID $CONSUMER_ID 2>/dev/null) &
TIMER_PID=$!

# Wait longer for model loading
echo "⏳ Waiting for models to load (this may take several minutes)..."
sleep 180

# Wait for both serving engines to be ready
echo "🔍 Checking server health..."
total_time_elapsed=0
until curl --fail -s http://localhost:8000/health && curl --fail -s http://localhost:8001/health; do
    if ! sudo docker ps -q --filter "id=$PRODUCER_ID" | grep -q .; then
        echo "❌ Producer container exited prematurely"
        sudo docker logs $PRODUCER_ID
        exit 1
    fi
    if ! sudo docker ps -q --filter "id=$CONSUMER_ID" | grep -q .; then
        echo "❌ Consumer container exited prematurely"
        sudo docker logs $CONSUMER_ID
        exit 1
    fi
    echo "Waiting for servers to become ready..."
    sleep 10
    total_time_elapsed=$((total_time_elapsed + 10))
done

echo "🔍 Checking if models are loaded..."
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL_URL" && curl --fail -s http://localhost:8001/v1/models | grep -q "$MODEL_URL"; do
    if ! sudo docker ps -q --filter "id=$PRODUCER_ID" | grep -q .; then
        echo "❌ Producer container exited prematurely"
        exit 1
    fi
    if ! sudo docker ps -q --filter "id=$CONSUMER_ID" | grep -q .; then
        echo "❌ Consumer container exited prematurely"
        exit 1
    fi
    echo "Waiting for model $MODEL_URL to be loaded on both engines..."
    sleep 10
    echo "--------------------------------"
    echo "Most recent producer (port 8000) logs:"
    echo "--------------------------------"
    sudo docker logs --tail 10 $PRODUCER_ID
    echo "--------------------------------"
    echo "Most recent consumer (port 8001) logs:"
    echo "--------------------------------"
    sudo docker logs --tail 10 $CONSUMER_ID
    echo "--------------------------------"
    total_time_elapsed=$((total_time_elapsed + 10))
done

echo "✅ Both LMCache serving engines are ready and models are loaded"
echo "🔧 Producer (KV storage): http://localhost:8000"
echo "🔧 Consumer (KV retrieval): http://localhost:8001"
echo "🔧 Redis server: localhost:6379"

# Store container IDs for cleanup scripts
echo "$PRODUCER_ID" > .lmcache-producer.pid  
echo "$CONSUMER_ID" > .lmcache-consumer.pid
echo "$TIMER_PID" > .lmcache-timer.pid

echo "✅ Dual LMCache setup completed successfully!"
echo "ℹ️  Use 'sudo docker kill $PRODUCER_ID $CONSUMER_ID' to stop containers"
echo "ℹ️  Use 'sudo systemctl stop redis-server' to stop Redis" 