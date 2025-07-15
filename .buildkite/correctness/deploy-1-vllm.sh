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
    
    # Kill any processes using the port directly
    sudo lsof -ti:$port | xargs -r sudo kill -9 2>/dev/null || true
    
    # Kill and remove containers using this port
    sudo docker ps -q --filter "publish=$port" | xargs -r sudo docker kill 2>/dev/null || true
    sudo docker ps -aq --filter "publish=$port" | xargs -r sudo docker rm -f 2>/dev/null || true
    
    # Wait a moment for port to be released
    sleep 3
    
    # Verify port is free
    if sudo lsof -i:$port >/dev/null 2>&1; then
        echo "⚠️ Port $port still in use after cleanup, waiting longer..."
        sleep 5
        sudo lsof -ti:$port | xargs -r sudo kill -9 2>/dev/null || true
        sleep 2
    fi
    
    # Final check
    if sudo lsof -i:$port >/dev/null 2>&1; then
        echo "❌ Failed to free port $port"
        echo "🔍 Processes still using port $port:"
        sudo lsof -i:$port || true
        return 1
    else
        echo "✅ Port $port is now free"
    fi
}

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# Clean up port
free_port 8000
if [ $? -ne 0 ]; then
    echo "❌ Failed to free port 8000, cannot continue"
    exit 1
fi

# Clean up container by name (in case it exists but isn't bound to port)
echo "🧹 Cleaning up any existing vLLM containers..."
sudo docker rm -f vllm-server 2>/dev/null || true
sudo docker rm -f lmcache-producer 2>/dev/null || true
sudo docker rm -f lmcache-consumer 2>/dev/null || true

# Deploy the vLLM serving engine (without LMCache)
echo "🔧 Starting vLLM serving engine on port 8000..."
CONTAINER_ID=$(sudo docker run -d --gpus all \
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

# MODEL="meta-llama/Llama-3.1-8B"
# MAX_MODEL_LEN=6000
# VLLM_MLA_DISABLE=0
# sudo docker run -d --runtime=nvidia --gpus all \
#     --env "HF_TOKEN=$HF_TOKEN" \
#     --env "LMCACHE_USE_EXPERIMENTAL=True" \
#     --env "LMCACHE_CHUNK_SIZE=256" \
#     --env "TORCH_USE_CUDA_DSA=1" \
#     --env "LMCACHE_LOCAL_CPU=True" \
#     --env "LMCACHE_MAX_LOCAL_CPU_SIZE=1.0" \
#     --env "LMCACHE_REMOTE_SERDE=naive" \
#     --env "CUDA_VISIBLE_DEVICES=0" \
#     --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
#     --env "VLLM_MLA_DISABLE=$VLLM_MLA_DISABLE" \
#     --env "CUDA_LAUNCH_BLOCKING=1" \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -p 8000:8000 \
#     lmcache/vllm-openai:latest-nightly \
#     $MODEL \
#     --max-model-len $MAX_MODEL_LEN \
#     --port 8000 \
#     --trust-remote-code \
#     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# MODEL="meta-llama/Llama-3.1-8B"
# MAX_MODEL_LEN=6000
# VLLM_MLA_DISABLE=0
# sudo docker run -d --runtime=nvidia --gpus all \
#     --env "HF_TOKEN=$HF_TOKEN" \
#     --env "LMCACHE_USE_EXPERIMENTAL=True" \
#     --env "LMCACHE_CHUNK_SIZE=256" \
#     --env "LMCACHE_LOCAL_CPU=True" \
#     --env "LMCACHE_MAX_LOCAL_CPU_SIZE=1.0" \
#     --env "LMCACHE_REMOTE_SERDE=naive" \
#     --env "CUDA_VISIBLE_DEVICES=0" \
#     --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
#     --env "VLLM_MLA_DISABLE=$VLLM_MLA_DISABLE" \
#     --env "CUDA_LAUNCH_BLOCKING=1" \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -p 8000:8000 \
#     lmcache/vllm-openai:latest-nightly \
#     $MODEL \
#     --max-model-len $MAX_MODEL_LEN \
#     --port 8000 \
#     --trust-remote-code \
#     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

echo "Started vLLM container: $CONTAINER_ID"

# Check if container started successfully
sleep 10
if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
    echo "❌ Container failed to start. Checking logs..."
    sudo docker logs $CONTAINER_ID
    exit 1
fi

# Wait longer for model loading
echo "⏳ Waiting for model to load (this may take a few minutes)..."
echo "📊 Monitoring container status and logs..."

# Wait with periodic status updates and early health checks
elapsed_seconds=0
early_health_success=false
while true; do
    elapsed_seconds=$((elapsed_seconds + 10))
    echo ""
    echo "🕐 Model loading progress: ${elapsed_seconds} seconds elapsed"
    
    # Check if container is still running
    if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
        echo "❌ vLLM container stopped unexpectedly!"
        echo "📋 Container logs:"
        sudo docker logs --tail 20 $CONTAINER_ID
        exit 1
    fi
    
    # Check health endpoint every 20 seconds after 30 seconds have elapsed
    if (( elapsed_seconds >= 30 && elapsed_seconds % 20 == 0 )); then
        echo "🔍 Early health check: testing http://localhost:8000/health"
        if curl --fail -s --max-time 5 http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Server responded to health check early! Breaking out of wait loop."
            early_health_success=true
            break
        else
            echo "⏳ Health check failed, continuing to wait..."
        fi
    fi
    
    # Show recent logs every 30 seconds
    if (( elapsed_seconds % 30 == 0 )); then
        echo "📋 Recent vLLM server logs:"
        sudo docker logs --tail 5 $CONTAINER_ID | sed 's/^/  /'
    fi
    
    # Break after 180 seconds (3 minutes) to avoid infinite loop
    if (( elapsed_seconds >= 180 )); then
        echo "⚠️ Model loading taking longer than expected (${elapsed_seconds}s), continuing to health checks..."
        break
    fi
    
    sleep 10
done

echo "✅ Model loading wait period completed"

# Wait until the vLLM server is ready AND the model is loaded
if [ "$early_health_success" = true ]; then
    echo "✅ Server is already healthy (confirmed during early checks)!"
else
    echo "🔍 Checking server health..."
    echo "📡 Testing health endpoint: http://localhost:8000/health"
    total_time_elapsed=0
    health_check_count=0
    until curl --fail -s http://localhost:8000/health; do
        health_check_count=$((health_check_count + 1))
        echo "⏳ Health check attempt ${health_check_count}: server not ready yet..."
        
        if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
            echo "❌ vLLM server container exited prematurely"
            sudo docker logs $CONTAINER_ID
            exit 1
        fi
        sleep 10
        total_time_elapsed=$((total_time_elapsed + 10))
        
        # Show recent logs every 60 seconds during health checks
        if (( health_check_count % 6 == 0 )); then
            echo "📋 Recent container logs (health check debugging):"
            sudo docker logs --tail 3 $CONTAINER_ID | sed 's/^/  /'
        fi
    done

    echo "✅ Server is healthy!"
fi

echo "🔍 Checking if model is loaded..."
echo "📡 Testing model endpoint for: $MODEL_URL"
model_check_count=0
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL_URL"; do
    model_check_count=$((model_check_count + 1))
    echo "⏳ Model check attempt ${model_check_count}: $MODEL_URL not loaded yet..."
    
    if ! sudo docker ps -q --filter "id=$CONTAINER_ID" | grep -q .; then
        echo "❌ vLLM server container exited prematurely"
        exit 1
    fi
    sleep 10
    total_time_elapsed=$((total_time_elapsed + 10))
    
    # Show detailed logs every 30 seconds during model loading
    if (( model_check_count % 3 == 0 )); then
        echo "📋 Model loading progress logs:"
        sudo docker logs --tail 5 $CONTAINER_ID | sed 's/^/  /'
        echo "--------------------------------"
    fi
done

echo "✅ vLLM serving engine is ready and model is loaded"
echo "🔧 Server: http://localhost:8000"

# Store container ID for cleanup scripts
echo "$CONTAINER_ID" > .vllm-server.pid

echo "✅ Single vLLM setup completed successfully!"
echo "ℹ️  Use 'sudo docker kill $CONTAINER_ID' to stop the container" 