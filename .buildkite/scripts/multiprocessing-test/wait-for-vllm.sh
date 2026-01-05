#!/bin/bash
# Wait for vLLM server to be ready
set -e

VLLM_PORT="${VLLM_PORT:-8000}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
VLLM_CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-mp-test}"

echo "=== Waiting for vLLM to be ready ==="
echo "Port: $VLLM_PORT"
echo "Max wait time: ${MAX_WAIT_SECONDS}s"

start_time=$(date +%s)
end_time=$((start_time + MAX_WAIT_SECONDS))

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ "$current_time" -ge "$end_time" ]; then
        echo "❌ Timeout: vLLM did not become ready within ${MAX_WAIT_SECONDS} seconds"
        echo ""
        echo "=== vLLM container logs ==="
        docker logs --tail 100 "$VLLM_CONTAINER_NAME" 2>&1 || true
        exit 1
    fi
    
    # Check if container is still running
    if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER_NAME}$"; then
        echo "❌ vLLM container is not running"
        echo ""
        echo "=== vLLM container logs ==="
        docker logs "$VLLM_CONTAINER_NAME" 2>&1 || true
        exit 1
    fi
    
    # Check if vLLM health endpoint is responding
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "✅ vLLM is ready! (took ${elapsed}s)"
        exit 0
    fi
    
    # Also try the models endpoint as a fallback
    if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "✅ vLLM is ready! (took ${elapsed}s)"
        exit 0
    fi
    
    echo "Waiting for vLLM... (${elapsed}s elapsed)"
    sleep 5
done

