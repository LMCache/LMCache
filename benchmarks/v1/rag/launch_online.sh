#!/usr/bin/env bash

# Parse command line arguments
MODE="both"  # Default mode
if [ $# -gt 0 ]; then
    MODE="$1"
fi

# Validate mode
if [[ "$MODE" != "server" && "$MODE" != "client" && "$MODE" != "both" ]]; then
    echo "Usage: $0 [server|client|both]"
    echo "  server: Start only the vLLM server"
    echo "  client: Run only the RAG benchmark client"
    echo "  both:   Start server and run client (default)"
    exit 1
fi

# Configuration for online RAG benchmark
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LMCache_CONFIG_FILE_PATH="blending.yaml"
DATASET_PATH=$HOME/datasets/musique_s.json
PROMPT_BUILD_METHOD=QA
MAX_TOKENS=32
TEMPERATURE=0.0
API_BASE="http://localhost:8200/v1"
API_KEY="dummy-key"
SERVER_PORT=8200
DATASET_NAME=$(echo $DATASET_PATH | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}')
OUTPUT_FILE="$DATASET_NAME"_online_rag.csv

# Function to start vLLM server
start_vllm_server() {
    local wait_for_ready=${1:-true}  # Default to waiting for server ready
    
    echo "Starting vLLM server on port $SERVER_PORT..."
    
    # Start vLLM server with OpenAI-compatible API
    # without tranfer (not PD disaggreated)
    # cache blending
    LMCACHE_CONFIG_FILE=$LMCache_CONFIG_FILE_PATH vllm serve "$MODEL_NAME" \
        --port $SERVER_PORT \
        --host localhost \
        --tokenizer-mode mistral \
        --api-key "$API_KEY" \
        --disable-log-requests \
        --max-model-len 8000 \
        --gpu-memory-utilization 0.8 \
        --no-enable-prefix-caching \
        --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}' \
        --served-model-name "$MODEL_NAME" &
    
    # without tranfer (PD disaggregated)
    # baseline

    # without tranfer (PD disaggreate)
    # prefix caching

    # without tranfer (PD disaggregated)
    # full re-use
    SERVER_PID=$!
    echo "vLLM server started with PID: $SERVER_PID"
    
    # Only wait for server to be ready if requested (not in server-only mode)
    if [ "$wait_for_ready" = "true" ]; then
        echo "Waiting for server to be ready..."
        for i in {1..600}; do
            if curl -s "$API_BASE/models" > /dev/null 2>&1; then
                echo "Server is ready!"
                break
            fi
            if [ "$i" -eq 600 ]; then
                echo "Server failed to start within 600 seconds"
                kill $SERVER_PID 2>/dev/null
                exit 1
            fi
            sleep 1
            echo -n "."
        done
    else
        echo "Server is starting in background. Use 'curl $API_BASE/models' to check readiness."
    fi
}

# Function to run RAG benchmark client
run_client() {
    echo "Running online RAG benchmark client..."
    
    # Run online RAG benchmark using the new --online flag
    LMCACHE_CONFIG_FILE=$LMCache_CONFIG_FILE_PATH python3 rag.py \
        --online \
        --start-index 0 \
        --end-index 20 \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_PATH" \
        --prompt-build-method $PROMPT_BUILD_METHOD \
        --max-tokens $MAX_TOKENS \
        --temperature $TEMPERATURE \
        --openai-api-base "$API_BASE" \
        --openai-api-key "$API_KEY" \
        --output "$OUTPUT_FILE" \
        --verbose
    
    echo "----------------------------------------"
    echo "Online benchmark completed. Results saved to $OUTPUT_FILE"
}

# Function to stop vLLM server
stop_vllm_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        echo "Server stopped."
    fi
}

# Set up trap for server mode
if [[ "$MODE" == "server" || "$MODE" == "both" ]]; then
    trap stop_vllm_server EXIT
fi

echo "Starting online RAG benchmark in '$MODE' mode..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "API Base: $API_BASE"
echo "Output: $OUTPUT_FILE"
echo "----------------------------------------"

# Execute based on mode
case "$MODE" in
    "server")
        echo "Starting server only..."
        start_vllm_server false  # Don't wait for readiness in server-only mode
        echo "Server is running in background. Press Ctrl+C to stop."
        echo "You can check server readiness with: curl $API_BASE/models"
        # Keep the script running
        while true; do
            sleep 1
        done
        ;;
    "client")
        echo "Running client only (assuming server is already running)..."
        run_client
        ;;
    "both")
        echo "Starting server and running client..."
        start_vllm_server true   # Wait for readiness before running client
        run_client
        # Server will be stopped automatically by the trap
        ;;
esac