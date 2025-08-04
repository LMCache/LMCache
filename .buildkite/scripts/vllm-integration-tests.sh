#!/usr/bin/bash
#
# This test script runs integration tests for the LMCache integration with vLLM.
# A lmcache/vllm-openai container image is built by this script from the LMCache code base 
# the script is running from and the latest nightly build of vLLM. It is therefore using the
# latest of both code bases to build the image which it then performs tests on.
#
# It is laid out as follows:
# - UTILITIES:  utility functions
# - TESTS:      test functions
# - SETUP:      environment setup steps
# - MAIN:       test execution steps
#
# It requires the following to be installed to run:
# - curl
# - docker engine (daemon running)
# - NVIDIA Container Toolkit:
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
# Note: The script should be run from the LMCache code base root.
# Note: L4 CI runners cannot use Flash Infer

set -ex
trap 'cleanup $?' EXIT

CID=
HF_TOKEN=
SERVER_WAIT_TIMEOUT=180
PORT=

#############
# UTILITIES #
#############

cleanup() {
    local code="${1:-0}"

    echo "→ Cleaning up Docker container and port..."
    if [[ -n "${CID:-}" ]]; then
        docker kill "$CID" &>/dev/null || true
        docker rm "$CID" &>/dev/null || true
    fi

    if [[ -n "${PORT:-}" ]]; then
        fuser -k "${PORT}/tcp" &>/dev/null || true
    fi
}

find_available_port() {
    local start_port=${1:-8000}
    local port=$start_port
    
    while [ $port -lt 65536 ]; do
        # Check if port is available using netstat
        if ! netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            # Double-check by trying to bind to the port with nc
            if timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
                # Port is in use, try next one
                ((port++))
                continue
            else
                # Port is available
                echo $port
                return 0
            fi
        fi
        ((port++))
    done
    
    echo "ERROR: No available ports found starting from $start_port" >&2
    return 1
}

build_lmcache_vllmopenai_image() {
    cp example_build.sh test-build.sh
    chmod 755 test-build.sh 
    ./test-build.sh
}

wait_for_openai_api_server(){
    if ! timeout $SERVER_WAIT_TIMEOUT bash -c "
        until curl 127.0.0.1:${PORT}/v1/models |grep '\"id\":\"meta-llama/Llama-3.2-1B-Instruct\"'; do
            echo 'waiting for OpenAI API server to start'
            sleep 30
        done
    "; then
        echo "OpenAI API server did not start"
        docker logs $CID
        return 1
    fi
}

run_lmcache_vllmopenai_container() {
    # Pick the GPU with the largest free memory
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" $PORT
    best_gpu="${CUDA_VISIBLE_DEVICES}"
    
    if [ -z "$HF_TOKEN" ]; then
        CID=$(docker run -d --runtime nvidia --gpus "device=${best_gpu}" \
            --env VLLM_USE_FLASHINFER_SAMPLER=0 \
            --env "LMCACHE_CHUNK_SIZE=256" \
            --env "LMCACHE_LOCAL_CPU=True" \
            --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
            --volume ~/.cache/huggingface:/root/.cache/huggingface \
            --network host \
            'lmcache/vllm-openai:build-latest' \
            'meta-llama/Llama-3.2-1B-Instruct' --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            --max-model-len 1024 \
            --gpu-memory-utilization '0.3' \
            --enforce-eager \
            --port $PORT)
    else
        CID=$(docker run -d --runtime nvidia --gpus "device=${best_gpu}" \
            --env VLLM_USE_FLASHINFER_SAMPLER=0 \
            --env HF_TOKEN=$HF_TOKEN \
            --env "LMCACHE_CHUNK_SIZE=256" \
            --env "LMCACHE_LOCAL_CPU=True" \
            --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
            --volume ~/.cache/huggingface:/root/.cache/huggingface \
            --network host \
            'lmcache/vllm-openai:build-latest' \
            'meta-llama/Llama-3.2-1B-Instruct' --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            --max-model-len 1024 \
            --gpu-memory-utilization '0.3' \
            --enforce-eager \
            --port $PORT)
    fi
    buildkite-agent meta-data set "docker-CID" "$CID"

    wait_for_openai_api_server

    LOGFILE="/tmp/vllm_${CID}.log"
    docker logs -f "$CID" &> "$LOGFILE" &
    LOG_PID=$!

    set +x
    end=$((SECONDS + 120))
    while [ $SECONDS -lt $end ]; do
        if grep -qi 'Starting vLLM API server' "$LOGFILE"; then
            echo "vLLM API server started."
            kill $LOG_PID
            break
        fi
        sleep 1
    done
    set -x

    if [ $SECONDS -ge $end ]; then
        echo "Timeout waiting for startup marker, dumping full log:"
        cat "$LOGFILE"
        kill $LOG_PID
        return 1
    fi

}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo " "
    echo "Options:"
    echo "  --hf-token|-hft              HuggingFace access token for downloading model(s)"
    echo "  --server-wait-timeout|-swt   Wait time in seconds for vLLM OpenAI server to start"
    echo "  --help|-h                    Print usage"
}

#########
# TESTS #
#########

test_vllmopenai_server_with_lmcache_integrated() {
    http_status_code=$(curl --max-time 60 http://localhost:${PORT}/v1/completions \
            -w "%{http_code}" -o response-file.txt \
            -H "Content-Type: application/json" \
            -d '{
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
                "max_tokens": 100,
                "temperature": 0.7
            }'
    )

    if [ "$http_status_code" -ne 200 ]; then
        echo "Model prompt request from OpenAI API server failed, HTTP status code: ${http_status_code}."
        cat response-file.txt
        docker logs -n 20 $CID
        return 1
    else
         echo "Model prompt request from OpenAI API server succeeded"
         cat response-file.txt
    fi
}

#########
# SETUP #
#########

while [ $# -gt 0 ]; do
  case "$1" in
    --hf-token*|-hft*)
      if [[ "$1" != *=* ]]; then shift; fi # Value is next arg if no `=`
      HF_TOKEN="${1#*=}"
      ;;
    --server-wait-timeout*|-swt*)
      if [[ "$1" != *=* ]]; then shift; fi
      SERVER_WAIT_TIMEOUT="${1#*=}"
      if ! [[ "$SERVER_WAIT_TIMEOUT" =~ ^[0-9]+$ ]]; then
            echo "server-wait-timeout is wait time in seconds - integer only"
            exit 1
      fi

      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      >&2 printf "Error: Invalid argument\n"
      usage
      exit 1
      ;;
  esac
  shift
done

ORIG_DIR="$PWD"

# Find an available port starting from 8000
PORT=$(find_available_port 8000)
if [ $? -ne 0 ]; then
    echo "Failed to find an available port"
    exit 1
fi
echo "Using port: $PORT"

# Need to run from docker directory
cd docker/

# Create the container image
build_lmcache_vllmopenai_image

# Start the OpenAI API server by running the container image
run_lmcache_vllmopenai_container

########
# MAIN #
########

# test that can inference model using vLLM OpenAI API (lmcache integrated)
test_vllmopenai_server_with_lmcache_integrated

exit 0
