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

set -ex

CID=
DOCKER_BUILD_FILE=
DOCKER_LOGFILE=
HF_TOKEN=
LOG_PID=
SERVER_WAIT_TIMEOUT=180
TEST_HTTP_RESPONSE_FILE=

#############
# UTILITIES #
#############

build_lmcache_vllmopenai_image() {
    DOCKER_BUILD_FILE="/tmp/lmcache_vllm_build.sh"
    cp example_build.sh $DOCKER_BUILD_FILE
    chmod 755 $DOCKER_BUILD_FILE
    $DOCKER_BUILD_FILE
}

wait_for_openai_api_server(){
    if ! timeout $SERVER_WAIT_TIMEOUT bash -c '
        until curl 127.0.0.1:8000/v1/models |grep "\"id\":\"meta-llama/Llama-3.2-1B-Instruct\""; do
            echo "waiting for OpenAI API server to start"
            sleep 30
        done
    '; then
        echo "OpenAI API server did not start"
        docker logs $CID
        cleanup 1
        exit 1
    fi
}

run_lmcache_vllmopenai_container() {
    # Pick the GPU with the largest free memory
    best_gpu=$(nvidia-smi --query-gpu=memory.free,index \
        --format=csv,noheader,nounits \
      | sort -t',' -k1 -nr \
      | head -n1 \
      | cut -d',' -f2)
    
    if [ -z "$HF_TOKEN" ]; then
        CID=$(docker run -d --runtime nvidia --gpus "device=${best_gpu}" \
            --env "LMCACHE_CHUNK_SIZE=256" \
            --env "LMCACHE_LOCAL_CPU=True" \
            --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
            --volume ~/.cache/huggingface:/root/.cache/huggingface \
            --network host \
            'lmcache/vllm-openai:build-latest' \
            'meta-llama/Llama-3.2-1B-Instruct' --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            --gpu-memory-utilization '0.5' \
            --enforce-eager)
    else
        CID=$(docker run -d --runtime nvidia --gpus "device=${best_gpu}" \
             --env HF_TOKEN=$HF_TOKEN \
            --env "LMCACHE_CHUNK_SIZE=256" \
            --env "LMCACHE_LOCAL_CPU=True" \
            --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
            --volume ~/.cache/huggingface:/root/.cache/huggingface \
            --network host \
            'lmcache/vllm-openai:build-latest' \
            'meta-llama/Llama-3.2-1B-Instruct' --kv-transfer-config \
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
            --gpu-memory-utilization '0.5' \
            --enforce-eager)
    fi
    
    wait_for_openai_api_server

    DOCKER_LOGFILE="/tmp/vllm_${CID}.log"
    docker logs -f "$CID" &> "$DOCKER_LOGFILE" &
    LOG_PID=$!

    set +x
    end=$((SECONDS + 120))
    api_srv_started=false
    while [ $SECONDS -lt $end ]; do
        if grep -qi 'Starting vLLM API server' "$DOCKER_LOGFILE"; then
            echo "vLLM API server started."
            api_srv_started=true
            break
        fi
        sleep 1
    done
    set -x

    if [ "$api_srv_started" = false ]; then
        echo "Timeout waiting for vLLM API server to start, dumping full log:"
        cat "$DOCKER_LOGFILE"
        cleanup 1
        exit 1
    fi

}

cleanup() {
    set +e
    if [ "${1:-0}" -ne 0 ]; then
        printf "\n\n"
        printf "\e[31m=%.0s\e[0m" {1..80}
        printf "\n\e[31mERROR OCCURRED\e[0m\n"
        printf "\e[31mFunction: %s\e[0m\n" "${FUNCNAME[1]}"
        printf "\e[31mExit Code: %s\e[0m\n" "$1"
        printf "\e[31m=%.0s\e[0m" {1..80}
        printf "\n\n"
    fi
    for cid in $CID; do
        if [ -n "$cid" ]; then
            docker stop $cid
            docker rm -v $cid
        fi
    done
    if [ -n "$LOG_PID" ]; then
        kill $LOG_PID
    fi

    rm -f $DOCKER_BUILD_FILE $TEST_HTTP_RESPONSE_FILE $DOCKER_LOGFILE
    set -e
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
    TEST_HTTP_RESPONSE_FILE="/tmp/lmcache_vllm_http_response.txt"
    http_status_code=$(curl http://localhost:8000/v1/completions \
            -w "%{http_code}" -o $TEST_HTTP_RESPONSE_FILE \
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
        echo "HTTP response:"
        cat $TEST_HTTP_RESPONSE_FILE
        echo "Snapshot of end of container logs:"
        docker logs -n 20 $CID
        cleanup 1
        exit 1
    else
         echo "Model prompt request from OpenAI API server succeeded, dumping server response:"
         cat $TEST_HTTP_RESPONSE_FILE
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
