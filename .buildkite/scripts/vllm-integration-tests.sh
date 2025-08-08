#!/usr/bin/bash
#
# This test script runs integration tests for the LMCache integration with vLLM.
# A lmcache/vllm-openai container image is built by this script from the LMCache code base
# the script is running from and the latest nightly build of vLLM. It is therefore using the
# latest of both code bases to build the image which it then performs tests on.
#
# It’s laid out as follows:
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

set -e
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

wait_for_openai_api_server() {
    if ! timeout "$SERVER_WAIT_TIMEOUT" bash -c "
        echo \"Curl /v1/models endpoint\"
        until curl -s 127.0.0.1:${PORT}/v1/models \
                | grep '\"id\":\"meta-llama/Llama-3.2-1B-Instruct\"'; do
            sleep 30
        done
    "; then
        echo "OpenAI API server did not start"
        docker logs $CID
        return 1
    fi
}

run_lmcache_vllmopenai_container() {
    local cfg_name="$1"
    LOGFILE="/tmp/build_${BUILD_ID}_${cfg_name}.log"
    # Pick the GPU with the largest free memory
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" $PORT
    best_gpu="${CUDA_VISIBLE_DEVICES}"

    # docker args
    docker_args=(
        --runtime nvidia
        --network host
        --gpus "device=${best_gpu}"
        --volume "${ORIG_DIR}/.buildkite/lmcache_configs:/configs:ro"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
        --env LMCACHE_CONFIG_FILE="/configs/${cfg_name}"
    )

    # vllm args
    cmd_args=(
        lmcache/vllm-openai:build-latest
        meta-llama/Llama-3.2-1B-Instruct
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        --port "$PORT"
    )
    if [ "$test_mode" = "dummy" ]; then
        cmd_args=("${cmd_args[@]}" --max-model-len 1024 --gpu-memory-utilization '0.35' --enforce-eager)
    fi

    CID=$(
        docker run -d \
            "${docker_args[@]}" \
            "${cmd_args[@]}"
    )

    buildkite-agent meta-data set "docker-CID" "$CID"

    wait_for_openai_api_server

    touch "$LOGFILE"
    docker logs -f "$CID" >>"$LOGFILE" 2>&1 &
    LOG_PID=$!

    end=$((SECONDS + 120))
    while [ $SECONDS -lt $end ]; do
        if grep -qi 'Starting vLLM API server' "$LOGFILE"; then
            echo "vLLM API server started."
            break
        fi
        sleep 1
    done

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
    echo "  --configs|-c                 Path to a file containing one config filename per line (required)"
    echo "  --tests|-t                   Test mode"
}

#########
# TESTS #
#########

test_vllmopenai_server_with_lmcache_integrated() {
    http_status_code=$(
        curl --max-time 60 http://localhost:${PORT}/v1/completions \
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

run_long_doc_qa() {
    local num_docs="${NUM_DOCUMENTS:-8}"
    local doc_len="${DOCUMENT_LENGTH:-20000}"
    local out_len="${OUTPUT_LEN:-100}"
    local repeat_count="${REPEAT_COUNT:-2}"
    local repeat_mode="${REPEAT_MODE:-random}"
    local shuffle_seed="${SHUFFLE_SEED:-0}"
    local max_inflight="${MAX_INFLIGHT_REQUESTS:-20}"
    local sleep_after="${SLEEP_TIME_AFTER_WARMUP:-0.0}"
    local expected_ttft_gain="${EXPECTED_TTFT_GAIN:-2.3}"
    local expected_latency_gain="${EXPECTED_LATENCY_GAIN:-3.5}"

    echo "→ Running long-doc-qa:"
    echo "   num_docs=${num_docs}, doc_len=${doc_len}, out_len=${out_len}"
    echo "   repeat=${repeat_mode}×${repeat_count}, seed=${shuffle_seed}"
    echo "   inflight=${max_inflight}, sleep_after=${sleep_after}s"
    echo "   expected_ttft_gain=${expected_ttft_gain}, expected_latency_gain=${expected_latency_gain}"

    if [ ! -d ".venv" ]; then
        UV_PYTHON=python3 uv -q venv
    fi
    source .venv/bin/activate
    uv -q pip install openai
    python3 "$ORIG_DIR/benchmarks/long-doc-qa/long-doc-qa.py" \
        --num-documents="$num_docs" \
        --document-length="$doc_len" \
        --output-len="$out_len" \
        --repeat-count="$repeat_count" \
        --repeat-mode="$repeat_mode" \
        --shuffle-seed="$shuffle_seed" \
        --max-inflight-requests="$max_inflight" \
        --sleep-time-after-warmup="$sleep_after" \
        --port="$PORT" \
        --model="meta-llama/Llama-3.2-1B-Instruct" \
        --output="response.txt" \
        --expected-ttft-gain="$expected_ttft_gain" \
        --expected-latency-gain="$expected_latency_gain"
}

#########
# SETUP #
#########

while [ $# -gt 0 ]; do
    case "$1" in
    --configs* | -c*)
        if [[ "$1" != *=* ]]; then shift; fi
        configs_arg="${1#*=}"
        ;;
    --tests* | -t*)
        if [[ "$1" != *=* ]]; then shift; fi
        test_mode="${1#*=}"
        ;;
    --hf-token* | -hft*)
        if [[ "$1" != *=* ]]; then shift; fi
        HF_TOKEN="${1#*=}"
        ;;
    --server-wait-timeout* | -swt*)
        if [[ "$1" != *=* ]]; then shift; fi
        SERVER_WAIT_TIMEOUT="${1#*=}"
        if ! [[ "$SERVER_WAIT_TIMEOUT" =~ ^[0-9]+$ ]]; then
            echo "server-wait-timeout is wait time in seconds - integer only"
            exit 1
        fi
        ;;
    --help | -h)
        usage
        exit 0
        ;;
    *)
        printf >&2 "Error: Invalid argument\n"
        usage
        exit 1
        ;;
    esac
    shift
done

ORIG_DIR="$PWD"
WORKLOAD_DIR="${ORIG_DIR}/.buildkite/workload_configs"

# Read the configs argument (always a file with one config per line)
if [[ ! -f "$configs_arg" ]]; then
    echo "Error: --configs file not found: $configs_arg" >&2
    exit 1
fi
mapfile -t CONFIG_NAMES < <(
  sed 's/[[:space:]]\+$//' "$configs_arg"
)

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

########
# MAIN #
########

for cfg_name in "${CONFIG_NAMES[@]}"; do
    echo -e "\033[1;33m===== Testing LMCache with ${cfg_name} =====\033[0m"

    if [ "$test_mode" = "dummy" ]; then
        run_lmcache_vllmopenai_container "$cfg_name" "$test_mode"
        test_vllmopenai_server_with_lmcache_integrated
    elif [ "$test_mode" = "long-doc-qa" ]; then
        # load workload override from YAML if present
        workload_file="${WORKLOAD_DIR}/${cfg_name}"
        if [[ -f "$workload_file" ]]; then
            echo "→ Loading workload parameters from ${workload_file}"
            NUM_DOCUMENTS="$(yq e '.num_docs' "$workload_file")"
            DOCUMENT_LENGTH="$(yq e '.doc_len' "$workload_file")"
            OUTPUT_LEN="$(yq e '.out_len' "$workload_file")"
            REPEAT_COUNT="$(yq e '.repeat_count' "$workload_file")"
            REPEAT_MODE="$(yq e '.repeat_mode' "$workload_file")"
            SHUFFLE_SEED="$(yq e '.shuffle_seed' "$workload_file")"
            MAX_INFLIGHT_REQUESTS="$(yq e '.max_inflight' "$workload_file")"
            SLEEP_TIME_AFTER_WARMUP="$(yq e '.sleep_after' "$workload_file")"
            EXPECTED_TTFT_GAIN="$(yq e '.expected_ttft_gain' "$workload_file")"
            EXPECTED_LATENCY_GAIN="$(yq e '.expected_latency_gain' "$workload_file")"
        else
            echo "❌ Error: workload YAML for ${cfg_name} not found at ${workload_file}" >&2
            exit 1
        fi

        run_lmcache_vllmopenai_container "$cfg_name" "$test_mode"
        run_long_doc_qa
    fi

    cleanup 0
done

exit 0
