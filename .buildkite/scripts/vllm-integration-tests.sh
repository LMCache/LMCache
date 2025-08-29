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
    local docker="$1"
    local vllm="$2"
    local cfg_name="$3"
    LOGFILE="/tmp/build_${BUILD_ID}_${cfg_name}.log"

    # Pick the GPU with the largest free memory
    source "$ORIG_DIR/.buildkite/scripts/pick-free-gpu.sh" $PORT
    best_gpu="${CUDA_VISIBLE_DEVICES}"

    # docker args
    docker_args=(
        --runtime nvidia
        --network host
        --gpus "device=${best_gpu}"
        --volume ~/.cache/huggingface:/root/.cache/huggingface
        --env VLLM_USE_FLASHINFER_SAMPLER=0
        --env HF_TOKEN="$HF_TOKEN"
    )
    while IFS= read -r line; do
        key="${line%%:*}" 
        val="${line#*:}"
        docker_args+=(--"$key" "$val")
    done < <(yq -r 'to_entries[] | .key as $k | .value[] | "\($k):\(. )"' <<<"$docker")

    # vllm args
    vllm_model="$(yq -r '.model' <<<"$vllm_args")"
    mapfile -t vllm_cli_args < <(yq -r '.args // [] | .[]' <<<"$vllm_args")
    cmd_args=(
        lmcache/vllm-openai:build-latest
        "$vllm_model"
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        --port "$PORT"
    )
    cmd_args+=("${vllm_cli_args[@]}")

    CID=$(
        docker run -d \
            "${docker_args[@]}" \
            "${cmd_args[@]}"
    )

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
    local workload_config="$1"

    echo "→ Running long_doc_qa with customed workload config:"
    printf '%s\n' "$workload_config"

    local workload_args=()
    mapfile -d '' -t workload_args < <(
    jq -j '
        to_entries[]
        | select(.value != null and (.value|tostring) != "")
        | "--\(.key)", "\u0000",
        (if (.value|type) == "string"
        then .value
        else (.value|tostring)
        end), "\u0000"
    ' <<<"$workload_yaml"
    )

    if [ ! -d ".venv" ]; then
        UV_PYTHON=python3 uv -q venv
    fi
    source .venv/bin/activate
    uv -q pip install openai
    python3 "$ORIG_DIR/benchmarks/long_doc_qa/long_doc_qa.py" \
        "${workload_args[@]}" \
        --port="$PORT" \
        --output="response.txt"
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
CONFIG_DIR="${ORIG_DIR}/.buildkite/configs"

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
    cfg_file="${CONFIG_DIR}/${cfg_name}"

    # Start server
    docker_args="$(yq '.docker' "$cfg_file")"
    vllm_args="$(yq '.vllm' "$cfg_file")"
    run_lmcache_vllmopenai_container "$docker_args" "$vllm_args" "$cfg_name"

    # Send request
    test_mode="$(yq -r '.workload.type' "$cfg_file")"
    if [ "$test_mode" = "dummy" ]; then
        test_vllmopenai_server_with_lmcache_integrated
    elif [ "$test_mode" = "long_doc_qa" ]; then
        workload_yaml="$(yq '(.workload * {"model": .vllm.model}) | del(.type)' "$cfg_file")"
        run_long_doc_qa "$workload_yaml"
    fi

    cleanup 0
done

exit 0
