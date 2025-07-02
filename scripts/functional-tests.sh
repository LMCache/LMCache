#!/usr/bin/bash
#
# This test script runs functional tests on the lmcache/vllm-openai container image.
# The image is built by this script from the LMCache code base the script is running from and
# the latest nightly build of vLLM. It is laid out as follows:
# - UTILITIES:  utility functions
# - TESTS:      test functions
# - SETUP:      environment setup steps
# - MAIN:       test execution steps
#

set -ex

#############
# UTILITIES #
#############

build_lmcache_vllmopenai_image() {
    cp example_build.sh test-build.sh
    chmod 755 test-build.sh 
    ./test-build.sh
}

wait_for_openai_api_server(){
    if ! timeout 180 bash -c '
        until curl 127.0.0.1:8000/v1/models |grep "\"id\":\"meta-llama/Llama-3.1-8B-Instruct\""; do
            echo "waiting for OpenAI API server to start"
            sleep 30
        done
    '; then
        echo "OpenAI API server did not start"
        docker logs $1
        cleanup 1
        exit 1
    fi
}

run_lmcache_vllmopenai_container() {
    CID=$(docker run -d --runtime nvidia --gpus all \
        --env "LMCACHE_CHUNK_SIZE=256" \
        --env "LMCACHE_LOCAL_CPU=True" \
        --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
        --volume ~/.cache/huggingface:/root/.cache/huggingface \
        --network host \
        'lmcache/vllm-openai:build-latest' \
        'meta-llama/Llama-3.1-8B-Instruct' --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')

    wait_for_openai_api_server $CID

    if ! timeout 10 bash -c '
        if ! docker logs $0 | grep -i "Starting vLLM API server"; then
            echo "container log file does not contain server started message"
            exit 1
        else
            docker logs $0
        fi
    ' $CID; then
        echo "container log file was not created"
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

    rm -f test-build.sh response-file.txt
    set -e
}


#########
# TESTS #
#########

test_lmcache_vllmopenai_server() {
    http_status_code=$(curl http://localhost:8000/v1/completions \
            -w "%{http_code}" -o response-file.txt \
            -H "Content-Type: application/json" \
            -d '{
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
                "max_tokens": 100,
                "temperature": 0.7
            }'
    )

    if [ "$http_status_code" -ne 200 ]; then
        echo "Model prompt request from OpenAI API server failed, HTTP status code: ${http_status_code}."
        cat response-file.txt
        cleanup 1
        exit 1
    else
         echo "Model prompt request from OpenAI API server succeeded"
         cat response-file.txt
    fi
}

#########
# SETUP #
#########

CID=

# Need to run from docker directory
cd docker/

# Create the container image
build_lmcache_vllmopenai_image

# Start the OpenAI API server by running the container image
run_lmcache_vllmopenai_container

########
# MAIN #
########

test_lmcache_vllmopenai_server

cleanup

exit 0