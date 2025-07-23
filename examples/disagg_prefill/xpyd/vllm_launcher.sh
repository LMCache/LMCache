#!/bin/bash

# vLLM Launch Script for Disaggregated Prefill
# Usage: ./vllm_launcher.sh <role> <host_id> <mode> [model_path] [localhost_mode] [FIRST_TOKEN_FROM]

ROLE=$1  # "prefiller" or "decoder"
HOST_ID=$2  # host identifier (e.g., "1", "2", etc.)
MODE=$3  # "xpyd" or "xp1d"
MODEL_PATH=${4:-"Qwen/Qwen2.5-0.5B"}
FIRST_TOKEN_FROM=${5:-"ftoken_from_p"}  # "ftoken_from_p" or "token_from_d"
LOCALHOST_MODE=${6:-"false"}  # "true" for single machine multi-GPU, "false" for multi-host

# Validate mode parameter
if [[ "$MODE" != "xpyd" && "$MODE" != "xp1d" ]]; then
    echo "Error: Mode must be 'xpyd' or 'xp1d'"
    echo "Usage: ./vllm_launcher.sh <role> <host_id> <mode> [model_path] [localhost_mode] [FIRST_TOKEN_FROM]"
    exit 1
fi
if [[ "$FIRST_TOKEN_FROM" != "ftoken_from_p" && "$FIRST_TOKEN_FROM" != "ftoken_from_d" ]]; then
    echo "Error: FIRST_TOKEN_FROM must be 'ftoken_from_p' or 'ftoken_from_d'"
    echo "Usage: ./vllm_launcher.sh <role> <host_id> <mode> [model_path] [localhost_mode] [FIRST_TOKEN_FROM]"
    exit 1
fi

# Common parameters
COMMON_ARGS="\
    --disable-log-requests \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max_model_len 512 \
    --max-num-seqs 10 \
    --gpu-memory-utilization 0.7"

#FIXME: below `lmcache_rpc_port` may not be necessary for multi-host use case
case $ROLE in
    "prefiller")
        CONFIG_FILE="configs/lmcache-prefiller-${MODE}-config.yaml"
        KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer_'"$HOST_ID"'"}}'
        # Port assignment: localhost mode uses different ports, multi-host mode uses same port
        if [[ "$LOCALHOST_MODE" == "true" ]]; then
            PORT=$((7100 + HOST_ID - 1))  # 7100, 7101, 7102, etc.
        else
            PORT=7100  # All prefillers use 7100 in multi-host mode
        fi
        ;;
    "decoder")
        CONFIG_FILE="configs/lmcache-decoder-${MODE}-config.yaml"
        # Add skip_last_n_tokens only for ftoken_from_p proxy type
        BASE_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer_'"$HOST_ID"'"'
        if [[ "$FIRST_TOKEN_FROM" == "ftoken_from_p" ]]; then
            KV_TRANSFER_CONFIG="${BASE_CONFIG}, \"skip_last_n_tokens\": 1}}"
        else
            KV_TRANSFER_CONFIG="${BASE_CONFIG}}}"
        fi
        # Port assignment: localhost mode uses different ports, multi-host mode uses same port
        if [[ "$LOCALHOST_MODE" == "true" ]]; then
            PORT=$((7200 + HOST_ID - 1))  # 7200, 7201, 7202, etc.
            NEW_CONFIG="/tmp/lmcache-decoder-${HOST_ID}-${MODE}-config.yaml"
            cp $CONFIG_FILE  $NEW_CONFIG
            CONFIG_FILE=$NEW_CONFIG
            # increase `nixl_peer_init_port`
            sed -i "s/7300/$((7300 + HOST_ID - 1))/g" $CONFIG_FILE
            # increase `nixl_peer_alloc_port`
            sed -i "s/7400/$((7400 + HOST_ID - 1))/g" $CONFIG_FILE
        else
            PORT=7200  # All decoders use 7200 in multi-host mode
        fi
        ;;
    *)
        echo "Error: Role must be 'prefiller' or 'decoder'"
        exit 1
        ;;
esac


set -x
# Launch vLLM
UCX_TLS=cuda_ipc,cuda_copy,tcp \
LMCACHE_LOG_LEVEL=DEBUG \
LMCACHE_CONFIG_FILE=$CONFIG_FILE \
vllm serve $MODEL_PATH \
    --port $PORT \
    $COMMON_ARGS \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
