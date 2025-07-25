#! /bin/bash

# Start for the RAG benchmark
# export LMCACHE_CHUNK_SIZE="256"
# export LMCACHE_ENABLE_BLENDING="True"
# export LMCACHE_BLEND_SPECIAL_STR=" # # "
# export LMCACHE_USE_LAYERWISE="True"
# export LMCACHE_LOCAL_CPU="True"
# export LMCACHE_MAX_LOCAL_CPU_SIZE="5"

# without tranfer (not PD disaggreated)
# full re-compute
LMCACHE_CONFIG_FILE=blending.yaml vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests \
    --port 8200 --host localhost \
    --tokenizer-mode "mistral" \
    --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}' \
    --max-model-len 8000 --gpu-memory-utilization 0.8 \
    --no-enable-prefix-caching \
    --override-generation-config '{"temperature": 0, "top_p": 0.95, "max_tokens": 10}'

# without tranfer (PD disaggreated)
# prefix caching
# LMCACHE_CONFIG_FILE=blending.yaml vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests \
#     --port 800 --host localhost \
#     --tensor-parallel-size 1 \
#     --max-model-len 8000 \
#     --gpu-memory-utilization 0.8 \
#     --enable-prefix-caching \
#     --kv-transfer-config '{"kv_cache_type": "LMCacheConnectorV1"}' \
#     --override-generation-config '{"temperature": 0, "top_p": 0.95, "max_tokens": 10}'

# without tranfer (PD disaggregated)
# full re-use
