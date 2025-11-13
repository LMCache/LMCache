export LM_CACHE_METRICS=1
export LMCACHE_DEBUG=1
export LMDEBUG=1
export LMCACHE_VERBOSE=1
export LMCACHE_CONFIG_FILE=lmcache_blend.yml
export LM_CACHE_CONFIG_FILE=lmcache_blend.yml   

rm -f server_log.log
# kill existing vllm serve process
pkill -f "vllm serve Qwen/Qwen2.5-VL-7B-Instruct"

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-log-requests \
  --max-num-batched-tokens 20480 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 128000 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' > server_log.log 2>&1 &
