#!/bin/bash
# Foreground vLLM pointed at the local LMCache MP server. Extra args are appended.
# Prefix caching is off so every LMCache hit is attributable to LMCache (turn it on for production).
source "$(dirname "$0")/common.sh"
export CUDA_VISIBLE_DEVICES="$GPU"
exec vllm serve "$MODEL" --port "$VLLM_PORT" \
  --no-enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$MP_PORT}}" \
  "$@"
