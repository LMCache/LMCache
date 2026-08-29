#!/bin/bash
# GLM-5.2 + LMCache MP mode startup script for node 43.
# Runs inside the sgl-lmcache container.
set -x

ulimit -l unlimited
sysctl -w vm.max_map_count=16777216 >/dev/null 2>&1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_OPT_USE_TOPK_V2=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m sglang.launch_server \
  --model-path /data/models/ZhipuAI/GLM-5.2-FP8 \
  --served-model-name glm-5.2 --api-key 456123 --tp 8 \
  --trust-remote-code --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.77 \
  --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
  --context-length 1000000 --max-running-requests 64 --max-queued-requests 32 \
  --max-total-tokens 1100000 --watchdog-timeout 1800 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-lmcache --lmcache-config-file /LMCache/lmcache-mp-glm52.yaml \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 31000 2>&1 | tee /tmp/sgl-glm52-lmcache-mp.log
