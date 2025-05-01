MODEL=deepseek-ai/DeepSeek-V2-Lite
PORT=8000

LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 \
VLLM_USE_V1=0 \
LMCACHE_CONFIG_FILE=.buildkite/mmlu_scripts/lmc-cpu.yaml \
python3 -m vllm.entrypoints.api_server \
  --model $MODEL \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port $PORT \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'