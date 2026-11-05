#!/usr/bin/env bash
#
# End-to-end example: LMCache MP server + disk L2 adapter + fp8lin serde + vLLM.
#
#   1. Fit an fp8lin coder for the model (skipped if CODER_PATH exists).
#   2. Start `lmcache server` (CPU L1 + disk L2 with the fp8lin serde).
#   3. Start vLLM with LMCacheMPConnector.
#   4. Request (cold): KV is computed, stored to L1, coded and written to L2.
#   5. Clear L1, request again: L2 prefetch, fp8lin decode, vLLM resumes from cache.
#   6. Compare both responses and report the bytes stored per token.
#
# With KV_CACHE_DTYPE=fp8 the KV cache is FP8 and fp8lin is lossless, so the two
# responses must match. With the default bf16 KV cache, fp8lin rounds to FP8 first
# (like the fp8 serde). SERDE=fp8 runs the same flow with the fp8 serde to compare.
# ATTENTION_BACKEND (e.g. TRITON_ATTN) is passed to vLLM when set.
#
# Requirements: vLLM, the lmcache CLI, one CUDA GPU, Triton.
set -e
set -o pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
GPU_DEVICE="${GPU_DEVICE:-0}"
SERDE="${SERDE:-fp8lin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
VLLM_PORT="${VLLM_PORT:-8000}"
L1_SIZE_GB="${L1_SIZE_GB:-8}"
TMP_DIR="${TMP_DIR:-/tmp/lmcache_fp8lin_example}"
L2_DISK_PATH="${L2_DISK_PATH:-${TMP_DIR}/disk_${SERDE}_${KV_CACHE_DTYPE}}"
KV_KIND=$([ "$KV_CACHE_DTYPE" = "fp8" ] && echo fp8 || echo bf16)
CODER_PATH="${CODER_PATH:-${TMP_DIR}/coder_${KV_KIND}}"
mkdir -p "$TMP_DIR" "$L2_DISK_PATH"
rm -f "$L2_DISK_PATH"/*

if [ "$SERDE" = "fp8lin" ] && [ ! -f "$CODER_PATH/meta.json" ]; then
    echo "=== Fitting an fp8lin coder for $MODEL ($KV_KIND KV) ==="
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" python -m lmcache.v1.fp8lin.fit \
        --model "$MODEL" --kv "$KV_KIND" --out "$CODER_PATH"
fi

if [ "$SERDE" = "fp8lin" ]; then
    SERDE_JSON="{\"type\": \"fp8lin\", \"coder_path\": \"${CODER_PATH}\", \"chunk_size\": 256}"
else
    SERDE_JSON='{"type": "fp8", "fp8_dtype": "float8_e4m3fn"}'
fi
L2_ADAPTER_JSON="{\"type\": \"fs\", \"base_path\": \"${L2_DISK_PATH}\", \"serde\": ${SERDE_JSON}}"

LMCACHE_PID=""
VLLM_PID=""
cleanup() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$LMCACHE_PID" ] && kill "$LMCACHE_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

wait_for_url() {
    local url="$1" timeout="${2:-300}" elapsed=0
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        [ "$elapsed" -ge "$timeout" ] && { echo "Timeout waiting for $url"; return 1; }
    done
    return 0
}

echo "=== Starting LMCache MP server (L2 serde: $SERDE) ==="
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" lmcache server \
    --l1-size-gb "$L1_SIZE_GB" \
    --eviction-policy LRU \
    --l2-store-policy default \
    --l2-prefetch-policy default \
    --l2-adapter "$L2_ADAPTER_JSON" \
    --port "$LMCACHE_PORT" \
    --http-port "$LMCACHE_HTTP_PORT" \
    > "$TMP_DIR/lmcache_${SERDE}.log" 2>&1 &
LMCACHE_PID=$!
wait_for_url "http://localhost:${LMCACHE_HTTP_PORT}/healthcheck" 120 || {
    tail -50 "$TMP_DIR/lmcache_${SERDE}.log"; exit 1; }

echo "=== Starting vLLM ($MODEL, kv-cache-dtype $KV_CACHE_DTYPE) ==="
KV_TRANSFER_CONFIG="{\"kv_connector\": \"LMCacheMPConnector\", \"kv_role\": \"kv_both\",
  \"kv_load_failure_policy\": \"recompute\",
  \"kv_connector_extra_config\": {\"lmcache.mp.port\": ${LMCACHE_PORT}, \"lmcache.mp.mq_timeout\": 10}}"
env -u VLLM_PORT CUDA_VISIBLE_DEVICES="$GPU_DEVICE" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONHASHSEED=0 \
vllm serve "$MODEL" \
    --port "$VLLM_PORT" \
    --no-enable-prefix-caching \
    --enforce-eager \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    ${ATTENTION_BACKEND:+--attention-backend "$ATTENTION_BACKEND"} \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.5}" \
    --max-model-len 8192 \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    > "$TMP_DIR/vllm_${SERDE}.log" 2>&1 &
VLLM_PID=$!
wait_for_url "http://localhost:${VLLM_PORT}/v1/models" 900 || {
    tail -50 "$TMP_DIR/vllm_${SERDE}.log"; exit 1; }

# A ~4k-token prompt from the example's README text, repeated with variations
PROMPT=$(python3 - <<'EOF'
import json
parts = [f"Section {i}: the history of computing spans mechanical calculators, vacuum tubes, "
         f"transistors, integrated circuits and accelerators; item {i * 7} discusses memory "
         f"hierarchies, caches and bandwidth, and why moving data costs more than computing on it. "
         for i in range(120)]
print(json.dumps("".join(parts) + "Summarize the main ideas above in one paragraph."))
EOF
)

request() {
    curl -s -o "$TMP_DIR/resp_$1.json" -w "%{time_total}" -X POST \
        "http://localhost:${VLLM_PORT}/v1/completions" -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": $PROMPT, \"max_tokens\": 48, \"temperature\": 0}"
}

T_COLD=$(request cold)
sleep 8   # let the store controller code and write the chunks
curl -s -X POST "http://localhost:${LMCACHE_HTTP_PORT}/cache/clear" > /dev/null
T_WARM=$(request warm)

python3 - "$TMP_DIR" "$L2_DISK_PATH" "$T_COLD" "$T_WARM" "$SERDE" "$KV_CACHE_DTYPE" <<'EOF'
import json, os, sys
tmp, disk, t_cold, t_warm, serde, kvd = sys.argv[1:]
cold = json.load(open(f"{tmp}/resp_cold.json"))
warm = json.load(open(f"{tmp}/resp_warm.json"))
files = [e for e in os.scandir(disk) if e.is_file() and not e.name.endswith(".tmp")]
stored = sum(e.stat().st_size for e in files)
prompt_tokens = cold["usage"]["prompt_tokens"]
print(json.dumps({
    "serde": serde, "kv_cache_dtype": kvd, "prompt_tokens": prompt_tokens,
    "l2_objects": len(files), "l2_bytes": stored,
    "l2_bytes_per_token": round(stored / max(1, (prompt_tokens // 256) * 256), 1),
    "request_s_cold": float(t_cold), "request_s_warm": float(t_warm),
    "same_response": cold["choices"][0]["text"] == warm["choices"][0]["text"],
    "response": cold["choices"][0]["text"][:120],
}, indent=1))
EOF
