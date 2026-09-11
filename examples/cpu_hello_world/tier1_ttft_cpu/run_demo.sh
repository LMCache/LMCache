#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Tier 1 -- LMCache reducing TTFT on CPU with a tiny open-weight model.
#
# Brings up an `lmcache server` (CPU L1 cache) and a CPU-only `vllm serve`
# wired to it through the LMCacheMPConnector, then sends three requests:
#   1. cold    -- first time this prompt is seen (KV computed + stored)
#   2. warm    -- identical prompt (KV retrieved from LMCache -> lower TTFT)
#   3. negative-- a different prompt (no shared prefix -> no cache hit)
#
# It reports TTFT for each and reads LMCache's Prometheus counters to prove
# the store (run 1) and the hit (run 2) actually happened. This mirrors the
# CPU e2e path in .github/scripts/run-cpu-e2e-validation.sh, with the model
# swapped to a redistributable Apache-2.0 model.
#
# Environment (all optional, defaults shown):
#   MODEL               HF model id (default: Qwen/Qwen2.5-0.5B-Instruct)
#   LMCACHE_ZMQ_PORT    LMCache RPC port      (default: 15557)
#   LMCACHE_HTTP_PORT   LMCache metrics port  (default: 18082)
#   VLLM_PORT           vLLM HTTP port        (default: 18000)
#   CHUNK_SIZE          LMCache chunk size    (default: 16, demo value)
#   MAX_MODEL_LEN       vLLM max context      (default: 2048)
#   VLLM_READY_TIMEOUT  seconds               (default: 600)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
LMCACHE_ZMQ_PORT="${LMCACHE_ZMQ_PORT:-15557}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-18082}"
VLLM_PORT="${VLLM_PORT:-18000}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-600}"
LMCACHE_READY_TIMEOUT="${LMCACHE_READY_TIMEOUT:-60}"
CPU_KVCACHE_SPACE_GB="${VLLM_CPU_KVCACHE_SPACE:-1}"

WORK_DIR="$(mktemp -d -t lmcache_tier1_XXXX)"
LMCACHE_LOG="${WORK_DIR}/lmcache.log"
VLLM_LOG="${WORK_DIR}/vllm.log"
PROMPT_SHARED="${WORK_DIR}/prompt_shared.txt"
PROMPT_NEGATIVE="${WORK_DIR}/prompt_negative.txt"
TTFT_OUT="${WORK_DIR}/ttft.jsonl"
LMCACHE_PID=""
VLLM_PID=""

WRITE_METRIC="lmcache_mp_l1_write_chunks_total"
READ_METRIC="lmcache_mp_l1_read_chunks_total"

cleanup() {
  for pid in "${VLLM_PID}" "${LMCACHE_PID}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

echo "==> Tier 1: LMCache TTFT reduction on CPU"
echo "    model:        ${MODEL}"
echo "    chunk-size:   ${CHUNK_SIZE}  (demo value; production default is 256)"
echo "    LMCache ZMQ/HTTP: ${LMCACHE_ZMQ_PORT}/${LMCACHE_HTTP_PORT}"
echo "    vLLM port:    ${VLLM_PORT}"
echo "    work dir:     ${WORK_DIR}"

# ------------------------------------------------------------------ #
# Build prompts. A long shared prefix guarantees several full chunks so
# there is visible cache traffic (a prompt shorter than one chunk caches
# nothing). cold and warm send the identical prompt; negative differs.
# ------------------------------------------------------------------ #
python3 - "${PROMPT_SHARED}" "${PROMPT_NEGATIVE}" <<'PY'
import sys

shared_path, negative_path = sys.argv[1], sys.argv[2]

def document(topic: str, marker: str) -> str:
    lines = [f"# Background briefing on {topic} ({marker})", ""]
    for i in range(1, 121):
        lines.append(
            f"{marker}-{i:03d}. This paragraph describes aspect {i} of "
            f"{topic}: how it is defined, why it matters for the system, "
            f"and the trade-offs an operator should weigh when configuring it."
        )
    lines.append("")
    lines.append("Question: In one sentence, summarize the briefing above.")
    lines.append("Answer:")
    return "\n".join(lines)

with open(shared_path, "w", encoding="utf-8") as fh:
    fh.write(document("KV cache management", "A"))
with open(negative_path, "w", encoding="utf-8") as fh:
    fh.write(document("network routing protocols", "Z"))
PY
echo "    shared prompt:   $(wc -c <"${PROMPT_SHARED}") chars"
echo "    negative prompt: $(wc -c <"${PROMPT_NEGATIVE}") chars"

# ------------------------------------------------------------------ #
# Start LMCache server (CPU L1). Small chunk-size so short demo prompts
# still form several cacheable chunks.
# ------------------------------------------------------------------ #
echo ""
echo "==> Starting lmcache server (log: ${LMCACHE_LOG})"
lmcache server \
  --port "${LMCACHE_ZMQ_PORT}" \
  --http-port "${LMCACHE_HTTP_PORT}" \
  --l1-size-gb 2 \
  --eviction-policy LRU \
  --chunk-size "${CHUNK_SIZE}" \
  >"${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
sleep 1
if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
  echo "!! lmcache server exited immediately. Last 40 lines:"; tail -n 40 "${LMCACHE_LOG}" || true; exit 1
fi
if ! wait_for_endpoint_contains \
    "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck" \
    "${LMCACHE_READY_TIMEOUT}" "" "lmcache server"; then
  tail -n 40 "${LMCACHE_LOG}" || true; exit 1
fi
echo "    lmcache server healthy"

# ------------------------------------------------------------------ #
# Start vLLM (CPU) wired to LMCache via the MP connector.
# vLLM's own prefix caching is disabled so LMCache is unambiguously the
# thing providing reuse.
# ------------------------------------------------------------------ #
export VLLM_DEVICE=cpu
export VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_KVCACHE_SPACE="${CPU_KVCACHE_SPACE_GB}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$(pick_loopback_iface)}"

KV_CACHE_BYTES="$(python3 -c "print(int(${CPU_KVCACHE_SPACE_GB} * 1024 * 1024 * 1024))")"
KV_TRANSFER_CONFIG="$(python3 - "${LMCACHE_ZMQ_PORT}" <<'PY'
import json
import sys

port = int(sys.argv[1])
print(json.dumps({
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
    "kv_connector_extra_config": {
        "lmcache.mp.host": "tcp://localhost",
        "lmcache.mp.port": port,
    },
}))
PY
)"

echo ""
echo "==> Starting vLLM (CPU) serving ${MODEL} (log: ${VLLM_LOG})"
echo "    First run downloads the model from Hugging Face (a few hundred MB)."
vllm serve "${MODEL}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --gpu-memory-utilization 0.1 \
  --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 4 \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
sleep 1
if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
  echo "!! vLLM exited immediately. Last 60 lines:"; tail -n 60 "${VLLM_LOG}" || true; exit 1
fi
echo "==> Waiting for vLLM readiness (timeout ${VLLM_READY_TIMEOUT}s; CPU startup is slow)"
if ! wait_for_endpoint_contains \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" \
    "${VLLM_READY_TIMEOUT}" "${MODEL}" "vLLM server"; then
  echo "!! Last 60 vLLM log lines:"; tail -n 60 "${VLLM_LOG}" || true; exit 1
fi
echo "    vLLM ready"

# ------------------------------------------------------------------ #
# Drive the three requests and read the counters between them.
# ------------------------------------------------------------------ #
API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
run_measure() {
  python3 "${SCRIPT_DIR}/measure_ttft.py" \
    --api-base "${API_BASE}" --model "${MODEL}" \
    --prompt-file "$1" --label "$2" --max-tokens 8 --out "${TTFT_OUT}"
}

echo ""
echo "==> [1/3] cold request (prompt seen for the first time)"
WRITE_BEFORE="$(scrape_metric "${WRITE_METRIC}" "${LMCACHE_HTTP_PORT}")"
TTFT_COLD="$(run_measure "${PROMPT_SHARED}" cold | sed -n 's/.*TTFT = \([0-9.]*\)s/\1/p')"
sleep 2
WRITE_AFTER="$(scrape_metric "${WRITE_METRIC}" "${LMCACHE_HTTP_PORT}")"

echo ""
echo "==> [2/3] warm request (identical prompt -> should hit LMCache)"
READ_BEFORE="$(scrape_metric "${READ_METRIC}" "${LMCACHE_HTTP_PORT}")"
TTFT_WARM="$(run_measure "${PROMPT_SHARED}" warm | sed -n 's/.*TTFT = \([0-9.]*\)s/\1/p')"
sleep 1
READ_AFTER="$(scrape_metric "${READ_METRIC}" "${LMCACHE_HTTP_PORT}")"

echo ""
echo "==> [3/3] negative request (different prompt -> should NOT hit)"
READ_NEG_BEFORE="$(scrape_metric "${READ_METRIC}" "${LMCACHE_HTTP_PORT}")"
run_measure "${PROMPT_NEGATIVE}" negative >/dev/null
sleep 1
READ_NEG_AFTER="$(scrape_metric "${READ_METRIC}" "${LMCACHE_HTTP_PORT}")"

# ------------------------------------------------------------------ #
# Report + validate.
# ------------------------------------------------------------------ #
echo ""
echo "======================= RESULTS ======================="
printf "  cold TTFT : %ss\n" "${TTFT_COLD:-?}"
printf "  warm TTFT : %ss\n" "${TTFT_WARM:-?}"
printf "  store: %s -> %s chunks written (cold)\n" "${WRITE_BEFORE}" "${WRITE_AFTER}"
printf "  hit  : %s -> %s chunks read    (warm)\n" "${READ_BEFORE}" "${READ_AFTER}"
printf "  negative read delta: %s -> %s\n" "${READ_NEG_BEFORE}" "${READ_NEG_AFTER}"
echo "  JSONL: ${TTFT_OUT}"
echo "======================================================="

FAIL=0
if [ "${WRITE_AFTER}" -le "${WRITE_BEFORE}" ]; then
  echo "!! Expected chunks to be WRITTEN on the cold request; none were."; FAIL=1
fi
if [ "${READ_AFTER}" -le "${READ_BEFORE}" ]; then
  echo "!! Expected chunks to be READ (cache hit) on the warm request; none were."; FAIL=1
fi
if awk "BEGIN{exit !(${TTFT_WARM:-0} < ${TTFT_COLD:-0})}"; then
  echo "==> TTFT dropped on the cache hit (warm < cold)."
else
  echo "!! Note: warm TTFT was not lower than cold. On a tiny model / tiny"
  echo "   machine the prefill saved can be small; the cache-hit counter above"
  echo "   is the authoritative proof that LMCache served the KV."
fi

if [ "${FAIL}" -ne 0 ]; then
  echo ""; echo "==> FAIL. LMCache log tail:"; tail -n 40 "${LMCACHE_LOG}" || true
  exit 1
fi
echo ""
echo "==> PASS: LMCache stored KV on the cold request and served it on the warm"
echo "    request, entirely on CPU."
