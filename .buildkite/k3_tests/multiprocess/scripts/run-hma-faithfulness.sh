#!/usr/bin/env bash
# HMA transfer-faithfulness test on a real hybrid model.
#
# One deterministic long-prefix request is sent against vLLM+LMCache twice:
#   1. Store phase: cache miss; vLLM computes the prefill and LMCache stores
#      the prefix KV. The completion is the ground truth (engine-computed KV).
#   2. Hit phase: after resetting vLLM's *local* prefix cache (APC) via the
#      dev-mode endpoint POST /reset_prefix_cache (LMCache preserved), the
#      byte-identical request; the whole prefix is served by LMCache
#      retrieve and consumed by the engine's own attention kernels.
#
# Assertions:
#   - The hit phase was served by LMCache: a retrieve delta in the server
#     log, and the request's own measured local/external split.
#   - The completion matches the ground truth exactly. Temperature 0 with a
#     single in-flight request is deterministic (GDN's batch-variance needs
#     concurrent load), so any divergence means the injected KV differs
#     from what the engine computed -- the silent-corruption class of #4247.
#
# Reading a failure: red on the FLASHINFER arm only points at the kernel
# sub-paging path (#4253's territory); red on both arms points at something
# common to every paged attention layout.
#
# Unlike a store->retrieve byte roundtrip inside LMCache, the engine is the
# judge here: a geometry misread that is symmetric between store and
# retrieve round-trips bytes perfectly while the model reads garbage, so
# only engine-consumed KV can detect it.
#
# The reset endpoint requires VLLM_SERVER_DEV_MODE=1 (set by
# launch-processes.sh).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
# Prefix length in rules; 60 rules is ~2.5k tokens, several 544-token chunks.
NUM_RULES="${NUM_RULES:-60}"
# Seconds to let async LMCache stores drain before the hit phase.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-10}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
# LMCache MP server log, scanned to confirm the hit phase hit LMCache.
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/build_${BUILD_ID}_lmcache.log}"

OUT_DIR="$RESULTS_DIR/hma_faithfulness"
mkdir -p "$OUT_DIR"

echo "=== HMA transfer-faithfulness test ==="
echo "Model: $MODEL"
echo "vLLM (LMCache) port: $VLLM_PORT"
echo "Rules in prefix: $NUM_RULES"
echo "Results dir: $OUT_DIR"
echo ""

# Count completed LMCache retrieves recorded in the server log so far.
count_retrieves() {
    if [ -f "$LMCACHE_LOG" ]; then
        grep -c "Retrieved" "$LMCACHE_LOG" || true
    else
        echo 0
    fi
}

# Read field $2 from the split sidecar hma_faithfulness_client.py writes for $1.
split_field() {
    python -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
        "$1.split.json" "$2"
}

echo "=== Store phase (cache miss, engine-computed ground truth) ==="
python "${SCRIPT_DIR}/hma_faithfulness_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --out "$OUT_DIR/store.txt"
echo ""

echo "Waiting ${STORE_DRAIN_SECONDS}s for async stores to drain..."
sleep "$STORE_DRAIN_SECONDS"

echo "=== Resetting vLLM local prefix cache on port $VLLM_PORT (LMCache preserved) ==="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")
if [ "$code" != "200" ]; then
    echo "Failed to reset prefix cache (HTTP $code). Is VLLM_SERVER_DEV_MODE=1?"
    exit 1
fi
echo "vLLM prefix cache reset."
echo ""

retrieves_before=$(count_retrieves)

echo "=== Hit phase (prefix served by LMCache retrieve) ==="
python "${SCRIPT_DIR}/hma_faithfulness_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --out "$OUT_DIR/hit.txt"
echo ""

retrieves_after=$(count_retrieves)

failures=0

if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "FAILED: LMCache served no retrieves during the hit phase" \
        "(before=$retrieves_before, after=$retrieves_after)"
    failures=$((failures + 1))
else
    echo "OK: hit phase served by LMCache" \
        "(retrieves: $retrieves_before -> $retrieves_after)"
fi

hit_prompt=$(split_field "$OUT_DIR/hit.txt" prompt_tokens)
hit_external=$(split_field "$OUT_DIR/hit.txt" external_tokens)
if [ "$hit_external" -eq 0 ]; then
    echo "FAILED: hit phase resolved with 0 external tokens" \
        "(prompt=$hit_prompt); the comparison would be local-vs-local"
    failures=$((failures + 1))
elif [ $((hit_external * 2)) -lt "$hit_prompt" ]; then
    echo "FAILED: hit phase was mostly local" \
        "(external=$hit_external of prompt=$hit_prompt); the retrieved KV" \
        "is not what the completion was built from"
    failures=$((failures + 1))
else
    echo "OK: hit phase resolved as an external hit" \
        "(external=$hit_external of prompt=$hit_prompt)"
fi

if ! diff -u "$OUT_DIR/store.txt" "$OUT_DIR/hit.txt"; then
    echo "FAILED: hit-phase completion diverges from the engine-computed" \
        "ground truth (see diff above) -- retrieved KV is not faithful."
    failures=$((failures + 1))
else
    echo "OK: hit-phase completion matches exactly."
fi
echo ""

if [ "$failures" -gt 0 ]; then
    echo "hma_faithfulness test failed ($failures assertion(s))"
    exit 1
fi
echo "hma_faithfulness test passed"
