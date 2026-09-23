#!/usr/bin/env bash
# HMA transfer-faithfulness probe on a real hybrid model.
#
# One deterministic long-prefix request is sent against vLLM+LMCache in
# three phases:
#   1. Store phase: cache miss; vLLM computes the prefill and LMCache stores
#      the prefix KV. The completion is the ground truth (engine-computed KV).
#   2. Hit phase: after resetting vLLM's *local* prefix cache (APC) via the
#      dev-mode endpoint POST /reset_prefix_cache (LMCache preserved), the
#      byte-identical request; the whole prefix is served by LMCache
#      retrieve and consumed by the engine's own attention kernels.
#   3. Mixed phase: after a second APC reset, a truncated strict prefix of
#      the prompt is sent first so the APC holds only the leading block(s);
#      the full request then extends that local prefix with a deeper
#      LMCache tail -- the ext > 0 composition path, which is where
#      local-plus-external divergence on recurrent (Mamba/GDN) groups has
#      been reported (#4247 follow-ups). Pure-external hits (phase 2) do
#      not exercise it.
#
# Assertions (per served phase):
#   - The phase was actually served by LMCache (retrieve delta > 0), so
#     the comparison cannot pass vacuously by silently recomputing.
#   - The completion matches the ground truth exactly. Temperature 0 with a
#     single in-flight request is deterministic (GDN's batch-variance needs
#     concurrent load), so any divergence means the injected KV differs
#     from what the engine computed -- the silent-corruption class of #4247.
#
# Reading a failure: phase 2 red on the FLASHINFER arm only points at the
# kernel sub-paging path (#4253's territory); phase 3 red on both backend
# arms points at the mixed local/external composition path instead.
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

PROBE_DIR="$RESULTS_DIR/hma_probe"
mkdir -p "$PROBE_DIR"

echo "=== HMA transfer-faithfulness probe ==="
echo "Model: $MODEL"
echo "vLLM (LMCache) port: $VLLM_PORT"
echo "Rules in prefix: $NUM_RULES"
echo "Results dir: $PROBE_DIR"
echo ""

# Count completed LMCache retrieves recorded in the server log so far.
count_retrieves() {
    if [ -f "$LMCACHE_LOG" ]; then
        grep -c "Retrieved" "$LMCACHE_LOG" || true
    else
        echo 0
    fi
}

echo "=== Store phase (cache miss, engine-computed ground truth) ==="
python "${SCRIPT_DIR}/hma_probe_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --out "$PROBE_DIR/store.txt"
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
python "${SCRIPT_DIR}/hma_probe_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --out "$PROBE_DIR/hit.txt"
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

if ! diff -u "$PROBE_DIR/store.txt" "$PROBE_DIR/hit.txt"; then
    echo "FAILED: hit-phase completion diverges from the engine-computed" \
        "ground truth (see diff above) -- retrieved KV is not faithful."
    failures=$((failures + 1))
else
    echo "OK: hit-phase completion matches exactly."
fi
echo ""

echo "=== Resetting vLLM local prefix cache again for the mixed phase ==="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")
if [ "$code" != "200" ]; then
    echo "Failed to reset prefix cache (HTTP $code)."
    exit 1
fi
echo "vLLM prefix cache reset."
echo ""

echo "=== Warmup (truncated strict prefix; APC holds only leading blocks) ==="
python "${SCRIPT_DIR}/hma_probe_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --cut-at-rule 30 --max-tokens 4 --out "$PROBE_DIR/warmup.txt"
echo ""

retrieves_before=$(count_retrieves)

echo "=== Mixed phase (local APC prefix + deeper LMCache tail) ==="
python "${SCRIPT_DIR}/hma_probe_client.py" \
    --port "$VLLM_PORT" --model "$MODEL" --num-rules "$NUM_RULES" \
    --out "$PROBE_DIR/mixed.txt"
echo ""

retrieves_after=$(count_retrieves)

if [ "$retrieves_after" -le "$retrieves_before" ]; then
    echo "FAILED: LMCache served no retrieves during the mixed phase" \
        "(before=$retrieves_before, after=$retrieves_after)"
    failures=$((failures + 1))
else
    echo "OK: mixed phase served by LMCache" \
        "(retrieves: $retrieves_before -> $retrieves_after)"
fi

if ! diff -u "$PROBE_DIR/store.txt" "$PROBE_DIR/mixed.txt"; then
    echo "FAILED: mixed-phase completion diverges from the engine-computed" \
        "ground truth (see diff above) -- the local-prefix-plus-external-" \
        "tail composition is not faithful."
    failures=$((failures + 1))
else
    echo "OK: mixed-phase completion matches exactly."
fi

if [ "$failures" -gt 0 ]; then
    echo "hma_probe test failed ($failures assertion(s))"
    exit 1
fi
echo "hma_probe test passed"
