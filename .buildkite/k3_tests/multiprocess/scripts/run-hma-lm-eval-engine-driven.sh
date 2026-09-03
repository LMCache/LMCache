#!/usr/bin/env bash
# Engine-driven HMA (hybrid memory allocator) correctness test.
#
# Identical to run-hma-lm-eval.sh in flow and coverage, but forces the
# multiprocess transfer path to Engine-driven (not the default LMCache-driven
# / AUTO selection). This ensures the Engine-driven path is exercised on CUDA
# workers with a real hybrid model.
#
# Forced via LMCACHE_MP_TRANSFER_MODE=engine_driven (environment variable
# respected by TransferContextFactory before AUTO detection runs).  The CI
# pipeline also sets lmcache.mp.mp_transfer_mode=engine_driven in
# kv_connector_extra_config for belt-and-suspenders coverage.
#
# Models (selected by run-single-test.sh):
#   - google/gemma-4-31B-it: sliding-window + full-attention hybrid whose full
#     layers have a larger head_dim, exercising per-group HMA store/retrieve
#     via Engine-driven path.
#   - Qwen/Qwen3.5-0.8B: Mamba/GDN + full-attention hybrid; exercising the
#     registration-time cache re-views (kv_cache_group_edits.py).
#
# Flow (single GPU, no baseline server):
#   1. Sanity-check the LMCache server log to confirm Engine-driven context was
#      created (not LMCache-driven).
#   2. vLLM run: lm_eval (gsm8k) against vLLM+LMCache, populating LMCache.
#   3. Reset vLLM's *local* prefix cache (APC) only, leaving LMCache intact, via
#      the dev-mode endpoint POST /reset_prefix_cache.
#   4. LMCache retrieve run: re-run lm_eval; vLLM APC misses → LMCache serves KV.
#   5. Assert the two runs' gsm8k scores match.
#   6. Assert LMCache actually served retrieves in the retrieve run (non-vacuous).
#
# The reset endpoint requires VLLM_SERVER_DEV_MODE=1 (set by launch-processes.sh).
# The LMCACHE_MP_TRANSFER_MODE and ENGINE_DRIVEN_TRANSPORT must be set before
# launching the servers (already done by the pipeline matrix command).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-google/gemma-4-31B-it}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-100}"
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0}"
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/build_${BUILD_ID}_lmcache.log}"

# Must be set: Engine-driven transfer path override.
LMCACHE_MP_TRANSFER_MODE="${LMCACHE_MP_TRANSFER_MODE:-engine_driven}"
ENGINE_DRIVEN_TRANSPORT="${ENGINE_DRIVEN_TRANSPORT:-pickle}"
case "$ENGINE_DRIVEN_TRANSPORT" in
    pickle|shm) ;;
    *)
        echo "ERROR: ENGINE_DRIVEN_TRANSPORT must be 'pickle' or 'shm', got '$ENGINE_DRIVEN_TRANSPORT'"
        exit 1
        ;;
esac

HMA_ED_DIR="$RESULTS_DIR/hma_lm_eval_engine_driven"
VLLM_RUN_DIR="$HMA_ED_DIR/vllm_run"
RETRIEVE_RUN_DIR="$HMA_ED_DIR/retrieve_run"

echo "=== Engine-driven HMA lm_eval correctness test ==="
echo "Model: $MODEL"
echo "Transfer mode: $LMCACHE_MP_TRANSFER_MODE"
echo "Engine-driven transport: $ENGINE_DRIVEN_TRANSPORT"
echo "vLLM (LMCache) port: $VLLM_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Score tolerance: $SCORE_TOLERANCE"
echo "Results dir: $HMA_ED_DIR"
echo ""

mkdir -p "$VLLM_RUN_DIR" "$RETRIEVE_RUN_DIR"

# ---------------------------------------------------------------------------
# Verify Engine-driven context selected the expected transport.
#
# The strategy is selected while registering the Engine-driven context. Fail
# fast so the matrix cannot pass with the wrong transport.
# ---------------------------------------------------------------------------
assert_engine_driven_transport_active() {
    echo "=== Asserting Engine-driven $ENGINE_DRIVEN_TRANSPORT transport is active ==="
    local marker
    case "$ENGINE_DRIVEN_TRANSPORT" in
        pickle) marker="Using pickle non-GPU transfer strategy" ;;
        shm) marker="Using shm non-GPU transfer strategy" ;;
    esac
    local max_wait=30
    local elapsed=0
    while [ "$elapsed" -lt "$max_wait" ]; do
        if [ -f "$LMCACHE_LOG" ] && grep -qi "$marker" "$LMCACHE_LOG" 2>/dev/null; then
            echo "Engine-driven $ENGINE_DRIVEN_TRANSPORT transport confirmed in LMCache log."
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo ""
    echo "ERROR: LMCache log did not contain '$marker' after ${max_wait}s."
    echo "       Ensure LMCACHE_MP_TRANSFER_MODE=engine_driven and"
    echo "       ENGINE_DRIVEN_TRANSPORT=$ENGINE_DRIVEN_TRANSPORT are set before startup."
    echo "       Log tail (last 20 lines):"
    tail -20 "$LMCACHE_LOG" 2>/dev/null || true
    return 1
}

# Run one lm_eval gsm8k pass against a vLLM OpenAI-compatible server.
run_lm_eval() {
    local port="$1"
    local output_dir="$2"
    local run_name="$3"

    echo "=== Running lm_eval ($run_name) on port $port ==="
    lm_eval --model local-completions --tasks gsm8k \
        --model_args "model=${MODEL},base_url=http://127.0.0.1:${port}/v1/completions,num_concurrent=${NUM_CONCURRENT},max_retries=3,tokenized_requests=False" \
        --limit "$LIMIT" \
        --seed 0 \
        -s --output_path "$output_dir" \
        --gen_kwargs '{"temperature": 0.0}'
    echo "$run_name completed"
    echo ""
}

# Reset a vLLM server's local prefix cache (APC) while preserving LMCache.
reset_vllm_prefix_cache() {
    local port="$1"
    echo "=== Resetting vLLM local prefix cache on port $port (LMCache preserved) ==="
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
        "http://127.0.0.1:${port}/reset_prefix_cache")
    if [ "$code" != "200" ]; then
        echo "Failed to reset prefix cache (HTTP $code). Is VLLM_SERVER_DEV_MODE=1?"
        return 1
    fi
    echo "vLLM prefix cache reset."
    echo ""
}

# Count completed LMCache retrieves in the server log.
count_retrieves() {
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

# ── 0. Assert Engine-driven is active before any evaluation ─
assert_engine_driven_transport_active

# ── 1. vLLM run: compute from scratch, populating LMCache ───
run_lm_eval "$VLLM_PORT" "$VLLM_RUN_DIR" "vLLM run (Engine-driven)"

# Let async stores drain to the LMCache server before invalidating the APC.
echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
sleep "$STORE_DRAIN_SECONDS"

retrieves_before=$(count_retrieves)

# ── 2. Invalidate vLLM's local prefix cache (keep LMCache) ──
reset_vllm_prefix_cache "$VLLM_PORT"

# ── 3. Retrieve run: vLLM APC misses → LMCache serves the KV ─
run_lm_eval "$VLLM_PORT" "$RETRIEVE_RUN_DIR" "LMCache Engine-driven retrieve run"

retrieves_after=$(count_retrieves)

# ── 4. Compare scores and verify LMCache was actually used ──
echo "============================================"
echo "=== Verifying Engine-driven HMA store/retrieve correctness ==="
echo "============================================"
echo "LMCache retrieves logged: before=${retrieves_before}, after=${retrieves_after}"

python3 - "$VLLM_RUN_DIR" "$RETRIEVE_RUN_DIR" \
    "$SCORE_TOLERANCE" "$retrieves_before" "$retrieves_after" <<'PYEOF'
import glob
import json
import os
import sys

vllm_run_dir, retrieve_run_dir, tol_s, before_s, after_s = sys.argv[1:6]
tol = float(tol_s)
retrieves_before = int(before_s)
retrieves_after = int(after_s)


def gsm8k_score_and_stderr(results_dir: str) -> tuple[float, float]:
    """Return the gsm8k (exact_match, stderr) from an lm_eval results directory.

    Prefers the strict-match variant; falls back to any non-stderr
    ``exact_match`` metric key paired with its ``exact_match_stderr`` twin.

    Args:
        results_dir: Directory passed to ``lm_eval --output_path``. Searched
            recursively for the newest ``results_*.json``.

    Returns:
        ``(score, stderr)``: the gsm8k ``exact_match`` accuracy in
        ``[0.0, 1.0]`` and its reported sampling stderr (0.0 if absent).

    Raises:
        SystemExit: If no ``results_*.json`` exists or the newest one has no
            ``exact_match`` metric for the gsm8k task.
    """
    files = glob.glob(os.path.join(results_dir, "**", "results_*.json"), recursive=True)
    if not files:
        raise SystemExit(f"No results_*.json under {results_dir}")
    latest = max(files, key=os.path.getmtime)
    with open(latest) as f:
        data = json.load(f)
    metrics = data["results"]["gsm8k"]
    preferred = "exact_match,strict-match"
    if preferred in metrics:
        stderr = float(metrics.get("exact_match_stderr,strict-match", 0.0))
        return float(metrics[preferred]), stderr
    for key, value in metrics.items():
        if key.startswith("exact_match,") and "stderr" not in key:
            variant = key.split(",", 1)[1]
            stderr = float(metrics.get(f"exact_match_stderr,{variant}", 0.0))
            return float(value), stderr
    raise SystemExit(f"No exact_match metric in {latest}: {sorted(metrics)}")


s_vllm, e_vllm = gsm8k_score_and_stderr(vllm_run_dir)
s_retrieve, e_retrieve = gsm8k_score_and_stderr(retrieve_run_dir)

print(f"  vLLM run (Engine-driven)   gsm8k exact_match = {s_vllm:.4f} +/- {e_vllm:.4f}")
print(f"  Engine-driven retrieve run gsm8k exact_match = {s_retrieve:.4f} +/- {e_retrieve:.4f}")
print(f"  tolerance = {tol}")

failures = []
if abs(s_vllm - s_retrieve) > tol:
    failures.append(
        f"score drift between runs: |{s_vllm:.4f} - {s_retrieve:.4f}| = "
        f"{abs(s_vllm - s_retrieve):.4f} > {tol}"
    )
if retrieves_after <= retrieves_before:
    failures.append(
        "Engine-driven LMCache served no retrieves during the retrieve run "
        f"(before={retrieves_before}, after={retrieves_after})"
    )

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: vLLM and Engine-driven LMCache-retrieve gsm8k scores match (tol={tol}); "
    f"LMCache served {retrieves_after - retrieves_before} retrieves."
)
PYEOF

echo ""
echo "============================================"
echo "=== Engine-driven HMA lm_eval correctness test passed ==="
echo "============================================"
