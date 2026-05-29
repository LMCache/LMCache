#!/usr/bin/env bash
# HMA (hybrid memory allocator) correctness test using a real hybrid model.
#
# openai/gpt-oss-20b interleaves sliding-window and full-attention layers, so
# vLLM keeps its hybrid KV cache manager ON for it (LMCacheMPConnector subclasses
# SupportsHMA, so vLLM does not auto-disable the hybrid manager). vLLM therefore
# exposes multiple KV cache groups and the connector exercises the multi-engine-
# group HMA store/retrieve path. gpt-oss-20b is ungated and uses standard paged
# attention for both layer families (so it is supported by LMCache's transfer
# kernels, unlike Mamba/linear-attention hybrids such as Qwen3.5/Qwen3-Next,
# whose state caches LMCache cannot yet transfer).
#
# Flow:
#   1. Run lm_eval (gsm8k) against vLLM+LMCache       -> populates LMCache (STORE).
#   2. Reset vLLM's *local* prefix cache (APC) only, leaving LMCache intact, via
#      the dev-mode endpoint POST /reset_prefix_cache (reset_external defaults to
#      false, so the LMCache-managed cache is preserved).
#   3. Re-run lm_eval                                  -> vLLM APC misses, so the
#      prefix KV is served by LMCache (RETRIEVE), exercising the HMA retrieve path.
#   4. Assert the two runs' gsm8k scores are aligned (LMCache retrieve is
#      numerically correct) and aligned with the no-LMCache baseline (full stack).
#   5. Assert LMCache actually served retrieves during run 2 (non-vacuous).
#
# Determinism comes from VLLM_BATCH_INVARIANT=1 (set by launch-processes.sh);
# the reset endpoint requires VLLM_SERVER_DEV_MODE=1 (also set there).
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
NUM_CONCURRENT="${NUM_CONCURRENT:-50}"
LIMIT="${LIMIT:-200}"
# Max allowed absolute difference in the gsm8k exact_match score between runs.
SCORE_TOLERANCE="${SCORE_TOLERANCE:-0.01}"
# Seconds to wait after run 1 so async LMCache stores drain before run 2.
STORE_DRAIN_SECONDS="${STORE_DRAIN_SECONDS:-20}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
# LMCache MP server log, scanned to confirm run 2 was served by LMCache retrieves.
LMCACHE_LOG="${LMCACHE_LOG:-/tmp/build_${BUILD_ID}_lmcache.log}"

HMA_DIR="$RESULTS_DIR/hma_lm_eval"
RUN1_DIR="$HMA_DIR/run1_store"
RUN2_DIR="$HMA_DIR/run2_retrieve"
BASELINE_DIR="$HMA_DIR/baseline"

echo "=== HMA lm_eval correctness test ==="
echo "Model: $MODEL"
echo "vLLM (LMCache) port: $VLLM_PORT"
echo "vLLM baseline port: $VLLM_BASELINE_PORT"
echo "Concurrent requests: $NUM_CONCURRENT"
echo "Limit: $LIMIT"
echo "Score tolerance: $SCORE_TOLERANCE"
echo "Results dir: $HMA_DIR"
echo ""

mkdir -p "$RUN1_DIR" "$RUN2_DIR" "$BASELINE_DIR"

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

reset_vllm_prefix_cache() {
    local port="$1"
    echo "=== Resetting vLLM local prefix cache on port $port (LMCache preserved) ==="
    # reset_external defaults to false -> only vLLM's APC is cleared.
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

count_retrieves() {
    # Number of completed LMCache retrieves logged so far (0 if log absent).
    # NB: ``grep -c`` prints 0 *and* exits 1 on no match, so guard the file
    # existence and use ``|| true`` (not ``|| echo 0``) to avoid emitting "0\n0".
    [ -f "$LMCACHE_LOG" ] || { echo 0; return; }
    grep -c "Retrieved" "$LMCACHE_LOG" 2>/dev/null || true
}

# ── 1. Cold run: compute + STORE into LMCache ───────────────
run_lm_eval "$VLLM_PORT" "$RUN1_DIR" "run1 LMCache STORE"

# Let async stores drain to the LMCache server before invalidating the APC.
echo "Waiting ${STORE_DRAIN_SECONDS}s for LMCache stores to drain..."
sleep "$STORE_DRAIN_SECONDS"

retrieves_before=$(count_retrieves)

# ── 2. Invalidate vLLM's local prefix cache (keep LMCache) ──
reset_vllm_prefix_cache "$VLLM_PORT"

# ── 3. Warm run: vLLM APC misses -> LMCache RETRIEVE ────────
run_lm_eval "$VLLM_PORT" "$RUN2_DIR" "run2 LMCache RETRIEVE"

retrieves_after=$(count_retrieves)

# ── 4. Baseline run: no LMCache, ground truth ──────────────
run_lm_eval "$VLLM_BASELINE_PORT" "$BASELINE_DIR" "baseline no LMCache"

# ── 5. Compare scores and verify LMCache was actually used ──
echo "============================================"
echo "=== Verifying HMA store/retrieve correctness ==="
echo "============================================"
echo "LMCache retrieves logged: before run2=${retrieves_before}, after run2=${retrieves_after}"

python3 - "$RUN1_DIR" "$RUN2_DIR" "$BASELINE_DIR" \
    "$SCORE_TOLERANCE" "$retrieves_before" "$retrieves_after" <<'PYEOF'
import glob
import json
import os
import sys

run1_dir, run2_dir, baseline_dir, tol_s, before_s, after_s = sys.argv[1:7]
tol = float(tol_s)
retrieves_before = int(before_s)
retrieves_after = int(after_s)


def gsm8k_exact_match(results_dir: str) -> float:
    """Return the gsm8k exact_match score from an lm_eval results dir.

    Prefers the strict-match variant; falls back to any non-stderr
    ``exact_match`` metric key.
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
        return float(metrics[preferred])
    for key, value in metrics.items():
        if key.startswith("exact_match,") and "stderr" not in key:
            return float(value)
    raise SystemExit(f"No exact_match metric in {latest}: {sorted(metrics)}")


s1 = gsm8k_exact_match(run1_dir)
s2 = gsm8k_exact_match(run2_dir)
sb = gsm8k_exact_match(baseline_dir)

print(f"  run1 (LMCache STORE)    gsm8k exact_match = {s1:.4f}")
print(f"  run2 (LMCache RETRIEVE) gsm8k exact_match = {s2:.4f}")
print(f"  baseline (no LMCache)   gsm8k exact_match = {sb:.4f}")
print(f"  tolerance = {tol}")

failures = []
if abs(s1 - s2) > tol:
    failures.append(
        f"LMCache retrieve score drift: |{s1:.4f} - {s2:.4f}| = "
        f"{abs(s1 - s2):.4f} > {tol}"
    )
if abs(s1 - sb) > tol:
    failures.append(
        f"Full-stack score drift vs baseline: |{s1:.4f} - {sb:.4f}| = "
        f"{abs(s1 - sb):.4f} > {tol}"
    )
# Non-vacuous: run 2 must have been served by LMCache retrieves, not recompute.
if retrieves_after <= retrieves_before:
    failures.append(
        "LMCache served no retrieves during run 2 "
        f"(before={retrieves_before}, after={retrieves_after}); "
        "the retrieve path was not exercised"
    )

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print(
    f"\nPASS: retrieve and baseline scores aligned within {tol}, and LMCache "
    f"served {retrieves_after - retrieves_before} retrieves during run 2."
)
PYEOF

echo ""
echo "============================================"
echo "=== HMA lm_eval correctness test passed ==="
echo "============================================"
