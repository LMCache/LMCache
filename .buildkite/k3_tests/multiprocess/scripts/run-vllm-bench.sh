#!/usr/bin/env bash
# Run vllm bench serve test against both vLLM servers.
# Compares performance between LMCache-enabled and baseline vLLM.
# Adapted from the old Docker-based run-vllm-bench.sh.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
LMCACHE_LOG_FILE="${LMCACHE_LOG_FILE:-/tmp/build_${BUILD_ID}_lmcache.log}"

# Expected values
EXPECTED_TOTAL_INPUT_TOKENS=$((NUM_PROMPTS * RANDOM_INPUT_LEN))
EXPECTED_COMPLETED=$NUM_PROMPTS
MAX_SLOWDOWN_PERCENT="${MAX_SLOWDOWN_PERCENT:-5}"

# Reproducible seed
RANDOM_SEED="${RANDOM_SEED:-$(date +%s)}"

# Output directory
VLLM_BENCH_DIR="$RESULTS_DIR/vllm_bench"
CACHE_HIT_DIR="$VLLM_BENCH_DIR/cache_hit_validation"

# A repeated long prompt is used after the random benchmark to validate that
# the LMCache-backed vLLM path performs a real retrieve on a warm request.
LONG_CACHE_HIT_CONTENT="Explain the history of computer science in great detail. $(printf 'The Turing machine is a fundamental concept in theoretical computer science that defines an abstract machine capable of manipulating symbols on a strip of tape according to a table of rules. %.0s' {1..20})"

echo "=== vLLM Bench Serve Test ==="
echo "Model: $MODEL"
echo "vLLM Port (with LMCache): $VLLM_PORT"
echo "vLLM Baseline Port (without LMCache): $VLLM_BASELINE_PORT"
echo "Number of prompts: $NUM_PROMPTS"
echo "Random input length: $RANDOM_INPUT_LEN"
echo "Random output length: $RANDOM_OUTPUT_LEN"
echo "Results dir: $VLLM_BENCH_DIR"
echo ""

mkdir -p "$VLLM_BENCH_DIR"
mkdir -p "$CACHE_HIT_DIR"

run_vllm_bench() {
    local port="$1"
    local result_filename="$2"
    local description="$3"
    local seed="$4"

    echo "=== Running vllm bench serve ($description) ==="
    echo "Port: $port, Seed: $seed"

    vllm bench serve \
        --seed "$seed" \
        --port "$port" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --ignore-eos \
        --backend openai-chat \
        --endpoint /v1/chat/completions \
        --result-dir "$VLLM_BENCH_DIR" \
        --result-filename "$result_filename" \
        --save-result

    echo "$description benchmark completed"
    echo ""
}

extract_json_field() {
    local json_file="$1"
    local field="$2"
    python3 -c "
import json
with open('$json_file', 'r') as f:
    data = json.load(f)
print(data.get('$field', 'null'))
"
}

verify_results() {
    local lmcache_result="$VLLM_BENCH_DIR/lmcache.json"
    local baseline_result="$VLLM_BENCH_DIR/baseline.json"

    echo "=== Verifying benchmark results ==="

    if [ ! -f "$lmcache_result" ]; then
        echo "LMCache result file not found: $lmcache_result"
        return 1
    fi
    if [ ! -f "$baseline_result" ]; then
        echo "Baseline result file not found: $baseline_result"
        return 1
    fi

    # Extract values
    lmcache_total_input_tokens=$(extract_json_field "$lmcache_result" "total_input_tokens")
    lmcache_completed=$(extract_json_field "$lmcache_result" "completed")
    lmcache_throughput=$(extract_json_field "$lmcache_result" "total_token_throughput")

    baseline_total_input_tokens=$(extract_json_field "$baseline_result" "total_input_tokens")
    baseline_completed=$(extract_json_field "$baseline_result" "completed")
    baseline_throughput=$(extract_json_field "$baseline_result" "total_token_throughput")

    echo "=== LMCache Results ==="
    echo "  total_input_tokens: $lmcache_total_input_tokens"
    echo "  completed: $lmcache_completed"
    echo "  total_token_throughput: $lmcache_throughput"
    echo ""
    echo "=== Baseline Results ==="
    echo "  total_input_tokens: $baseline_total_input_tokens"
    echo "  completed: $baseline_completed"
    echo "  total_token_throughput: $baseline_throughput"
    echo ""

    local failed=0

    echo "=== Verification ==="

    # vLLM's random dataset decodes and re-encodes token sequences, which can
    # drift slightly from the requested length (see RandomDataset in
    # vllm/benchmarks/datasets.py). Allow 1% tolerance.
    local token_tolerance=$((EXPECTED_TOTAL_INPUT_TOKENS / 100))

    check_input_tokens() {
        local label="$1"
        local actual="$2"
        local diff=$((actual - EXPECTED_TOTAL_INPUT_TOKENS))
        local abs_diff=${diff#-}
        if [ "$abs_diff" -le "$token_tolerance" ] 2>/dev/null; then
            echo "$label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance) PASS"
        else
            echo "$label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance) FAIL"
            failed=1
        fi
    }

    check_input_tokens "LMCache" "$lmcache_total_input_tokens"
    check_input_tokens "Baseline" "$baseline_total_input_tokens"

    if [ "$lmcache_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED) PASS"
    else
        echo "LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED) FAIL"
        failed=1
    fi

    if [ "$baseline_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED) PASS"
    else
        echo "Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED) FAIL"
        failed=1
    fi

    # Throughput comparison
    throughput_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
max_slowdown = $MAX_SLOWDOWN_PERCENT
min_acceptable = baseline_tp * (1 - max_slowdown / 100.0)
if baseline_tp > 0:
    slowdown_pct = ((baseline_tp - lmcache_tp) / baseline_tp) * 100
else:
    slowdown_pct = 0
if lmcache_tp >= min_acceptable:
    print(f'PASS|{slowdown_pct:.2f}')
else:
    print(f'FAIL|{slowdown_pct:.2f}')
")

    throughput_status=$(echo "$throughput_check" | cut -d'|' -f1)
    slowdown_pct=$(echo "$throughput_check" | cut -d'|' -f2)

    if [ "$throughput_status" = "PASS" ]; then
        echo "Throughput: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%) PASS"
    else
        echo "Throughput: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%) FAIL"
        failed=1
    fi

    # Sanity check: on a random (no-reuse) workload, LMCache should NOT be
    # significantly faster than baseline. If it is, the benchmark setup is
    # asymmetric and the results are unreliable as a regression test.
    local max_speedup_pct=10
    speedup_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
if baseline_tp > 0:
    speedup_pct = ((lmcache_tp - baseline_tp) / baseline_tp) * 100
else:
    speedup_pct = 0
if speedup_pct > $max_speedup_pct:
    print(f'WARN|{speedup_pct:.2f}')
else:
    print(f'OK|{speedup_pct:.2f}')
")

    local speedup_status speedup_pct
    speedup_status=$(echo "$speedup_check" | cut -d'|' -f1)
    speedup_pct=$(echo "$speedup_check" | cut -d'|' -f2)

    if [ "$speedup_status" = "WARN" ]; then
        echo "WARNING: LMCache is ${speedup_pct}% faster than baseline on random workload (max expected: ${max_speedup_pct}%)"
        echo "This suggests a measurement asymmetry, not a real cache benefit."
        failed=1
    else
        echo "Speedup sanity check: LMCache is ${speedup_pct}% faster (max expected: ${max_speedup_pct}%) OK"
    fi

    echo ""
    return "$failed"
}

warmup_server() {
    local port="$1"
    local description="$2"
    local num_warmup="${3:-3}"

    echo "=== Warming up $description (port $port) ==="
    # Send a few chat completion requests to warm up the tokenizer,
    # chat template (Jinja2), and engine pipeline. Without this, the
    # first-ever batch of requests incurs ~25s of cold-start overhead
    # (BPE compilation, template compilation, etc.) which skews the
    # benchmark since lm-eval (Step 3) only warms the LMCache server.
    for i in $(seq 1 "$num_warmup"); do
        curl -s -X POST "http://localhost:${port}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"${MODEL}\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Warmup request ${i}. The quick brown fox jumps over the lazy dog.\"}],
                \"max_tokens\": 1
            }" > /dev/null 2>&1
    done
    echo "$description warmup complete"
}

count_retrieve_log_lines() {
    if [ ! -f "$LMCACHE_LOG_FILE" ]; then
        echo 0
        return 0
    fi
    python3 - <<'PY' "$LMCACHE_LOG_FILE"
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
pattern = re.compile(r"Retrieved \d+ tokens in ")
count = sum(1 for line in log_path.read_text(errors="ignore").splitlines() if pattern.search(line))
print(count)
PY
}

send_cache_hit_request() {
    local label="$1"
    local output_file="$2"

    echo "--- Sending ${label} cache-hit validation request ---"
    local http_code
    http_code=$(curl -s -o "$output_file" -w "%{http_code}" \
        -X POST "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json; print(json.dumps('$LONG_CACHE_HIT_CONTENT'))")}],
            \"max_tokens\": 1,
            \"kv_transfer_params\": {\"cached_token_stats\": true}
        }")

    if [ "$http_code" -ne 200 ]; then
        echo "${label} request returned HTTP $http_code"
        cat "$output_file" 2>/dev/null || true
        return 1
    fi
    echo "${label} request completed (HTTP 200)"
}

validate_cache_hit_response() {
    local label="$1"
    local response_file="$2"

    python3 - <<'PY' "$label" "$response_file"
import json
import sys

label = sys.argv[1]
response_file = sys.argv[2]
with open(response_file) as f:
    data = json.load(f)

stats = (data.get("kv_transfer_params") or {}).get("cached_token_stats")
if stats is None:
    print(f"{label}: missing kv_transfer_params.cached_token_stats")
    sys.exit(1)

required_keys = [
    "num_vllm_cached_tokens",
    "num_lmcache_cached_tokens",
    "num_lmcache_extra_cached_tokens",
]
missing = [k for k in required_keys if k not in stats]
if missing:
    print(f"{label}: missing cached_token_stats keys: {missing}")
    sys.exit(1)

for key in required_keys:
    value = stats[key]
    if not isinstance(value, int) or value < 0:
        print(f"{label}: {key} should be a non-negative integer, got {value!r}")
        sys.exit(1)

print(
    f"{label}: vLLM cached={stats['num_vllm_cached_tokens']}, "
    f"LMCache cached={stats['num_lmcache_cached_tokens']}, "
    f"LMCache extra={stats['num_lmcache_extra_cached_tokens']}"
)
PY
}

validate_cache_hit_progression() {
    local cold_file="$1"
    local warm_file="$2"

    python3 - <<'PY' "$cold_file" "$warm_file"
import json
import sys

cold_file, warm_file = sys.argv[1:3]
with open(cold_file) as f:
    cold = json.load(f)
with open(warm_file) as f:
    warm = json.load(f)

cold_stats = cold["kv_transfer_params"]["cached_token_stats"]
warm_stats = warm["kv_transfer_params"]["cached_token_stats"]

cold_lmcache = cold_stats["num_lmcache_cached_tokens"]
warm_lmcache = warm_stats["num_lmcache_cached_tokens"]

print(f"Cold LMCache cached tokens: {cold_lmcache}")
print(f"Warm LMCache cached tokens: {warm_lmcache}")

if warm_lmcache <= cold_lmcache:
    print("Warm request should report more LMCache cached tokens than cold request")
    sys.exit(1)

if warm_lmcache == 0:
    print("Warm request reported 0 LMCache cached tokens")
    sys.exit(1)
PY
}

validate_retrieve_log_growth() {
    local retrieve_before="$1"
    local retrieve_after="$2"

    echo "LMCache retrieve log lines before replay: $retrieve_before"
    echo "LMCache retrieve log lines after replay:  $retrieve_after"

    if [ "$retrieve_after" -le "$retrieve_before" ]; then
        echo "Expected LMCache log to record at least one retrieve during warm replay"
        if [ -f "$LMCACHE_LOG_FILE" ]; then
            echo "--- Last 80 LMCache log lines ---"
            tail -n 80 "$LMCACHE_LOG_FILE"
        fi
        return 1
    fi
}

verify_cache_hit_replay() {
    local cold_file="$CACHE_HIT_DIR/cold_response.json"
    local warm_file="$CACHE_HIT_DIR/warm_response.json"
    local retrieve_before
    local retrieve_after

    echo "============================================"
    echo "=== Cache Hit Replay Validation ==="
    echo "============================================"

    retrieve_before=$(count_retrieve_log_lines)
    send_cache_hit_request "Cold" "$cold_file"
    validate_cache_hit_response "Cold" "$cold_file"

    # Allow the asynchronous store path to commit objects before replaying the
    # same prompt. Without this short delay the second request can race the
    # offload and spuriously observe a cold cache.
    sleep 2

    send_cache_hit_request "Warm" "$warm_file"
    validate_cache_hit_response "Warm" "$warm_file"
    sleep 2

    retrieve_after=$(count_retrieve_log_lines)
    validate_cache_hit_progression "$cold_file" "$warm_file"
    validate_retrieve_log_growth "$retrieve_before" "$retrieve_after"
    echo ""
}

echo "Using random seed: $RANDOM_SEED"
echo ""

# Warm up the active benchmark server(s). In single-instance mode the baseline
# server is intentionally disabled, so skip the warmup/benchmark phases that
# require it.
echo "============================================"
echo "=== Warming up servers ==="
echo "============================================"
warmup_server "$VLLM_PORT" "vLLM with LMCache"
if [ "${LAUNCH_BASELINE:-true}" = "true" ]; then
    warmup_server "$VLLM_BASELINE_PORT" "vLLM baseline"
fi
echo ""

if [ "${LAUNCH_BASELINE:-true}" = "true" ]; then
    echo "============================================"
    echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
    echo "============================================"
    run_vllm_bench "$VLLM_BASELINE_PORT" "baseline.json" "Baseline vLLM" "$RANDOM_SEED"
fi

# LMCache
echo "============================================"
echo "=== Benchmark: vLLM with LMCache ==="
echo "============================================"
run_vllm_bench "$VLLM_PORT" "lmcache.json" "vLLM with LMCache" "$RANDOM_SEED"

# Verify
echo "============================================"
echo "=== Verifying benchmark results ==="
echo "============================================"
if [ "${LAUNCH_BASELINE:-true}" = "true" ]; then
    if ! verify_results; then
        echo "Verification failed"
        exit 1
    fi
else
    echo "Single-instance benchmark mode: skipping baseline-vs-LMCache comparison"
fi

if ! verify_cache_hit_replay; then
    echo "Cache-hit replay validation failed"
    exit 1
fi

echo "============================================"
echo "=== vLLM Bench test completed ==="
echo "============================================"
