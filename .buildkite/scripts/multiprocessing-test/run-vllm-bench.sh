#!/bin/bash
# Run vllm bench serve test against both vLLM servers
# Compares performance between LMCache-enabled and baseline vLLM

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "$SCRIPT_DIR/common.sh"

# Configuration
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-10000}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1}"
LMCACHE_LOG_FILE="${LMCACHE_LOG_FILE:-/tmp/build_${BUILD_ID}_lmcache.log}"

# Expected values
EXPECTED_TOTAL_INPUT_TOKENS=$((NUM_PROMPTS * RANDOM_INPUT_LEN))
EXPECTED_COMPLETED=$NUM_PROMPTS
MAX_SLOWDOWN_PERCENT="${MAX_SLOWDOWN_PERCENT:-5}"

# Generate a random seed once for reproducibility across both benchmarks
RANDOM_SEED="${RANDOM_SEED:-$(date +%s)}"

# Output directory (subdirectory of shared RESULTS_DIR)
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
echo "Virtual env: $VENV_DIR"
echo "Build ID: $BUILD_ID"
echo "Results dir: $VLLM_BENCH_DIR"
echo ""

# Create results directory
mkdir -p "$VLLM_BENCH_DIR"
mkdir -p "$CACHE_HIT_DIR"

# Run vllm bench serve
run_vllm_bench() {
    local port="$1"
    local result_filename="$2"
    local description="$3"
    local seed="$4"
    
    echo "=== Running vllm bench serve ($description) ==="
    echo "Port: $port"
    echo "Seed: $seed"
    echo "Result file: $VLLM_BENCH_DIR/$result_filename"
    
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
    
    echo "✅ $description benchmark completed"
    echo ""
}

# Extract a numeric field from JSON file
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

# Verify benchmark results
verify_results() {
    local lmcache_result="$VLLM_BENCH_DIR/lmcache.json"
    local baseline_result="$VLLM_BENCH_DIR/baseline.json"
    
    echo "=== Verifying benchmark results ==="
    
    # Check if result files exist
    if [ ! -f "$lmcache_result" ]; then
        echo "❌ LMCache result file not found: $lmcache_result"
        return 1
    fi
    
    if [ ! -f "$baseline_result" ]; then
        echo "❌ Baseline result file not found: $baseline_result"
        return 1
    fi
    
    echo "LMCache result: $lmcache_result"
    echo "Baseline result: $baseline_result"
    echo ""
    
    # Extract values from LMCache result
    lmcache_total_input_tokens=$(extract_json_field "$lmcache_result" "total_input_tokens")
    lmcache_completed=$(extract_json_field "$lmcache_result" "completed")
    lmcache_throughput=$(extract_json_field "$lmcache_result" "total_token_throughput")
    
    # Extract values from baseline result
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
    
    # Verification
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
            echo "✅ $label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance)"
        else
            echo "❌ $label total_input_tokens: $actual (expected: $EXPECTED_TOTAL_INPUT_TOKENS ±$token_tolerance)"
            failed=1
        fi
    }

    check_input_tokens "LMCache" "$lmcache_total_input_tokens"
    check_input_tokens "Baseline" "$baseline_total_input_tokens"
    
    # Check completed for LMCache
    if [ "$lmcache_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "✅ LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED)"
    else
        echo "❌ LMCache completed: $lmcache_completed (expected: $EXPECTED_COMPLETED)"
        failed=1
    fi
    
    # Check completed for baseline
    if [ "$baseline_completed" -eq "$EXPECTED_COMPLETED" ] 2>/dev/null; then
        echo "✅ Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED)"
    else
        echo "❌ Baseline completed: $baseline_completed (expected: $EXPECTED_COMPLETED)"
        failed=1
    fi
    
    # Check throughput comparison against the configured slowdown allowance.
    throughput_check=$(python3 -c "
lmcache_tp = $lmcache_throughput
baseline_tp = $baseline_throughput
max_slowdown = $MAX_SLOWDOWN_PERCENT

# Calculate the minimum acceptable throughput.
min_acceptable = baseline_tp * (1 - max_slowdown / 100.0)

# Calculate actual slowdown percentage
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
        echo "✅ Throughput comparison: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%)"
    else
        echo "❌ Throughput comparison: LMCache is ${slowdown_pct}% slower (max allowed: ${MAX_SLOWDOWN_PERCENT}%)"
        failed=1
    fi
    
    echo ""
    
    if [ "$failed" -eq 1 ]; then
        return 1
    fi
    
    return 0
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
    sleep 2

    send_cache_hit_request "Warm" "$warm_file"
    validate_cache_hit_response "Warm" "$warm_file"
    sleep 2

    retrieve_after=$(count_retrieve_log_lines)
    validate_cache_hit_progression "$cold_file" "$warm_file"
    validate_retrieve_log_growth "$retrieve_before" "$retrieve_after"
    echo ""
}

# Main execution
main() {
    setup_venv vllm openai
    
    echo "Using random seed: $RANDOM_SEED"
    echo ""
    
    # Run benchmark against baseline vLLM (without LMCache)
    echo "============================================"
    echo "=== Benchmark: Baseline vLLM (without LMCache) ==="
    echo "============================================"
    run_vllm_bench "$VLLM_BASELINE_PORT" "baseline.json" "Baseline vLLM" "$RANDOM_SEED"
    
    # Run benchmark against vLLM with LMCache
    echo "============================================"
    echo "=== Benchmark: vLLM with LMCache ==="
    echo "============================================"
    run_vllm_bench "$VLLM_PORT" "lmcache.json" "vLLM with LMCache" "$RANDOM_SEED"
    
    # Verify results
    echo "============================================"
    echo "=== Verifying benchmark results ==="
    echo "============================================"
    if ! verify_results; then
        echo "❌ Verification failed"
        exit 1
    fi

    if ! verify_cache_hit_replay; then
        echo "❌ Cache-hit replay validation failed"
        exit 1
    fi
    
    echo "============================================"
    echo "=== ✅ vLLM Bench test completed ==="
    echo "============================================"
    echo "Results saved to: $VLLM_BENCH_DIR"
    echo "  - LMCache: $VLLM_BENCH_DIR/lmcache.json"
    echo "  - Baseline: $VLLM_BENCH_DIR/baseline.json"
}

main "$@"
