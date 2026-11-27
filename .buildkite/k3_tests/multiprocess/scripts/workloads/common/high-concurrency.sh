#!/usr/bin/env bash
# High-concurrency workload for an OpenAI-compatible inference engine.
set -euo pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"

NUM_REQUESTS="${NUM_REQUESTS:-100}"
REQUEST_LENGTH="${REQUEST_LENGTH:-10000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
KV_CACHE_VOLUME="${KV_CACHE_VOLUME:-15}"
OUTPUT_DIR="${RESULTS_DIR}/high_concurrency"
SUMMARY_FILE="${OUTPUT_DIR}/bench_summary.json"

mkdir -p "$OUTPUT_DIR"
echo "=== High-concurrency workload ==="
echo "Engine: $ENGINE_NAME"
echo "Requests: $NUM_REQUESTS"
echo "Request length: $REQUEST_LENGTH"

timeout "$TIMEOUT_SECONDS" lmcache bench engine \
    --engine-url "http://127.0.0.1:${ENGINE_PORT}" \
    --model "$MODEL" \
    --workload random-prefill \
    --rp-num-requests "$NUM_REQUESTS" \
    --rp-request-length "$REQUEST_LENGTH" \
    --kv-cache-volume "$KV_CACHE_VOLUME" \
    --json \
    --no-csv \
    --no-interactive \
    --quiet \
    --output-dir "$OUTPUT_DIR"

if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "High-concurrency summary not found: $SUMMARY_FILE" >&2
    exit 1
fi

python3 - "$SUMMARY_FILE" "$NUM_REQUESTS" <<'PY'
import json
import sys

summary_file, expected = sys.argv[1], int(sys.argv[2])
with open(summary_file) as result_file:
    metrics = json.load(result_file).get("results", {})
successful = int(metrics.get("successful_requests", 0))
failed = int(metrics.get("failed_requests", 0))
print(f"successful_requests={successful}, failed_requests={failed}")
if successful != expected or failed != 0:
    raise SystemExit(
        f"Expected {expected} successful requests and 0 failures, "
        f"got {successful} successful and {failed} failed"
    )
print("High-concurrency workload passed")
PY
