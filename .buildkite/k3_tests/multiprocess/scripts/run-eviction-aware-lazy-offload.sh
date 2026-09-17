#!/usr/bin/env bash
# Verify EVICTION_AWARE lazy offload against a live GPU vLLM + LMCache MP
# deployment.
#
# run-lazy-offload.sh covers FIFO, which drains once enough requests have
# finished. EVICTION_AWARE ignores request counts and decides on GPU block
# pressure: a buffered store comes due when one of its blocks sits within the
# danger depth of the free queue, and on an idle engine that depth is zero, so
# nothing drains at all. Those are two separate behaviors and each needs its
# own workload.
#
# Phase 1 sends three small requests to an otherwise idle engine and requires
# that nothing is written to L1. This is the deferral half, and it also tells
# the policies apart: eager would write after each request, and FIFO at the
# harness default threshold of 2 would write on the third.
#
# Phase 2 sends a long-document workload whose working set is twice the pinned
# GPU block pool, so the pool turns over and the deferred stores come due. L1
# writes must rise, and the policy's own counter ledger must show that it
# emitted and that the counts close.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="${RESULTS_DIR}/eviction_aware_lazy_offload"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
LMCACHE_DIR="${LMCACHE_DIR:-$REPO_ROOT}"

# Set by run-single-test.sh; repeated here so the script also runs by hand.
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-2048}"
# vLLM's default paged-block size. The pool is this many tokens per block.
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"

# Phase 2 workload. The product below is deliberately twice the pool: one pass
# fills it and the second forces the turnover the policy reacts to.
DOCUMENT_LENGTH="${DOCUMENT_LENGTH:-8000}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-10}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-2}"

# Store operations the policy must have released by the end. One proves the
# drain path ran end to end; raise it once a green run shows the real figure.
MIN_EMITTED="${MIN_EMITTED:-1}"

mkdir -p "${TEST_DIR}"

# Same scrape as run-lazy-offload.sh. Kept local rather than shared so that
# test keeps passing untouched.
scrape_l1_write_chunks() {
    python3 - <<EOF
import urllib.request

body = urllib.request.urlopen(
    "http://localhost:${LMCACHE_HTTP_PORT}/metrics", timeout=10
).read().decode()
total = 0.0
for line in body.splitlines():
    if line.startswith("#") or not line.startswith("lmcache_mp_l1_write_chunks_total"):
        continue
    try:
        total += float(line.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        pass
print(int(total))
EOF
}

send_small_request() {
    local request_number="$1"
    local first_token="$2"
    local body_file="${TEST_DIR}/request_${request_number}.json"
    local response_file="${TEST_DIR}/response_${request_number}.json"
    local status_code

    python3 - "${body_file}" "${MODEL}" "${first_token}" <<'PY'
import json
import sys

path, model, first_token = sys.argv[1:4]
# A distinct first token keeps the three requests from sharing a cache key,
# so each one buffers a store of its own.
prompt = " ".join([first_token] + ["cache"] * 1024)
with open(path, "w") as output:
    json.dump(
        {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0},
        output,
    )
PY

    status_code="$(curl -sS -o "${response_file}" -w "%{http_code}" \
        -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        --data-binary "@${body_file}")"
    if [ "${status_code}" != "200" ]; then
        echo "Request ${request_number} failed with HTTP ${status_code}" >&2
        cat "${response_file}" >&2 || true
        return 1
    fi
}

# Read the last complete counter ledger the policy logged.
#
# The ledger is logged periodically and again at shutdown, and the engine core
# and API server share one fd, so a line can arrive spliced or cut short. Only
# a line carrying both the first and the last counter is complete, which
# matters because a truncated line would read as an unbalanced ledger, the
# very thing this check exists to find.
check_ledger() {
    python3 - "${VLLM_LOG}" "${MIN_EMITTED}" "${TEST_DIR}/ledger.json" <<'PY'
import json
import re
import sys

log_path, min_emitted, json_out = sys.argv[1:4]
min_emitted = int(min_emitted)

LEDGER = re.compile(r"Lazy offload (?:final )?counters: (?P<body>.*)")
FIELD = re.compile(r"(?P<name>[a-z_]+)=(?P<value>\d+)")
BOUNDS = {"admitted", "pending"}

ledger, ledger_lines = {}, 0
with open(log_path, errors="replace") as handle:
    for line in handle:
        found = LEDGER.search(line)
        if found is None:
            continue
        fields = {
            m.group("name"): int(m.group("value"))
            for m in FIELD.finditer(found.group("body"))
        }
        if BOUNDS.issubset(fields):
            ledger_lines += 1
            ledger = fields

failures = []
if ledger_lines == 0:
    failures.append("policy logged no counter ledger: it never drained")
else:
    # Every dropped_* counter is a term of the equation, so the check is
    # written against the prefix and stays honest when the policy grows a new
    # drop reason. rejected_* counters are turned away before admission and
    # emitted_overdue is a weight on emitted, so neither is a term.
    admitted = ledger.get("admitted", 0)
    emitted = ledger.get("emitted", 0)
    accounted = ledger.get("pending", 0) + emitted
    accounted += sum(v for k, v in ledger.items() if k.startswith("dropped_"))
    if admitted != accounted:
        failures.append(
            f"ledger does not close: admitted={admitted}, "
            f"pending+emitted+dropped={accounted}"
        )
    if emitted < min_emitted:
        failures.append(f"emitted={emitted}, expected at least {min_emitted}")
    if ledger.get("dropped_failed_store", 0):
        failures.append(
            f"dropped_failed_store={ledger['dropped_failed_store']}: "
            "a submitted store was reported failed by a worker"
        )

with open(json_out, "w") as handle:
    json.dump({"ledger": ledger, "lines": ledger_lines, "failures": failures}, handle)

print(f"ledger lines: {ledger_lines}")
print(f"final ledger: {ledger or 'none'}")
for failure in failures:
    print(f"FAIL: {failure}")
sys.exit(1 if failures else 0)
PY
}

pool_tokens=$((NUM_GPU_BLOCKS_OVERRIDE * VLLM_BLOCK_SIZE))
working_set=$((DOCUMENT_LENGTH * NUM_DOCUMENTS))

echo "=== GPU EVICTION_AWARE Lazy Offload Integration Test ==="
echo "Model: ${MODEL}"
echo "vLLM: http://localhost:${VLLM_PORT}"
echo "LMCache metrics: http://localhost:${LMCACHE_HTTP_PORT}/metrics"
echo "GPU block pool: ${NUM_GPU_BLOCKS_OVERRIDE} blocks = ${pool_tokens} tokens"
echo "Phase 2 working set: ${working_set} tokens"

if [ "${working_set}" -lt $((pool_tokens * 2)) ]; then
    echo "FAIL: the workload must be at least twice the pool or it never turns over"
    exit 1
fi

if ! grep -q "lazy offload enabled with EVICTION_AWARE policy" "${VLLM_LOG}"; then
    echo "FAIL: vLLM did not enable EVICTION_AWARE lazy offload"
    tail -100 "${VLLM_LOG}" || true
    exit 1
fi

curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT}/metrics/reset" >/dev/null

echo ""
echo "=== Phase 1: an idle engine must not drain ==="
writes_before="$(scrape_l1_write_chunks)"
for pair in "1 zebra" "2 yak" "3 xenon"; do
    set -- ${pair}
    echo "Sending small request $1"
    send_small_request "$1" "$2"
    # The response is complete, but the worker reports an asynchronous store
    # afterwards. Settle before reading the counter.
    sleep 3
done
writes_after="$(scrape_l1_write_chunks)"
idle_delta=$((writes_after - writes_before))
echo "Idle L1 write chunks delta = ${idle_delta}"
if [ "${idle_delta}" -ne 0 ]; then
    echo "FAIL: ${idle_delta} chunks written with no eviction pressure;"
    echo "      EVICTION_AWARE must hold every store while the pool is idle"
    exit 1
fi
echo "PASS: nothing was written while the engine was idle"

echo ""
echo "=== Phase 2: pool turnover must drain ==="
python3 "${LMCACHE_DIR}/benchmarks/long_doc_qa/long_doc_qa.py" \
    --port "${VLLM_PORT}" \
    --model "${MODEL}" \
    --document-length "${DOCUMENT_LENGTH}" \
    --num-documents "${NUM_DOCUMENTS}" \
    --output-len "${OUTPUT_LEN}" \
    --repeat-count 2 \
    --repeat-mode tile \
    --max-inflight-requests "${MAX_INFLIGHT_REQUESTS}" \
    --output "${TEST_DIR}/long_doc_qa_output.txt" \
    > "${TEST_DIR}/long_doc_qa.log" 2>&1 || {
        echo "FAIL: the long document workload did not complete"
        tail -50 "${TEST_DIR}/long_doc_qa.log" || true
        exit 1
    }

# The drain is asynchronous: the policy releases on a scheduler step and the
# worker reports the store afterwards.
writes_final="${writes_after}"
for _ in $(seq 1 30); do
    writes_final="$(scrape_l1_write_chunks)"
    if [ "$((writes_final - writes_after))" -gt 0 ]; then
        break
    fi
    sleep 1
done
pressure_delta=$((writes_final - writes_after))
echo "Under-pressure L1 write chunks delta = ${pressure_delta}"
if [ "${pressure_delta}" -le 0 ]; then
    echo "FAIL: the pool turned over and nothing was written;"
    echo "      the policy buffered stores it never released"
    tail -100 "${VLLM_LOG}" || true
    exit 1
fi

echo ""
echo "=== Checking the policy's counter ledger ==="
if ! check_ledger; then
    tail -100 "${VLLM_LOG}" || true
    exit 1
fi

echo ""
echo "PASS: nothing drained while idle; ${pressure_delta} chunks written once the pool turned over"
