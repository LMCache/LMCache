#!/usr/bin/env bash
# Verify lazy offload against a live GPU vLLM + LMCache MP deployment.
#
# FIFO retains its threshold behavior check. EVICTION_AWARE verifies that
# allocation pressure emits a store, a warm replay reads it from L1 with
# byte-identical output, and the policy counter ledger closes.
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="${RESULTS_DIR}/lazy_offload"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
LMCACHE_CHUNK_SIZE="${CHUNK_SIZE:-16}"
LAZY_OFFLOAD_POLICY="${LMCACHE_MP_LAZY_OFFLOAD_POLICY:-FIFO}"
LMCACHE_LOG="/tmp/build_${BUILD_ID}_lmcache.log"

mkdir -p "${TEST_DIR}"

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

write_request_body() {
    local request_number="$1"
    local word_count="$2"
    local body_file="${TEST_DIR}/request_${request_number}.json"
    local first_token

    case "${request_number}" in
        1) first_token="zebra" ;;
        2) first_token="yak" ;;
        3) first_token="xenon" ;;
        *) echo "Unknown request number: ${request_number}" >&2; return 1 ;;
    esac

    python3 - "${body_file}" "${MODEL}" "${first_token}" "${word_count}" <<'PY'
import json
import sys

path, model, first_token, word_count = sys.argv[1:]
# A distinct first token prevents the three requests from sharing a cache key.
prompt = " ".join([first_token] + ["cache"] * int(word_count))
with open(path, "w") as output:
    json.dump(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        },
        output,
    )
PY
}

tokenize_request() {
    local request_number="$1"
    local body_file="${TEST_DIR}/request_${request_number}.json"
    local tokenize_file="${TEST_DIR}/tokenize_${request_number}.json"

    python3 - "${body_file}" "${tokenize_file}" <<'PY'
import json
import sys

request = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as output:
    json.dump({"model": request["model"], "prompt": request["prompt"]}, output)
PY
    curl -fsS -X POST "http://localhost:${VLLM_PORT}/tokenize" \
        -H "Content-Type: application/json" \
        --data-binary "@${tokenize_file}" \
        | python3 -c '
import json
import sys

response = json.load(sys.stdin)
tokens = response.get("tokens")
if not isinstance(tokens, list):
    raise ValueError(f"Unexpected /tokenize response: {response}")
print(len(tokens))
'
}

prepare_request() {
    local request_number="$1"
    local word_count="$2"
    local token_count

    # A generated output token can add at most one token before the request
    # completes. Avoid a prompt one token short of the next chunk boundary so
    # floor(prompt_tokens / chunk_size) remains the exact stored-chunk count.
    while true; do
        write_request_body "${request_number}" "${word_count}"
        token_count="$(tokenize_request "${request_number}")"
        if [ $((token_count % LMCACHE_CHUNK_SIZE)) -ne $((LMCACHE_CHUNK_SIZE - 1)) ]; then
            break
        fi
        word_count=$((word_count + 1))
    done

    local chunk_count=$((token_count / LMCACHE_CHUNK_SIZE))
    if [ "${chunk_count}" -lt 1 ]; then
        echo "Request ${request_number} does not fill an LMCache chunk" >&2
        return 1
    fi
    echo "Request ${request_number}: ${token_count} prompt tokens -> ${chunk_count} LMCache chunks" >&2
    echo "${chunk_count}"
}

send_request() {
    local request_number="$1"
    local body_file="${TEST_DIR}/request_${request_number}.json"
    local response_file="${TEST_DIR}/response_${request_number}.json"

    local status_code
    status_code="$(curl -sS -o "${response_file}" -w "%{http_code}" \
        -X POST "http://localhost:${VLLM_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        --data-binary "@${body_file}")"
    if [ "${status_code}" != "200" ]; then
        echo "Request ${request_number} failed with HTTP ${status_code}" >&2
        cat "${response_file}" >&2 || true
        return 1
    fi
    python3 - "${response_file}" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
assert response["choices"], "vLLM response had no choices"
PY
}

count_retrieves() {
    python3 - "${LMCACHE_LOG}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    pattern = re.compile(r"Retrieved \d+ tokens in ")
    print(sum(bool(pattern.search(line)) for line in path.read_text(errors="ignore").splitlines()))
PY
}

wait_for_write_delta() {
    local writes_before="$1"
    local minimum_delta="$2"
    local writes_after="$writes_before"

    for _ in $(seq 1 30); do
        writes_after="$(scrape_l1_write_chunks)"
        if [ $((writes_after - writes_before)) -ge "${minimum_delta}" ]; then
            echo "${writes_after}"
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for ${minimum_delta} lazy-offload L1 write chunks" >&2
    return 1
}

enable_cached_token_stats() {
    local request_number="$1"
    local body_file="${TEST_DIR}/request_${request_number}.json"

    python3 - "${body_file}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as source:
    request = json.load(source)
request["kv_transfer_params"] = {"cached_token_stats": True}
with open(path, "w") as output:
    json.dump(request, output)
PY
}

reset_vllm_prefix_cache() {
    local status_code
    status_code="$(curl -sS -o /dev/null -w "%{http_code}" -X POST \
        "http://127.0.0.1:${VLLM_PORT}/reset_prefix_cache")"
    if [ "${status_code}" != "200" ]; then
        echo "reset_prefix_cache failed with HTTP ${status_code}" >&2
        return 1
    fi
}

validate_warm_replay() {
    local cold_file="$1"
    local warm_file="$2"
    local retrieves_before="$3"
    local retrieves_after="$4"

    python3 - "${cold_file}" "${warm_file}" "${retrieves_before}" "${retrieves_after}" <<'PY'
import json
import sys

cold_path, warm_path, before_text, after_text = sys.argv[1:5]
with open(cold_path) as source:
    cold = json.load(source)
with open(warm_path) as source:
    warm = json.load(source)

cold_text = cold["choices"][0]["text"]
warm_text = warm["choices"][0]["text"]
if cold_text != warm_text:
    raise AssertionError(
        f"warm output differs from cold output: {cold_text!r} != {warm_text!r}"
    )

stats = (warm.get("kv_transfer_params") or {}).get("cached_token_stats")
if stats is None:
    raise AssertionError("warm response has no cached_token_stats")
cached = stats.get("num_lmcache_cached_tokens", 0)
if not isinstance(cached, int) or cached <= 0:
    raise AssertionError(f"warm replay reported no LMCache-cached tokens: {stats}")

before = int(before_text)
after = int(after_text)
if after <= before:
    raise AssertionError(
        f"LMCache retrieve log did not grow during warm replay: {before} -> {after}"
    )
print(f"Warm replay matched exactly and retrieved {cached} cached tokens")
PY
}

validate_eviction_aware_ledger() {
    python3 - "${VLLM_LOG}" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
lines = path.read_text(errors="ignore").splitlines()
ledgers = []
for line in lines:
    if "Lazy offload counters:" not in line and "Lazy offload final counters:" not in line:
        continue
    ledgers.append({key: int(value) for key, value in re.findall(r"([a-z_]+)=(\d+)", line)})

ledger = next((item for item in reversed(ledgers) if item.get("emitted", 0) > 0), None)
if ledger is None:
    raise AssertionError("no eviction-aware counter ledger recorded an emitted store")
drops = sum(
    ledger.get(key, 0)
    for key in (
        "dropped_evicted",
        "dropped_on_request_drop",
        "dropped_failed_store",
        "dropped_id_reuse",
    )
)
accounted = ledger.get("pending", 0) + ledger["emitted"] + drops
if ledger["admitted"] != accounted:
    raise AssertionError(f"lazy-offload ledger does not close: {ledger}")
unexpected = {
    key: ledger.get(key, 0)
    for key in (
        "emitted_overdue",
        "dropped_evicted",
        "rejected_unhashed",
        "rejected_prefix_broken",
        "dropped_on_request_drop",
        "dropped_failed_store",
        "dropped_id_reuse",
    )
    if ledger.get(key, 0)
}
if unexpected:
    raise AssertionError(f"eviction-aware test recorded unexpected outcomes: {unexpected}")
print(f"Eviction-aware ledger closes: {ledger}")
PY
}

assert_no_runtime_faults() {
    python3 - "${VLLM_LOG}" "${LMCACHE_LOG}" <<'PY'
import pathlib
import sys

markers = (
    "gpu fault",
    "memory access fault",
    "illegal memory access",
    "hsa_status_error_exception",
    "block hashes missing or mismatched",
)
for filename in sys.argv[1:]:
    path = pathlib.Path(filename)
    text = path.read_text(errors="ignore").lower() if path.exists() else ""
    found = [marker for marker in markers if marker in text]
    if found:
        raise AssertionError(f"{path} contains runtime fault markers: {found}")
PY
}

run_eviction_aware_test() {
    local target_chunks
    local writes_before
    local writes_after
    local retrieves_before
    local retrieves_after
    local cold_response="${TEST_DIR}/eviction_aware_cold.json"
    local warm_response="${TEST_DIR}/eviction_aware_warm.json"

    if ! grep -q "lazy offload enabled with EVICTION_AWARE policy" "${VLLM_LOG}"; then
        echo "FAIL: vLLM did not enable EVICTION_AWARE lazy offload"
        tail -100 "${VLLM_LOG}" || true
        return 1
    fi

    curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT}/metrics/reset" >/dev/null
    target_chunks="$(prepare_request 1 256)"
    enable_cached_token_stats 1
    prepare_request 2 512 >/dev/null

    writes_before="$(scrape_l1_write_chunks)"
    send_request 1
    cp "${TEST_DIR}/response_1.json" "${cold_response}"

    # The stats logger is throttled to five seconds. Waiting here ensures the
    # pressure step records the emission ledger rather than only admission.
    sleep 6
    send_request 2
    writes_after="$(wait_for_write_delta "${writes_before}" "${target_chunks}")"
    echo "Eviction pressure wrote $((writes_after - writes_before)) L1 chunks"

    sleep 2
    reset_vllm_prefix_cache
    sleep 2
    retrieves_before="$(count_retrieves)"
    send_request 1
    cp "${TEST_DIR}/response_1.json" "${warm_response}"
    sleep 2
    retrieves_after="$(count_retrieves)"

    validate_warm_replay \
        "${cold_response}" "${warm_response}" \
        "${retrieves_before}" "${retrieves_after}"
    validate_eviction_aware_ledger
    assert_no_runtime_faults
    echo "PASS: eviction pressure emitted stores and warm L1 replay matched exactly"
}

echo "=== GPU ${LAZY_OFFLOAD_POLICY} Lazy Offload Integration Test ==="
echo "Model: ${MODEL}"
echo "vLLM: http://localhost:${VLLM_PORT}"
echo "LMCache metrics: http://localhost:${LMCACHE_HTTP_PORT}/metrics"
echo "LMCache chunk size: ${LMCACHE_CHUNK_SIZE}"

if [ "${LAZY_OFFLOAD_POLICY}" = "EVICTION_AWARE" ]; then
    run_eviction_aware_test
    exit 0
fi

if [ "${LAZY_OFFLOAD_POLICY}" != "FIFO" ]; then
    echo "Unknown lazy-offload policy: ${LAZY_OFFLOAD_POLICY}" >&2
    exit 2
fi

if ! grep -q "lazy offload enabled with FIFO policy, offload threshold: 2" "${VLLM_LOG}"; then
    echo "FAIL: vLLM did not enable FIFO lazy offload with threshold 2"
    tail -100 "${VLLM_LOG}" || true
    exit 1
fi

curl -fsS -X POST "http://localhost:${LMCACHE_HTTP_PORT}/metrics/reset" >/dev/null

declare -a expected_chunks
for request_number in 1 2 3; do
    # Deliberately different request sizes make a third-step write delta an
    # identity check: only request 1's chunk count is accepted at that point.
    expected_chunks[${request_number}]="$(prepare_request "${request_number}" "$((request_number * 64))")"
    if [ "${request_number}" -gt 1 ] \
        && [ "${expected_chunks[${request_number}]}" -le "${expected_chunks[$((request_number - 1))]}" ]; then
        echo "FAIL: request chunk counts must be strictly increasing" >&2
        exit 1
    fi
    writes_before="$(scrape_l1_write_chunks)"
    echo "Sending lazy-offload request ${request_number} (writes before: ${writes_before}, expected chunks: ${expected_chunks[${request_number}]})"
    send_request "${request_number}"
    # The response is complete, but the worker reports an asynchronous store
    # afterward. The first two requests only need one settling period; the
    # third waits longer for the expected FIFO drain to reach the L1 counter.
    if [ "${request_number}" -lt 3 ]; then
        sleep 3
        writes_after="$(scrape_l1_write_chunks)"
    else
        writes_after="${writes_before}"
        for _ in $(seq 1 15); do
            writes_after="$(scrape_l1_write_chunks)"
            if [ $((writes_after - writes_before)) -ge "${expected_chunks[1]}" ]; then
                break
            fi
            sleep 1
        done
    fi
    write_delta=$((writes_after - writes_before))
    echo "Request ${request_number}: L1 write chunks delta = ${write_delta}"

    if [ "${request_number}" -lt 3 ] && [ "${write_delta}" -ne 0 ]; then
        echo "FAIL: request ${request_number} stored before the FIFO threshold"
        exit 1
    fi
    if [ "${request_number}" -eq 3 ] && [ "${write_delta}" -ne "${expected_chunks[1]}" ]; then
        echo "FAIL: request 3 wrote ${write_delta} chunks; expected request 1's ${expected_chunks[1]} chunks"
        exit 1
    fi
done

echo "PASS: requests 1 and 2 did not offload; request 3 wrote exactly request 1's ${expected_chunks[1]} chunks"
