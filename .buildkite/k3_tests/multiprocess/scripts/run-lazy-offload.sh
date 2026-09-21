#!/usr/bin/env bash
# Verify FIFO lazy offload against a live GPU vLLM + LMCache MP deployment.
#
# The launcher configures threshold=2 and select_count=1. This test sends
# three completed, cacheable requests and checks the L1-write metric after
# each: requests 1 and 2 must not write; request 3 must drain request 1.
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
BUILD_ID="${BUILD_ID:-local_$$}"
RESULTS_DIR="${RESULTS_DIR:-/tmp/lmcache_ci_results_${BUILD_ID}}"
TEST_DIR="${RESULTS_DIR}/lazy_offload"
VLLM_LOG="/tmp/build_${BUILD_ID}_vllm.log"
LMCACHE_CHUNK_SIZE="${CHUNK_SIZE:-16}"

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

echo "=== GPU FIFO Lazy Offload Integration Test ==="
echo "Model: ${MODEL}"
echo "vLLM: http://localhost:${VLLM_PORT}"
echo "LMCache metrics: http://localhost:${LMCACHE_HTTP_PORT}/metrics"
echo "LMCache chunk size: ${LMCACHE_CHUNK_SIZE}"

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
