#!/usr/bin/env bash
# Shared helpers for SGLang + LMCache MP CI tests.
# Source from each scripts/run-*.sh.

# ── Fixed test configuration ────────────────────────────────
# 7B model gives a robust TTFT differential between LMCache hit and full
# prefill (~45ms on Blackwell). The 1.5B model's differential is ~4ms,
# inside CI noise.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DAEMON_PORT="${DAEMON_PORT:-6200}"
DAEMON_HTTP_PORT="${DAEMON_HTTP_PORT:-7200}"

# Tracking for cleanup.
DAEMON_PID=""
SGLANG_PID=""

cleanup_all() {
    if [[ -n "${SGLANG_PID}" ]]; then
        echo "--- Killing SGLang (PID=${SGLANG_PID})"
        kill -9 "${SGLANG_PID}" 2>/dev/null || true
        pkill -9 -P "${SGLANG_PID}" 2>/dev/null || true
        pkill -9 -f "sglang::scheduler" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        SGLANG_PID=""
    fi
    if [[ -n "${DAEMON_PID}" ]]; then
        echo "--- Killing LMCache daemon (PID=${DAEMON_PID})"
        kill -9 "${DAEMON_PID}" 2>/dev/null || true
        wait "${DAEMON_PID}" 2>/dev/null || true
        DAEMON_PID=""
    fi
    sleep 2
}

# Launch the LMCache MP daemon. Writes its log to $1.
launch_daemon() {
    local log_file="$1"
    echo "--- :rocket: Launching LMCache daemon (log=${log_file})"
    lmcache server \
        --host 127.0.0.1 --port "${DAEMON_PORT}" --http-port "${DAEMON_HTTP_PORT}" \
        --chunk-size 256 --l1-size-gb 4 --eviction-policy LRU \
        > "${log_file}" 2>&1 &
    DAEMON_PID=$!
    # Daemon prints "ZMQ cache server is running" when ready.
    for ((i = 0; i < 60; i++)); do
        if grep -q "ZMQ cache server is running" "${log_file}" 2>/dev/null; then
            echo "  daemon ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: daemon did not start within 60s" >&2
    tail -n 200 "${log_file}" >&2
    return 1
}

# Launch an SGLang server.
#   $1 = port
#   $2 = log file
#   $3 = "lmcache" to enable LMCache, "no-lmcache" to disable
launch_sglang() {
    local port="$1"
    local log_file="$2"
    local mode="$3"

    local lmcache_args=()
    if [[ "${mode}" == "lmcache" ]]; then
        lmcache_args=(
            --enable-lmcache
            --lmcache-mp-host 127.0.0.1
            --lmcache-mp-port "${DAEMON_PORT}"
        )
    fi

    echo "--- :rocket: Launching SGLang (port=${port}, mode=${mode}, log=${log_file})"
    # Blackwell SM 12 workarounds — drop on other hardware.
    python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --host 127.0.0.1 --port "${port}" \
        --max-total-tokens 4096 \
        --disable-cuda-graph --disable-piecewise-cuda-graph \
        --attention-backend triton \
        "${lmcache_args[@]}" \
        > "${log_file}" 2>&1 &
    SGLANG_PID=$!

    wait_sglang_ready "${port}" "${log_file}"
}

# Wait for an SGLang server to report healthy.
wait_sglang_ready() {
    local port="$1"
    local log_file="$2"
    local timeout="${3:-240}"
    echo "  waiting for SGLang on port ${port} (timeout=${timeout}s)..."
    for ((i = 0; i < timeout; i++)); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "  SGLang ready on port ${port} (${i}s)"
            return 0
        fi
        if grep -q "SIGQUIT received\|FATAL\|RuntimeError" "${log_file}" 2>/dev/null; then
            echo "FAIL: SGLang crashed during startup" >&2
            tail -n 200 "${log_file}" >&2
            return 1
        fi
        sleep 1
    done
    echo "FAIL: SGLang did not become ready on port ${port} within ${timeout}s" >&2
    tail -n 200 "${log_file}" >&2
    return 1
}

# Send a non-streaming chat completion. Echoes the assistant message text.
# Args: $1=port  $2=prompt  $3=max_tokens
chat_completion() {
    local port="$1" prompt="$2" max_tokens="${3:-64}"
    curl -sf "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json, sys
print(json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': sys.argv[1]}],
    'max_tokens': int(sys.argv[2]),
    'temperature': 0.0,
    'stream': False,
}))" "${prompt}" "${max_tokens}")" \
        | python3 -c "
import json, sys
resp = json.load(sys.stdin)
print(resp['choices'][0]['message']['content'], end='')
"
}

# Measure wall-clock latency in seconds for a max_tokens=1 chat completion.
# This approximates TTFT — the request returns after prefill + one decode step,
# and decode for one token is small relative to prefill on prefill-heavy prompts.
measure_latency_seconds() {
    local port="$1" prompt="$2"
    local t0 t1
    t0=$(python3 -c "import time; print(time.perf_counter())")
    chat_completion "${port}" "${prompt}" 1 >/dev/null
    t1=$(python3 -c "import time; print(time.perf_counter())")
    python3 -c "print(${t1} - ${t0})"
}

# Count occurrences of 'Retrieved N tokens' lines in the daemon log.
# `grep -c` prints "0" but exits 1 on zero matches, so capture exit code
# separately and always emit a single-line count.
count_retrievals() {
    local log_file="$1"
    local count
    if count=$(grep -c "Retrieved [0-9]* tokens" "${log_file}" 2>/dev/null); then
        echo "${count}"
    else
        echo "${count:-0}"
    fi
}

# Two deterministic ~2500-token prompts (A and B). Each fits within
# --max-total-tokens 4096 after chat-template + max_tokens overhead, but
# together they exceed the pool — so after A→B, A's radix entry is evicted
# and a follow-up A becomes a radix miss / LMCache hit. Used by the
# correctness test.
generate_prompt() {
    local variant="${1:-a}"
    python3 -c "
import sys
sentences = {
    'a': 'The quick brown fox jumps over the lazy dog while a curious cat watches from the windowsill on a quiet afternoon in early autumn. ',
    'b': 'An old man walks slowly along a foggy riverside path at dawn carrying a fishing rod and dreaming of his grandchildren far away. ',
}
print(sentences[sys.argv[1]] * 80 + 'Summarize the scene above in one short sentence.')
" "${variant}"
}

# Run long_doc_qa.py against a server and echo the Query-round mean TTFT
# in seconds. The workload is 4 distinct ~1500-token docs × 2 tiles —
# combined KV (~6000 tokens) exceeds --max-total-tokens 4096, so during
# the warmup round SGLang's radix evicts older entries; the query round
# re-queries them. With LMCache, the radix misses are served by RETRIEVE;
# without LMCache they fall back to full prefills. Repeating the test
# script directly is simpler than the connector's manual A→B→A pattern.
run_long_doc_qa_query_ttft() {
    local port="$1"
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    python3 "${repo_root}/benchmarks/long_doc_qa/long_doc_qa.py" \
        --num-documents 4 --document-length 1500 --output-len 100 \
        --repeat-count 2 --repeat-mode tile --max-inflight-requests 1 \
        --host 127.0.0.1 --port "${port}" \
        --model "${MODEL}" \
        2>&1 \
        | tee "/tmp/perf_bench_${port}.log" \
        | grep -oE "Query round mean TTFT: [0-9.]+" \
        | awk '{print $NF}'
}
