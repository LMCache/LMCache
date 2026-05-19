#!/usr/bin/env bash
# Shared helpers for SGLang + LMCache MP CI tests.

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DAEMON_PORT="${DAEMON_PORT:-6200}"
DAEMON_HTTP_PORT="${DAEMON_HTTP_PORT:-7200}"
LMC_CONFIG_FILE="${LMC_CONFIG_FILE:-/tmp/lmcache_mp_ci.yaml}"

DAEMON_PID=""
SGLANG_PID=""

cleanup_all() {
    if [[ -n "${SGLANG_PID}" ]]; then
        kill -9 "${SGLANG_PID}" 2>/dev/null || true
        pkill -9 -P "${SGLANG_PID}" 2>/dev/null || true
        pkill -9 -f "sglang::scheduler" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        SGLANG_PID=""
    fi
    if [[ -n "${DAEMON_PID}" ]]; then
        kill -9 "${DAEMON_PID}" 2>/dev/null || true
        wait "${DAEMON_PID}" 2>/dev/null || true
        DAEMON_PID=""
    fi
    sleep 2
}

launch_daemon() {
    local log_file="$1"
    cat > "${LMC_CONFIG_FILE}" <<EOF
mp_host: 127.0.0.1
mp_port: ${DAEMON_PORT}
EOF
    lmcache server \
        --host 127.0.0.1 --port "${DAEMON_PORT}" --http-port "${DAEMON_HTTP_PORT}" \
        --chunk-size 256 --l1-size-gb 4 --eviction-policy LRU \
        > "${log_file}" 2>&1 &
    DAEMON_PID=$!
    for ((i = 0; i < 60; i++)); do
        if grep -q "ZMQ cache server is running" "${log_file}" 2>/dev/null; then
            echo "  daemon ready (${i}s, log=${log_file})"
            return 0
        fi
        sleep 1
    done
    echo "FAIL: daemon did not start within 60s" >&2
    tail -n 200 "${log_file}" >&2
    return 1
}

# Launch SGLang. $1=port, $2=log file, $3="lmcache" or "no-lmcache".
launch_sglang() {
    local port="$1" log_file="$2" mode="$3"
    local lmcache_args=()
    if [[ "${mode}" == "lmcache" ]]; then
        lmcache_args=(--enable-lmcache --lmcache-config-file "${LMC_CONFIG_FILE}")
    fi
    python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --host 127.0.0.1 --port "${port}" \
        --max-total-tokens 4096 \
        "${lmcache_args[@]}" \
        > "${log_file}" 2>&1 &
    SGLANG_PID=$!
    wait_sglang_ready "${port}" "${log_file}"
}

wait_sglang_ready() {
    local port="$1" log_file="$2" timeout="${3:-240}"
    for ((i = 0; i < timeout; i++)); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "  sglang ready on port ${port} (${i}s, mode=${mode:-?}, log=${log_file})"
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

# Non-streaming chat completion. Args: port, prompt, max_tokens.
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

# Count "Retrieved N tokens" lines in the daemon log. grep -c prints 0 but
# exits 1 on zero matches, so capture both and always emit one line.
count_retrievals() {
    local log_file="$1" count
    if count=$(grep -c "Retrieved [0-9]* tokens" "${log_file}" 2>/dev/null); then
        echo "${count}"
    else
        echo "${count:-0}"
    fi
}

# Two ~2500-token prompts. Together they exceed --max-total-tokens 4096,
# so after A→B, A's radix entry is evicted and a follow-up A is a radix
# miss / LMCache hit. Used by run-correctness.sh.
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

# Run long_doc_qa.py against $1 and echo the Query-round mean TTFT in
# seconds. 4 distinct ~1500-token docs × 2 tiles — combined KV exceeds
# --max-total-tokens 4096, so query-round prompts miss the radix and
# (with LMCache) hit RETRIEVE.
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
        | tee "perf-bench-${port}.log" \
        | grep -oE "Query round mean TTFT: [0-9.]+" \
        | awk '{print $NF}'
}
