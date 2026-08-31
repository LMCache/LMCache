#!/usr/bin/env bash
# Shared helpers for the containerized MUSA model-serving E2E tests.
set -euo pipefail

E2E_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${E2E_SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT_PATH="${MUSA_CI_ARTIFACT_DIR:-musa-ci-artifacts/e2e}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi

PYTHON_BIN="${MUSA_CI_PYTHON:-python3}"
MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
MODEL="${MUSA_E2E_MODEL:-}"
MAX_TOKENS="${MUSA_E2E_MAX_TOKENS:-8}"
TOP_K="${MUSA_E2E_TOP_K:-1}"
PROMPT_FILE="${ARTIFACT_DIR}/prompt.txt"
MAX_WAIT_SECONDS="${MUSA_E2E_STARTUP_TIMEOUT:-300}"
HIT_PATTERN="${MUSA_E2E_HIT_PATTERN:-LMCache.*(retrieve|retrieved|cache hit|hit tokens)}"
E2E_PIDS=()

log() {
    echo "--- :musa: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[musa-e2e] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    for log_file in "${ARTIFACT_DIR}"/*.log; do
        [[ -f "${log_file}" ]] || continue
        echo "--- tail ${log_file}" >&2
        tail -n 120 "${log_file}" >&2 || true
    done
    exit 1
}

require_model() {
    [[ -n "${MODEL}" ]] || fail \
        "MUSA_E2E_MODEL must be set to a model ID or local model path"
    if [[ "${MODEL}" == /* && ! -d "${MODEL}" ]]; then
        fail "configured model path does not exist: ${MODEL}"
    fi
    if [[ "${MODEL}" == /* && ! -f "${MODEL}/config.json" ]]; then
        fail "configured model path has no config.json: ${MODEL}"
    fi
    log "Using model ${MODEL}"
}

prepare_prompt() {
    mkdir -p "${ARTIFACT_DIR}"
    if [[ -n "${MUSA_E2E_PROMPT:-}" ]]; then
        printf '%s' "${MUSA_E2E_PROMPT}" > "${PROMPT_FILE}"
    else
        "${PYTHON_BIN}" - "${PROMPT_FILE}" <<'PY'
from pathlib import Path
import sys

sentence = (
    "Moore Threads MUSA LMCache deterministic cache reuse validation. "
    "The same prefix must produce the same completion after a cache retrieve. "
)
Path(sys.argv[1]).write_text(sentence * 64 + "Summarize this validation in one sentence.\n")
PY
    fi
    [[ -s "${PROMPT_FILE}" ]] || fail "E2E prompt is empty"
}

prepare_variant_prompt() {
    local variant_file="$1"
    "${PYTHON_BIN}" - "${PROMPT_FILE}" "${variant_file}" <<'PY'
from pathlib import Path
import sys

base = Path(sys.argv[1]).read_text()
prefix, separator, _ = base.rpartition("\n")
if not separator:
    prefix = base
Path(sys.argv[2]).write_text(
    prefix + "\nSummarize this second scene in one sentence.\n"
)
PY
    [[ -s "${variant_file}" ]] || fail "variant E2E prompt is empty"
}

register_pid() {
    E2E_PIDS+=("$1")
}

kill_process_tree() {
    local pid="$1"
    local signal="$2"
    local child
    for child in $(pgrep -P "${pid}" 2>/dev/null || true); do
        kill_process_tree "${child}" "${signal}"
    done
    kill -"${signal}" "${pid}" 2>/dev/null || true
}

stop_pid() {
    local pid="$1"
    [[ -n "${pid}" ]] || return 0
    if kill -0 "${pid}" 2>/dev/null; then
        kill_process_tree "${pid}" TERM
        for _ in $(seq 1 30); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill_process_tree "${pid}" KILL
        fi
    fi
    wait "${pid}" 2>/dev/null || true
    local remaining=()
    local candidate
    for candidate in "${E2E_PIDS[@]:-}"; do
        [[ "${candidate}" == "${pid}" ]] || remaining+=("${candidate}")
    done
    E2E_PIDS=("${remaining[@]}")
}

stop_all() {
    local pid
    for pid in "${E2E_PIDS[@]:-}"; do
        stop_pid "${pid}"
    done
    E2E_PIDS=()
}

wait_for_http() {
    local port="$1"
    local description="$2"
    local log_file="$3"
    local start now elapsed
    start="$(date +%s)"
    log "Waiting for ${description} on port ${port}"
    while true; do
        now="$(date +%s)"
        elapsed=$((now - start))
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || \
            curl -fsS "http://127.0.0.1:${port}/healthcheck" >/dev/null 2>&1 || \
            curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            log "${description} is healthy after ${elapsed}s"
            return 0
        fi
        if [[ -f "${log_file}" ]] && grep -Eqi \
            'FATAL|Traceback|RuntimeError|Address already in use' "${log_file}"; then
            tail -n 160 "${log_file}" >&2 || true
            fail "${description} failed during startup"
        fi
        if (( elapsed >= MAX_WAIT_SECONDS )); then
            tail -n 160 "${log_file}" >&2 || true
            fail "${description} did not become healthy within ${MAX_WAIT_SECONDS}s"
        fi
        sleep 2
    done
}

request_completion() {
    local port="$1"
    local model="$2"
    local output_file="$3"
    "${PYTHON_BIN}" "${E2E_SCRIPT_DIR}/e2e_client.py" completion \
        --url "http://127.0.0.1:${port}/v1/completions" \
        --model "${model}" \
        --prompt-file "${PROMPT_FILE}" \
        --max-tokens "${MAX_TOKENS}" \
        --seed "${MUSA_E2E_SEED:-0}" \
        --temperature "${MUSA_E2E_TEMPERATURE:-0}" \
        --top-k "${TOP_K}" \
        --output "${output_file}"
}

request_completion_with_prompt() {
    local port="$1"
    local model="$2"
    local output_file="$3"
    local prompt_file="$4"
    "${PYTHON_BIN}" "${E2E_SCRIPT_DIR}/e2e_client.py" completion \
        --url "http://127.0.0.1:${port}/v1/completions" \
        --model "${model}" \
        --prompt-file "${prompt_file}" \
        --max-tokens "${MAX_TOKENS}" \
        --seed "${MUSA_E2E_SEED:-0}" \
        --temperature "${MUSA_E2E_TEMPERATURE:-0}" \
        --top-k "${TOP_K}" \
        --output "${output_file}"
}

request_chat_completion() {
    local port="$1"
    local model="$2"
    local output_file="$3"
    local prompt_file="${4:-${PROMPT_FILE}}"
    "${PYTHON_BIN}" "${E2E_SCRIPT_DIR}/e2e_client.py" chat-completion \
        --url "http://127.0.0.1:${port}/v1/chat/completions" \
        --model "${model}" \
        --prompt-file "${prompt_file}" \
        --max-tokens "${MAX_TOKENS}" \
        --seed "${MUSA_E2E_SEED:-0}" \
        --temperature "${MUSA_E2E_TEMPERATURE:-0}" \
        --top-k "${TOP_K}" \
        --output "${output_file}"
}

model_id() {
    local port="$1"
    "${PYTHON_BIN}" "${E2E_SCRIPT_DIR}/e2e_client.py" model \
        --url "http://127.0.0.1:${port}/v1/models"
}

compare_completion_text() {
    local left="$1"
    local right="$2"
    "${PYTHON_BIN}" "${E2E_SCRIPT_DIR}/e2e_client.py" compare \
        --left "${left}" --right "${right}"
}

log_hit_count() {
    local count=0
    local log_file
    local log_files=("${ARTIFACT_DIR}"/*.log)
    if [[ "$#" -gt 0 ]]; then
        log_files=("$@")
    fi
    for log_file in "${log_files[@]}"; do
        [[ -f "${log_file}" ]] || continue
        count=$((count + $(grep -Eio "${HIT_PATTERN}" "${log_file}" 2>/dev/null | wc -l | tr -d ' ')))
    done
    echo "${count}"
}

cleanup_e2e() {
    local exit_code=$?
    set +e
    stop_all
    return "${exit_code}"
}

mkdir -p "${ARTIFACT_DIR}"
trap cleanup_e2e EXIT
