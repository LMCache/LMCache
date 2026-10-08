#!/usr/bin/env bash
# Wait for inference-engine servers to be ready (native processes, no Docker).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

INFERENCE_ENGINE="${INFERENCE_ENGINE:-vllm}"
ENGINE_ADAPTER="${SCRIPT_DIR}/engines/${INFERENCE_ENGINE}.sh"
if [[ ! -f "$ENGINE_ADAPTER" ]]; then
    echo "Unsupported inference engine '${INFERENCE_ENGINE}': $ENGINE_ADAPTER not found" >&2
    exit 1
fi
source "$ENGINE_ADAPTER"
engine_configure_defaults

MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"

read_pid_from_file() {
    local index="$1"

    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi

    # PID order from launch-processes.sh:
    # 1) LMCache, 2) engine with LMCache, 3) baseline (optional)
    sed -n "${index}p" "$PID_FILE" 2>/dev/null || true
}

process_alive() {
    local pid="$1"

    if [ -z "$pid" ]; then
        return 0
    fi

    kill -0 "$pid" 2>/dev/null
}

print_process_diagnostics() {
    local pid="$1"
    local role="$2"

    echo "=== Process diagnostics: ${role} (pid=${pid}) ==="

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo "Process not running: pid=${pid}"
        return 0
    fi

    echo "--- ps summary ---"
    ps -p "$pid" -o pid,ppid,stat,etime,pcpu,pmem,comm,args || true

    echo "--- /proc status (selected) ---"
    grep -E "^(Name|State|Tgid|Pid|PPid|Threads|VmRSS|VmSize|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):" \
        "/proc/${pid}/status" || true

    echo "--- thread view (top CPU first) ---"
    ps -Lp "$pid" -o pid,tid,psr,pcpu,stat,wchan:32,comm --sort=-pcpu | head -n 40 || true

    if [ -r "/proc/${pid}/wchan" ]; then
        echo "--- wchan ---"
        cat "/proc/${pid}/wchan" || true
    fi

    if [ -r "/proc/${pid}/stack" ]; then
        echo "--- kernel stack ---"
        cat "/proc/${pid}/stack" || true
    fi

    if command -v py-spy >/dev/null 2>&1; then
        echo "--- py-spy dump ---"
        timeout 20s py-spy dump --pid "$pid" --native || true
    elif command -v gdb >/dev/null 2>&1; then
        echo "--- gdb thread backtrace (best effort) ---"
        timeout 20s gdb -q -n -batch \
            -ex "set pagination off" \
            -ex "thread apply all bt" \
            -p "$pid" || true
    else
        echo "Neither py-spy nor gdb is available for user-space backtraces"
    fi

    echo "=== End process diagnostics: ${role} (pid=${pid}) ==="
}

print_log_diagnostics() {
    local logfile="$1"

    if [ ! -f "$logfile" ]; then
        echo "Log file not found: $logfile"
        return 0
    fi

    echo "=== Matched error markers ==="
    grep -nE "ERROR|Error|Traceback|Exception|RuntimeError|ValueError|OSError|Timeout|Failed|failed|huggingface|download|Downloading" "$logfile" | tail -120 || true
    echo ""

    echo "=== Last 200 lines (full) ==="
    tail -200 "$logfile" || true
    echo ""

    echo "=== Last 200 lines (without engine-wait noise) ==="
    grep -vE "Waiting for [0-9]+ local, [0-9]+ remote core engine proc\(s\) to start" "$logfile" | tail -200 || true
    echo ""

    echo "=== First 120 lines (startup context) ==="
    sed -n '1,120p' "$logfile" || true
}

# Wait for an engine server using the adapter's readiness endpoints.
wait_for_engine_server() {
    local port="$1"
    local description="$2"
    local logfile="$3"
    local expected_pid="${4:-}"
    local -a ready_urls=()

    mapfile -t ready_urls < <(engine_ready_urls "$port")
    if [[ ${#ready_urls[@]} -eq 0 ]]; then
        echo "${ENGINE_NAME} adapter returned no readiness URLs" >&2
        return 1
    fi

    echo "=== Waiting for $description to be ready ==="
    echo "Port: $port, Max wait: ${MAX_WAIT_SECONDS}s"

    local start_time end_time
    start_time=$(date +%s)
    end_time=$((start_time + MAX_WAIT_SECONDS))

    while true; do
        local current_time elapsed
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if ! process_alive "$expected_pid"; then
            echo "$description exited before becoming ready (pid=${expected_pid:-unknown})"
            echo ""
            echo "=== $description log diagnostics ==="
            print_log_diagnostics "$logfile"
            echo ""
            echo "=== $description process diagnostics ==="
            if declare -F engine_print_timeout_diagnostics >/dev/null; then
                engine_print_timeout_diagnostics "$logfile"
            fi
            return 1
        fi

        # Probe before enforcing the deadline so a server that becomes ready
        # during the final polling interval is not reported as timed out.
        # Bypass proxy for localhost checks; CI often exports http_proxy.
        local ready_url
        for ready_url in "${ready_urls[@]}"; do
            if curl --noproxy '*' -sf "$ready_url" > /dev/null 2>&1; then
                echo "$description is ready! (took ${elapsed}s)"
                return 0
            fi
        done

        if [ "$current_time" -ge "$end_time" ]; then
            echo "Timeout: $description did not become ready within ${MAX_WAIT_SECONDS}s"
            echo ""
            echo "=== $description log diagnostics ==="
            print_log_diagnostics "$logfile"
            echo ""
            echo "=== $description process diagnostics ==="
            if declare -F engine_print_timeout_diagnostics >/dev/null; then
                engine_print_timeout_diagnostics "$logfile"
            fi
            return 1
        fi

        echo "Waiting for $description... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Wait for both engine servers (they start simultaneously).
ENGINE_LMCACHE_PID="$(read_pid_from_file 2 || true)"
ENGINE_BASELINE_PID="$(read_pid_from_file 3 || true)"

if ! wait_for_engine_server "$ENGINE_PORT" "${ENGINE_NAME} with LMCache" \
    "$ENGINE_LOG_FILE" "$ENGINE_LMCACHE_PID"; then
    exit 1
fi

# The baseline server only exists for 2-GPU tests; 1-GPU tests set
# LAUNCH_BASELINE=false in launch-processes.sh and never start it.
if [[ "${LAUNCH_BASELINE:-true}" == "true" ]]; then
    if ! wait_for_engine_server "$ENGINE_BASELINE_PORT" \
            "${ENGINE_NAME} baseline (without LMCache)" \
            "$ENGINE_BASELINE_LOG_FILE" "$ENGINE_BASELINE_PID"; then
        exit 1
    fi
fi

echo ""
echo "=== All ${ENGINE_NAME} servers are ready ==="
