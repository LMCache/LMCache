#!/usr/bin/env bash
# Wait for vLLM servers to be ready (native processes, no Docker).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASELINE_PORT="${VLLM_BASELINE_PORT:-9000}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"
BUILD_ID="${BUILD_ID:-local_$$}"
PID_FILE="/tmp/lmcache_mp_pids_${BUILD_ID}"

read_pid_from_file() {
    local index="$1"

    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi

    # PID order from launch-processes.sh:
    # 1) LMCache, 2) vLLM with LMCache, 3) baseline (optional)
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

print_engine_timeout_diagnostics() {
    local logfile="$1"

    if [ ! -f "$logfile" ]; then
        return 0
    fi

    local engine_pid
    engine_pid=$(grep -oE "\(EngineCore pid=[0-9]+\)" "$logfile" | tail -n 1 | sed -E 's/.*pid=([0-9]+).*/\1/' || true)
    if [ -n "$engine_pid" ]; then
        print_process_diagnostics "$engine_pid" "EngineCore"
    else
        echo "EngineCore PID not found in $logfile"
    fi

    local api_pid
    api_pid=$(grep -oE "\(APIServer pid=[0-9]+\)" "$logfile" | tail -n 1 | sed -E 's/.*pid=([0-9]+).*/\1/' || true)
    if [ -n "$api_pid" ]; then
        print_process_diagnostics "$api_pid" "APIServer"
    else
        echo "APIServer PID not found in $logfile"
    fi
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

# Wait for a vLLM server with health check
wait_for_vllm_server() {
    local port="$1"
    local description="$2"
    local logfile="$3"
    local expected_pid="${4:-}"
    local health_url="http://127.0.0.1:${port}/health"
    local models_url="http://127.0.0.1:${port}/v1/models"

    echo "=== Waiting for $description to be ready ==="
    echo "Port: $port, Max wait: ${MAX_WAIT_SECONDS}s"

    local start_time end_time
    start_time=$(date +%s)
    end_time=$((start_time + MAX_WAIT_SECONDS))

    while true; do
        local current_time elapsed
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        if [ "$current_time" -ge "$end_time" ]; then
            echo "Timeout: $description did not become ready within ${MAX_WAIT_SECONDS}s"
            echo ""
            echo "=== $description log diagnostics ==="
            print_log_diagnostics "$logfile"
            echo ""
            echo "=== $description process diagnostics ==="
            print_engine_timeout_diagnostics "$logfile"
            return 1
        fi

        if ! process_alive "$expected_pid"; then
            echo "$description exited before becoming ready (pid=${expected_pid:-unknown})"
            echo ""
            echo "=== $description log diagnostics ==="
            print_log_diagnostics "$logfile"
            echo ""
            echo "=== $description process diagnostics ==="
            print_engine_timeout_diagnostics "$logfile"
            return 1
        fi

        # Bypass proxy for localhost checks; CI often exports http_proxy.
        if curl --noproxy '*' -sf "$health_url" > /dev/null 2>&1; then
            echo "$description is ready! (took ${elapsed}s)"
            return 0
        fi

        if curl --noproxy '*' -sf "$models_url" > /dev/null 2>&1; then
            echo "$description is ready! (took ${elapsed}s)"
            return 0
        fi

        echo "Waiting for $description... (${elapsed}s elapsed)"
        sleep 5
    done
}

# Wait for both servers (they start simultaneously)
VLLM_PID="$(read_pid_from_file 2 || true)"
VLLM_BASELINE_PID="$(read_pid_from_file 3 || true)"

if ! wait_for_vllm_server "$VLLM_PORT" "vLLM with LMCache" \
    "/tmp/build_${BUILD_ID}_vllm.log" "$VLLM_PID"; then
    exit 1
fi

# The baseline server only exists for 2-GPU tests; 1-GPU tests set
# LAUNCH_BASELINE=false in launch-processes.sh and never start it.
if [[ "${LAUNCH_BASELINE:-true}" == "true" ]]; then
    if ! wait_for_vllm_server "$VLLM_BASELINE_PORT" "vLLM baseline (without LMCache)" \
            "/tmp/build_${BUILD_ID}_vllm_baseline.log" "$VLLM_BASELINE_PID"; then
        exit 1
    fi
fi

echo ""
echo "=== All vLLM servers are ready ==="
