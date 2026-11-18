#!/usr/bin/env bash
# Verify recovery on the same engine: re-registration, store, and retrieve.
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"
source "${COMMON_WORKLOAD_DIR}/../verification/l1-round-trip-verification.sh"

LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
RECOVER_TIMEOUT="${RR_RECOVER_TIMEOUT:-150}"
PID_FILE="${MP_TEST_PID_FILE:-/tmp/lmcache_mp_pids_${BUILD_ID}}"
OUT_DIR="$RESULTS_DIR/restart_recovery"
LMCACHE_LOG="${LMCACHE_RESTART_LOG:-/tmp/build_${BUILD_ID}_lmcache_restart.log}"
mkdir -p "$OUT_DIR"

start_lmcache() {
    if [[ -n "${MP_TEST_LMCACHE_LAUNCH_SCRIPT:-}" ]]; then
        bash "$MP_TEST_LMCACHE_LAUNCH_SCRIPT" > "$LMCACHE_LOG" 2>&1 &
    else
        local -a server_environment=()
        if [[ "${ENGINE_USE_ALL_DEVICES:-false}" != "true" ]]; then
            server_environment+=("${DEVICE_AFFINITY_VAR}=${GPU_FOR_ENGINE}")
        fi
        engine_add_lmcache_server_environment server_environment
        local -a transfer_args=(
            --supported-transfer-mode "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"
        )
        if [[ -n "${CHUNK_SIZE:-}" ]]; then
            transfer_args+=(--chunk-size "$CHUNK_SIZE")
        fi
        if [[ "${L1_USE_LAZY:-true}" == "false" ]]; then
            transfer_args+=(--no-l1-use-lazy)
            if [[ "${LMCACHE_MP_TRANSFER_MODE:-}" == "engine_driven" ]]; then
                transfer_args+=(--shm-name "mp_${BUILD_ID}")
            fi
        fi
        env "${server_environment[@]}" lmcache server \
            --transport "$LMCACHE_REQUEST_TRANSPORT" \
            --l1-size-gb "${CPU_BUFFER_SIZE:-80}" \
            --eviction-policy LRU --max-workers "${MAX_WORKERS:-4}" \
            --port "$LMCACHE_PORT" --http-port "$LMCACHE_HTTP_PORT" \
            "${transfer_args[@]}" > "$LMCACHE_LOG" 2>&1 &
    fi
    local new_pid=$!
    python3 - "$PID_FILE" "$new_pid" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
lines[0] = sys.argv[2]
path.write_text("\n".join(lines) + "\n")
PY
}

wait_for_registrations() {
    python3 - "$LMCACHE_HTTP_PORT" "$RECOVER_TIMEOUT" \
        "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" "$OUT_DIR/registrations.json" "$1" <<'PY'
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

port, timeout, mode, snapshot, action = sys.argv[1:]
key = "non_cuda_context_meta" if mode == "engine_driven" else "cache_context_meta"
if action == "recover":
    expected = set(json.loads(Path(snapshot).read_text()))
    if not expected:
        raise SystemExit("No original Worker registrations were captured")
else:
    expected = set()
deadline = time.monotonic() + float(timeout)
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/status", timeout=5) as response:
            status = json.load(response)
        ids = set(status.get(key, {}))
        if ids and (action == "snapshot" or expected <= ids):
            Path(snapshot + "." + action).write_text(json.dumps(status, indent=2))
            if action == "snapshot":
                Path(snapshot).write_text(json.dumps(sorted(ids)))
            print(f"{action}: {key}={sorted(ids)}", flush=True)
            sys.exit(0)
    except (urllib.error.URLError, TimeoutError):
        pass
    time.sleep(1)
raise SystemExit(f"Timed out waiting for {action} of registrations {sorted(expected)}")
PY
}

verify_cache_service() {
    local phase="$1"
    PROMPT_FILE="$OUT_DIR/${phase}_prompt.txt"
    export OUT_A="$OUT_DIR/${phase}_cold.txt"
    export OUT_B="$OUT_DIR/${phase}_hit.txt"
    export MAX_TOKENS=8
    # A unique prefix prevents a previous workload from supplying the cold run.
    printf '%s %s\n' "$BUILD_ID" "$phase" > "$PROMPT_FILE"
    local body="$OUT_DIR/prompt_body.txt"
    l1_round_trip_build_prompt "$body" "$phase" 4 || return 1
    cat "$body" >> "$PROMPT_FILE"
    l1_round_trip_verify_cache_service "$phase" || return 1
}

# A preceding fault-tolerance workload may already have stopped LMCache.
if [[ "${RR_SERVER_STOPPED:-0}" != "1" ]]; then
    verify_cache_service before_restart || exit 1
    wait_for_registrations snapshot || exit 1
else
    python3 - "$RESULTS_DIR/fault_tolerance/registrations.json" \
        "$OUT_DIR/registrations.json" "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" <<'PYREG'
import json
from pathlib import Path
import sys

source, destination, mode = sys.argv[1:]
key = "non_cuda_context_meta" if mode == "engine_driven" else "cache_context_meta"
Path(destination).write_text(json.dumps(sorted(json.loads(Path(source).read_text())[key])))
PYREG
    if [[ $? -ne 0 ]]; then exit 1; fi
fi

old_pid=$(sed -n '1p' "$PID_FILE")
if [[ -z "$old_pid" ]]; then
    echo "FAIL: no LMCache PID in $PID_FILE"
    exit 1
fi
log_offset="${MP_TEST_DEGRADED_LOG_OFFSET:-$(wc -c < "$ENGINE_LOG_FILE")}"
python3 "${COMMON_WORKLOAD_DIR}/../../stop-processes.py" "$old_pid" \
    --ports "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT" || exit 1

# Allow default 10s heartbeats to time out on every adapter before recovery.
sleep "${RR_OUTAGE_SECONDS:-30}"
# This test deliberately observes a failed heartbeat; fast restarts are separate.
python3 - "$ENGINE_LOG_FILE" "$log_offset" "$RECOVER_TIMEOUT" <<'PY'
from pathlib import Path
import sys
import time

log, offset, timeout = sys.argv[1:]
deadline = time.monotonic() + float(timeout)
while time.monotonic() < deadline:
    with Path(log).open("rb") as stream:
        stream.seek(int(offset))
        if b"entering degraded mode" in stream.read():
            print("Observed unhealthy transition before restarting LMCache", flush=True)
            break
    time.sleep(1)
else:
    raise SystemExit("No unhealthy transition observed while LMCache was stopped")
PY
if [[ $? -ne 0 ]]; then exit 1; fi

start_lmcache || exit 1
wait_for_registrations recover || exit 1
verify_cache_service after_restart || exit 1

echo "PASS: original workers re-registered; cache store, hit and retrieve succeeded"
