#!/usr/bin/env bash
# Verify recovery on the same engine: re-registration, write volume, and retrieve.
set -o pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"
source "${COMMON_WORKLOAD_DIR}/../verification/l1-round-trip-verification.sh"

LMCACHE_PORT="${LMCACHE_PORT:-6555}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
NUM_REQUESTS="${RR_NUM_REQUESTS:-100}"
REQUEST_LEN="${RR_REQUEST_LEN:-5000}"
KV_CACHE_VOLUME="${RR_KV_CACHE_VOLUME:-5}"
RECOVER_TIMEOUT="${RR_RECOVER_TIMEOUT:-150}"
PID_FILE="${MP_TEST_PID_FILE:-/tmp/lmcache_mp_pids_${BUILD_ID}}"
OUT_DIR="$RESULTS_DIR/restart_recovery"
LMCACHE_LOG="${LMCACHE_RESTART_LOG:-/tmp/build_${BUILD_ID}_lmcache_restart.log}"
mkdir -p "$OUT_DIR"

run_bench_round() {
    local label="$1"
    local seed=$2
    local out_subdir="$OUT_DIR/$label"
    local -a token_budget_args=()
    if [[ -n "${RR_TOKENS_PER_GB:-}" ]]; then
        token_budget_args=(--tokens-per-gb-kvcache "$RR_TOKENS_PER_GB")
    fi
    mkdir -p "$out_subdir"
    echo "--- bench round: $label ---"

    if ! lmcache bench engine \
        --engine-url "http://localhost:${ENGINE_PORT}" \
        --lmcache-url "http://localhost:${LMCACHE_HTTP_PORT}" \
        --workload random-prefill \
        --rp-num-requests "$NUM_REQUESTS" \
        --rp-request-length "$REQUEST_LEN" \
        --kv-cache-volume "$KV_CACHE_VOLUME" \
        "${token_budget_args[@]}" \
        --no-interactive \
        --no-csv \
        --json \
        --quiet \
        --seed $seed \
        --output-dir "$out_subdir" \
        2>&1 | tee "$out_subdir/bench.log"; then
        echo "FAIL: bench round '$label' returned non-zero (failed requests)"
        return 1
    fi

    local successful failed
    successful=$(python3 -c "import json; print(json.load(open('$out_subdir/bench_summary.json'))['results']['successful_requests'])") || return 1
    failed=$(python3 -c "import json; print(json.load(open('$out_subdir/bench_summary.json'))['results']['failed_requests'])") || return 1
    echo "$label: successful=$successful failed=$failed"
    if [ "$failed" -ne 0 ] || [ "$successful" -ne "$NUM_REQUESTS" ]; then
        echo "FAIL: '$label' completed $successful/$NUM_REQUESTS requests with $failed failures"
        return 1
    fi
    return 0
}

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
        "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" "$OUT_DIR/registrations.json" "$1" \
        "${RR_EXCLUDE_REGISTRATIONS:-}" <<'PY'
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

port, timeout, mode, snapshot, action, excluded_snapshot = sys.argv[1:]
key = "non_cuda_context_meta" if mode == "engine_driven" else "cache_context_meta"
if action == "recover":
    expected = set(json.loads(Path(snapshot).read_text()))
    if not expected:
        raise SystemExit("No original Worker registrations were captured")
else:
    expected = set()
excluded = set()
if excluded_snapshot:
    excluded = set(json.loads(Path(excluded_snapshot).read_text()).get(key, {}))
deadline = time.monotonic() + float(timeout)
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/status", timeout=5) as response:
            status = json.load(response)
        ids = set(status.get(key, {})) - excluded
        Path(snapshot + "." + action).write_text(json.dumps(status, indent=2))
        if ids and (action == "snapshot" or expected <= ids):
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

measure_bench_writes() {
    local label="$1" seed="$2" before after
    engine_clear_local_cache "$ENGINE_PORT" || return 1
    before=$(l1_round_trip_metric lmcache_mp_l1_write_chunks_total) || return 1
    run_bench_round "$label" "$seed" || return 1
    sleep "${STORE_DRAIN_SECONDS:-5}"
    after=$(l1_round_trip_metric lmcache_mp_l1_write_chunks_total) || return 1
    printf '%s\n' "$((after - before))" > "$OUT_DIR/${label}_writes.txt"
}

# CPU prepares round 1 before fault tolerance, then resumes the same outage.
if [[ "${RR_SERVER_STOPPED:-0}" != "1" ]]; then
    measure_bench_writes round1 41 || exit 1
    wait_for_registrations snapshot || exit 1
    if [[ "${RR_PREPARE_ONLY:-0}" == "1" ]]; then exit 0; fi
fi
ROUND1_WRITES=$(cat "$OUT_DIR/round1_writes.txt") || exit 1
if ((ROUND1_WRITES <= 0)); then
    echo "FAIL: round 1 produced no completed L1 writes"
    exit 1
fi

old_pid=$(sed -n '1p' "$PID_FILE")
if [[ -z "$old_pid" ]]; then
    echo "FAIL: no LMCache PID in $PID_FILE"
    exit 1
fi
python3 "${COMMON_WORKLOAD_DIR}/../../stop-processes.py" "$old_pid" \
    --ports "$LMCACHE_PORT" "$LMCACHE_HTTP_PORT" || exit 1

# Allow default 10s heartbeats to time out on every adapter before recovery.
sleep "${RR_OUTAGE_SECONDS:-30}"
start_lmcache || exit 1
wait_for_registrations recover || exit 1
# Registration is visible before the Worker finishes publishing recovery.
# Wait for a real store, clearing local KV so fallback cannot hide readiness.
deadline=$((SECONDS + RECOVER_TIMEOUT))
PROMPT_FILE="$OUT_DIR/recovery_probe.txt"
printf '%s recovery probe\n' "$BUILD_ID" > "$PROMPT_FILE"
l1_round_trip_build_prompt "$OUT_DIR/probe_body.txt" recovery 4 || exit 1
cat "$OUT_DIR/probe_body.txt" >> "$PROMPT_FILE"
MAX_TOKENS=8
writes_before=$(l1_round_trip_metric lmcache_mp_l1_write_chunks_total) || exit 1
while :; do
    engine_clear_local_cache "$ENGINE_PORT" || exit 1
    l1_round_trip_send_completion "$OUT_DIR/recovery_probe_output.txt" recovery_probe || exit 1
    sleep 1
    writes_after=$(l1_round_trip_metric lmcache_mp_l1_write_chunks_total) || exit 1
    if ((writes_after > writes_before)); then break; fi
    if ((SECONDS >= deadline)); then
        echo "FAIL: Worker registered but no completed L1 writes after recovery"
        exit 1
    fi
done

measure_bench_writes round2 42 || exit 1
ROUND2_WRITES=$(cat "$OUT_DIR/round2_writes.txt") || exit 1
if ((ROUND2_WRITES <= 0 || ROUND2_WRITES * 10 < ROUND1_WRITES * 8)); then
    echo "FAIL: round 2 writes=$ROUND2_WRITES; expected >=80% of round 1=$ROUND1_WRITES"
    exit 1
fi
echo "PASS: round 2 writes=$ROUND2_WRITES >=80% of round 1=$ROUND1_WRITES"
engine_clear_local_cache "$ENGINE_PORT" || exit 1
verify_cache_service after_restart || exit 1

echo "PASS: original workers re-registered; cache store, hit and retrieve succeeded"
