#!/usr/bin/env bash
# Verify prefill fallback when the MP server dies with a retrieve RPC outstanding.
set -eo pipefail

COMMON_WORKLOAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${COMMON_WORKLOAD_DIR}/../helpers.sh"
source "${WORKLOAD_DIR}/verification/l1-round-trip-verification.sh"

FAULT_DIR="${LMCACHE_RETRIEVE_FAULT_DIR:?Launch via run-single-test.sh fault_tolerance}"
LMCACHE_HTTP_PORT="${LMCACHE_HTTP_PORT:-8080}"
PROMPT_FILE="$FAULT_DIR/prompt.txt"
MAX_TOKENS=8
COMPLETION_TIMEOUT=180
expected_rpc=RETRIEVE
if [[ "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" == engine_driven ]]; then
    expected_rpc=PREPARE_RETRIEVE
fi

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

l1_writes() {
    curl --noproxy '*' -fsS --max-time 5 "http://127.0.0.1:${LMCACHE_HTTP_PORT}/metrics" |
        awk '/^lmcache_mp_l1_write_chunks_total([ {])/ { total += $NF }
             END { printf "%.0f\n", total }'
}

echo "=== Baseline and LMCache warmup ==="
l1_round_trip_build_prompt "$PROMPT_FILE" $'\nComplete with the next number: one, two, three,'
writes_before="$(l1_writes)" || fail "could not read LMCache L1 write metrics"
l1_round_trip_send_completion "$FAULT_DIR/baseline.txt" "baseline"
[[ -s "$FAULT_DIR/baseline.txt" ]] || fail "baseline output is empty"

deadline=$((SECONDS + 60))
while true; do
    writes_after="$(l1_writes)" || fail "could not read LMCache L1 write metrics"
    if (( writes_after > writes_before )); then
        break
    fi
    (( SECONDS < deadline )) || fail "no L1 writes observed after warmup"
    sleep 0.25
done
engine_clear_local_cache "$ENGINE_PORT" || fail "could not clear ${ENGINE_NAME}'s local cache"

echo "=== Kill LMCache during retrieve ==="
read -r lmcache_pid < "/tmp/lmcache_mp_pids_${BUILD_ID}"
[[ "$lmcache_pid" =~ ^[1-9][0-9]+$ ]] || fail "invalid MP server PID: $lmcache_pid"
kill -0 "$lmcache_pid" || fail "MP server already exited"

touch "$FAULT_DIR/armed"
l1_round_trip_send_completion "$FAULT_DIR/recomputed.txt" "in-flight retrieve" &
completion_pid=$!
deadline=$((SECONDS + 60))
until [[ -f "$FAULT_DIR/retrieve-pending" ]]; do
    kill -0 "$completion_pid" 2>/dev/null || fail "request finished before reaching $expected_rpc"
    (( SECONDS < deadline )) || fail "no $expected_rpc reached the MP server"
    sleep 0.05
done
read -r marker_pid rpc request_id < "$FAULT_DIR/retrieve-pending"
[[ "$marker_pid" == "$lmcache_pid" && "$rpc" == "$expected_rpc" && -n "$request_id" ]] || fail "invalid $expected_rpc marker"
kill -0 "$completion_pid" 2>/dev/null || fail "request finished before fault injection"
echo "Killing LMCache PID $lmcache_pid during $rpc for request $request_id"
kill -KILL "$lmcache_pid" || fail "could not kill MP server"

# The MP server belongs to the launcher, so this shell cannot wait(1) on it.
deadline=$((SECONDS + 10))
while process_state="$(ps -o stat= -p "$lmcache_pid")" &&
        [[ "${process_state//[[:space:]]/}" != Z* ]]; do
    (( SECONDS < deadline )) || fail "MP server did not exit after SIGKILL"
    sleep 0.1
done

wait "$completion_pid" || fail "original request failed after losing the MP server"
if ! cmp -s "$FAULT_DIR/baseline.txt" "$FAULT_DIR/recomputed.txt"; then
    diff -u "$FAULT_DIR/baseline.txt" "$FAULT_DIR/recomputed.txt" || true
    fail "recomputed output differs from the baseline"
fi
echo "Original request completed with the baseline output."

echo "=== Send a new request with LMCache offline ==="
PROMPT_FILE="$FAULT_DIR/offline-prompt.txt"
# Use a distinct prefix long enough to reach the connector's lookup path.
awk 'BEGIN {
    for (line = 0; line < 256; line++) print "The alphabet starts with A, B, and C."
    print "List the first three letters of the alphabet:"
}' > "$PROMPT_FILE"
l1_round_trip_send_completion "$FAULT_DIR/offline.txt" "LMCache offline"
[[ -s "$FAULT_DIR/offline.txt" ]] || fail "offline output is empty"
echo "PASS: the in-flight request recovered and a new request completed while offline."
