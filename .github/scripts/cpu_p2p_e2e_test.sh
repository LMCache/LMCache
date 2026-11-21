#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only P2P end-to-end test: a coordinator and two MP servers (s0, s1)
# on loopback share KV through the P2P L2 adapter (NIXL over UCX TCP).
# `lmcache bench server --mode cpu` stands in for the engine: it stores
# sequences on s1 and reads them cold on s0, so every s0 cold hit is a P2P
# hit. Needs no GPU and no vLLM, only `pip install -r requirements/nixl.txt`.
#
# Scenarios run in order on the same cluster:
#   basic               s0 gets every chunk from s1; bytes match s1's.
#   peer_killed         kill -9 s1: s0 lookups miss within the P2P timeouts,
#                       s0 recomputes and drops the dead peer.
#   peer_restart        s1 restarts (same id, OS-assigned P2P RPC port):
#                       s0 reconnects and gets P2P hits again.
#   coordinator_restart kill -9 the coordinator and restart it on its port:
#                       the servers re-register and P2P hits come back.
#
# Usage (Linux): bash .github/scripts/cpu_p2p_e2e_test.sh
# Logs: ${LOG_DIR}/cpu_device_p2p_*.log (default /tmp), tails printed on failure.

set -euo pipefail

LOG_DIR="${LMCACHE_P2P_LOG_DIR:-/tmp}"
NUM_SEQS=4
NUM_TOKENS=2048 # 8 chunks of 256 tokens per sequence
TOTAL_CHUNKS=$((NUM_SEQS * NUM_TOKENS / 256))
# Bound on any single cold lookup, including one against a dead peer
# (--p2p-lookup-timeout and --p2p-load-timeout are 2 s below).
MAX_LOOKUP_MS=10000
# Lower bound on the slowest cold lookup against the dead peer: proves s0
# still held its adapter and waited out --p2p-lookup-timeout.
MIN_DEAD_PEER_LOOKUP_MS=1000
WAIT_TIMEOUT=60

# NIXL rejects tcp,self when it sees an NVIDIA GPU, even a hidden one;
# where cuda_copy is unavailable UCX only warns and skips it.
export UCX_TLS="${UCX_TLS:-tcp,self,cuda_copy}"
export CUDA_VISIBLE_DEVICES="" LMCACHE_DISABLE_BANNER=1 PYTHONUNBUFFERED=1

read -r COORD_PORT MQ0 MQ1 HTTP0 HTTP1 XFER0 XFER1 < <(python3 -c '
import socket
socks = [socket.socket() for _ in range(7)]
for s in socks:
    s.bind(("127.0.0.1", 0))
print(*(s.getsockname()[1] for s in socks))
')
MQ=("${MQ0}" "${MQ1}")
HTTP=("${HTTP0}" "${HTTP1}")
XFER=("${XFER0}" "${XFER1}")
COORD_URL="http://127.0.0.1:${COORD_PORT}"
FAIL_LOG="${LOG_DIR}/cpu_device_p2p_fail.log"
COORD_PID=""
SERVER_PID=("" "")

log() { echo "==> $*"; }

# kill9 <pid>: crash one of our processes and reap it. A killed server
# cannot unlink its POSIX SHM segments, so remove them here.
kill9() {
  kill -9 "$1" 2>/dev/null || true
  wait "$1" 2>/dev/null || true
  rm -f "/dev/shm/lmcache_l1_pool_$1" "/dev/shm/torch_$1_"*
}

cleanup() {
  local status=$? pids
  # Only processes this script started and that are still running.
  pids=$(jobs -p)
  for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
  for pid in ${pids}; do
    for _ in $(seq 1 10); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    kill9 "${pid}"
  done
  if [ "${status}" != 0 ]; then
    for f in "${LOG_DIR}"/cpu_device_p2p_*.log; do
      echo "==> Last 40 lines of ${f}:"
      tail -n 40 "${f}" || true
    done
    # Repeat the cause after the tails so it is the last thing in the log.
    cat "${FAIL_LOG}" >&2 2>/dev/null || true
  fi
}
trap cleanup EXIT

fail() {
  # fail can run inside $(...), so keep the reason in a file for cleanup.
  echo "!! FAIL: $*" | tee -a "${FAIL_LOG}" >&2
  exit 1
}

# wait_until <seconds> <description> <command...>
wait_until() {
  local deadline=$((SECONDS + $1)) what=$2
  shift 2
  until "$@"; do
    [ "${SECONDS}" -lt "${deadline}" ] || fail "timed out waiting for ${what}"
    sleep 0.5
  done
}

start_coordinator() {
  lmcache coordinator --host 127.0.0.1 --port "${COORD_PORT}" \
    --instance-timeout 20 --health-check-interval 1 \
    >>"${LOG_DIR}/cpu_device_p2p_coordinator.log" 2>&1 &
  COORD_PID=$!
}

# start_server <index> [extra lmcache server args...]
start_server() {
  local i=$1
  shift
  lmcache server --host 127.0.0.1 --port "${MQ[i]}" \
    --http-host 127.0.0.1 --http-port "${HTTP[i]}" --instance-id "s${i}" \
    --l1-size-gb 0.5 --eviction-policy LRU --chunk-size 256 \
    --coordinator-url "${COORD_URL}" --coordinator-advertise-ip 127.0.0.1 \
    --coordinator-heartbeat-interval 1 \
    --p2p-advertise-url "127.0.0.1:${XFER[i]}" \
    --p2p-lookup-timeout 2 --p2p-load-timeout 2 "$@" \
    >>"${LOG_DIR}/cpu_device_p2p_s${i}.log" 2>&1 &
  SERVER_PID[i]=$!
}

healthy() { curl -fs --max-time 2 "http://127.0.0.1:${HTTP[$1]}/healthcheck" >/dev/null 2>&1; }

# Fail at once if the server exited instead of waiting for the timeout.
server_up() {
  kill -0 "${SERVER_PID[$1]}" 2>/dev/null || fail "s$1 exited, see cpu_device_p2p_s$1.log"
  healthy "$1"
}

# s0 has an adapter for s1 pointing at s1's currently registered RPC port.
s0_sees_s1() {
  local port
  port=$(curl -fs --max-time 2 "${COORD_URL}/instances" | python3 -c '
import json, sys
print(next(i["mq_port"] for i in json.load(sys.stdin)["instances"] if i["instance_id"] == "s1"))
' 2>/dev/null) || return 1
  curl -fs --max-time 2 "http://127.0.0.1:${HTTP[0]}/status" | python3 -c '
import json, sys
def urls(o):
    if isinstance(o, dict):
        yield o.get("peer_mq_server_url")
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            yield from urls(v)
sys.exit(sys.argv[1] not in urls(json.load(sys.stdin)))
' "grpc://127.0.0.1:${port}"
}

s0_has_no_peer() {
  curl -fs --max-time 2 "http://127.0.0.1:${HTTP[0]}/status" |
    grep -q '"p2p_peer_count": *0'
}

# bench <name> <server index> <first sequence id>
bench() {
  local out="${LOG_DIR}/cpu_device_p2p_bench_$1.log"
  timeout 300 lmcache bench server \
    --rpc-url "tcp://127.0.0.1:${MQ[$2]}" --url "http://127.0.0.1:${HTTP[$2]}" \
    --mode cpu --transfer-mode lmcache_driven \
    --kvcache-shape-spec '(2,256,16,1,64):float16:2' \
    --num-tokens "${NUM_TOKENS}" --interval 0.5 \
    --start "$3" --end "$(($3 + NUM_SEQS))" >"${out}" 2>&1 ||
    fail "bench $1 exited with status $?"
  echo "${out}"
}

cold_hits() {
  grep -o '/cold\] LOOKUP: [0-9]*/[0-9]*' "$1" |
    awk '{split($3, a, "/"); h += a[1]; t += a[2]} END {print h + 0 "/" t + 0}'
}

max_cold_lookup_ms() {
  grep -o '/cold\] LOOKUP: .* chunks hit ([0-9.]* ms)' "$1" |
    sed 's/.*(\([0-9.]*\) ms)/\1/' | sort -n | tail -n 1
}

# The bench's own cold-store vs warm-retrieve check on one server.
expect_local_checksums() {
  grep -q "CHECKSUM MISMATCH" "$1" && fail "checksum mismatch in $1"
  local ok
  ok=$(grep -c "CHECKSUM MATCH OK" "$1" || true)
  [ "${ok}" = "${NUM_SEQS}" ] || fail "${ok}/${NUM_SEQS} checksum matches in $1"
}

# Every chunk s0 read through P2P equals what the bench stored on s1:
# s1's cold (store) digests must equal s0's warm (retrieve) digests.
expect_same_bytes() {
  local stored read
  stored=$(grep -o 'seq [0-9]*/cold\] CHECKSUM: [0-9a-f]*' "$1" | sed 's|/cold\]||')
  read=$(grep -o 'seq [0-9]*/warm\] CHECKSUM: [0-9a-f]*' "$2" | sed 's|/warm\]||')
  [ "$(echo "${stored}" | wc -l)" = "${NUM_SEQS}" ] || fail "missing store digests in $1"
  [ "${stored}" = "${read}" ] || fail "bytes read on s0 differ from s1's ($1 vs $2)"
}

# expect_p2p_hits <name> <first sequence id>: s1 stores, s0 must get every chunk via P2P.
expect_p2p_hits() {
  local fill read hits
  fill=$(bench "$1_fill_s1" 1 "$2")
  expect_local_checksums "${fill}"
  read=$(bench "$1_read_s0" 0 "$2")
  hits=$(cold_hits "${read}")
  [ "${hits}" = "${TOTAL_CHUNKS}/${TOTAL_CHUNKS}" ] ||
    fail "$1: s0 cold P2P hits ${hits}, want ${TOTAL_CHUNKS}/${TOTAL_CHUNKS}"
  expect_same_bytes "${fill}" "${read}"
  log "$1: P2P hits ${hits}, bytes match"
}

scenario_basic() {
  expect_p2p_hits basic 0
}

scenario_peer_killed() {
  local fill read hits ms errors_before
  fill=$(bench peer_killed_fill_s1 1 100)
  expect_local_checksums "${fill}"
  kill9 "${SERVER_PID[1]}"
  # s0 still holds an adapter for the dead peer here: the coordinator keeps
  # s1 for --instance-timeout (20 s), asserted by the lower lookup bound.
  read=$(bench peer_killed_read_s0 0 100)
  hits=$(cold_hits "${read}")
  [ "${hits}" = "0/${TOTAL_CHUNKS}" ] || fail "peer_killed: cold hits ${hits} from a dead peer"
  expect_local_checksums "${read}"
  ms=$(max_cold_lookup_ms "${read}")
  [ -n "${ms}" ] || fail "peer_killed: no cold lookup times in ${read}"
  awk -v ms="${ms}" -v max="${MAX_LOOKUP_MS}" 'BEGIN {exit !(ms + 0 <= max)}' ||
    fail "peer_killed: cold lookup took ${ms} ms (bound ${MAX_LOOKUP_MS} ms)"
  awk -v ms="${ms}" -v min="${MIN_DEAD_PEER_LOOKUP_MS}" 'BEGIN {exit !(ms + 0 >= min)}' ||
    fail "peer_killed: max cold lookup ${ms} ms < ${MIN_DEAD_PEER_LOOKUP_MS} ms, the dead peer was never looked up"
  wait_until "${WAIT_TIMEOUT}" "s0 to drop the dead peer" s0_has_no_peer
  errors_before=$(grep -c "LMCache ERROR" "${LOG_DIR}/cpu_device_p2p_s0.log" || true)
  read=$(bench peer_killed_after_s0 0 110)
  expect_local_checksums "${read}"
  [ "$(grep -c "LMCache ERROR" "${LOG_DIR}/cpu_device_p2p_s0.log" || true)" = "${errors_before}" ] ||
    fail "peer_killed: s0 logged new errors after dropping the dead peer"
  healthy 0 || fail "peer_killed: s0 unhealthy"
  log "peer_killed: s0 recomputed (max cold lookup ${ms} ms) and dropped the peer"
}

# Start the crashed s1 again with the same instance id and URLs (OS-assigned
# P2P RPC port) and wait until s0 targets the new process.
restart_s1() {
  start_server 1
  wait_until "${WAIT_TIMEOUT}" "restarted s1 to be healthy" server_up 1
  wait_until "${WAIT_TIMEOUT}" "s0 to reconnect to restarted s1" s0_sees_s1
}

scenario_peer_restart() {
  restart_s1
  expect_p2p_hits peer_restart 200
}

scenario_coordinator_restart() {
  kill9 "${COORD_PID}"
  wait_until "${WAIT_TIMEOUT}" "s0 to drop s1 while the coordinator is down" s0_has_no_peer
  start_coordinator
  wait_until "${WAIT_TIMEOUT}" "s0 to rediscover s1 after the coordinator restart" s0_sees_s1
  expect_p2p_hits coordinator_restart 300
}

rm -f "${LOG_DIR}"/cpu_device_p2p_*.log
log "Starting coordinator and servers (UCX_TLS=${UCX_TLS})"
start_coordinator
wait_until "${WAIT_TIMEOUT}" "coordinator" curl -fs --max-time 2 -o /dev/null "${COORD_URL}/instances"
start_server 0
start_server 1
wait_until "${WAIT_TIMEOUT}" "s0 to be healthy" server_up 0
wait_until "${WAIT_TIMEOUT}" "s1 to be healthy" server_up 1
wait_until "${WAIT_TIMEOUT}" "s0 to discover s1" s0_sees_s1

for scenario in basic peer_killed peer_restart coordinator_restart; do
  start=${SECONDS}
  log "Scenario ${scenario}"
  "scenario_${scenario}"
  log "Scenario ${scenario} passed in $((SECONDS - start)) s"
done
log "CPU P2P e2e passed."
