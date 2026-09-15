#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run one dedicated MP server and validation worker until interrupted.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ADVERTISE_IP INSTANCE_ID COORDINATOR_URL" >&2
  echo 'Set START_COORDINATOR=1 on the coordinator host.' >&2
  exit 2
fi
advertise_ip="$1"
instance_id="$2"
coordinator_url="$3"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
log_dir="${LOG_DIR:-move-logs-${instance_id}}"
mkdir -p "$log_dir"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-DEBUG}"
if [[ "${START_COORDINATOR:-0}" == 1 ]]; then
  lmcache coordinator --host "$advertise_ip" --port "${COORDINATOR_PORT:-19300}" \
    --extra-config '{"move_poll_interval_s":0.2}' > "$log_dir/coordinator.log" 2>&1 &
  pids+=("$!")
fi
lmcache server --host 0.0.0.0 --port "${SERVER_PORT:-19555}" \
  --http-port "${HTTP_PORT:-19755}" --chunk-size 256 \
  --l1-size-gb 2 --eviction-policy LRU --l1-align-bytes 65536 \
  --instance-id "$instance_id" --coordinator-url "$coordinator_url" \
  --coordinator-advertise-ip "$advertise_ip" --coordinator-event-reporting \
  --p2p-advertise-url "$advertise_ip:${P2P_PORT:-19855}" \
  > "$log_dir/server.log" 2>&1 &
pids+=("$!")
"${PYTHON:-python3}" "$script_dir/kv_worker.py" --host "$advertise_ip" \
  --port "${WORKER_PORT:-19955}" --server-url "tcp://127.0.0.1:${SERVER_PORT:-19555}" \
  > "$log_dir/worker.log" 2>&1 &
pids+=("$!")
printf 'Services started. Logs: %s. Ctrl-C stops this harness.\n' "$log_dir"
while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Service $pid exited; inspect $log_dir" >&2
      exit 1
    fi
  done
  sleep 1
done
