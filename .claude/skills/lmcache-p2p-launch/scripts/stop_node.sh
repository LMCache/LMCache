#!/bin/bash
# Stop vLLM and the LMCache server started by start_node.sh (and the coordinator with --with-coordinator).
source "$(dirname "$0")/common.sh"
names="vllm lmcache"; [ "${1:-}" = "--with-coordinator" ] && names="$names coordinator"
for n in $names; do
  f="$LOGDIR/$n.pid"; [ -f "$f" ] || continue
  pid=$(cat "$f"); pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null && echo "stopped $n ($pid)"; rm -f "$f"
done
sleep 3
# vLLM engine-core subprocesses can outlive the API server: kill whatever still owns our ports.
ports="$VLLM_PORT $MP_PORT $HTTP_PORT $P2P_PORT"; [ "${1:-}" = "--with-coordinator" ] && ports="$ports $COORD_PORT"
for p in $ports; do
  for pid in $(ss -ltnp "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do kill -KILL "$pid" 2>/dev/null && echo "killed port $p owner $pid"; done
done
true
