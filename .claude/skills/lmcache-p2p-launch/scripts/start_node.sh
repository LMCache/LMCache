#!/bin/bash
# Start the LMCache server and vLLM for this env in the background (logs/pids in $P2P_HOME/logs).
# usage: start_node.sh [--with-coordinator]   (GPU / L1_GB can be overridden in the environment)
source "$(dirname "$0")/common.sh"
D="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = "--with-coordinator" ]; then
  nohup "$D/launch_coordinator.sh" > "$LOGDIR/coordinator.log" 2>&1 & echo $! > "$LOGDIR/coordinator.pid"
  for _ in $(seq 1 20); do curl -s -o /dev/null "$COORD_URL/instances" && break; sleep 1; done
  echo "coordinator pid $(cat "$LOGDIR/coordinator.pid") on $COORD_URL"
fi
nohup "$D/launch_lmcache.sh" > "$LOGDIR/lmcache.log" 2>&1 & echo $! > "$LOGDIR/lmcache.pid"
sleep 5
if ! kill -0 "$(cat "$LOGDIR/lmcache.pid")" 2>/dev/null; then echo "lmcache server died; tail of $LOGDIR/lmcache.log:"; grep -v flexio "$LOGDIR/lmcache.log" | tail -15; exit 1; fi
nohup "$D/launch_vllm.sh" > "$LOGDIR/vllm.log" 2>&1 & echo $! > "$LOGDIR/vllm.pid"
echo "started lmcache pid $(cat "$LOGDIR/lmcache.pid"), vllm pid $(cat "$LOGDIR/vllm.pid") (GPU $GPU, L1 ${L1_GB} GB); logs in $LOGDIR"
