#!/bin/bash
# Wait until this node's vLLM answers and its LMCache server reports P2P registered with >= 1 peer.
# usage: wait_ready.sh [--timeout SECONDS] [--peers N]      exit 0 when ready, 1 on timeout
source "$(dirname "$0")/common.sh"
TIMEOUT=600; PEERS=1
while [ $# -gt 0 ]; do case "$1" in --timeout) TIMEOUT=$2; shift 2;; --peers) PEERS=$2; shift 2;; *) shift;; esac; done
t0=$(date +%s)
while :; do
  code=$(curl -s --max-time 2 -o /dev/null -w '%{http_code}' "$VLLM_URL/v1/models")
  st=$(curl -s --max-time 2 "$HTTP_URL/status" 2>/dev/null)
  state=$(printf '%s' "$st" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("p2p_state"), d.get("p2p_peer_count"))' 2>/dev/null)
  if [ "$code" = 200 ] && [ "${state%% *}" = registered ] && [ "${state##* }" -ge "$PEERS" ] 2>/dev/null; then
    echo "READY after $(( $(date +%s) - t0 ))s: vllm=$VLLM_URL p2p_state=registered peers=${state##* }"; exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
    echo "TIMEOUT after ${TIMEOUT}s: vllm http=$code lmcache status='$state'"; echo "lmcache log tail:"; grep -v flexio "$LOGDIR/lmcache.log" | tail -5; echo "vllm log tail:"; tail -5 "$LOGDIR/vllm.log"; exit 1
  fi
  [ -f "$LOGDIR/lmcache.pid" ] && ! kill -0 "$(cat "$LOGDIR/lmcache.pid")" 2>/dev/null && { echo "lmcache server exited; see $LOGDIR/lmcache.log"; grep -v flexio "$LOGDIR/lmcache.log" | grep -i -E "error|Traceback" | tail -5; exit 1; }
  sleep 5
done
