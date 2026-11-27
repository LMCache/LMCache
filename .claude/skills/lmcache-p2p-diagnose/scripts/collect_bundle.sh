#!/bin/bash
# Collect everything needed to report a P2P problem into one tar.gz: versions, env.sh, probe
# output, coordinator /instances, /status and /metrics of both nodes, log tails, UCX/NIXL env
# of the running server, RDMA device state, and optional peer logs over ssh.
# usage: collect_bundle.sh [--out FILE.tar.gz] [--lines 400]
source "$(dirname "$0")/../../lmcache-p2p-launch/scripts/common.sh"
D="$(cd "$(dirname "$0")" && pwd)"; OUT="$RESULTS_DIR/p2p-bundle-$(hostname -s)-$(date +%Y%m%d-%H%M%S).tar.gz"; LINES=400
while [ $# -gt 0 ]; do case "$1" in --out) OUT=$2; shift 2;; --lines) LINES=$2; shift 2;; *) shift;; esac; done
T=$(mktemp -d); B="$T/bundle"; mkdir -p "$B"
{ echo "host: $(hostname) $(date -Iseconds)"; python -c "import torch,vllm,lmcache,nixl;print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__,'lmcache',lmcache.__version__)" 2>&1; lmcache --version 2>&1 | tail -1; } > "$B/versions.txt"
cp "$P2P_ENV" "$B/env.sh" 2>/dev/null
"$D/../../lmcache-p2p-launch/scripts/probe_env.sh" --model "$MODEL" > "$B/probe_env.txt" 2>&1
"$D/../../lmcache-p2p-launch/scripts/status.sh" > "$B/status.txt" 2>&1
curl -s --max-time 5 "$COORD_URL/instances" > "$B/coordinator_instances.json" 2>/dev/null
for u in "$HTTP_URL" "$PEER_HTTP_URL"; do n=$(echo "$u" | tr -c 'a-zA-Z0-9' '_'); curl -s --max-time 5 "$u/status" > "$B/status_$n.json"; curl -s --max-time 5 "$u/metrics" | grep -E "^lmcache_mp_(lookup|l2_|l1_(usage|evicted|allocation)|active_p2p|event_bus|timeout)" > "$B/metrics_$n.txt"; done
for f in coordinator lmcache vllm; do [ -f "$LOGDIR/$f.log" ] && { grep -v flexio "$LOGDIR/$f.log" | tail -n "$LINES" > "$B/$f.log.tail"; grep -a -i -E "error|exception|traceback|warn" "$LOGDIR/$f.log" | grep -v -E "flexio|PQC|deprecat" | tail -100 > "$B/$f.log.errors"; }; done
if [ -f "$LOGDIR/lmcache.pid" ] && kill -0 "$(cat "$LOGDIR/lmcache.pid")" 2>/dev/null; then tr '\0' '\n' < "/proc/$(cat "$LOGDIR/lmcache.pid")/environ" | grep -E "^(UCX_|NIXL_|LMCACHE_|HF_)" > "$B/server_env.txt"; else echo "lmcache server not running" > "$B/server_env.txt"; fi
"$D/check_rdma.sh" --no-bw > "$B/rdma.txt" 2>&1
ls "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.log 2>/dev/null | head -20 | xargs -r -I{} cp {} "$B/" 2>/dev/null
if [ -n "$PEER_SSH" ]; then mkdir -p "$B/peer"; ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER_SSH" "cd \"$P2P_HOME/logs\" 2>/dev/null && for f in lmcache vllm; do [ -f \$f.log ] && grep -v flexio \$f.log | tail -n $LINES; echo; done" > "$B/peer/logs.tail" 2>&1; fi
tar -C "$T" -czf "$OUT" bundle && rm -rf "$T"
echo "wrote $OUT"; tar -tzf "$OUT" | sed 's/^/  /'
