#!/bin/bash
# Foreground LMCache MP server with P2P enabled. Extra args are appended (e.g. --l2-adapter ...).
source "$(dirname "$0")/common.sh"
NUMA=""; command -v numactl >/dev/null && NUMA="numactl --interleave=all"
# --http-host 0.0.0.0: the default 127.0.0.1 hides /status and /metrics from other hosts.
# --prometheus-port is ignored by the MP server; /metrics is served on --http-port.
exec $NUMA lmcache server \
  --max-workers 4 \
  --host 0.0.0.0 --port "$MP_PORT" --http-host 0.0.0.0 --http-port "$HTTP_PORT" \
  --l1-size-gb "$L1_GB" --eviction-policy LRU --l1-align-bytes 65536 \
  --chunk-size 256 \
  --instance-id "$INSTANCE_ID" \
  --coordinator-url "$COORD_URL" \
  --coordinator-advertise-ip "$NODE_IP" \
  --p2p-advertise-url "$NODE_IP:$P2P_PORT" \
  --p2p-listen-url "0.0.0.0:$P2P_PORT" \
  "$@"
