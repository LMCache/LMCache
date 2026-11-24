#!/bin/bash
# Run Prometheus in docker (host network) scraping both LMCache servers, the coordinator and both vLLMs.
# usage: start_prometheus.sh [--port 9710] [--interval 2s]      (stop: docker rm -f p2p-prometheus)
source "$(dirname "$0")/../../lmcache-p2p-launch/scripts/common.sh"
PORT=9710; INTERVAL=2s
while [ $# -gt 0 ]; do case "$1" in --port) PORT=$2; shift 2;; --interval) INTERVAL=$2; shift 2;; *) shift;; esac; done
command -v docker >/dev/null || { echo "docker not found; run prometheus another way with the config below" >&2; }
mkdir -p "$P2P_HOME/prometheus"
cat > "$P2P_HOME/prometheus/prometheus.yml" <<YML
global:
  scrape_interval: $INTERVAL
scrape_configs:
  - job_name: lmcache
    static_configs:
      - targets: ["$NODE_IP:$HTTP_PORT"]
        labels: {node: "$NODE_IP"}
      - targets: ["$PEER_IP:$PEER_HTTP_PORT"]
        labels: {node: "$PEER_IP"}
  - job_name: coordinator
    static_configs:
      - targets: ["$COORD_IP:$COORD_PORT"]
  - job_name: vllm
    static_configs:
      - targets: ["$NODE_IP:$VLLM_PORT"]
        labels: {node: "$NODE_IP"}
      - targets: ["$PEER_IP:$PEER_VLLM_PORT"]
        labels: {node: "$PEER_IP"}
YML
docker rm -f p2p-prometheus >/dev/null 2>&1
docker run -d --name p2p-prometheus --network host -v "$P2P_HOME/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:latest --config.file=/etc/prometheus/prometheus.yml --web.listen-address=:$PORT --storage.tsdb.retention.time=2d >/dev/null \
  && echo "Prometheus UI: http://$NODE_IP:$PORT  (config: $P2P_HOME/prometheus/prometheus.yml)"
sleep 3; curl -s "http://127.0.0.1:$PORT/api/v1/targets" | python3 -c '
import sys, json
for t in json.load(sys.stdin)["data"]["activeTargets"]:
    print("  %-12s %-40s %s %s" % (t["labels"]["job"], t["scrapeUrl"], t["health"], t.get("lastError", "")[:60]))
' 2>/dev/null
echo "export PROM_URL=http://127.0.0.1:$PORT"
