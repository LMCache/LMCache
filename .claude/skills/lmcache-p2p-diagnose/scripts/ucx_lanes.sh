#!/bin/bash
# Summarize a UCX trace log: lane map of the inter-node endpoints and the per-lane split of the
# multi-lane RMA protocols. Produce the log by restarting the LMCache server with
#   UCX_LOG_LEVEL=trace UCX_LOG_FILE=/path/ucx_%h_%p.log   and running one P2P exchange.
# usage: ucx_lanes.sh <ucx log file>
L=${1:?usage: ucx_lanes.sh <ucx log>}
echo "== UCX version / env"; grep -a -m1 -oE "UCX  INFO  Version [^ ]+" "$L"; grep -a -m1 -oE "UCX_\* env variables: .*" "$L" | cut -c1-300
echo "== lanes per endpoint (rma_bw = lanes a multi-rail RMA may use)"
grep -a -oE "ep 0x[0-9a-f]+: lane\[[0-9]+\]: +[0-9]+:[a-z0-9_]+/[a-z0-9_]+:[0-9.]+ .*" "$L" | sed -E 's/ md\[[0-9]+\].*seg [0-9]+ / /' | sort -V | cut -c1-110
echo "== devices with an rma_bw lane:"; grep -a -oE "lane\[[0-9]+\]: +[0-9]+:[a-z0-9_]+/[a-z0-9_]+:[0-9]+\.[0-9]+ .*rma_bw" "$L" | grep -oE "/[a-z0-9_]+:[0-9]+\.[0-9]+" | sort | uniq -c | sed 's/^/  /'
echo "== multi-lane protocol split (ucp_put*/ucp_get* multi): % per lane"
grep -a -A14 -E "remote memory (write|read) by ucp_(put|get)\*\(multi\)" "$L" | grep -a "% on" | sed -E 's/^.*\| *([0-9]+\.\.[a-z0-9]+) *\| *([a-z -]+) *\| */  \1 \2: /' | sort -u | cut -c1-400
echo "(a lane list that names only some devices, or 13% x 8 lanes on 4 LAG bonds, means UCX_MAX_RMA_RAILS is too small: rails count paths)"
