#!/bin/bash
# Which RDMA devices carry P2P traffic? Diffs port_xmit_data/port_rcv_data on every RDMA device
# of this host (and the peer over ssh) and the Ethernet byte counters around one P2P exchange,
# so a TCP fallback or a single-rail configuration is visible at once.
# usage: rail_check.sh [--peer-ssh HOST] [--cmd "command that triggers one P2P exchange"] [--eth eth0]
#   default --cmd runs lmcache-p2p-test/scripts/basic_test.py with the deployment's python.
source "$(dirname "$0")/../../lmcache-p2p-launch/scripts/common.sh"
D="$(cd "$(dirname "$0")" && pwd)"; CMD="python $D/../../lmcache-p2p-test/scripts/basic_test.py"; ETH=""
while [ $# -gt 0 ]; do case "$1" in --peer-ssh) PEER_SSH=$2; shift 2;; --cmd) CMD=$2; shift 2;; --eth) ETH=$2; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
ETH=${ETH:-$(ip -o -4 addr show | awk -v ip="$NODE_IP" '$4 ~ "^"ip"/" {print $2; exit}')}
snap() { for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue; printf "%s %s %s " "$(basename "$d")" "$(cat "$d"/ports/1/counters/port_xmit_data)" "$(cat "$d"/ports/1/counters/port_rcv_data)"; done; echo; }
eth() { awk -v i="$1" '$1 == i":" {print $2, $10}' /proc/net/dev; }
L0=$(snap); E0=$(eth "$ETH"); R0=""; [ -n "$PEER_SSH" ] && R0=$(ssh -o BatchMode=yes "$PEER_SSH" "$(declare -f snap); snap")
echo "== running: $CMD"; eval "$CMD" > "$RESULTS_DIR/rail_check_cmd.log" 2>&1; echo "   exit $? (output in $RESULTS_DIR/rail_check_cmd.log)"
L1=$(snap); E1=$(eth "$ETH"); R1=""; [ -n "$PEER_SSH" ] && R1=$(ssh -o BatchMode=yes "$PEER_SSH" "$(declare -f snap); snap")
python3 - "$L0" "$L1" "$R0" "$R1" "$E0" "$E1" "$ETH" <<'PY'
import sys
def parse(s):
    t = s.split(); return {t[i]: (int(t[i+1]), int(t[i+2])) for i in range(0, len(t), 3)}
def show(name, a, b):
    A, B = parse(a), parse(b); used = 0
    for dev in sorted(A):
        dx = (B[dev][0]-A[dev][0])*4/1e9; dr = (B[dev][1]-A[dev][1])*4/1e9   # counters are in 4-byte units
        if dx > 0.01 or dr > 0.01: print(f"  {name:6} {dev:14} xmit +{dx:6.2f} GB  rcv +{dr:6.2f} GB"); used += 1
    print(f"  {name:6} devices carrying traffic: {used} of {len(A)}")
    return used
print("== RDMA bytes moved during the exchange (local)"); n = show("local", sys.argv[1], sys.argv[2])
if sys.argv[3]: print("== peer"); show("peer", sys.argv[3], sys.argv[4])
e0, e1 = sys.argv[5].split(), sys.argv[6].split()
if len(e0) == 2 and len(e1) == 2:
    rx = (int(e1[0]) - int(e0[0]))/1e9; tx = (int(e1[1]) - int(e0[1]))/1e9
    print(f"== Ethernet {sys.argv[7]} (TCP path): rx +{rx:.3f} GB  tx +{tx:.3f} GB")
    if rx + tx > 0.1 and n == 0: print("VERDICT: traffic went over TCP, not RDMA (UCX fell back; check UCX_TLS / RDMA reachability)")
    elif n == 0: print("VERDICT: no RDMA traffic and no TCP traffic: did the command trigger a P2P load? (see cmd log)")
if n and n < len(parse(sys.argv[1])):
    print(f"VERDICT: only {n} RDMA device(s) used; for all of them see lmcache-p2p-launch/references/ucx-multirail.md (UCX_MAX_RMA_RAILS counts paths, 2 per LAG bond)")
elif n: print("VERDICT: RDMA on every device")
PY
