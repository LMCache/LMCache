#!/bin/bash
# Interconnect readiness for P2P: RDMA devices with NUMA placement, IPv6/IPv4 addressing and
# routes to the peer's RDMA addresses, and an ib_write_bw point-to-point test over one device.
# usage: check_rdma.sh [--peer-ssh HOST] [--peer-ip IP] [--device mlx5_bond_0] [--gid-index 3] [--no-bw]
#   (defaults from env.sh: PEER_SSH / PEER_IP). Needs ib_write_bw (perftest) on both hosts for the bw test.
if [ -f "${P2P_ENV:-${P2P_HOME:-$HOME/lmcache-p2p}/env.sh}" ]; then source "$(dirname "$0")/../../lmcache-p2p-launch/scripts/common.sh"; fi
PEER_SSH=${PEER_SSH:-}; PEER_IP=${PEER_IP:-}; DEV=""; GID=3; BW=1
while [ $# -gt 0 ]; do case "$1" in
  --peer-ssh) PEER_SSH=$2; shift 2;; --peer-ip) PEER_IP=$2; shift 2;; --device) DEV=$2; shift 2;; --gid-index) GID=$2; shift 2;; --no-bw) BW=0; shift;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
devs() {  # device, link layer, state, rate, NUMA node, global GID (index $1)
  for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue
    printf "  %-14s %-9s %-7s rate=%-5s numa=%s gid[%s]=%s\n" "$(basename "$d")" "$(cat "$d"/ports/1/link_layer)" "$(awk '{print $2}' "$d"/ports/1/state)" "$(awk '{print $1}' "$d"/ports/1/rate)" "$(cat "$d"/device/numa_node)" "$1" "$(cat "$d"/ports/1/gids/"$1" 2>/dev/null)"
  done
  command -v ibdev2netdev >/dev/null && ibdev2netdev | sed 's/^/  /'
  ip -br addr 2>/dev/null | awk 'NF > 2 && $1 !~ /^(lo|docker|br-|veth|eth0)/ {print "  " $0}'
}
rdma_addrs() { ip -6 -br addr 2>/dev/null | awk 'NF > 2 && $1 !~ /^(lo|docker|br-|veth)/' | grep -oE "2[0-9a-f]{3}:[0-9a-f:]+/[0-9]+" | cut -d/ -f1 | head -8; }
echo "== $(hostname): RDMA devices, netdevs and addresses"; devs "$GID"
if [ -n "$PEER_SSH" ]; then
  echo "== peer $PEER_SSH"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER_SSH" "$(declare -f devs); devs $GID" 2>&1 | head -30
  echo "== routes from here to the peer's global RDMA addresses"
  for a in $(ssh -o BatchMode=yes "$PEER_SSH" "$(declare -f rdma_addrs); rdma_addrs" 2>/dev/null); do
    printf "  %-30s -> %s\n" "$a" "$(ip -6 route get "$a" 2>&1 | head -1 | grep -oE 'dev [a-z0-9]+|unreachable|No route')"; done
fi
if [ "$BW" = 1 ] && [ -n "$PEER_SSH" ] && [ -n "$PEER_IP" ]; then
  DEV=${DEV:-$(ls /sys/class/infiniband | head -1)}
  if command -v ib_write_bw >/dev/null; then
    echo "== ib_write_bw $DEV (1 MiB x 2000, gid index $GID), handshake over $PEER_IP"
    ssh -o BatchMode=yes "$PEER_SSH" "nohup timeout 60 ib_write_bw -d $DEV -x $GID -F -s 1048576 -n 2000 >/tmp/ibw-server.log 2>&1 &"; sleep 2
    timeout 60 ib_write_bw -d "$DEV" -x "$GID" -F -s 1048576 -n 2000 "$PEER_IP" 2>&1 | grep -E "^ *[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+|error|Unable|failed" | sed 's/^/  /'
    echo "  (columns: bytes, iterations, BW peak MiB/s, BW average MiB/s, msg rate; 200 Gb/s -> ~23000 MiB/s)"
  else echo "WARN: ib_write_bw not installed (perftest); skipping bandwidth test"; fi
fi
