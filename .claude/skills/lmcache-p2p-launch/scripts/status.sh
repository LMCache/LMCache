#!/bin/bash
# Print the coordinator fleet and the P2P state of this node and its peer.
source "$(dirname "$0")/common.sh"
echo "== coordinator $COORD_URL/instances"
curl -s --max-time 3 "$COORD_URL/instances" | python3 -c '
import sys, json
for i in json.load(sys.stdin)["instances"]:
    print("  %-24s ip=%s http=%s p2p=%s" % (i["instance_id"], i["ip"], i["http_port"], i.get("p2p_advertised_url")))
' 2>/dev/null || echo "  (unreachable)"
for u in "$HTTP_URL" "$PEER_HTTP_URL"; do
  printf "== %s/status: " "$u"
  curl -s --max-time 3 "$u/status" | python3 -c 'import sys,json; d=json.load(sys.stdin); print({k: d[k] for k in sorted(d) if k.startswith("p2p") or k=="active_p2p_lookup_jobs"})' 2>/dev/null || echo "(unreachable)"
done
