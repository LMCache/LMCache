---
name: lmcache-p2p-diagnose
description: Diagnose LMCache P2P KV sharing problems: peers not registered, server dying at startup (NIXL_ERR_BACKEND), zero cached tokens on the peer, low hit ratio (l1_oom, l1_contended), slow or single-NIC transfers, TCP fallback, unreachable /status or /metrics; includes an RDMA readiness check, a NIXL/UCX probe, a per-NIC traffic check, UCX trace reading and a debug-bundle collector. Use when a P2P deployment misbehaves or before filing a P2P issue.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "[symptom] | rdma | rails | bundle"
---

# LMCache P2P: diagnose

Works on a deployment described by the `env.sh` from `lmcache-p2p-launch` (`$P2P_ENV`).
Start from the symptom in `references/decision-tree.md`; the scripts below gather the facts
each branch needs. Never guess at UCX tunables: the trace (`ucx_lanes.sh`) shows what UCX
actually chose. Collect a bundle before changing anything if the problem may need an issue.

## Triage order

1. **Is the fleet up?** `../lmcache-p2p-launch/scripts/status.sh`: coordinator lists both
   instances, both `/status` show `p2p_state=registered` and `p2p_peer_count >= 1`. If an
   instance is missing, read its `$P2P_HOME/logs/lmcache.log` (ignore `flexio` lines): the
   decision tree covers `NIXL_ERR_BACKEND`, coordinator refused, port in use, P2P refused
   without `--coordinator-url`, GDS/DAX L1.
2. **Does a transfer happen?** `../lmcache-p2p-test/scripts/basic_test.py`. Zero cached tokens
   on the peer with the counter `lmcache_mp_l2_load_completed_requests_total{l2_name="p2p"}`
   growing means the client was streaming (stats are non-streaming only). Counter flat: see
   "no P2P hit" in the tree.
3. **Is it RDMA, and on how many NICs?** `scripts/rail_check.sh [--peer-ssh HOST]` diffs the
   RDMA port counters and the Ethernet counters around one exchange and prints a verdict
   (TCP fallback / N of M devices / all devices). `scripts/check_rdma.sh` reports devices,
   NUMA placement, routes to the peer and an `ib_write_bw` point-to-point number (about
   23 GB/s per 200 Gb/s link).
4. **Why did UCX choose that?** Restart the server once with
   `UCX_LOG_LEVEL=trace UCX_LOG_FILE=$P2P_HOME/logs/ucx_%h_%p.log`, run one exchange, then
   `scripts/ucx_lanes.sh <log>`: lane map per endpoint and the per-lane % of the multi-lane
   RMA protocol. Interpretation: `references/reading-ucx-traces.md`.
5. **Will a candidate UCX setting even start?** `scripts/nixl_probe.py --tls rc,sm,self,cuda_copy
   --env UCX_MAX_RMA_RAILS=16` creates a NIXL UCX agent in two seconds with that environment and
   prints the backend's own error if it would fail at server start.
6. **Low hit ratio with no errors**: read
   `lmcache_mp_l2_prefetch_failure_chunks_total{reason}` on the reading node; the tree maps
   `l1_oom` and `l1_contended` to fixes.
7. **Report**: `scripts/collect_bundle.sh` writes one tar.gz with versions, env, probe,
   coordinator and status JSON, metrics, log tails and error greps, the running server's
   UCX/NIXL environment, RDMA state and (over ssh) the peer's log tails. Attach it to the issue
   with the symptom, the exact command, and the decision-tree branch you reached.

## Eval prompts

- "P2P peer count is 0 on node B" -> status, B's log, coordinator `/instances`; usually
  `--coordinator-advertise-ip` unreachable, coordinator down, or the server died at start.
- "lmcache server exits right after start with NIXL_ERR_BACKEND" -> `UCX_TLS` lacks
  `cuda_copy`; prove with `nixl_probe.py`, fix env.sh, restart.
- "Only 4 of 8 NICs are used" -> `rail_check.sh`, then `ucx_lanes.sh`: rails count LAG paths,
  set `UCX_MAX_RMA_RAILS=16`.
- "Hit ratio dropped to 0.8 under load" -> failure reasons; `l1_oom` -> raise `L1_GB`.
