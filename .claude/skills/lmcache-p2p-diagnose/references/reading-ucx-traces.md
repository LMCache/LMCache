# Reading a UCX trace for P2P lane selection

Produce: restart the LMCache server with
`UCX_LOG_LEVEL=trace UCX_LOG_FILE=$P2P_HOME/logs/ucx_%h_%p.log`, run one P2P exchange
(`basic_test.py`), then `scripts/ucx_lanes.sh $P2P_HOME/logs/ucx_<host>_<pid>.log`. The file
stays small (a few hundred KB): trace logs setup, not the data path.

What to look at:

1. **Version and env**: `UCX  INFO  Version 1.23.0` and `UCX_* env variables: ...` confirm the
   variables reached the process (`collect_bundle.sh` also dumps `/proc/<pid>/environ`).
2. **Lane map** lines, one per lane of each endpoint:
   `ep 0x...: lane[9]: 26:rc_mlx5/mlx5_bond_5:1.0 md[9] -> addr[6].md[9]/ib/sysdev[25] ... rma_bw#10`.
   `rma_bw#N` marks lanes a multi-rail RMA may use; `:1.0` / `:1.1` are the two paths of a LAG
   bond; the `inter-node` endpoint is the one to the peer (`self` endpoints also exist). All
   devices appearing here only means they were *eligible*.
3. **Protocol tables** printed when a protocol is first selected, e.g.
   `remote memory write by ucp_put*(multi) from host memory to host` followed by rows like
   `| 6130..inf | zero-copy | 13% on rc_mlx5/mlx5_bond_0:1/path0, 13% on ... |`. These
   percentages are the actual split of each operation across lanes. Eight lanes at 13% on
   four bonds = `UCX_MAX_RMA_RAILS=8` consumed by paths; sixteen at ~7% = all eight bonds.
   The `ucp_get*(multi) ... from host` table is the P2P read path; it is printed on first
   use, so it may be missing if the log was captured before the first read completed.
4. **Memory registration**: `KSM-mt multi-thread-key memory registration status "Success"`
   once per device confirms the L1 buffer was registered on every NIC.

Caveats: UCX truncates its own one-line `inter-node cfg#N rma(...) am(...)` summary at about
250 characters, so use the lane map lines instead. Percentages are planned shares from UCX's
bandwidth model (NUMA-far devices get a lower estimate); confirm the real split with
`rail_check.sh` counters.
