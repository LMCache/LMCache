# UCX multi-rail for P2P (nixl transfer engine)

The default P2P transfer engine is nixl on UCX. With no UCX variables set, UCX picks one
device (`mlx5_bond_0` on the test nodes) and every transfer is capped by that link
(about 23 GB/s on a 200 Gb/s bond). Three things were needed to use all eight bonds.

## 1. `UCX_TLS` must include `cuda_copy`

nixl 1.5 validates `UCX_TLS` when it creates the UCX backend. The old recipe
`UCX_TLS=rc,sm,self` fails on any host with a GPU:

```
config.cpp:84] Invalid UCX_TLS=rc,sm,self for NIXL UCX backend: NVIDIA GPU(s) are present,
but this setting does not enable CUDA memory support. Add cuda_copy ...
nixl_agent.cpp:342] createBackend: backend creation failed for 'UCX'
nixlBackendError: NIXL_ERR_BACKEND      -> "Application startup failed", server exits
```

Use `UCX_TLS=rc,sm,self,cuda_copy`. The `lmcache-p2p-diagnose` skill's `nixl_probe.py`
checks a candidate setting in two seconds without restarting anything.

## 2. `UCX_MAX_RMA_RAILS` counts paths, not devices

Each `mlx5_bond_*` LAG device exposes two UCX paths (`mlx5_bond_0:1.0` and `:1.1`).
`UCX_MAX_RMA_RAILS=8` therefore splits every RMA over bonds 0-3 x 2 paths and never
touches bonds 4-7. The UCX trace shows it: 16 `rma_bw` lanes exist but the
`ucp_put*(multi)` protocol table lists eight lanes at 13% each. With
`UCX_MAX_RMA_RAILS=16` the table shows 16 lanes at about 7% and every bond carries
an equal share. `gen_env.sh` computes `rails = devices x (2 if LAG else 1)`, capped at 16.

Things that did **not** help and are not needed: `UCX_IB_PREFER_NEAREST_DEVICE=n`,
`UCX_IB_NUM_PATHS=1`, `UCX_MULTI_PATH_RATIO`. `UCX_NET_DEVICES=<all devices>` and
`UCX_MULTI_LANE_MAX_RATIO=1000` are set as belt and braces (explicit device list; do not
drop NUMA-far lanes because their estimated bandwidth is lower).

## 3. Verify with NIC counters, not with logs

`port_xmit_data` / `port_rcv_data` under `/sys/class/infiniband/<dev>/ports/1/counters/`
(units of 4 bytes) before and after one P2P exchange tell you which devices moved bytes;
`/proc/net/dev` for the Ethernet interface tells you whether anything fell back to TCP.
The diagnose skill's `rail_check.sh` does this diff. Expected: every bond gains the same
amount, the TCP interface gains nothing.

## Effect (two H200 nodes, 80 GB L1, Qwen3-8B)

| run | 1 rail peer p50 TTFT / prefill tok/s | 8 rails |
|---|---|---|
| 24 docs x 7k tokens, concurrency 32 | 0.57 s / 359k | 0.38 s / 544k |
| 8 docs x 23k tokens, concurrency 8 | 0.54 s / 322k | 0.38 s / 360k |
| 32 docs x 7k, concurrency 16, 7 rounds | 0.29 s / 377k | 0.21 s / 480k |

The per-load throughput histogram (`lmcache_mp_l2_load_throughput_GB_per_second`) peaks
near 120 GB/s with 8 rails against 14-40 GB/s with one. Workloads that are bound by
L1-to-GPU copies or by queueing (15k-token prompts at 16 in flight in `lmcache bench`)
do not change.

## Reading the UCX lane map yourself

Restart the server with `UCX_LOG_LEVEL=trace UCX_LOG_FILE=/path/ucx_%h_%p.log` (small file,
setup only), run one transfer, then grep `lane\[` for the lane map and the
`ucp_put*(multi)` / `ucp_get*(multi)` tables for the per-lane split. UCX truncates its own
`inter-node cfg#` summary line at about 250 characters, so do not rely on that one.
