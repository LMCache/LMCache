# P2P symptom -> cause -> fix

Each entry: how it shows up, how to confirm, what to do. Logs are `$P2P_HOME/logs/*.log`;
`flexio ... Illegal application argument: NULL` lines are DOCA noise and never the cause.

## Server does not come up

| symptom | confirm | fix |
|---|---|---|
| `lmcache.log` ends with `nixlBackendError: NIXL_ERR_BACKEND`, `Application startup failed`, `Fatal Python error: Aborted` | the `E... config.cpp:84] Invalid UCX_TLS=... Add cuda_copy` line above it; `scripts/nixl_probe.py` reproduces it | `UCX_TLS=rc,sm,self,cuda_copy` (nixl >= 1.5 on GPU hosts); or unset `UCX_TLS` |
| `refuses to start` mentioning `--coordinator-url` | `--p2p-advertise-url` given without `--coordinator-url` | add `--coordinator-url` (env.sh `COORD_URL`) |
| server refuses P2P with GDS or Device-DAX L1 | `--gds-l1-path` / `--l1-devdax-path` in the command | P2P needs a single DRAM L1 region; drop those tiers on P2P nodes |
| `address already in use` | `probe_env.sh` port check; another instance on this host | every instance needs distinct `--port`, `--http-port`, P2P port and vLLM port |
| vLLM `Timed out waiting for engine core` | check the API server start time: the line may be the **previous** vLLM writing into the truncated log during a restart | if the current process is alive and `/v1/models` answers, ignore |
| vLLM never ready, engine core dead | `vllm.log` for CUDA OOM (`GPU_MEM_UTIL`, `MAX_MODEL_LEN`), model not in HF cache (`HF_HUB_OFFLINE=1`) | fix the model path / lower memory settings |

## Peers do not see each other

| symptom | confirm | fix |
|---|---|---|
| `/status` has `p2p_state` != `registered` or `p2p_peer_count: 0` | `curl $COORD_URL/instances`: is the instance listed with the right `ip` and `p2p_advertised_url`? | coordinator unreachable: `COORD_IP`/port, firewall; wrong advertise ip: `NODE_IP` must be routable from the peer; one peer only registers after its vLLM is up: wait |
| `/status` unreachable from another host, Prometheus target down, `lmcache kvcache clear --url` fails | `ss -ltnp | grep 7700` shows `127.0.0.1:7700` | `--http-host 0.0.0.0` (set by the launch scripts); `--prometheus-port` is ignored, `/metrics` is on the HTTP port |
| instance listed twice / stale after a restart | same `--instance-id` re-registered | harmless; the coordinator expires the stale entry after `--instance-timeout` |

## No P2P hit although peers are registered

| symptom | confirm | fix |
|---|---|---|
| `num_lmcache_cached_tokens` absent or 0 on the peer, but TTFT dropped | client used `stream=True` | stats come back only on non-streaming responses; use `basic_test.py` |
| cached tokens 0, `l2_load_completed_requests_total{l2_name="p2p"}` flat on the reader | the prompt differs (chat template, system prompt, `enable_thinking`), the home node had not finished storing (`--gap`), prefix < one chunk (256 tokens) | send byte-identical prompts, wait a few seconds after the cold request, use prompts > 1k tokens |
| hit on the home node is a `lookup_hit_l1` but the peer shows nothing | the peer's lookup never reached P2P: `lmcache_mp_l2_prefetch_lookup_requests_total` flat | peer adapter missing: `Added L2 adapter 0 (p2p)` absent in the reader's log; check `p2p_peers` in `/status` |
| works A->B but not B->A | B's `--p2p-advertise-url` / port not reachable from A (firewall, wrong ip) | `rail_check.sh` from B; fix `NODE_IP`/`P2P_PORT` on B |

## Hit ratio below the chunk-aligned maximum, zero errors

| reason label on `lmcache_mp_l2_prefetch_failure_chunks_total` | meaning | fix |
|---|---|---|
| `l1_oom` | reader could not reserve temporary L1 for the load (`Failed to batched allocate N memory blocks` warnings) | `L1_GB >= resident KV + concurrency x prompt KV`; 20 GB -> 80 GB removed it at 32 in flight |
| `l1_contended` | the same keys were being loaded by another request on this node; the second recomputes | expected at a few % with duplicate prompts in flight; reduce duplicate concurrency or accept |
| misses only right after the warm phase | home node still storing | longer `--gap` |

## Slow transfers

| symptom | confirm | fix |
|---|---|---|
| peer TTFT barely below cold; `rail_check.sh` says `traffic went over TCP` | no RDMA devices, or UCX could not use them (no reachable GID/route, `UCX_TLS` without `rc`) | `check_rdma.sh`: devices Active, routes to the peer's GIDs; let UCX default or set `UCX_TLS=rc,sm,self,cuda_copy` |
| `rail_check.sh`: 1 of N devices | UCX defaults to one device | set the multi-rail variables from `lmcache-p2p-launch/references/ucx-multirail.md` (gen_env.sh does) |
| `rail_check.sh`: exactly half the devices, all on one NUMA node | `UCX_MAX_RMA_RAILS` too small: rails count paths and LAG bonds have two | `UCX_MAX_RMA_RAILS = devices x 2` (16 for 8 bonds); confirm with `ucx_lanes.sh` (16 lanes at ~7%) |
| rails are right but TTFT does not move | workload not fabric-bound: `vllm_wait` high, L1-to-GPU copies, 15k-token prompts at 16 in flight | compare against the local-L1-hit baseline (`bench_ldqa.sh --with-local-baseline`); tune concurrency / engine, not UCX |
| `lmcache bench engine --lmcache-url` fails: `Model '...' not found on LMCache server. Available: /path/snapshots/...` | server reports the snapshot path | pass `--tokens-per-gb-kvcache` (`tokens_per_gb.py`) and drop `--lmcache-url` |

## Operational

- Restarting the LMCache server requires restarting its vLLM (the connector does not reconnect).
- `stop_node.sh` kills whatever still owns the ports; vLLM engine-core children survive SIGTERM.
- A `$P2P_HOME/logs/*.log` is truncated by each `start_node.sh`; copy it first if you need history.
- `/metrics/reset` (POST on the HTTP port) zeroes counters between experiments.
