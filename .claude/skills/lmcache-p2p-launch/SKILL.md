---
name: lmcache-p2p-launch
description: Set up and launch LMCache multiprocess P2P KV sharing (coordinator + one LMCache server and one vLLM per node, nixl/UCX over RDMA) on one or more nodes and verify the peers see each other. Use when asked to install, configure, start, restart or verify a P2P deployment, or to get all RDMA NICs used.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "[--single-node | --coordinator-ip IP --peer-ip IP --peer-ssh HOST] [--model ID] [--gpu N] [--l1-gb N]"
---

# LMCache P2P: set up, launch, verify

P2P lets every node's LMCache server read KV from a peer's L1 over RDMA instead of
recomputing it. One deployment = one `lmcache coordinator` + per node one `lmcache server`
(`--p2p-advertise-url`) + one `vllm serve` with `LMCacheMPConnector`. The reference is
`docs/source/mp/p2p.rst` (flags, single-node recipe, how it works) and `examples/p2p/`;
read those first for anything this file does not cover.

Every script here sources one `env.sh` per LMCache instance (written by `gen_env.sh`, found
through `$P2P_ENV` or `$P2P_HOME/env.sh`, default `~/lmcache-p2p/env.sh`). The
`lmcache-p2p-test` and `lmcache-p2p-diagnose` skills read the same file, so write it once.

Verified with LMCache dev `45a0bb420` (0.5.7rc3), vLLM 0.32.0, torch 2.13.0+cu130,
nixl 1.5.2 / UCX 1.23, two H200 nodes with 8 RoCE LAG bonds each, Qwen/Qwen3-8B.

## Steps

1. **Probe each node**: `scripts/probe_env.sh --python <venv>/bin/python --model <id>`.
   It reports GPUs, RDMA devices with NUMA placement, tool availability, the Python stack
   (torch / vllm / lmcache / nixl / `lmcache.cuda_ops`), free ports and whether the model is
   in the HF cache. Fix every `WARN` before going on. No RDMA devices means P2P will run
   over TCP: fine for a functional test, not for performance.
2. **Install if needed**: `scripts/setup_venv.sh --dir <dir> [--lmcache pypi|source --ref dev]`
   builds a fresh uv venv (vLLM from PyPI, LMCache release wheels or an editable CUDA build,
   nixl). Skip it when the probe shows a working stack. Models must be present on every node
   (`HF_HUB_OFFLINE=1` is set; copy the HF cache dir with `tar | ssh tar` when `rsync` is
   missing). Disk: the venv is about 12 GB.
3. **Write env.sh on every node** with `scripts/gen_env.sh`:
   - multi-node: `--coordinator-ip <ip of the node that runs the coordinator> --peer-ip <other node>
     --peer-ssh <ssh alias of the other node> --model <id> --gpu <free gpu> --l1-gb 80`
   - single node (two instances, two GPUs): `--single-node a --gpu 0` and `--single-node b --gpu 1`
     (presets ports 6700/7700/8700/8300 and 6701/7701/8701/8301, RDMA off).
   It picks the UCX multi-rail settings from the RDMA devices it finds; see
   `references/ucx-multirail.md` before changing them. Ports and flags: `references/flags-and-ports.md`.
4. **Start the coordinator on one node, then every node**:
   `scripts/start_node.sh --with-coordinator` on the coordinator node, `scripts/start_node.sh`
   elsewhere (over ssh: `P2P_ENV=... <skill>/scripts/start_node.sh`). Logs and pids go to
   `$P2P_HOME/logs/`. The LMCache server must stay alive for 5 s or the script prints its log.
5. **Wait and verify**: `scripts/wait_ready.sh --peers <n-1>` on each node blocks until vLLM
   answers and `/status` shows `p2p_state=registered` with that many peers; `scripts/status.sh`
   prints the coordinator fleet and both nodes' P2P state. vLLM takes 1-3 min (torch.compile).
6. **Prove a transfer** with the test skill's `basic_test.py` (cold prompt on one node, same
   prompt on the other, `num_lmcache_cached_tokens` > 0 on the second). Then hand over to
   `lmcache-p2p-test` for stress and monitoring, or `lmcache-p2p-diagnose` if anything is off.

`scripts/stop_node.sh [--with-coordinator]` stops what `start_node.sh` started (it also kills
vLLM engine-core children that hold the ports). Changing `L1_GB`, UCX variables or any server
flag needs a restart of both the LMCache server and its vLLM (the connector does not reconnect).

## Sizing and settings that matter

- **L1 size**: every in-flight P2P load needs prompt-sized temporary L1 on the reading node on
  top of the resident set (default retention policy keeps nothing from L2/P2P loads). Rule of
  thumb: `L1_GB >= resident_KV + concurrency x prompt_KV`; 20 GB with 32 in-flight 7k-token
  prompts of an 8B model produced `l1_oom` fallbacks, 80 GB did not.
- **`--http-host 0.0.0.0`** (set by `launch_lmcache.sh`): the default binds 127.0.0.1 and hides
  `/status` and `/metrics` from other hosts. `--prometheus-port` is ignored by the MP server.
- **`--l1-align-bytes 65536`** (set): bigger, aligned RDMA reads.
- **Prefix caching is disabled in vLLM** by `launch_vllm.sh` so hits are attributable to
  LMCache; re-enable it for production runs.
- **RDMA fabric**: `UCX_TLS` must include `cuda_copy` on GPU hosts with nixl >= 1.5 or the
  server dies at start with `NIXL_ERR_BACKEND`; `UCX_MAX_RMA_RAILS` counts paths, two per
  LAG bond. Details and verification: `references/ucx-multirail.md`.
- On a shared box pick free GPUs and ports (`probe_env.sh`) and never kill processes you
  did not start.

## Known noise

`Error while registering kernel ... flexio ... Illegal application argument: NULL` lines at
server start come from DOCA/flexio on some NICs and are harmless.

## Eval prompts

- "Bring up LMCache P2P between host A and host B with Qwen3-8B" -> probe both, write env.sh on
  both, start coordinator + nodes, `wait_ready.sh` on both, report `status.sh` output.
- "Why does only one NIC carry P2P traffic?" -> `references/ucx-multirail.md`, then the
  diagnose skill's `rail_check.sh`.
- "Run two P2P nodes on this one machine" -> `gen_env.sh --single-node a|b`, both GPUs.
