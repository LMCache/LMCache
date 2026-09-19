# Dynamo + LMCache MP integration

Recipes for running [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) vLLM
serving with KV cache offloaded to an **LMCache multiprocess (mp) server**, in
both **aggregated** and **disaggregated** modes.

In mp mode LMCache runs as an out-of-process cache engine. vLLM workers attach
to it through the `LMCacheMPConnector` and share KV tensors over CUDA IPC, so
`sharedMemory` is disabled (LMCache owns `/dev/shm`) and `hostIPC` is enabled.

## Layout

| Path | What it is |
|------|------------|
| [`local/docker-compose.yml`](local/docker-compose.yml) | Defines NATS, etcd, and the Dynamo runtime container. |
| [`local/nats-server.conf`](local/nats-server.conf) | NATS configuration mounted by Docker Compose. |
| [`local/agg_lmcache_mp.sh`](local/agg_lmcache_mp.sh) | Local single-node launch script, aggregated (1 GPU). |
| [`local/disagg_lmcache_mp.sh`](local/disagg_lmcache_mp.sh) | Local single-node launch script, disaggregated (2 GPUs). |
| [`local/serve.sh`](local/serve.sh) | Container entry point that waits for dependencies and starts LMCache and Dynamo. |
| [`kubernetes/lmcache_engine.yaml`](kubernetes/lmcache_engine.yaml) | `LMCacheEngine` CR for the shared MP server. Apply **before** the workers. |
| [`kubernetes/agg_lmcache_mp.yaml`](kubernetes/agg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, aggregated (single worker). |
| [`kubernetes/disagg_lmcache_mp.yaml`](kubernetes/disagg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, disaggregated (prefill + decode workers). |

## Local

Use a Linux host with NVIDIA GPUs, Docker Compose 2.30 or newer, and the
NVIDIA Container Toolkit installed. From the root of the LMCache repository,
run one script.

For aggregated serving on one GPU:

```bash
./examples/dynamo_integration/local/agg_lmcache_mp.sh
```

For separate prefill and decode workers on two GPUs in the same node, stop
the aggregated deployment first, then run:

```bash
./examples/dynamo_integration/local/disagg_lmcache_mp.sh
```

Each script starts NATS and etcd, creates a GPU-enabled Dynamo container,
and launches LMCache, the Dynamo frontend, and the vLLM workers. Press
`Ctrl+C` to stop the whole demo, including NATS and etcd.

Both modes serve `Qwen/Qwen3-0.6B` with 16 GiB of CPU cache. They use
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2`, which includes LMCache 0.5.2.
You can find the latest image tags on
[NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/ai-dynamo/containers/vllm-runtime/-/tags).

To start each process yourself, follow the
[manual startup steps](../../docs/source/production/dynamo_coordination.rst#start-processes-manually)
instead of running a launch script. The
[Dynamo integration guide](../../docs/source/production/dynamo_coordination.rst#check-the-deployment)
also includes inference and cache-hit checks.

## Kubernetes

### LMCacheEngine

[`kubernetes/lmcache_engine.yaml`](kubernetes/lmcache_engine.yaml) declares the MP
server that both the aggregated and disaggregated workers attach to. The
LMCache operator reconciles the CR into:

- a per-node **manager DaemonSet** pod (the LMCache server that holds the CPU
  KV pool), and
- a **ConfigMap** named `<metadata.name>-connection` (i.e. `lmcache-mp-connection`),
  which the vLLM workers mount at `/etc/lmcache` and read as
  `--kv-transfer-config`.

It must live in the **same namespace** as the workers (`default`) so the
connection ConfigMap is created where the workers can mount it.

**Version pinning**: the server's bundled lmcache must be wire-compatible with
the worker's. The worker image `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`
ships lmcache `0.4.4` (vLLM `0.20.1`), and the recipe pairs it with the
guide-validated server build `nightly-2026-04-25` (lmcache `0.4.5.dev31`, a
pre-stable build wire-compatible with that worker). Replace `my-tag` in each
manifest accordingly.

### Deploy

The `kubernetes/` manifests are applied with `kubectl` against a cluster that
already has the Dynamo platform and the LMCache operator installed. Apply the
`LMCacheEngine` first, then one of the worker manifests. Run these commands
from the root of the LMCache repository:

```bash
kubectl apply -n default -f examples/dynamo_integration/kubernetes/lmcache_engine.yaml
kubectl apply -n default -f examples/dynamo_integration/kubernetes/disagg_lmcache_mp.yaml   # or agg_lmcache_mp.yaml
```

See the [Dynamo integration guide](../../docs/source/production/dynamo_coordination.rst)
for cluster prerequisites, manifest settings, and verification commands.

> The Kubernetes manifests pin the worker image to `my-tag` and the
> `LMCacheEngine` image separately; keep the two wire-compatible (see the
> version pairing above).
