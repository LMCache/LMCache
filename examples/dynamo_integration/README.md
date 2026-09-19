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
| [`local/docker-compose.yml`](local/docker-compose.yml) | Starts NATS and etcd on the host. |
| [`local/nats-server.conf`](local/nats-server.conf) | NATS configuration mounted by Docker Compose. |
| [`local/agg_lmcache_mp.sh`](local/agg_lmcache_mp.sh) | Local single-node launch script, aggregated (1 GPU). |
| [`local/disagg_lmcache_mp.sh`](local/disagg_lmcache_mp.sh) | Local single-node launch script, disaggregated (2 GPUs). |
| [`kubernetes/lmcache_engine.yaml`](kubernetes/lmcache_engine.yaml) | `LMCacheEngine` CR for the shared MP server. Apply **before** the workers. |
| [`kubernetes/agg_lmcache_mp.yaml`](kubernetes/agg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, aggregated (single worker). |
| [`kubernetes/disagg_lmcache_mp.yaml`](kubernetes/disagg_lmcache_mp.yaml) | Kubernetes `DynamoGraphDeployment`, disaggregated (prefill + decode workers). |

## Local

Start NATS and etcd on the host. From the root of the LMCache repository, run:

```bash
docker compose -f examples/dynamo_integration/local/docker-compose.yml up -d
```

From the same directory on the host, start the Dynamo container. Replace
`my-tag` with the tag of a `vllm-runtime` image that includes LMCache:

```bash
docker run --rm -it --name dynamo-lmcache \
    --gpus all --network host --ipc host \
    --ulimit memlock=-1 \
    -v "$PWD:/workspace/LMCache:ro" \
    nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag bash
```

The container mounts your LMCache repository at `/workspace/LMCache` and
includes Dynamo's launch helpers under `/workspace/examples`. In the
container shell, copy the scripts into Dynamo's launch directory:

```bash
cp /workspace/LMCache/examples/dynamo_integration/local/*_lmcache_mp.sh \
    /workspace/examples/backends/vllm/launch/
cd /workspace/examples/backends/vllm
```

For aggregated serving on one GPU:

```bash
LMCACHE_L1_SIZE_GB=16 ./launch/agg_lmcache_mp.sh
```

For separate prefill and decode workers on two GPUs in the same node, stop
the aggregated deployment first, then run:

```bash
LMCACHE_L1_SIZE_GB=16 ./launch/disagg_lmcache_mp.sh
```

Each script starts LMCache, the Dynamo frontend, and the vLLM workers. Press
`Ctrl+C` to stop those processes. NATS and etcd run separately through Docker
Compose and remain running.

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
