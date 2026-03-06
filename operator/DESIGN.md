# LMCache Kubernetes Operator Design

## Overview

LMCache multiprocess mode runs as a separate server process that vLLM instances connect to for KV cache offloading. Today this is deployed manually via raw K8s manifests (a DaemonSet for LMCache + a Deployment for vLLM). A K8s operator would automate lifecycle management, enforce best practices (`hostIPC`, resource sizing), and expose connection info for vLLM discovery.

### Current Pain Points the Operator Solves

- Manual `hostIPC` setup and node-local service discovery
- No automated connection info propagation to vLLM
- No Prometheus ServiceMonitor integration
- No validation of configuration parameters

### How the Operator Addresses These

The operator introduces a single CRD (`LMCacheEngine`) that declaratively captures the full LMCache server configuration. The controller reconciles this CR into the underlying K8s resources (DaemonSet, Service, ConfigMap, ServiceMonitor), automatically injecting the required pod-level settings that are easy to forget in hand-written manifests.

**Auto-injected pod settings eliminate manual boilerplate.** The controller always sets `hostIPC: true` and `--host 0.0.0.0` — settings that the current `lmcache-daemonset.yaml` requires users to specify by hand. Getting any of these wrong (e.g., forgetting `hostIPC`) causes silent connectivity failures or CUDA IPC errors that are hard to debug.

**A node-local Service with connection ConfigMap provides a stable discovery contract for vLLM.** The operator creates a ClusterIP Service with `internalTrafficPolicy=Local`, which ensures kube-proxy routes traffic only to the LMCache pod on the same node. A ConfigMap (`<name>-connection`) contains the `kv-transfer-config` JSON pointing to this service's cluster DNS name. vLLM deployments simply mount the ConfigMap — no downward API or shell substitution needed. When the LMCache CR changes, the ConfigMap updates automatically and vLLM pods pick up the new values on restart.

**Prometheus integration is declarative.** When `prometheus.serviceMonitor.enabled` is set, the operator creates a ServiceMonitor CR that the Prometheus Operator discovers automatically. Without the operator, users must manually create ServiceMonitor resources and keep labels/ports in sync with the DaemonSet.

**CRD validation catches misconfigurations at apply time.** OpenAPI schema validation and a validating webhook enforce constraints (e.g., `l1.sizeGB > 0`, `eviction.triggerWatermark` in `(0.0, 1.0]`) before any pods are created. Today, invalid CLI flags only surface as runtime crashes inside the container.

**Resource sizing is auto-computed from the L1 cache size.** The operator derives `memoryRequest` (`l1.sizeGB + 5 GiB`) and `memoryLimit` (`1.5x request`) automatically, eliminating the mental math that currently leads to either OOM kills (under-provisioned) or wasted node capacity (over-provisioned). Users can override with explicit values.

---

## API Group & CRD

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
```

- **Group** `lmcache.ai`
- **v1alpha1** — alpha maturity; shape will evolve as L2 backends stabilize
- **LMCacheEngine** — represents the per-node cache engine. Future CRDs can
cover other concerns (e.g., `LMCacheKeyManager` for global key management,
`LMCacheMonitor` for engine state monitoring)

---

## CRD Spec (Complete)

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: string
  namespace: string
spec:
  # -- Container image --
  image:
    repository: string        # default: "lmcache/standalone"
    tag: string               # default: "nightly"
    pullPolicy: string        # default: IfNotPresent
  imagePullSecrets: []LocalObjectReference

  # -- Server config (maps to server.py argparse) --
  server:
    port: int                 # default: 5555
    chunkSize: int            # default: 256 tokens
    maxWorkers: int           # default: 1
    hashAlgorithm: string     # default: blake3 (builtin | sha256_cbor | blake3)

  # -- L1 cache (maps to L1MemoryManagerConfig + L1ManagerConfig) --
  # Internal tuning knobs (useLazy, alignBytes, writeTTLSeconds,
  # readTTLSeconds) use server defaults and can be overridden via the
  # env escape hatch if needed.
  l1:
    sizeGB: float             # REQUIRED

  # -- Eviction (maps to EvictionConfig) --
  eviction:
    policy: string            # default: "LRU" (only supported value)
    triggerWatermark: float   # default: 0.8 (range 0.0-1.0)
    evictionRatio: float      # default: 0.2 (range 0.0-1.0)

  # -- Monitoring (maps to PrometheusConfig from mp_observability/config.py) --
  # Note: the CRD uses `enabled: true` by default; the CLI equivalent is
  # the absence of the `--disable-prometheus` flag.
  prometheus:
    enabled: bool             # default: true  (CLI: omit --disable-prometheus)
    port: int                 # default: 9090
    serviceMonitor:
      enabled: bool           # default: false
      interval: string        # default: "30s"
      labels: map[string]string
  # ServiceMonitor is a CRD from the Prometheus Operator (kube-prometheus-stack).
  # When enabled, the operator creates a ServiceMonitor resource that tells
  # Prometheus to automatically discover and scrape LMCache metrics endpoints.
  # Without it, you'd need to manually configure Prometheus scrape targets.
  # If you don't use the Prometheus Operator, leave serviceMonitor.enabled=false
  # and use the pod annotations (prometheus.io/scrape, prometheus.io/port) instead.

  # -- L2 storage backends (extensible, mirrors l2_adapters registry) --
  # Note: the CLI expects a flat JSON with "type" as a peer key
  # (e.g. --l2-adapter '{"type":"disk","path":"/data"}').
  # The operator must merge {type} + config into a flat dict when
  # serializing to container args.
  l2Backends:
    - type: string            # adapter type name (mock, disk, redis, s3, p2p)
      config: map[string]any  # type-specific config as free-form map

  # -- Resources (auto-computed, no user input needed) --
  # The operator derives resource requests/limits from l1.sizeGB:
  #   memoryRequest = ceil(l1.sizeGB + 5) Gi
  #   memoryLimit   = ceil(memoryRequest * 1.5) Gi
  #   cpuRequest    = "4"
  # To override, use the resourceOverrides escape hatch below.
  resourceOverrides: ResourceRequirements  # optional, raw K8s resources override

  # -- Logging --
  logLevel: string            # default: INFO (DEBUG|INFO|WARNING|ERROR)

  # -- Scheduling --
  nodeSelector: map[string]string
  affinity: Affinity
  tolerations: []Toleration
  # nodeSelector determines which nodes get an LMCache instance.
  # Use nodeSelector: {nvidia.com/gpu.present: "true"} to target all GPU
  # nodes. When new GPU nodes join the cluster, the DaemonSet controller
  # automatically schedules an LMCache pod on them.

  # -- Overrides --
  env: []EnvVar               # additional environment variables
  volumes: []Volume           # additional volumes (e.g. for L2 disk backend)
  volumeMounts: []VolumeMount
  podAnnotations: map[string]string
  podLabels: map[string]string
  serviceAccountName: string
  priorityClassName: string

  # -- Extra CLI flags --
  extraArgs: []string         # appended last, can override any auto-generated flag
```

---

## Auto-managed Pod Settings (not in CRD spec)

The operator always injects these into the pod spec:

- **`hostIPC: true`** — **required for CUDA IPC between LMCache and vLLM.** LMCache uses `CudaIPCWrapper` which calls PyTorch's `_share_cuda_()` to get a GPU driver-level IPC handle. The handle is serialized and sent over ZMQ TCP. The receiving process reconstructs the tensor via `cudaIpcOpenMemHandle` at the driver level. This call requires both processes to share the same IPC namespace — without `hostIPC: true`, `cudaIpcOpenMemHandle` fails with `cudaErrorMapBufferObjectFailed`. **Both the LMCache pods and vLLM pods must have `hostIPC: true`.**
- **`--host 0.0.0.0`** — always passed as a container arg. The server defaults to `--host localhost` which only binds to loopback; the server must bind to all interfaces so the node-local Service can route traffic to it.
- **No `hostNetwork`** — the operator does **not** use `hostNetwork`. Instead, it creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy ensures that traffic to the service is routed only to the LMCache pod on the same node. This avoids occupying host ports and reduces the privileged surface area.
- **No `/dev/shm` emptyDir mount** — the operator intentionally does *not* mount an emptyDir at `/dev/shm`. With `hostIPC: true`, the container already sees the host's `/dev/shm`. Mounting an emptyDir would shadow the host's `/dev/shm` with a private tmpfs, breaking CUDA IPC (`cudaIpcOpenMemHandle` fails because IPC handles written by one pod are invisible to others). If your workload needs a larger `/dev/shm` for non-IPC purposes, add it via `spec.volumes` / `spec.volumeMounts`.

> **Security implications of `hostIPC`:** This setting exposes the host's IPC namespace (System V IPC, POSIX message queues) to the container. This means any process in the container can interact with IPC resources from other processes on the same host. Only deploy in trusted environments. Clusters using Pod Security Standards must allow the `privileged` profile for the LMCache namespace — the `baseline` and `restricted` profiles reject `hostIPC`.

---

## Validation Rules

| Field | Rule |
|---|---|
| `l1.sizeGB` | Required, must be `> 0` |
| `eviction.policy` | Must be `"LRU"` (if set) |
| `eviction.triggerWatermark` | Must be in `(0.0, 1.0]` |
| `eviction.evictionRatio` | Must be in `(0.0, 1.0]` |
| `server.port` | Must be in `[1024, 65535]` |

---

## CRD Status

```yaml
status:
  phase: Pending | Running | Degraded | Failed
  observedGeneration: int64

  desiredInstances: int
  readyInstances: int

  # Per-node connection info (for kubectl visibility)
  endpoints:
    - nodeName: string
      hostIP: string
      podName: string
      port: int
      metricsPort: int
      ready: bool

  # Standard conditions
  conditions:
    - type: Available          # at least one instance ready
    - type: AllInstancesReady  # all desired instances ready
    - type: ConfigValid        # spec validation passed
  # Connection ConfigMap is always named <metadata.name>-connection;
  # no need to store the ref in status.
```

---

## Examples

### Minimal Deployment

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: my-cache
spec:
  l1:
    sizeGB: 60
```

This deploys a DaemonSet with 60GB L1 cache, LRU eviction, blake3 hashing, port 5555, auto-computed 65Gi memory request / 98Gi limit, Prometheus on 9090, and a connection ConfigMap for vLLM. The operator auto-injects `hostIPC` and `--host 0.0.0.0`, and creates a node-local Service for vLLM discovery.

### Production Deployment (all GPU nodes)

```yaml
apiVersion: lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: production-cache
  namespace: llm-serving
spec:
  # Target all GPU nodes — new GPU nodes automatically get an LMCache pod
  nodeSelector:
    nvidia.com/gpu.present: "true"

  image:
    repository: lmcache/standalone
    tag: v0.1.0
    pullPolicy: IfNotPresent

  server:
    port: 6555
    chunkSize: 256
    maxWorkers: 4
    hashAlgorithm: blake3

  l1:
    sizeGB: 60

  eviction:
    triggerWatermark: 0.8
    evictionRatio: 0.2

  prometheus:
    enabled: true
    port: 9090
    serviceMonitor:
      enabled: true
      labels:
        release: kube-prometheus-stack

  logLevel: INFO
  podAnnotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
  priorityClassName: system-node-critical
```

---

## Resources Created by the Operator

For `LMCacheEngine` named `production-cache`:

| Resource | Name | Purpose |
|---|---|---|
| DaemonSet | `production-cache` | Runs LMCache server pods |
| Service (ClusterIP, `internalTrafficPolicy=Local`) | `production-cache` | Node-local service discovery for vLLM |
| ConfigMap | `production-cache-connection` | kv-transfer-config JSON pointing to the lookup Service |
| Service (headless) | `production-cache-metrics` | Prometheus scrape target |
| ServiceMonitor (optional) | `production-cache` | Prometheus Operator integration |

### ConfigMap Content (for vLLM discovery)

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://<name>.<namespace>.svc.cluster.local",
    "lmcache.mp.port": "<spec.server.port, default 5555>"
  }
}
```

The ConfigMap uses the lookup Service's cluster DNS name. Because the Service has `internalTrafficPolicy=Local`, kube-proxy routes traffic only to the LMCache pod on the same node as the vLLM pod. vLLM pods mount this ConfigMap and pass the JSON to `--kv-transfer-config` — no downward API or shell variable substitution required. vLLM pods should also set `PYTHONHASHSEED=0` for deterministic token hashing.

---

## Auto-injected Pod Template Details

In addition to the auto-managed pod settings above (`hostIPC`, `--host 0.0.0.0`),
the operator injects:

- Container command: `/opt/venv/bin/python3 -m lmcache.v1.multiprocess.server`
- Container args: serialized from spec fields
- Env: `LMCACHE_LOG_LEVEL` from `spec.logLevel`
- Probes:
  - **Startup:** TCP on server port, `initialDelay=5s`, `period=5s`, `failureThreshold=30`
  - **Liveness:** TCP on server port, `period=10s`
  - **Readiness:** TCP on server port, `period=5s`

---

## Reconciliation Logic

```
OnEvent(LMCacheEngine create/update/delete):

1. VALIDATE spec -> set condition ConfigValid
2. COMPUTE derived values (unless overridden):
   - memoryRequest = ceil(l1.sizeGB + 5) Gi
   - memoryLimit = ceil(memoryRequest * 1.5) Gi
   - containerArgs from all spec fields
3. RECONCILE DaemonSet (CreateOrUpdate, ownerRef)
   - Always inject: hostIPC, --host 0.0.0.0
4. RECONCILE node-local lookup Service (internalTrafficPolicy=Local)
5. RECONCILE headless Service for metrics
6. RECONCILE connection ConfigMap
7. RECONCILE ServiceMonitor (if enabled)
8. UPDATE status:
   - Query workload for ready/desired counts
   - Enumerate pods -> build endpoints list
   - Set phase: Running | Degraded | Pending | Failed
   - Set conditions, observedGeneration
```

**Secondary watches:** DaemonSet, Pods (readiness changes → update endpoints), Nodes (new GPU node → DaemonSet auto-schedules)

**Finalizer** `lmcache.ai/cleanup`: on CR deletion, cascading delete of owned resources.

---

## Future Extensibility

- **L2 backends:** `l2Backends` uses free-form config maps matching the codebase's adapter registry pattern. New adapter types (disk, Redis, S3, P2P) need no CRD schema changes. The operator can create PVCs or inject Secret env vars based on type.
- **Blend mode:** Future `LMCacheEngine` field `blend.enabled` to switch entrypoint from `server.py` to `blend_server.py` (deferred from v1alpha1).
- **Update strategy:** Future `spec.updateStrategy` field for `RollingUpdate`/`OnDelete` control on the DaemonSet.
- **Additional CRDs:** `LMCacheKeyManager` (global key management), `LMCacheMonitor` (engine state monitoring), `LMCacheFederation` (cross-cluster P2P topology).

---

## Key Source Files Referenced

| File | What it defines |
|---|---|
| `lmcache/v1/distributed/config.py` | `L1MemoryManagerConfig`, `L1ManagerConfig`, `EvictionConfig`, `StorageManagerConfig`, argparse |
| `lmcache/v1/mp_observability/config.py` | `PrometheusConfig`, `add_prometheus_args`, `parse_args_to_prometheus_config` |
| `lmcache/v1/multiprocess/server.py` | `MPCacheEngine`, server CLI entry point, argparse (lines 629–653) |
| `lmcache/v1/multiprocess/http_server.py` | HTTP server with `/api/healthcheck` endpoint (FastAPI + ZMQ) |
| `lmcache/v1/distributed/l2_adapters/config.py` | L2 adapter registry pattern, `L2AdapterConfigBase`, `L2AdaptersConfig` |
| `examples/multi_process/lmcache-daemonset.yaml` | Reference DaemonSet manifest |
| `examples/multi_process/vllm-deployment.yaml` | Reference vLLM deployment with kv-transfer-config |
