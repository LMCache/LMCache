# LMCache Kubernetes Operator Design

## Overview

LMCache multiprocess mode runs as a separate server process that vLLM instances connect to for KV cache offloading. Today this is deployed manually via raw K8s manifests (a DaemonSet for LMCache + a Deployment for vLLM). A K8s operator would automate lifecycle management, enforce best practices (`/dev/shm` sharing for CUDA IPC, resource sizing), and expose connection info for vLLM discovery.

### Current Pain Points the Operator Solves

- Manual `/dev/shm`-sharing setup and node-local service discovery
- No automated connection info propagation to vLLM
- No Prometheus ServiceMonitor integration
- No validation of configuration parameters

### How the Operator Addresses These

The operator introduces a single CRD (`LMCacheEngine`) that declaratively captures the full LMCache server configuration. The controller reconciles this CR into the underlying K8s resources (DaemonSet, Service, ConfigMap, ServiceMonitor), automatically injecting the required pod-level settings that are easy to forget in hand-written manifests.

**Auto-injected pod settings eliminate manual boilerplate.** The controller always mounts the host's `/dev/shm` (hostPath) and sets `--host 0.0.0.0` — settings that the current `lmcache-daemonset.yaml` requires users to specify by hand. Getting any of these wrong (e.g., forgetting the `/dev/shm` mount) causes silent connectivity failures or CUDA IPC errors that are hard to debug.

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

  # -- L2 storage backend (single adapter) --
  # Currently only one L2 adapter is supported at a time.
  # LMCache MP mode supports multiple adapters, but this is not yet
  # fully tested. Once validated, the operator will support multiple.
  # Exactly one of resp or raw must be set.
  l2Backend:
    # Option A: Native RESP (Redis/Valkey) adapter
    resp:
      host: string              # REQUIRED
      port: int                 # REQUIRED, 1-65535
      numWorkers: int           # default: 8
      maxCapacityGB: float      # default: 0 (disabled)
      authSecretRef:            # optional, Secret with "username"/"password" keys
        name: string
    # Option B: Raw escape hatch for other adapter types
    raw:
      type: string              # adapter type name (nixl_store, fs, mock, raw_block, etc.)
      config: map[string]any    # type-specific config as free-form map
    # Optional: serde transform on KV bytes to/from the L2 adapter. Renders
    # a "serde" sub-dict into the --l2-adapter JSON; applies to whichever
    # adapter is configured. Exactly one serde type must be set (only aesgcm
    # today; other server-side serdes like fp8/turboquant can gain typed
    # fields here later, or be set via raw.config.serde).
    serde:
      # aesgcm: at-rest encryption keyed per cache_salt. The referenced
      # Secret is mounted directly into the engine pods, read-only at
      # /etc/lmcache/keys/master (only the "master" data key is projected).
      # The operator never reads or writes the Secret and never generates
      # key material; provenance/rotation stay with the user. The reference
      # is same-namespace only, so CR-create permission cannot be leveraged
      # to read Secrets from other namespaces (confused deputy). See
      # docs/design/v1/distributed/serde/aesgcm.md for the threat/key models.
      aesgcm:
        masterKeySecretRef:     # REQUIRED, user-created Secret with a "master" data key,
          name: string          # in the engine's namespace (same-namespace only)
        keyProvider: string     # default: hkdf (only implemented provider)
        aesBits: int            # default: 128 (or 256)

  # -- Connection-injection webhook defaults (optional) --
  # Read by the LMCache mutating webhook for pods bound to this engine. When
  # unset, the webhook wires only the connection (--kv-transfer-config, the
  # /dev/shm sharing, PYTHONHASHSEED). Set injection.payloadImage to ALSO stage an lmcache code
  # tree into the vLLM container (emptyDir + init container + readOnly mount +
  # PYTHONPATH=/lmcache-payload), reusing the CacheBlend payload-staging
  # mechanism so the vLLM client and the engine server run one lmcache build.
  injection:
    payloadImage:               # SEPARATE image: ships lmcache under /payload
      repository: string        # REQUIRED (no valid default)
      tag: string               # default: latest
      pullPolicy: string        # default: IfNotPresent
    imagePullSecrets: []LocalObjectReference  # for a private payload image
    targetContainer: string     # default: first container

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

  # -- Security --
  isolatedIPC: bool           # unset = auto: true for gpuVendor nvidia, false
                              # for amd. Runs the server with --isolated-ipc:
                              # KV caches and completion events travel over
                              # driver-level CUDA IPC, needing neither hostIPC
                              # nor any /dev/shm sharing. Overrides hostIPC
                              # when it resolves to true. Set to false for
                              # LMCache images without --isolated-ipc or for
                              # CUDA-VMM-backed allocators (vLLM sleep mode /
                              # expandable segments), which driver-level CUDA
                              # IPC cannot share.
  hostIPC: bool               # default: false; only applies when isolatedIPC
                              # resolves to false: run the pod in the host IPC
                              # namespace instead of mounting the host's
                              # /dev/shm via hostPath. Needed only where the
                              # hostPath mount is unavailable or for AMD (HIP
                              # IPC is unverified without it) — see
                              # "Auto-managed Pod Settings" below.
  privileged: bool            # default: false; run the engine container in
                              # privileged mode. Needed only on clusters where
                              # the engine cannot see the node GPUs otherwise
                              # (see "Auto-managed Pod Settings" below).

  # -- Extra CLI flags --
  extraArgs: []string         # appended last, can override any auto-generated flag
```

`podLabels` adds labels to LMCacheEngine, CacheBlendEngine, and
LMCacheCoordinator pods. The operator's selector labels
(`app.kubernetes.io/name`, `app.kubernetes.io/instance`, and
`app.kubernetes.io/managed-by`) take precedence over conflicting entries.
On workload updates, existing selector labels also take precedence, keeping
the pod template compatible with the immutable selector.

---

## Auto-managed Pod Settings (not in CRD spec)

The operator always injects these into the pod spec:

- **IPC wiring (mode-dependent)** — under the default **isolated IPC** (`spec.isolatedIPC`, auto-on for nvidia) the server runs with `--isolated-ipc`, KV caches and completion events travel over driver-level CUDA IPC that rendezvouses in the kernel driver, and the engine pod gets **no** IPC wiring at all: no `hostIPC`, no `/dev/shm` volume (the injection webhook gives opted-in vLLM pods a pod-private memory-backed `/dev/shm` for their own workers). With `spec.isolatedIPC: false`, the legacy wiring applies: a **host `/dev/shm` mount (hostPath, volume `lmcache-dev-shm`)** — required for torch-storage CUDA IPC between LMCache and vLLM. Those CUDA IPC handles are opened via files the driver and PyTorch create in `/dev/shm`, so both pods must see the same `/dev/shm` tmpfs; without it, `cudaIpcOpenMemHandle` fails with `cudaErrorMapBufferObjectFailed`. A shared IPC namespace is **not** required — only the shared `/dev/shm` tmpfs is. The injection webhook mirrors the mount onto opted-in vLLM pods. `spec.hostIPC: true` restores the host-IPC-namespace mode — for clusters that block hostPath volumes, or for `gpuVendor: amd` (unverified without it). A user-supplied `/dev/shm` mount suppresses the injection in every mode.
- **`runtimeClassName: nvidia`** — uses the NVIDIA container runtime, which injects the host's NVIDIA driver libraries and device files into the container. This is required for CUDA to function inside the pod.
- **`NVIDIA_VISIBLE_DEVICES=all`** — gives the engine access to all GPUs on the node for CUDA IPC and custom data transfer kernels without claiming any via `nvidia.com/gpu` resource requests (which would make those GPUs unavailable to the serving engine). The combination of `runtimeClassName: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` lets the container see all GPUs without consuming device-plugin resources, so the serving engine (e.g., vLLM) can still request all GPUs on the node. On most clusters this is sufficient; on some, the engine cannot see the GPUs unless the pod is also privileged — set `spec.privileged: true` to run the engine container privileged (default `false`).
- **`NVIDIA_VISIBLE_DEVICES=all`** and **`NVIDIA_DRIVER_CAPABILITIES=all`** — env vars that instruct the NVIDIA container runtime to expose all GPUs and all driver capabilities to the container.
- **`--host 0.0.0.0`** — always passed as a container arg. The server defaults to `--host localhost` which only binds to loopback; the server must bind to all interfaces so the node-local Service can route traffic to it.
- **No `hostNetwork`** — the operator does **not** use `hostNetwork`. Instead, it creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy ensures that traffic to the service is routed only to the LMCache pod on the same node. This avoids occupying host ports and reduces the privileged surface area.
- **No `/dev/shm` emptyDir mount on the engine** — in legacy mode (`isolatedIPC: false`) the operator intentionally never mounts an emptyDir at `/dev/shm`: it would shadow the host's `/dev/shm` with a private tmpfs, breaking CUDA IPC (`cudaIpcOpenMemHandle` fails because IPC handles written by one pod are invisible to others). Under isolated IPC the engine pod simply has no `/dev/shm` volume; only webhook-injected vLLM pods get a private emptyDir there (their own workers need it).

> **Security implications:** Under the default isolated IPC the pods carry no host-level IPC grants at all — no hostPath `/dev/shm`, no `hostIPC` — so `baseline` Pod Security Standards can admit them (barring other spec fields). With `spec.isolatedIPC: false` the LMCache pods mount the host's `/dev/shm` (hostPath), which exposes the host's shared-memory tmpfs — a narrower grant than the whole host IPC namespace, but still host-level access. With `spec.hostIPC: true` they instead join the host IPC namespace, and with `spec.privileged: true` they additionally run privileged, granting full device access. Only deploy those modes in trusted environments; clusters using Pod Security Standards must then allow the `privileged` profile for the LMCache namespace — hostPath volumes, `hostIPC`, and `privileged` are all rejected by the `baseline` and `restricted` profiles.

---

## Validation Rules

| Field | Rule |
|---|---|
| `l1.sizeGB` | Required, must be `> 0` |
| `eviction.policy` | Must be `"LRU"` (if set) |
| `eviction.triggerWatermark` | Must be in `(0.0, 1.0]` |
| `eviction.evictionRatio` | Must be in `(0.0, 1.0]` |
| `server.port` | Must be in `[1024, 65535]` |
| `workloadAwareScheduling.workloadSelector` | Required and non-empty when `workloadAwareScheduling.enabled: true` |
| `workloadAwareScheduling.releaseDelaySeconds` | Default `300`; must be `>= 0` |

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

This deploys a DaemonSet with 60GB L1 cache, LRU eviction, blake3 hashing, port 5555, auto-computed 65Gi memory request / 98Gi limit, Prometheus on 9090, and a connection ConfigMap for vLLM. The operator auto-injects the `/dev/shm` hostPath mount and `--host 0.0.0.0`, and creates a node-local Service for vLLM discovery.

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
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://<name>.<namespace>.svc.cluster.local",
    "lmcache.mp.port": "<spec.server.port, default 5555>"
  }
}
```

The ConfigMap uses the lookup Service's cluster DNS name. Because the Service has `internalTrafficPolicy=Local`, kube-proxy routes traffic only to the LMCache pod on the same node as the vLLM pod. vLLM pods mount this ConfigMap and pass the JSON to `--kv-transfer-config` — no downward API or shell variable substitution required. The explicit `kv_connector_module_path` makes the external LMCache MP connector path load-bearing and avoids silent fallback to an older vendored builtin connector path.

---

## Auto-injected Pod Template Details

In addition to the auto-managed pod settings above (the `/dev/shm` mount, `--host 0.0.0.0`),
the operator injects:

- Container command: `/opt/venv/bin/lmcache server`
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
   - Always inject: the lmcache-dev-shm hostPath mount (or hostIPC when spec.hostIPC=true), runtimeClassName: nvidia, NVIDIA_VISIBLE_DEVICES=all, --host 0.0.0.0 (privileged only when spec.privileged=true)
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

**Deletion / cleanup**: every child resource the operator creates
(DaemonSet, lookup Service, metrics Service, connection ConfigMap,
managed RESP auth Secret, optional ServiceMonitor) carries an
`ownerReference` to the LMCacheEngine, so Kubernetes garbage
collection cascade-deletes them when the CR goes away. **No finalizer
is used.** An earlier design added a `lmcache.ai/cleanup` finalizer
to mirror that GC behavior, but it was a no-op that only created
deadlocks when the controller pod was not running (e.g. during
cluster issues or a single-step `kubectl delete -k config/default`).
The reconciler now actively strips that legacy finalizer from any CR
it sees, so migration from older operator versions is automatic.
Finalizers will return when we need to clean up state K8s GC cannot
reach (Redis L2 keys, federation deregistration, etc.).

---

## Workload-Aware Placement (`spec.workloadAwareScheduling`)

### The problem

The engine DaemonSet is created on every node matching `spec.nodeSelector`,
and co-location with the workload is enforced by a *required* pod affinity
(`requiredDuringSchedulingIgnored...`). On shared GPU pools this produces two
known warts:

1. **Parked Pending pods.** Every node in the pool without the workload gets
   a Pending engine pod — on a pool hosting several models, one per
   non-hosting node per engine. They reserve nothing, but they trip
   pod-alerting, add `desiredNumberToSchedule` noise, and look broken.
2. **Lingering engines.** The reverse leak: an engine that was scheduled onto
   a node keeps running after the workload leaves (`IgnoredDuringExecution`),
   pinning its memory — often tens of GiB of warm cache — indefinitely.

Both have the same root cause: the DaemonSet placement does not know where
the workload actually runs.

### The feature

```yaml
spec:
  workloadAwareScheduling:
    enabled: true
    workloadSelector:
      matchLabels:
        serving.kserve.io/inferenceservice: my-model
        component: predictor
    releaseDelaySeconds: 300
```

When enabled, the controller mirrors a per-engine label
(`lmcache.ai/wlas-<hash>`, hash of the CR's `namespace/name`) onto every node
currently hosting a **bound** (`spec.nodeName` set), non-terminating pod
matching `workloadSelector`, and adds that label to the DaemonSet
nodeSelector. Consequences:

- Engine pods are only created where the workload runs — no Pending pods on
  non-hosting nodes.
- When the workload leaves a node, the label is removed after
  `releaseDelaySeconds` and the DaemonSet pod there is deleted.

Semantics worth knowing:

- **`enabled: false` or absent = previous behavior, byte for byte.** The
  feature is strictly opt-in; existing CRs change nothing.
- **Bound, not Ready.** Labels apply at pod binding so the engine boots while
  the workload still loads weights; both fit comfortably inside the KV
  connector's server handshake timeout.
- **`releaseDelaySeconds` defaults to 300.** A workload restart on the same
  node (rolling update) keeps the engine and its warm KV cache; a genuine
  departure still cleans up. `0` removes the label on the next reconcile —
  always clean, but re-rolls rebuild the cache cold. The window runs from
  the **first observed absence**, so a controller outage that missed the
  actual departure still grants the full delay, and a labeled node whose
  bookkeeping predates this operator gets one full window from its first
  handling rather than an instant cleanup.
- **Failure direction.** A node that must be labeled but momentarily cannot be
  fails loudly (requeue). A node that should lose its label but cannot be is
  just retried — sticky engines are the safe side of the asymmetry, and the
  only side affinity-only placement ever had.
- **Toggling rolls engine pods once.** Changing `enabled` changes the
  DaemonSet pod template (the managed nodeSelector key appears/disappears),
  so the DaemonSet controller performs one rolling update of all engine
  pods — including the ones on nodes that keep their workload. CRs that
  never enable the feature are byte-for-byte previous behavior; whether on
  or off, the toggle itself is not cache-preserving.
- **Disable ordering is enforced, not merely sequenced.** On the disabling
  path the reconcile re-reads the live DaemonSet after updating it and
  refuses to purge node labels while its nodeSelector still references the
  managed key — so no watch race or partial reconcile can produce the
  "selector points at a label no node carries" state that would delete
  every engine pod. Cleanup therefore always lands after the operator
  DaemonSet has accepted the label-free selector.
- **Node watch.** A node losing its managed label (by anything other than
  the operator) is repaired on the next reconcile of the owning engines;
  the operator re-watches nodes for managed-prefix label changes so that
  repair happens immediately instead of on the next pod event.
- **Deletion.** Node labels are cluster-scoped, so ownership garbage
  collection cannot clean them. While enabled, the CR carries the
  `lmcache.ai/workload-placement-cleanup` finalizer; deletion first strips
  every managed label, then proceeds. Disabling the feature also strips
  residue immediately (after the label-free selector has landed, per the
  ordering note above). **This is a deliberate trade-off:** the finalizer
  blocks CR deletion while the operator is down or uninstalling
  (`kubectl delete -k config/default` hangs on WLAS-enabled CRs until the
  operator comes back and strips the labels) — the alternative is
  leaking per-engine labels on cluster nodes, which no GC can remove.
  Uninstall orders that delete the operator Deployment before the CRs
  must account for it (scale the operator to zero only after the CRs are
  gone, or strip the labels by hand).
- **Namespace scoping.** `workloadSelector` matches pods in the ENGINE's
  namespace only. A workload in another namespace (KServe components are
  commonly namespaced apart) matches nothing: no node is ever labeled
  and the engine stays Pending with no pods — the status's
  `workloadNodes` staying empty is the signal. Deploy the engine CR in
  the same namespace as its workload (a namespace selector is a possible
  follow-up if this proves limiting).
- **RBAC footprint.** The controller needs `nodes: get, list, watch, patch`.
  Cluster-scoped `nodes: patch` cannot be narrowed by RBAC to label keys
  only — the grant covers taints, unschedulable and any label, not just
  `lmcache.ai/wlas-*`. Clusters that care can add a
  ValidatingAdmissionPolicy restricting the operator's service account
  to label changes on the `lmcache.ai/wlas-` prefix. The controller also
  holds a Pod watch (cluster-wide; placement-mapped per event) and a
  Node watch filtered to managed-label changes — on clusters with many
  pods, the Pod informer carries the usual memory cost of caching every
  pod in the cluster.

- **Status.** `status.workloadLabelKey` names the managed key (for dashboards), `status.workloadNodes` the current labeled set, and `status.nodeAbsentSince` the release-window bookkeeping (see `releaseDelaySeconds` above for the first-observed-absence semantics). With DRA on, `status.nodeTierChanges` tracks pending tier changes.

### Why a node label at all

A DaemonSet's nodeSelector is the only variable that decides *where DS pods
are created*, and it only reads node labels. Required pod affinity cannot
prevent creation — only park it Pending, which is the original problem.
Mirroring the workload's presence onto the node label space is the only
bridge available without changing the engine's object form.


### Dynamic resource allocation (`spec.workloadAwareScheduling.dynamicResourceAllocation`)

Workload-aware placement gives every node the same engine spec. On nodes
where the co-located workload is small (or shares the node with another
engine's workload), a larger L1 would fit and serve more cache. DRA
derives a per-node **tier multiplier** from the GPU requests of the
matching workload pods and runs one DaemonSet per active tier:

```yaml
spec:
  workloadAwareScheduling:
    enabled: true
    workloadSelector:
      matchLabels: {app: predictor}
    dynamicResourceAllocation:
      enabled: true
      gpusPerUnit: 1        # multiplier = ceil(node GPUs / gpusPerUnit)
      maxMultiplier: 8     # clamp; see sizing note
      tierChangeDelaySeconds: 300
```

**Tier derivation.** For every node hosting ≥1 matching pod:
`multiplier = min(maxMultiplier, max(1, ceil(totalGPUs / gpusPerUnit)))`,
where `totalGPUs` sums the regular containers' `Requests["nvidia.com/gpu"`
(or `amd.com/gpu`, per `gpuVendor`) over the same filtered pod list that
placement uses (bound, non-terminating, not Failed/Succeeded — one shared
filter function, so node selection and GPU summation cannot drift). Init
containers do not count: they release their GPUs before app containers
run. Fractional requests are summed, then rounded up once.

**One node label, tier in the value.** The existing `lmcache.ai/wlas-<hash>`
label's value becomes the tier: `"1"` with DRA off (identical semantics to
the previous `"true"`), `"1".."N"` with it. Presence and tier are stamped
atomically in one label write, so a DS can never reference a tier value
before the node carries it. The fnv64a key has no collision detection —
the probability is negligible, but two colliding engines (especially DRA
engines with different tiers) would flap each other's label values.

**DaemonSet fleet.** The base DS keeps its name and its selector
(`instance=<engine>`) unchanged — `spec.selector` is immutable, so enabling
DRA on an existing engine only patches mutable template fields (the base
nodeSelector pins the label at `"1"`). Each active tier N ≥ 2 gets a
`<engine>-<N>x` DS: L1 size (`--l1-size-gb`), memory request
(`ceil(N * l1.sizeGB + 5) Gi`, limit 1.5×) and startup-probe window all
scale by N; its selector adds `lmcache.ai/dra-tier: <dsName>`; its
nodeSelector pins the placement label at `"<N>"`. All tier templates keep
the shared `instance=<engine>` label, so the node-local lookup Service and
metrics Service match every tier's pods unchanged. The base selector
therefore *overlaps* tier pods — safe, because the DS controller adopts
only orphaned pods and tier pods carry a controller ownerRef to their tier
DS. Tier DSs are matched for cleanup and status by ownerRef plus the
`dra-tier` template label, never by name pattern (a CR named `s1-2x` must
not collide with `s1`'s tier DS). Heterogeneous L1 sizes across pods of
one engine are safe: cache sharing is pull-based P2P with no capacity
negotiation.

**Tier-change hysteresis.** A node's FIRST label lands immediately
(placement — the workload needs its engine). A CHANGED tier is applied
only after it has been continuously observed for
`tierChangeDelaySeconds` (default 300, 0 = immediate): the relabel
restarts the engine pod and drops its warm L1, and workload rolling
updates transiently change the GPU sum (surge pods). Bookkeeping lives in
`status.nodeTierChanges` (pending tier + first-observed timestamp, so a
changed target restarts the window), mirroring `nodeAbsentSince`; the pending
deadline feeds the same RequeueAfter as release windows. Changing
`gpusPerUnit`/`maxMultiplier` while enabled is allowed and damped the
same way.

**Status.** `desiredInstances` sums `DesiredNumberScheduled` over all
owned DSs; `readyInstances` counts ready pods from the pod list (per-DS
`NumberReady` is ambiguous under the overlapping base selector). Both
fields mean "across the whole fleet" now, not "the one DS".

**Limitations / sizing.** `maxMultiplier` must be sized so
`ceil(maxMultiplier * l1.sizeGB + 5) Gi` fits a node's allocatable
memory: an oversized tier pod stays Pending, which surfaces via the
`TierSchedulable` status condition (False, reason `TierPodPending`) but
the node it should cover runs WITHOUT an engine until memory frees up —
the base DS already released it. The operator does not auto-demote the
tier (that would flap against the hysteresis window); size
`maxMultiplier` for the smallest node in the fleet. Every tier change is
also a deliberate node-local engine outage: the old pod dies before the
new tier pod pins its L1 (~1 GB/s pin rate — tier 8 at l1=60GB is about
eight minutes). During that window the vLLM pods on the node have no
KV connector server; serving requests fall back to vLLM's local cache /
origin recomputation until the engine is Ready. The hysteresis window
(`tierChangeDelaySeconds`) bounds how OFTEN this happens, not how long
it lasts — budget it into rollout planning. `spec.resourceOverrides` is
rejected with DRA enabled — tier DSs scale the auto-computed memory,
and overrides would silently bypass it; scaling overrides
proportionally is a possible follow-up. `gpuVendor` must be `nvidia` or
`amd`, and the engine name must leave room for the `-<N>x` suffix (the
`dra-tier` label value is capped at 63 characters; validation rejects
over-long names). Only the standard resource names `nvidia.com/gpu` /
`amd.com/gpu` count toward the tier; MIG (`nvidia.com/mig-*`),
ResourceClaim-based GPUs (core-k8s DRA, resource.k8s.io), and
GPU-Operator time-slicing without per-pod claims are invisible to the
tier math — those nodes stay at tier 1, biased small, never
over-provisioning. The field name `dynamicResourceAllocation` refers to
this per-node tiered sizing only and is unrelated to core Kubernetes
DRA (resource.k8s.io). With `spec.hostNetwork: true`, a tier change
briefly contends the old pod and the new tier pod for the same host
ports; the scheduler holds the new pod Pending until the old one exits,
so expect a short engine gap on the node during the swap.

## Validation

`workloadSelector` is required when enabled, and must be non-empty and
syntactically valid: an empty `LabelSelector` matches every pod (one
malformed CR would label every node running anything in the namespace),
and unknown operators or `In`-without-values are rejected at validation
time via upstream `LabelSelector` syntax validation, surfacing as
`ConfigValid=False` at reconcile time, rather than failing per-event
in the controller.

---

## CacheBlend: `CacheBlendEngine` CRD + Injection Webhook

CacheBlend reuses cached KV at shifted positions. It has two halves the operator
manages together: a **GPU-resident blend engine** (server side) and a
**vLLM-side plugin** that must be loaded into the serving container. The operator
ships both as a second CRD plus a mutating admission webhook.

> This implements what was previously deferred as a future `blend.enabled` field
> on `LMCacheEngine`. It is instead a **separate `CacheBlendEngine` kind** (with
> its own controller) plus an injection webhook — cleaner separation, and no
> behavior change to `LMCacheEngine`.

### `CacheBlendEngine` CRD

Group `lmcache.lmcache.ai`, `v1alpha1`, kind `CacheBlendEngine` (shortName `cbe`).
The spec **mirrors `LMCacheEngineSpec`** (image, server, l1, eviction, prometheus,
l2Backend, scheduling, overrides, imagePullSecrets) and adds:

- `blend.checkLayer` (default 1), `blend.recompRatio` (default 0.15), and
  `blend.partialBucket` (unset by default — emits `cb.partial_bucket`, the
  PARTIAL row-count padding bucket fp8-MoE models need to keep the Triton
  autotuner from re-tuning per CB step) — CB tunables fed to the vLLM
  connector.
- `injection` — what the webhook injects into vLLM pods: `payloadImage` (an
  `ImageSpec` — `repository`/`tag`/`pullPolicy`, like `spec.image` — for the
  private `lmcache-cacheblend` init-container image; set `repository` explicitly,
  the inherited engine-image default is not a valid payload), `imagePullSecrets`
  (appended to the vLLM pod so the private payload image can pull — the Secret
  must exist in the vLLM pod's namespace), `targetContainer` (default: first
  container), `cudagraph` (`eager`|`piecewise`|`full_decode_only`, default
  `eager`). Env vars need no injection field: the webhook only
  prepends PYTHONPATH and never touches other env, so pod authors set model
  env vars (e.g. `VLLM_USE_FLASHINFER_MOE_FP8=0`) directly on their container.
- `server.chunkSize` defaults to **256** and is validated to equal 256 (the blend
  matcher requires `chunk_size == 256`).

### The blend engine (controller)

`CacheBlendEngineReconciler` mirrors `LMCacheEngineReconciler` and reconciles a
DaemonSet running `lmcache server --engine-type blend` (plus
`--l1-align-bytes 16777216`), a node-local lookup Service, a metrics Service, and
a `<name>-connection` ConfigMap. **GPU model is identical to `LMCacheEngine`**:
`runtimeClassName: nvidia` + `NVIDIA_VISIBLE_DEVICES=all` + the lmcache-dev-shm hostPath
mount (or `hostIPC` when `spec.hostIPC=true`; `privileged` when
`spec.privileged=true`), with **no `nvidia.com/gpu`
device-plugin claim** — the engine
*shares* the vLLM GPU rather than reserving one, because the blend server scatters
re-RoPE'd KV directly into vLLM's paged KV over **same-device CUDA IPC**. The
engine resource builders are the same name/spec-keyed cores used by
`LMCacheEngine`.

The `<name>-connection` ConfigMap carries the **`CBKVConnector`**
`kv-transfer-config` (vs `LMCacheMPConnector` for `LMCacheEngine`) — same node-local
`tcp://` host/port shape, plus the `cb.*` tunables:

```json
{
  "kv_connector": "CBKVConnector",
  "kv_connector_module_path": "lmcache_cacheblend.connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://<name>.<namespace>.svc.cluster.local",
    "lmcache.mp.port": "<server.port>",
    "cb.check_layer": <blend.checkLayer>,
    "cb.recomp_ratio": <blend.recompRatio>,
    "cb.partial_bucket": <blend.partialBucket, only when set>
  }
}
```

With `spec.pd` set, the ConfigMap instead carries three role-keyed configs
(same key names as the `LMCacheEngine` PD ConfigMap, selected by the webhook
from the pod's `lmcache.ai/pd-role` annotation):

| key | config |
|---|---|
| `kv-transfer-config.json` | bare `CBKVConnector` (fallback, identical to the non-PD JSON above) |
| `kv-transfer-config-prefiller.json` | `MultiConnector` (`kv_producer`) wrapping `NixlConnector` + the `CBKVConnector` above |
| `kv-transfer-config-decoder.json` | bare `NixlConnector` (`kv_consumer`) |

The roles are asymmetric — unlike `LMCacheEngine`, where both roles get a
MultiConnector — because blending is a prefill-time operation: the prefiller
blends cached KV and pushes the result to the decoder over NIXL, while the
decoder only receives KV and has no use for a CacheBlend connector (or the
engine) at all.

Co-location works exactly like `LMCacheEngine`: one engine per GPU node
(DaemonSet), and the node-local Service (`internalTrafficPolicy: Local`) routes a
vLLM pod to the same-node engine. The control-plane RPC is TCP via that Service;
the data-plane KV write is CUDA IPC on the shared GPU.

### The injection webhook

A mutating admission webhook (`/mutate--v1-pod`, `CREATE`, `failurePolicy: Ignore`)
injects the `lmcache-cacheblend` plugin into opted-in pods so a **stock vLLM image
needs no rebuild**. A pod opts in with label `lmcache.ai/cacheblend-inject: "true"`
and binds to an engine with annotation `lmcache.ai/cacheblend-engine: <name>`. The
webhook then applies:

| Mutation | What |
|---|---|
| host `/dev/shm` mount (or `hostIPC: true` when the engine opts in) | required for CUDA IPC with the node-local engine |
| `cb-plugin` emptyDir + payload init container | the busybox payload `cp -a`'s the pure-Python plugin tree onto the shared volume |
| readOnly mount + `PYTHONPATH=/cb-plugin` on the vLLM container | vLLM discovers the plugin via its `vllm.general_plugins` entry point |
| append required vLLM args | `--kv-transfer-config <from the connection ConfigMap>`, `--pipeline-parallel-size 1`, `--no-enable-chunked-prefill`, `--enforce-eager` (or the configured cudagraph) |
| append `injection.imagePullSecrets` | so the private payload image can pull |
| stamp `lmcache.ai/cacheblend-injected: "true"` | idempotency guard |

The webhook **skips** (stamping `lmcache.ai/cacheblend-skip-reason`) when: the
target container overrides `command` (a `sh -c` wrapper — appended args wouldn't
reach `vllm serve`); the user already supplies `--kv-transfer-config` (not
clobbered); the named engine's connection ConfigMap doesn't exist; the engine's
`injection.payloadImage` resolves to an empty reference (`payload-image-unset`);
the pod's `lmcache.ai/pd-role` annotation carries a value other than
`prefiller`/`decoder` (`unknown-pd-role` — a typo'd role must not silently
receive the non-PD fallback config; shared with the `LMCacheEngine` injector);
or the requested `targetContainer`/`cacheblend-container` annotation names a
container that does not exist on the pod (`target-container-not-found`). It does
**not** gate on engine readiness — like `LMCacheEngine`, the connector connects
when the engine comes up. Args are emitted in two-token form
(`--pipeline-parallel-size 1`); the replace-not-duplicate dedup still recognizes
a user-supplied `--flag=value`.

**PD roles.** With `spec.pd` set, the pod's `lmcache.ai/pd-role` annotation
selects the kv-transfer-config (see the ConfigMap contract above). Prefiller
pods receive the full mutation table; **decoder pods receive only**
`--kv-transfer-config` and the NIXL side-channel env vars
(`VLLM_NIXL_SIDE_CHANNEL_HOST` from `status.podIP`,
`VLLM_NIXL_SIDE_CHANNEL_PORT` from `spec.pd.nixlSideChannelPort`). A decoder
runs a bare `NixlConnector` and never loads the CacheBlend plugin, so payload
staging would be dead weight, the CacheBlend vLLM flags would be harmful
(`--enforce-eager` disables CUDA graphs on the decode role — the exact role
PD exists to optimize), and there is no CUDA IPC connection to the engine to
wire `/dev/shm` for. Consequently the `payload-image-unset` skip does not
apply to decoder pods, and `injection.payloadImage` is not required for
decoder-only engines.

> **Shared with the LMCache injector.** The emptyDir + payload init container +
> readOnly mount + `PYTHONPATH` staging (rows 2–3 above) is generic — the standard
> `LMCacheEngine` injector (`/mutate-lmcache--v1-pod`) reuses it (under
> `internal/webhook/payload_staging.go`) to optionally stage an `lmcache` build
> when `LMCacheEngine.spec.injection.payloadImage` is set. There the staging is
> **additive and optional**: the connection wiring always runs, and an unset (or
> absent) `injection.payloadImage` simply stages nothing — it never skips the
> whole injection. The LMCache staging uses its own names (`lmcache-payload`
> volume, `/lmcache-payload` mount) so both injectors can fire on one pod.

### Prerequisites

- **cert-manager** — the webhook's serving cert is a cert-manager `Issuer` +
  `Certificate` (caBundle injected via `inject-ca-from`); install it before
  `make deploy`.
- **`make deploy`, not `make run`** — `make run` sets `ENABLE_WEBHOOKS=false` and
  installs no `MutatingWebhookConfiguration`; it is controller-only. The webhook
  needs the operator running as an in-cluster pod.
- **Pod Security Standards** — the injected hostPath `/dev/shm` mount (and
  `hostIPC`/`privileged` when opted in) is rejected by the `baseline`/`restricted`
  profiles, so the engine's and the vLLM pod's namespaces must be labeled
  `pod-security.kubernetes.io/enforce=privileged`.

### Resources created (for a `CacheBlendEngine` named `cb`)

| Resource | Name | Purpose |
|---|---|---|
| DaemonSet | `cb` | `lmcache server --engine-type blend` on GPU nodes |
| Service (node-local) | `cb` | same-node discovery for vLLM (`CBKVConnector`) |
| Service (headless) | `cb-metrics` | Prometheus scrape target |
| ConfigMap | `cb-connection` | `CBKVConnector` kv-transfer-config |
| MutatingWebhookConfiguration | (operator-wide) | injects the plugin into opted-in vLLM pods |

---

## Coordinator: `LMCacheCoordinator` CRD

The **coordinator** (`lmcache/v1/mp_coordinator/`) is the fleet-level HTTP service
that engine servers register/heartbeat against, and which drives L2 quota
eviction and global CacheBlend lookups. Unlike the engines (one DaemonSet pod per
GPU node), the coordinator is a single fleet-wide service, so `LMCacheCoordinator`
reconciles a **Deployment + ClusterIP Service** instead of a DaemonSet. The
controller carries no finalizer — owner-reference GC cascade-deletes the children.

### `LMCacheCoordinatorSpec`

The spec mirrors `MPCoordinatorConfig` (`lmcache/v1/mp_coordinator/config.py`); the
controller renders each field into the matching `lmcache coordinator` CLI flag:
`host`, `port` (9300), `instanceTimeout` (30), `healthCheckInterval` (10),
`evictionCheckInterval` (5), `evictionRatio` (0.2), `triggerWatermark` (1.0),
`blendChunkSize` (256), `blendProbeStride` (1). It also carries `replicas`,
`image`, `prometheus`, and the usual pod-shaping fields (resources, scheduling,
env, etc.).

The global-CacheBlend knobs (`blendChunkSize`, `blendProbeStride`) render into the
`--blend-chunk-size` / `--blend-probe-stride` flags and default to 256 / 1. Note
`blendChunkSize` **must equal** the blend servers' chunk size.

> **Metrics caveat:** the coordinator process exposes only `/healthz`, not a
> Prometheus `/metrics` endpoint. ServiceMonitor support is wired for parity but
> defaults **disabled**; enabling it is only useful once a metrics endpoint is
> added to the coordinator app.

### Connecting engines to a coordinator

`LMCacheEngineSpec` and `CacheBlendEngineSpec` gained a `coordinator` block
(`CoordinatorConnectionSpec`) that maps to the server's coordinator-client flags
(`lmcache/v1/multiprocess/config.py`: `add_coordinator_args`):

- `ref` — name of an `LMCacheCoordinator` in the same namespace. The controller
  resolves it to the coordinator's Service URL (`http://<name>.<ns>.svc:<port>`)
  before building the DaemonSet, so `BuildContainerArgs` stays a pure function.
- `url` — explicit escape hatch for a coordinator the operator does not manage.
  Exactly one of `ref`/`url` is required (enforced in validation).
- `advertiseIP` — **do not set this in almost every case.** It defaults to the
  pod IP via the downward API env `LMCACHE_COORDINATOR_ADVERTISE_IP`, which is
  correct for normal in-cluster deployments. Only override it if you know exactly
  what you are doing (e.g. the coordinator runs outside the cluster and must
  reach the server through a specific externally-routable address); an incorrect
  value silently breaks coordinator-to-server connectivity.
- `heartbeatInterval`, `l2EventReporting`, `l2EventFlushInterval`.

When `coordinator` is unset, the server emits no `--coordinator-url` and does not
register (unchanged behavior).

### Resources created (for an `LMCacheCoordinator` named `coord`)

| Resource | Name | Purpose |
|---|---|---|
| Deployment | `coord` | `lmcache coordinator` HTTP server (port 9300) |
| Service (ClusterIP) | `coord` | fleet-wide discovery endpoint for engines |
| Service (headless) | `coord-metrics` | Prometheus scrape target (only if `prometheus.serviceMonitor.enabled`) |
| ServiceMonitor | `coord` | Prometheus scrape config (gated, disabled by default) |

---

## Future Extensibility

- **L2 backends:** The RESP (Redis/Valkey) adapter is natively supported with typed CRD fields and Secret-based auth injection. Other adapter types (nixl_store, fs, mock, mooncake_store, raw_block) can be configured via the `raw` escape hatch. Currently only a single L2 adapter is supported at a time. LMCache MP mode is designed to support multiple adapters in cascade, but this is not yet fully tested — once validated, the operator will support multiple adapters.
- **Blend mode:** Implemented as the separate `CacheBlendEngine` CRD + injection webhook — see [CacheBlend](#cacheblend-cacheblendengine-crd--injection-webhook) above. (This supersedes the earlier idea of a `blend.enabled` field on `LMCacheEngine`.)
- **Update strategy:** Future `spec.updateStrategy` field for `RollingUpdate`/`OnDelete` control on the DaemonSet.
- **Coordinator:** Implemented as the separate `LMCacheCoordinator` CRD (Deployment + Service) with engine-side `coordinator` connection wiring — see [Coordinator](#coordinator-lmcachecoordinator-crd) above.
- **Additional CRDs:** `LMCacheKeyManager` (global key management), `LMCacheMonitor` (engine state monitoring), `LMCacheFederation` (cross-cluster P2P topology).

---

## Key Source Files Referenced

| File | What it defines |
|---|---|
| `lmcache/v1/distributed/config.py` | `L1MemoryManagerConfig`, `L1ManagerConfig`, `EvictionConfig`, `StorageManagerConfig`, argparse |
| `lmcache/v1/mp_observability/config.py` | `PrometheusConfig`, `add_prometheus_args`, `parse_args_to_prometheus_config` |
| `lmcache/v1/multiprocess/server.py` | `MPCacheServer`, server CLI entry point, argparse (lines 629–653) |
| `lmcache/v1/multiprocess/http_server.py` | HTTP server with `/healthcheck` endpoint (FastAPI + ZMQ) |
| `lmcache/v1/mp_coordinator/config.py` | `MPCoordinatorConfig` (coordinator knobs mapped by `LMCacheCoordinator`) |
| `lmcache/cli/commands/coordinator.py` | `lmcache coordinator` CLI (flags rendered into the coordinator Deployment) |
| `lmcache/v1/multiprocess/config.py` | `add_coordinator_args` (engine `coordinator` connection flags) |
| `lmcache/v1/distributed/l2_adapters/config.py` | L2 adapter registry pattern, `L2AdapterConfigBase`, `L2AdaptersConfig` |
| `examples/multi_process/lmcache-daemonset.yaml` | Reference DaemonSet manifest |
| `examples/multi_process/vllm-deployment.yaml` | Reference vLLM deployment with kv-transfer-config |
