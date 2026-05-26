# LMCache Kubernetes Operator

A Kubernetes operator that automates the deployment and lifecycle management of [LMCache](https://github.com/LMCache/LMCache) multiprocess cache servers. It manages a single CRD (`LMCacheEngine`) and reconciles it into a DaemonSet, ConfigMap, Service, and optional ServiceMonitor.

See [DESIGN.md](DESIGN.md) for architecture details, reconciliation logic, and CRD spec reference.

## Prerequisites

- Kubernetes 1.20+
- `kubectl` configured to access your cluster
- For NVIDIA GPUs (default): NVIDIA GPU Operator with the `nvidia` RuntimeClass available on GPU nodes
- For AMD GPUs: set `spec.gpuVendor: amd` in your `LMCacheEngine` (see [AMD GPUs (ROCm)](#amd-gpus-rocm) below)
- (Optional) [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) for ServiceMonitor support

> [!IMPORTANT]
> By default the operator runs LMCache pods with `runtimeClassName: nvidia` and `privileged: true` to gain GPU visibility without consuming GPU resources via the device plugin. This allows the serving engine (e.g., vLLM) to claim all GPUs on the node. Clusters using Pod Security Standards must allow the `privileged` profile for the LMCache namespace.
>
> On AMD ROCm clusters, `spec.gpuVendor: amd` omits `runtimeClassName` and skips NVIDIA-specific env vars.

## Quick Start

### 1. Install the Operator

**Option A: One-line install from release (recommended)**

Install the latest stable release:

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
```

Or use the nightly build from the `dev` branch:

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-nightly-latest/install.yaml
```

**Option B: Build from source**

```bash
cd operator
make build
make install
make deploy IMG=<your-registry>/lmcache-operator:latest
```

### 2. Deploy an LMCacheEngine

A minimal CR deploys a DaemonSet with 60 GB L1 cache on every node:

```yaml
# lmcache-engine.yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: my-cache
spec:
  l1:
    sizeGB: 60
```

```bash
kubectl apply -f lmcache-engine.yaml
```

The operator automatically handles `hostIPC`, GPU visibility (`runtimeClassName: nvidia`, `privileged: true`), node-local service routing, resource sizing, and Prometheus metrics — see [DESIGN.md](DESIGN.md) for details.

### 3. Connect vLLM to LMCache

The operator creates a ConfigMap named `<engine-name>-connection` containing the `kv-transfer-config` JSON that vLLM needs. Use it in your vLLM Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      # Required for CUDA IPC between vLLM and LMCache
      hostIPC: true
      containers:
        - name: vllm
          image: lmcache/vllm-openai:<pinned-tag>
          command: ["/bin/sh", "-c"]
          args:
            - |
              exec python3 -m vllm.entrypoints.openai.api_server \
                --model <your-model> \
                --port 8000 \
                --gpu-memory-utilization 0.8 \
                --kv-transfer-config "$(cat /etc/lmcache/kv-transfer-config.json)"
          ports:
            - name: http
              containerPort: 8000
          volumeMounts:
            - name: kv-transfer-config
              mountPath: /etc/lmcache
              readOnly: true
          resources:
            limits:
              nvidia.com/gpu: "1"
      volumes:
        - name: kv-transfer-config
          configMap:
            name: my-cache-connection  # Must match your LMCacheEngine name + "-connection"
```

Key points for vLLM pods:

- **`hostIPC: true` is required** — CUDA IPC (`cudaIpcOpenMemHandle`) needs a shared IPC namespace between vLLM and LMCache. Without this, GPU memory mapping fails.
- **ConfigMap mount** — the `$(cat ...)` pattern reads the connection JSON and passes it inline to `--kv-transfer-config`. The ConfigMap name is always `<LMCacheEngine name>-connection`.
- **External LMCache connector required** — the operator-generated config now sets `kv_connector_module_path=lmcache.integration.vllm.lmcache_mp_connector` so vLLM loads the external LMCache MP connector instead of silently resolving a vendored builtin path.
- **No `hostNetwork` needed** — the operator creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy routes traffic to the LMCache pod on the same node automatically. The ConfigMap points to the service DNS name, so neither LMCache nor vLLM pods need `hostNetwork`.

> [!IMPORTANT]
> Use a pinned vLLM image that is new enough to honor `kv_connector_module_path` for KV connector loading. In practice, that means a build that includes the external-module selection fix from vLLM PR #38301 (merged April 7, 2026). Builds that also include vLLM PR #42596 (merged May 15, 2026) are preferred because they default LMCache MP to the external connector with builtin fallback. If your existing vLLM build predates those changes or you are unsure, upgrade it before enabling this operator path.

> [!WARNING]
> **Do NOT mount an emptyDir at `/dev/shm`** on either LMCache or vLLM pods. With `hostIPC: true`, both pods share the host's `/dev/shm`. Mounting an emptyDir (even with `medium: Memory`) shadows it with a private tmpfs, breaking CUDA IPC — `cudaIpcOpenMemHandle` fails because IPC handles from one pod become invisible to the other.

### 4. Verify the Deployment

```bash
# Check LMCacheEngine status
kubectl get lmc
```

```
NAME       PHASE     READY   DESIRED   AGE
my-cache   Running   3       3         5m
```

```bash
# Check the connection ConfigMap
kubectl get configmap my-cache-connection -o yaml

# Check LMCache pods
kubectl get pods -l app.kubernetes.io/managed-by=lmcache-operator

# Check detailed status with endpoints
kubectl describe lmc my-cache
```

## Examples

### Target Only GPU Nodes

Use `nodeSelector` to run LMCache only on GPU nodes. New GPU nodes automatically get an LMCache pod:

```yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: my-cache
spec:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  l1:
    sizeGB: 60
```

### AMD GPUs (ROCm)

Set `spec.gpuVendor: amd` to run on AMD GPU nodes. The operator omits `runtimeClassName` from the pod spec and skips the NVIDIA env vars. AMD GPU nodes don't have a universal label equivalent to `nvidia.com/gpu.present`, so supply a `nodeSelector` that matches the label your platform exposes (e.g. `feature.node.kubernetes.io/amd-gpu: "true"` when using the [ROCm/gpu-operator](https://github.com/ROCm/gpu-operator)):

```yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: amd-cache
spec:
  gpuVendor: amd
  nodeSelector:
    feature.node.kubernetes.io/amd-gpu: "true"
  l1:
    sizeGB: 60
```

vLLM connects to LMCache via HIP IPC over `hostIPC` exactly the same way as CUDA IPC on NVIDIA — the `hostIPC: true` and `PYTHONHASHSEED=0` requirements above apply unchanged. Use a ROCm-built LMCache image for `spec.image`.

### Custom Server Port

If the default port (5555) conflicts with other services:

```yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: my-cache
spec:
  server:
    port: 6555
  l1:
    sizeGB: 60
```

The connection ConfigMap updates automatically — vLLM pods pick up the new port on restart.

### Production with Prometheus Monitoring

```yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: production-cache
  namespace: llm-serving
spec:
  nodeSelector:
    nvidia.com/gpu.present: "true"
  image:
    repository: lmcache/standalone
    tag: v0.1.0
  server:
    port: 6555
    chunkSize: 256
    maxWorkers: 4
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
  podAnnotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
  priorityClassName: system-node-critical
```

### L2 Storage: Redis/Valkey

Add a Redis L2 adapter for persistent KV cache storage beyond L1 memory:

```yaml
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: cache-with-redis
spec:
  l1:
    sizeGB: 60
  l2Backend:
    resp:
      host: redis.default.svc.cluster.local
      port: 6379
      numWorkers: 8
```

For Redis authentication, create a Secret with `username` and `password` keys and reference it. Credentials are injected as environment variables and never appear in pod args or `kubectl describe` output. The Secret can live in a different namespace — the operator creates a managed copy automatically:

```yaml
# Create the secret (or reference an existing one in another namespace):
# kubectl create secret generic redis-auth \
#   --from-literal=username=myuser \
#   --from-literal=password=mypassword
spec:
  l2Backend:
    resp:
      host: redis.default.svc.cluster.local
      port: 6379
      authSecretRef:
        name: redis-auth
        namespace: redis    # omit if the Secret is in the same namespace
```

### L2 Storage: Other Adapters (Raw Escape Hatch)

For adapter types not yet natively supported by the operator (e.g. `nixl_store`, `fs`, `mock`, `raw_block`), use the `raw` escape hatch. The JSON is passed through to `--l2-adapter` as-is:

```yaml
spec:
  l2Backend:
    raw:
      type: nixl_store
      config:
        backend: "POSIX"
        backend_params:
          file_path: "/data/lmcache/l2"
          use_direct_io: "false"
        pool_size: 64
```

Example `raw_block` configuration via the same escape hatch:

```yaml
spec:
  l2Backend:
    raw:
      type: raw_block
      config:
        device_path: "/dev/nvme0n1"
        slot_bytes: 1048576
        block_align: 4096
        header_bytes: 4096
        meta_total_bytes: 268435456
        use_odirect: true
        num_store_workers: 2
        num_lookup_workers: 1
        num_load_workers: 4
```

Use an unmounted raw block device or a dedicated file path reserved for LMCache. With `use_odirect: true`, the LMCache server's `--l1-align-bytes` setting must be at least `block_align`.

> [!NOTE]
> Currently only a single L2 adapter is supported at a time. While LMCache multiprocess mode is designed to support multiple L2 adapters in cascade, this functionality is not yet fully tested. Once the multi-adapter pipeline is validated and performance is confirmed, the operator will be updated to support multiple adapters.

### Override Auto-Computed Resources

By default, the operator derives memory requests/limits from `l1.sizeGB`. To override:

```yaml
spec:
  l1:
    sizeGB: 60
  resourceOverrides:
    requests:
      memory: "70Gi"
      cpu: "8"
    limits:
      memory: "100Gi"
```

## Development

```bash
make generate     # Generate DeepCopy methods
make manifests    # Generate CRD YAML + RBAC
make build        # Compile operator binary
make fmt          # go fmt
make vet          # go vet
make test         # Run unit tests (envtest, CPU-only)
make lint         # Run golangci-lint
```

### End-to-End Tests

Four `make` targets cover the e2e tiers. The `-kind` variants create
a throwaway Kind cluster and tear it down on exit; the `-cluster`
variants run against whatever your current `KUBECONFIG` points at.
M2 (GPU) targets additionally run a runtime HTTP check against the
LMCache server and a vLLM round-trip that proves the KV cache
stores on the first request and retrieves on the second.

```bash
make test-e2e-kind                                   # local Kind, no GPU, ~5 min
make test-e2e-cluster        IMG=<registry/image:tag>  # existing cluster, no GPU
make test-e2e-gpu-kind                               # local Kind, GPU, ~30 min
make test-e2e-gpu-cluster    IMG=<registry/image:tag>  # existing GPU cluster
```

#### Tool prerequisites

| Target | Tools to install |
|---|---|
| `test-e2e-kind` | `kind`, `kubectl`, `docker` |
| `test-e2e-cluster` | `kubectl` (cluster access via `KUBECONFIG`) |
| `test-e2e-gpu-kind` | `kind`, `kubectl`, `docker`, `helm` (v3) |
| `test-e2e-gpu-cluster` | `kubectl`, `helm` (cluster access via `KUBECONFIG`) |

```bash
# Install kind (the other tools are distro-specific)
go install sigs.k8s.io/kind/cmd/kind@latest
```

#### One-time host setup for `test-e2e-gpu-kind`

Beyond the tools above, your host needs the NVIDIA driver and
`nvidia-container-toolkit` installed (distro-specific), plus two
`nvidia-ctk` commands that flip docker's default runtime to `nvidia`
and toggle the volume-mount-based GPU injection mechanism the
target's inline Kind config relies on:

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk config --set accept-nvidia-visible-devices-as-volume-mounts=true --in-place
sudo systemctl restart docker
```

The target fails fast with a copy-pasteable fix command if either
piece of host config is missing. Note that flipping docker's default
runtime is a **host-wide** change — every container on the host then
starts through `nvidia-container-runtime`. Non-GPU workloads still
work but go through one extra hook.

The target installs the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
inside the Kind cluster to handle the toolkit / containerd reconfig
that lets pods scheduled by Kind's inner containerd see the driver
libraries. That install takes 5-10 min, which is the bulk of the
`test-e2e-gpu-kind` runtime. (`nvkind` is **not** required — we
tried it and the current target uses a hand-rolled Kind config
instead.)

#### Cluster prerequisites for `test-e2e-gpu-cluster`

The existing cluster must have:

- at least one node labeled `nvidia.com/gpu.present=true`
- the `nvidia` `RuntimeClass` installed (GPU Operator or equivalent)
- the operator image pushed to a registry the cluster's image puller can reach (pass as `IMG=…`)

#### Knobs (env vars)

| Variable | Default | Used by |
|---|---|---|
| `KIND_CLUSTER` | `operator-test-e2e-<id>` | `test-e2e-kind` — set to target an existing Kind cluster |
| `GPU_KIND_CLUSTER` | `operator-test-e2e-gpu-<id>` | `test-e2e-gpu-kind` — same idea |
| `KEEP_CLUSTER_ON_FAILURE` | unset | `test-e2e-gpu-kind` — set to `1` to keep the cluster alive after a failure for live debugging |
| `VLLM_MODEL` | `Qwen/Qwen2.5-0.5B` | vLLM integration spec — Hugging Face model id |
| `VLLM_IMAGE` | `lmcache/vllm-openai:latest` | vLLM integration spec |
| `SKIP_VLLM_INTEGRATION` | unset | set to `true` to skip the heavyweight vLLM spec but still run the runtime HTTP check |

If a GPU run fails during setup, use the diagnostic target:

```bash
make diagnose-test-e2e-gpu-kind GPU_KIND_CLUSTER=<name>
```

It dumps `ClusterPolicy` status, GPU-Operator pod state, events,
toolkit/device-plugin daemonset logs, `/dev/nvidia*` inside the Kind
worker, and the `nvidia` stanza of the worker's containerd config.

For deeper design notes (why GPU Operator over the bare device plugin,
how the volume-mount marker propagates GPUs into the Kind worker, the
test specs themselves), see [`AGENTS.md`](./AGENTS.md).

### Pushing a Custom Operator Image

```bash
# Docker Hub
docker login
make docker-build docker-push IMG=docker.io/<your-user>/lmcache-operator:latest
make deploy IMG=docker.io/<your-user>/lmcache-operator:latest

# Private registry
docker login <your-registry>
make docker-build docker-push IMG=<your-registry>/lmcache-operator:latest
make deploy IMG=<your-registry>/lmcache-operator:latest

# Multi-platform (amd64 + arm64)
make docker-buildx IMG=<your-registry>/lmcache-operator:latest
```

If your cluster needs pull credentials, create a secret and reference it in `config/manager/manager.yaml`:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n lmcache-operator-system
```

## License

Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
