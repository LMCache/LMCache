# LMCache Kubernetes Operator

A Kubernetes operator that automates the deployment and lifecycle management of [LMCache](https://github.com/LMCache/LMCache) multiprocess cache servers. It manages a single CRD (`LMCacheEngine`) and reconciles it into a DaemonSet, ConfigMap, Service, and optional ServiceMonitor.

See [DESIGN.md](DESIGN.md) for architecture details, reconciliation logic, and CRD spec reference.

## Prerequisites

- Kubernetes 1.20+
- `kubectl` configured to access your cluster
- (Optional) [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator) for ServiceMonitor support

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

The operator automatically handles `hostIPC`, node-local service routing, resource sizing, and Prometheus metrics — see [DESIGN.md](DESIGN.md) for details.

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
          image: lmcache/vllm-openai:latest
          env:
            # Deterministic hashing required by LMCache
            - name: PYTHONHASHSEED
              value: "0"
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
- **`PYTHONHASHSEED=0`** — ensures deterministic token hashing so vLLM and LMCache produce consistent cache keys.
- **ConfigMap mount** — the `$(cat ...)` pattern reads the connection JSON and passes it inline to `--kv-transfer-config`. The ConfigMap name is always `<LMCacheEngine name>-connection`.
- **No `hostNetwork` needed** — the operator creates a ClusterIP Service with `internalTrafficPolicy=Local`. kube-proxy routes traffic to the LMCache pod on the same node automatically. The ConfigMap points to the service DNS name, so neither LMCache nor vLLM pods need `hostNetwork`.

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
make test         # Run unit tests
make lint         # Run golangci-lint
```

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
