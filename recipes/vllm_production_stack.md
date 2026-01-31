# LMCache + vLLM Production Stack: Enterprise Deployment

## 1. Introduction

**Target workload**
- Kubernetes-based vLLM deployments with multiple replicas
- Shared system prompts or RAG prefixes across users
- Enterprise environments with observability and routing requirements
- Autoscaling where cache persistence matters across pods

**LMCache mode**
- **Storage Mode**
- Multi-pod vLLM deployment via Production Stack
- CPU offloading (local hot tier) enabled through Helm values

This recipe shows how to enable LMCache in the **vLLM Production Stack** Helm chart so KV cache can be offloaded to CPU and reused across requests routed to the same model instances.

**Expected outcome**
- Production Stack pods start with LMCache enabled
- Router forwards requests to vLLM backends with session routing
- Warm requests show LMCache hits and reduced TTFT

## 2. When to Use LMCache in Production Stack

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Large prompts, limited GPU memory | **LMCache enabled** | Offload KV to CPU for larger working sets |
| Multi-replica serving | **LMCache + session routing** | Reuse KV with stable session affinity |
| Single-node dev cluster | **Optional** | Native prefix caching may be faster |
| Strict cache isolation | **Per-tenant deployment** | Avoid cross-tenant cache reuse |
| Need cache persistence across pod restarts | **Add disk or remote tier** | CPU tier alone is ephemeral |

## 3. Installing vLLM Production Stack + LMCache

```bash
git clone https://github.com/vllm-project/production-stack.git
cd production-stack
helm repo add vllm https://vllm-project.github.io/production-stack
```

Ensure your Kubernetes cluster has GPU nodes and the NVIDIA device plugin installed.

## 4. LMCache Configuration

Create `recipes/vllm_production_stack.yaml` (Helm values override):

```yaml
servingEngineSpec:
  modelSpec:
  - name: "mistral"
    repository: "lmcache/vllm-openai"
    tag: "latest"
    modelURL: "mistralai/Mistral-7B-Instruct-v0.2"
    replicaCount: 1
    requestCPU: 10
    requestMemory: "40Gi"
    requestGPU: 1
    pvcStorage: "50Gi"
    vllmConfig:
      enableChunkedPrefill: false
      enablePrefixCaching: true
      maxModelLen: 16384
      gpuMemoryUtilization: 0.90
    lmcacheConfig:
      enabled: true
      cpuOffloadingBufferSize: "30"
    hf_token: "<YOUR_HF_TOKEN>"

routerSpec:
  enableRouter: true
  routingLogic: "session"
  sessionKey: "x-session-id"
```

**Integration points between Production Stack and LMCache**
- `servingEngineSpec.modelSpec[].lmcacheConfig.enabled`: turns LMCache on for the vLLM pods.
- `servingEngineSpec.modelSpec[].lmcacheConfig.cpuOffloadingBufferSize`: CPU cache size (GB).
- `servingEngineSpec.modelSpec[].vllmConfig.enablePrefixCaching`: keep **true** for best performance.
- `routerSpec.routingLogic: session`: enables session affinity for higher cache reuse across requests.

**Sizing guidance**
- Set `cpuOffloadingBufferSize` to ~1.5x the GPU KV cache budget per replica.
- Ensure each vLLM pod has enough host RAM for the CPU cache plus model overhead.

## 5. Launching the Stack (with LMCache)

```bash
helm install vllm vllm/vllm-stack -f recipes/vllm_production_stack.yaml
```

To update an existing installation:

```bash
helm upgrade vllm vllm/vllm-stack -f recipes/vllm_production_stack.yaml
```

## 6. Startup Validation

Check pods and logs for LMCache initialization:

```bash
kubectl get pods
kubectl logs -f <vllm-pod-name>
```

Expected logs include LMCache configuration messages:

```
INFO ... lmcache_connector.py:... Initializing LMCacheConfig ... kv_connector='LMCacheConnector'
INFO LMCache: Creating LMCacheEngine instance vllm-instance
```

## 7. Inference and Cache Validation

### 7.1 Port-forward the router service

```bash
kubectl port-forward svc/vllm-router-service 30080:80
```

### 7.2 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:30080/v1/completions \
  -H "Content-Type: application/json" \
  -H "x-session-id: demo-session" \
  -d @-
import json
prompt = "System: You are helpful.\n" + ("LMCache reuse test. " * 400)
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected LMCache logs (cold):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.3 Warm request (same session)

Repeat the same request with the same `x-session-id` header.

Expected LMCache logs (warm):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

## 8. Benchmarking

Use `vllm bench serve` against the router to observe TTFT improvements:

```bash
vllm bench serve --model mistralai/Mistral-7B-Instruct-v0.2 \
  --base-url http://localhost:30080 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run once to warm the cache, then run again to observe TTFT drop on warm hits.

## 9. Optimizing Performance

- **Session routing**: keep `routerSpec.routingLogic: session` and pass a stable session header.
- **CPU cache sizing**: increase `cpuOffloadingBufferSize` if hit rate is low.
- **GPU memory utilization**: tune `gpuMemoryUtilization` to balance throughput and cache capacity.
- **Prefix caching**: keep vLLM native prefix caching enabled for best latency.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No LMCache logs | `lmcacheConfig.enabled` false | Set to true in Helm values |
| Warm requests still cold | Session routing disabled | Use `routingLogic: session` and pass `x-session-id` |
| CPU OOM in pod | Buffer too large | Lower `cpuOffloadingBufferSize` or raise pod memory |
| Low hit rate | Prompts vary per request | Normalize system prompts or reduce `chunk_size` |
| Router 404 | Port-forward missing | Re-run `kubectl port-forward` |

## 11. Additional Resources

- Production Stack docs: https://docs.vllm.ai/projects/production-stack
- Production Stack LMCache tutorial: https://github.com/vllm-project/production-stack/blob/main/tutorials/05-offload-kv-cache.md
- LMCache configuration guide: `docs/source/api_reference/configurations.rst`
- vLLM + LMCache single node: `recipes/dense_instruct_cpu_hot_cache.md`
