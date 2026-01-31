# LMCache + KServe: vLLM Runtime with KV Offload

## 1. Introduction

**Target workload**
- Kubernetes deployments using KServe InferenceService
- vLLM models with long prompts and large working sets
- Multi-tenant clusters with session routing needs
- Production environments that require standard KServe APIs

**LMCache mode**
- **Storage Mode**
- KServe-managed vLLM runtime
- CPU hot cache (local) enabled via LMCache config

This recipe shows how to run **vLLM on KServe** with **LMCache enabled** for KV offloading. It uses a custom container in an InferenceService with LMCache config mounted via ConfigMap.

**Expected outcome**
- KServe starts the vLLM container with LMCache enabled
- Cold request stores KV into LMCache
- Warm request shows LMCache hits and reduced TTFT

## 2. When to Use LMCache with KServe

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Long prompts on GPU-limited nodes | **Enable LMCache** | Offload KV to CPU for larger working sets |
| Multi-tenant inference service | **Enable LMCache + session routing** | Increases reuse of hot prefixes |
| Short prompts, single model | **Optional** | Native prefix caching may be sufficient |
| Strict isolation between tenants | **Separate InferenceServices** | Avoid cross-tenant cache reuse |
| Need persistence across restarts | **Add disk or remote tier** | CPU hot cache is ephemeral |

## 3. Installing KServe + vLLM

Prerequisites:
- Kubernetes cluster with GPU nodes
- KServe installed (see https://kserve.github.io/)
- NVIDIA device plugin installed

## 4. LMCache Configuration

Create `recipes/kserve_vllm_lmcache.yaml` (ConfigMap + InferenceService):

```yaml
apiVersion: v1
kind: List
items:
- apiVersion: v1
  kind: ConfigMap
  metadata:
    name: lmcache-config
  data:
    lmcache.yaml: |
      chunk_size: 256
      local_cpu: true
      # Size CPU cache to ~1.5x GPU KV cache budget
      max_local_cpu_size: 48
      use_layerwise: false
      save_unfull_chunk: true
- apiVersion: serving.kserve.io/v1beta1
  kind: InferenceService
  metadata:
    name: vllm-lmcache
  spec:
    predictor:
      containers:
      - name: vllm
        image: lmcache/vllm-openai:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8080
        env:
        - name: HF_TOKEN
          value: "<YOUR_HF_TOKEN>"
        - name: LMCACHE_CONFIG_FILE
          value: /etc/lmcache/lmcache.yaml
        command: ["bash", "-lc"]
        args:
        - |
          export PYTHONHASHSEED=0
          vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
            --host 0.0.0.0 \
            --port 8080 \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.85 \
            --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
        resources:
          limits:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
          requests:
            nvidia.com/gpu: 1
            cpu: "8"
            memory: "32Gi"
        volumeMounts:
        - name: lmcache-config
          mountPath: /etc/lmcache
      volumes:
      - name: lmcache-config
        configMap:
          name: lmcache-config
```

**Custom runtime configuration notes**
- `LMCACHE_CONFIG_FILE` points to the mounted ConfigMap.
- `--kv-transfer-config` enables LMCacheConnectorV1 in vLLM.
- Use `PYTHONHASHSEED=0` for deterministic chunk hashing.
- For production, keep vLLM prefix caching enabled (do not add `--no-enable-prefix-caching`).

## 5. Launching the Service (with LMCache)

```bash
kubectl apply -f recipes/kserve_vllm_lmcache.yaml
```

Monitor the InferenceService:

```bash
kubectl get inferenceservice vllm-lmcache
kubectl get pods -l serving.kserve.io/inferenceservice=vllm-lmcache
```

## 6. Startup Validation

Check the vLLM pod logs for LMCache init:

```bash
kubectl logs -f <vllm-pod-name>
```

Expected LMCache logs:

```
LMCache INFO: Loading LMCache config file /etc/lmcache/lmcache.yaml
LMCache INFO: Creating LMCacheEngine with config: {'chunk_size': 256, 'local_cpu': True, ...}
```

## 7. Inference and Cache Validation

### 7.1 Port-forward the predictor service

```bash
kubectl get svc | grep vllm-lmcache
kubectl port-forward svc/vllm-lmcache-predictor 8080:80
```

### 7.2 Cold request (first run)

```bash
python - <<'PY' | curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d @-
import json
prompt = "You are helpful.\n" + ("LMCache reuse test. " * 400)
payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": prompt,
    "max_tokens": 32,
}
print(json.dumps(payload))
PY
```

Expected logs (cold, stores to LMCache):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.3 Warm request (second run)

Run the same request again.

Expected logs (warm, retrieves from LMCache):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

## 8. Benchmarking

Use `vllm bench serve` against the KServe endpoint to measure TTFT improvements:

```bash
vllm bench serve --model mistralai/Mistral-7B-Instruct-v0.2 \
  --base-url http://localhost:8080 \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run once to warm the cache, then run again to observe TTFT reduction.

## 9. Optimizing Performance

- Increase `max_local_cpu_size` when hit rate is low.
- Use smaller `chunk_size` (128) if prompts share partial prefixes.
- Keep KServe routing sticky (same session header) when possible.
- Consider adding disk or remote tiers if cache must persist across restarts.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No LMCache logs | ConfigMap not mounted | Verify volumeMounts and `LMCACHE_CONFIG_FILE` |
| Warm requests still cold | Prompts differ | Ensure identical prompts/tokenization |
| Pod OOM | CPU cache too large | Lower `max_local_cpu_size` or raise pod memory |
| Service unreachable | Wrong service name | Check `kubectl get svc` and port-forward |
| GPU not allocated | Missing device plugin | Install NVIDIA device plugin |

## 11. Additional Resources

- KServe documentation: https://kserve.github.io/
- vLLM OpenAI server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- LMCache configuration guide: `docs/source/api_reference/configurations.rst`
- LMCache vLLM recipe: `recipes/dense_instruct_cpu_hot_cache.md`
