# LMCache + llm-d: ModelService with KV Offload

## 1. Introduction

**Target workload**
- Kubernetes deployments using llm-d ModelService
- vLLM models with long prompts and large working sets
- Workloads that benefit from cache reuse and CPU offload
- Environments that already use llm-d routing and scheduling

**LMCache mode**
- **Storage Mode**
- llm-d ModelService (Helm)
- CPU hot cache enabled via vLLM args and LMCache config

This recipe shows how to run **llm-d ModelService** with **LMCache enabled** for KV offloading by configuring a custom vLLM command in the Helm values file.

**Expected outcome**
- ModelService starts vLLM with LMCacheConnectorV1
- Cold request stores KV to LMCache
- Warm request shows LMCache hits and lower TTFT

## 2. When to Use LMCache with llm-d

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Long prompts, limited GPU memory | **Enable LMCache** | Offload KV to CPU to extend cache capacity |
| High overlap in prompts | **Enable LMCache** | Improves TTFT on warm hits |
| Small models, short prompts | **Optional** | Native prefix caching may suffice |
| Need persistence across restarts | **Add disk/remote tier** | CPU hot cache is ephemeral |
| Multi-tenant isolation | **Separate deployments** | Avoid cross-tenant cache sharing |

## 3. Installing llm-d + ModelService

Prerequisites:
- Kubernetes cluster with GPU nodes
- llm-d infra stack installed (inference gateway + CRDs)

Install the ModelService Helm repo:

```bash
helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/
helm repo update
```

## 4. LMCache Configuration

Create `recipes/llm_d_lmcache.yaml`:

```yaml
multinode: false

modelArtifacts:
  name: mistralai/Mistral-7B-Instruct-v0.2
  labels:
    llm-d.ai/inference-serving: "true"
    llm-d.ai/model: mistral-7b-instruct
  uri: hf://"{{ .Values.modelArtifacts.name }}"
  size: 50Gi

routing:
  servicePort: 8000
  proxy:
    secure: false

accelerator:
  type: nvidia

decode:
  replicas: 1
  containers:
  - name: vllm
    image: "ghcr.io/llm-d/llm-d-cuda:latest"
    modelCommand: custom
    command: ["bash", "-lc"]
    args:
      - |
        cat <<'EOF' > /tmp/lmcache.yaml
        chunk_size: 256
        local_cpu: true
        # Size CPU cache to ~1.5x GPU KV cache budget
        max_local_cpu_size: 48
        use_layerwise: false
        save_unfull_chunk: true
        EOF
        export LMCACHE_CONFIG_FILE=/tmp/lmcache.yaml
        export PYTHONHASHSEED=0
        vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
          --host 0.0.0.0 \
          --port 8000 \
          --max-model-len 8192 \
          --gpu-memory-utilization 0.85 \
          --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
    env:
      - name: HF_TOKEN
        value: "<YOUR_HF_TOKEN>"
    ports:
      - containerPort: 8000
        protocol: TCP
    resources:
      limits:
        nvidia.com/gpu: 1
        cpu: "8"
        memory: "32Gi"
      requests:
        nvidia.com/gpu: 1
        cpu: "8"
        memory: "32Gi"
    mountModelVolume: true
```

**Integration approach**
- llm-d ModelService does not expose LMCache knobs directly, so we use a **custom vLLM command**.
- LMCache config is written at container start and referenced via `LMCACHE_CONFIG_FILE`.

## 5. Launching the Service (with LMCache)

```bash
helm install llm-d-lmcache llm-d-modelservice/llm-d-modelservice -f recipes/llm_d_lmcache.yaml \
  --namespace llm-d --create-namespace
```

## 6. Startup Validation

Check the decode pod logs for LMCache init:

```bash
kubectl get pods -n llm-d -l llm-d.ai/role=decode
kubectl logs -n llm-d -f <decode-pod-name>
```

Expected LMCache logs:

```
LMCache INFO: Loading LMCache config file /tmp/lmcache.yaml
LMCache INFO: Creating LMCacheEngine with config: {'chunk_size': 256, 'local_cpu': True, ...}
```

## 7. Inference and Cache Validation

### 7.1 Route requests through Gateway

Create an HTTPRoute (example from llm-d ModelService docs) that points to the InferencePool, then send requests to the gateway. Example path routing:

```bash
curl -X POST http://<GATEWAY_HOST>/mymodel/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistralai/Mistral-7B-Instruct-v0.2","prompt":"Hello","max_tokens":16}'
```

### 7.2 Cold request (first run)

```bash
python - <<'PY' | curl http://<GATEWAY_HOST>/mymodel/v1/completions \
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

Expected logs (cold):

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 0, need to load: 0
LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.2461 GB
```

### 7.3 Warm request (second run)

Repeat the same request to observe LMCache hits:

```
LMCache INFO: Reqid: ..., Total tokens 2000, LMCache hit tokens: 1792, need to load: 1792
LMCache INFO: Retrieved 1792 out of 1792 required tokens. size: 0.2461 gb
```

## 8. Benchmarking

Run `vllm bench serve` against the gateway with a cache-friendly dataset:

```bash
vllm bench serve --model mistralai/Mistral-7B-Instruct-v0.2 \
  --base-url http://<GATEWAY_HOST>/mymodel \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 6144 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 1 \
  --prefix-repetition-output-len 32 \
  --num-prompts 50 --request-rate 0.5 --max-concurrency 1
```

Run once to warm, then run again for TTFT improvement.

## 9. Optimizing Performance

- Increase `max_local_cpu_size` if hit rate is low.
- Use smaller `chunk_size` (128) for partial prefix reuse.
- Keep routing sticky for repeat prompts when possible.
- For persistence across restarts, add disk or remote tiers in LMCache config.

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No LMCache logs | Command not used | Ensure `modelCommand: custom` and args include LMCache |
| Warm requests still cold | Prompt mismatch | Send identical prompts/tokenization |
| HF auth failure | Missing token | Set `HF_TOKEN` in values file |
| Gateway 404 | HTTPRoute not configured | Deploy HTTPRoute to target InferencePool |
| Pod OOM | CPU cache too large | Lower `max_local_cpu_size` or increase pod memory |

## 11. Additional Resources

- llm-d docs: https://www.llm-d.ai
- llm-d ModelService: https://github.com/llm-d-incubation/llm-d-modelservice
- LMCache configuration guide: `docs/source/api_reference/configurations.rst`
- vLLM + LMCache single node: `recipes/dense_instruct_cpu_hot_cache.md`
