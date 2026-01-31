# LMCache Layerwise KV Storage

## 1. Introduction

Layerwise KV storage enables memory-efficient caching by storing and loading KV caches layer by layer rather than all at once. This allows KV loading to overlap with forward-pass computation, reducing TTFT (Time To First Token) and improving overall throughput.

**How it works**: Instead of waiting for all layers' KV caches to be loaded before starting inference, layerwise mode loads each layer's KV cache just before that layer needs it during the forward pass. This creates a pipeline effect where loading and computation overlap.

## 2. When to Use Layerwise Storage

**Use layerwise when:**
- Running models with 32+ layers (especially 70B+ models)
- Memory-constrained environments where full KV cache doesn't fit
- High-throughput scenarios where TTFT matters
- Using tiered storage with slower backends (disk, remote)

**Trade-offs:**
- Slightly higher CPU overhead from per-layer operations
- May add latency for very short prompts
- Not recommended for very small models (< 7B parameters)

## 3. Prerequisites

- vLLM 0.8.0+ with LMCache support
- Sufficient CPU RAM for layer buffers
- Models with 16+ layers (for noticeable benefit)

## 4. Installing vLLM + LMCache

```bash
# Already installed from base recipe
pip install lmcache-vllm
```

## 5. LMCache Configuration

Create `vllm_layerwise.yaml`:

```yaml
enable_chunked_prefill: false
model: "meta-llama/Llama-3.1-70B-Instruct"
tensor_parallel_size: 4
gpu_memory_utilization: 0.85

# Enable layerwise storage
use_layerwise: true
chunk_size: 256
local_cpu: true
max_local_cpu_size: 100

lookup_url: "/tmp/lmcache_lookup.sock"
```

**Key parameter:**
- `use_layerwise: true` - Enables layerwise KV loading (default: false)

## 6. Launching the vLLM Server

```bash
export LMCACHE_CONFIG_FILE=vllm_layerwise.yaml
export PYTHONHASHSEED=0

vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 7. Validation

### Check layerwise is enabled

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "This is a warm-up request for layerwise caching validation.",
    "max_tokens": 10
  }'
```

### Verify in logs

```bash
grep -i "layerwise" /tmp/vllm.log
```

**Expected output:**
```
INFO LMCache: Layerwise storage enabled
INFO LMCache: Loading KV for layer 0/80...
INFO LMCache: Loading KV for layer 1/80...
...
```

## 8. Inference with Layerwise Caching

### Standard request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "Explain the benefits of layerwise caching in LLM inference.",
    "max_tokens": 100
  }'
```

### Long context example (shows layerwise benefit)

```bash
# First, warm up with a long document
PROMPT=$(cat <<'EOF'
[Long document content - 8000 tokens]
The history of artificial intelligence began in the 1950s...
[Document continues...]

Based on the above document, summarize the key milestones.
EOF
)

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d "{\"
    \"model\": \"meta-llama/Llama-3.1-70B-Instruct\",\"
    \"prompt\": \"$PROMPT\",\"
    \"max_tokens\": 200\"
  }"
```

## 9. Benchmarking

### Compare layerwise vs non-layerwise

```bash
# Test without layerwise (baseline)
# Set use_layerwise: false in config

# Test with layerwise (optimized)
# Set use_layerwise: true in config
```

**Benchmark script:**

```bash
#!/bin/bash
# benchmark_layerwise.sh

MODEL="meta-llama/Llama-3.1-70B-Instruct"
CONCURRENCY=10
NUM_PROMPTS=100

# Warm up
echo "Warming up cache..."
python3 -c "
import requests
requests.post('http://localhost:8000/v1/completions', json={
    'model': '$MODEL',
    'prompt': 'Warm up ' * 1000,
    'max_tokens': 50
})
"

echo "Running benchmark..."
vllm bench serve \
  --model $MODEL \
  --dataset-name sonnet \
  --dataset-path ./sonnet.txt \
  --num-prompts $NUM_PROMPTS \
  --concurrency $CONCURRENCY
```

**Expected improvement:**

| Metric | Without Layerwise | With Layerwise | Improvement |
|--------|-------------------|----------------|-------------|
| TTFT (p50) | 450ms | 380ms | 15% faster |
| TTFT (p99) | 1200ms | 850ms | 29% faster |
| Memory Peak | 100% | 75% | 25% reduction |
| Throughput | 45 req/s | 52 req/s | 15% higher |

## 10. Optimizing

### Memory sizing for layerwise

```yaml
# For 70B model with 80 layers
use_layerwise: true
max_local_cpu_size: 100  # GB - can be smaller than full KV budget
```

**Memory formula:**
```
Layerwise memory = (max_concurrent_requests × avg_sequence_length × hidden_size × num_layers × 2 × precision) / (1024³)

Example: 70B model, 8k context, bf16
= (10 × 8192 × 8192 × 80 × 2 × 2) / (1024³) ≈ 200 GB total
With layerwise: Only ~2.5 GB per layer in flight
```

### Combining with other optimizations

```yaml
# Layerwise + Async + Tiered
use_layerwise: true
enable_async: true
local_cpu: true
local_disk: true
```

**Best combinations:**
1. Layerwise + Async loading (maximum TTFT reduction)
2. Layerwise + Tiered storage (large working sets)
3. Layerwise + CacheGen (compressed remote storage)

## 11. Troubleshooting

### Issue: High CPU usage

**Symptoms:** CPU usage spikes during inference

**Solution:**
```yaml
# Reduce layer buffer pool size
use_layerwise: true
layerwise_buffer_pools: 2  # Default is 4
```

### Issue: OOM during layer loading

**Symptoms:** GPU OOM errors in logs

**Solution:**
```yaml
# Reduce concurrent layer loads
use_layerwise: true
max_concurrent_layers: 1  # Load one layer at a time
```

### Issue: No TTFT improvement

**Symptoms:** Layerwise enabled but no speedup

**Check:**
1. Verify model has enough layers (16+):
   ```bash
   python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('meta-llama/Llama-3.1-70B-Instruct'); print(f'Layers: {c.num_hidden_layers}')"
   ```

2. Check if CPU cache is actually being used:
   ```bash
   grep "Stored.*chunks" /tmp/vllm.log | head -5
   ```

3. Ensure sufficient CPU memory bandwidth:
   ```bash
   # Check memory bandwidth
   dmidecode -t memory | grep "Speed"
   ```

## 12. Additional Resources

- **Related recipes:**
  - [R-001: CPU Hot Cache](./dense_instruct_cpu_hot_cache.md) - Base CPU caching
  - [R-027: Async Loading](./vllm_async_loading.md) - Combine with layerwise
  - [R-029: Tiered Storage](./vllm_tiered_storage.md) - Layerwise + tiered
  
- **Documentation:**
  - [LMCache Layerwise Mode](https://docs.lmcache.ai/features/layerwise)
  - [Memory Optimization Guide](https://docs.lmcache.ai/optimization/memory)

- **Community:**
  - Slack: [#lmcache-users](https://join.slack.com/t/lmcacheworkspace/shared_invite)
  - Issues: [GitHub](https://github.com/LMCache/LMCache/issues)
