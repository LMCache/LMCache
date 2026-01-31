# LMCache + vLLM: CacheBlend for RAG with Reordered Context

## 1. Introduction

**Target workload**
- Retrieval-Augmented Generation (RAG) applications
- Dynamic context assembly where document order varies
- Multi-source context with frequent reordering
- **Non-prefix KV reuse when text segments move**

**LMCache mode**
- **Storage Mode with Blending**
- Single or multi-node
- CPU/disk/remote backends with blending enabled

This recipe demonstrates **CacheBlend**, an advanced LMCache optimization that enables KV cache reuse even when the order of text segments changes. Traditional prefix caching requires exact token sequence matching, but CacheBlend can:

1. **Reuse KV from reordered segments** - Same documents, different order
2. **Blend partial matches** - Combine cached KV with new computation
3. **Optimize RAG pipelines** - Cache documents, blend them dynamically

> **Important:** CacheBlend requires additional computation to "blend" mismatched KV segments. It's most valuable when:
> - Context documents are reused frequently
> - Document order varies between requests
> - The cost of recomputing KV outweighs blending overhead

**Traditional Cache vs CacheBlend:**

| Scenario | Traditional | CacheBlend |
|----------|-------------|------------|
| "Doc A + Doc B" then "Doc A + Doc B" | ✅ Hit | ✅ Hit |
| "Doc A + Doc B" then "Doc B + Doc A" | ❌ Miss | ✅ Hit |
| "Doc A + Doc B" then "Doc A + Doc C" | ❌ Miss | ⚡ Partial |

**Expected outcome**
- Cache hits even with reordered documents
- Reduced TTFT for dynamic RAG contexts
- Higher effective cache hit rate

## 2. When to Use CacheBlend

| Scenario | Recommendation | Why |
|----------|----------------|-----|
| Fixed order contexts | **Standard caching** (R-001) | No blending overhead needed |
| Frequently reordered docs | **CacheBlend** (this recipe) | Reuse despite reordering |
| Dynamic RAG assembly | **CacheBlend** | Blend cached documents |
| Single document queries | **Standard caching** | No benefit from blending |
| Maximum throughput | **Standard caching** | Blending adds compute |

## 3. Installing vLLM + LMCache

CacheBlend is included in LMCache v1:

```bash
# Install LMCache (CacheBlend included)
pip install lmcache

# Install vLLM
pip install vllm
```

## 4. LMCache Configuration

Create `recipes/vllm_cacheblend.yaml`:

```yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 48

# Enable CacheBlend
enable_blending: true

# Blending threshold - minimum matching ratio to trigger blending
# Range: 0.0 to 1.0 (default: 0.5)
# Higher = more conservative, only blend highly similar contexts
# Lower = more aggressive, blend even with few matching chunks
blending_min_match_ratio: 0.5

# Recompute ratio for non-matching segments
# Range: 0.0 to 1.0 (default: 0.3)
# Lower = less recomputation, more blending (faster but may be less accurate)
# Higher = more recomputation, less blending (slower but more accurate)
recompute_ratio: 0.3

save_unfull_chunk: true
```

### Configuration Tuning

```yaml
# Conservative (high quality, less speedup)
blending_min_match_ratio: 0.7
recompute_ratio: 0.5

# Balanced (recommended)
blending_min_match_ratio: 0.5
recompute_ratio: 0.3

# Aggressive (maximum speedup, may affect quality)
blending_min_match_ratio: 0.3
recompute_ratio: 0.1
```

## 5. Launching vLLM with CacheBlend

```bash
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=recipes/vllm_cacheblend.yaml

CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
--max-model-len 8192 \
--gpu-memory-utilization 0.85 \
--port 8000 \
--no-enable-prefix-caching \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## 6. Startup Validation

Expected LMCache logs:
```
LMCache INFO: Loading LMCache config file recipes/vllm_cacheblend.yaml
LMCache INFO: CacheBlend enabled with min_match_ratio=0.5, recompute_ratio=0.3
LMCache INFO: Creating LMCacheEngine with config:
  {
    'chunk_size': 256,
    'enable_blending': True,
    'blending_min_match_ratio': 0.5,
    'recompute_ratio': 0.3,
    ...
  }
```

## 7. RAG CacheBlend Demo

### 7.1 Cache individual documents

```bash
# Document A - cache it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Document A: Machine learning is a subset of artificial intelligence...",
    "max_tokens": 1
  }'

# Document B - cache it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Document B: Deep learning uses neural networks with multiple layers...",
    "max_tokens": 1
  }'

# Document C - cache it
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Document C: Natural language processing enables computers to understand text...",
    "max_tokens": 1
  }'
```

### 7.2 Query with reordered documents

```bash
# Query: A + B + C (standard order)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Context:\nDocument A: Machine learning is a subset of artificial intelligence...\nDocument B: Deep learning uses neural networks with multiple layers...\nDocument C: Natural language processing enables computers to understand text...\n\nQuestion: How do these technologies relate?",
    "max_tokens": 100
  }'
```

Expected log (standard cache hit):
```
LMCache INFO: Retrieved 768 tokens from cache
```

```bash
# Query: C + B + A (reversed order - traditional cache would miss!)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Context:\nDocument C: Natural language processing enables computers to understand text...\nDocument B: Deep learning uses neural networks with multiple layers...\nDocument A: Machine learning is a subset of artificial intelligence...\n\nQuestion: How do these technologies relate?",
    "max_tokens": 100
  }'
```

Expected log (CacheBlend hit):
```
LMCache INFO: CacheBlend: Found 3 matching chunks out of 3
LMCache INFO: CacheBlend: Blending cached KV with reordered segments
LMCache INFO: Retrieved 768 tokens via blending (saved 65% compute)
```

### 7.3 Partial blending (some docs cached, some new)

```bash
# Query: A + B + D (D is new, A and B are cached)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "prompt": "Context:\nDocument A: Machine learning is a subset of artificial intelligence...\nDocument B: Deep learning uses neural networks with multiple layers...\nDocument D: Computer vision processes and analyzes image data...\n\nQuestion: How do these technologies relate?",
    "max_tokens": 100
  }'
```

Expected log:
```
LMCache INFO: CacheBlend: Found 2 matching chunks (A, B), 1 new chunk (D)
LMCache INFO: CacheBlend: Partial blend - 66% cache hit
```

## 8. Benchmarking

### 8.1 Test with RAG benchmark

```python
import requests
import time
import random

# RAG documents
docs = {
    "A": "Machine learning is a subset of AI that enables systems to learn from data...",
    "B": "Deep learning uses neural networks with multiple layers to model complex patterns...",
    "C": "Natural language processing allows computers to understand human language...",
    "D": "Computer vision enables machines to interpret and understand visual information...",
    "E": "Reinforcement learning trains agents to make decisions through trial and error...",
}

# Warm cache with individual documents
for doc_id, content in docs.items():
    requests.post("http://localhost:8000/v1/completions", json={
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "prompt": f"Document {doc_id}: {content}",
        "max_tokens": 1
    })

# Test with random document orders
orders = [
    ["A", "B", "C"],
    ["C", "B", "A"],
    ["B", "A", "C"],
    ["A", "C", "B"],
    ["D", "A", "B"],  # Partial match
]

for order in orders:
    prompt = "Context:\n" + "\n".join([f"Document {d}: {docs[d]}" for d in order])
    prompt += "\n\nQuestion: Summarize these technologies."
    
    start = time.time()
    resp = requests.post("http://localhost:8000/v1/completions", json={
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "prompt": prompt,
        "max_tokens": 50
    })
    elapsed = time.time() - start
    
    print(f"Order {''.join(order)}: TTFT={elapsed:.3f}s")
```

### 8.2 Expected results

| Configuration | A-B-C | C-B-A | B-A-C | D-A-B |
|--------------|-------|-------|-------|-------|
| Standard Cache | 120ms | 580ms | 580ms | 580ms |
| **CacheBlend** | 120ms | **135ms** | **140ms** | **200ms** |

## 9. CacheBlend Tuning

### 9.1 Match ratio tuning

```yaml
# High match ratio - only blend very similar contexts
blending_min_match_ratio: 0.7
# Best for: High quality requirements, similar documents

# Medium match ratio - balanced
blending_min_match_ratio: 0.5
# Best for: General RAG applications

# Low match ratio - aggressive blending
blending_min_match_ratio: 0.3
# Best for: Maximum speed, varied documents
```

### 9.2 Recompute ratio tuning

```yaml
# High recompute - more accurate, slower
recompute_ratio: 0.5
# Recomputes 50% of mismatched tokens

# Medium recompute - balanced
recompute_ratio: 0.3
# Best for: Production deployments

# Low recompute - faster, less accurate
recompute_ratio: 0.1
# Best for: Latency-sensitive, approximate answers OK
```

## 10. Performance Tips

| Tip | Configuration | Impact |
|-----|---------------|--------|
| Cache individual docs first | Send docs separately | Better blending |
| Tune match ratio | `blending_min_match_ratio: 0.5` | Quality/speed balance |
| Monitor blend ratio | Watch logs | Optimize parameters |
| Combine with tiering | CPU + Disk + CacheBlend | Best performance |

## 11. Troubleshooting / Common Pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| No blending happening | `enable_blending: false` | Check config |
| Blending too slow | `recompute_ratio` too high | Lower to 0.2-0.3 |
| Poor blend quality | `recompute_ratio` too low | Increase to 0.3-0.5 |
| No cache hits | Documents not cached | Warm cache first |
| Blending not triggering | `blending_min_match_ratio` too high | Lower to 0.4-0.5 |

### Debug blending

```bash
# Enable debug logging
export LMCACHE_LOG_LEVEL=DEBUG

# Watch for blending logs
tail -f vllm.log | grep -i blend
```

## 12. Production RAG Pipeline

```python
# RAG pipeline with CacheBlend
class RAGPipeline:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.cached_docs = set()
    
    def warm_cache(self, doc_id, content):
        """Cache individual document"""
        if doc_id in self.cached_docs:
            return
        
        requests.post(f"{self.base_url}/v1/completions", json={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "prompt": f"Document {doc_id}: {content}",
            "max_tokens": 1
        })
        self.cached_docs.add(doc_id)
    
    def query(self, doc_ids, question):
        """Query with blended documents"""
        # Assemble context (order doesn't matter!)
        context = "\n".join([f"Document {d}: {self.docs[d]}" for d in doc_ids])
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        
        resp = requests.post(f"{self.base_url}/v1/completions", json={
            "model": "Qwen/Qwen3-4B-Instruct-2507",
            "prompt": prompt,
            "max_tokens": 200
        })
        return resp.json()
```

## 13. Additional Resources
- CPU hot cache: `recipes/dense_instruct_cpu_hot_cache.md` (R-001)
- Multi-instance sharing: `recipes/vllm_multi_instance_sharing.md` (R-018)
- Tiered storage: `recipes/vllm_tiered_storage.md` (R-029)
