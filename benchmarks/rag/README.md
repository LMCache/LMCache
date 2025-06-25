# RAG Workload Generator

Benchmarks RAG workloads on serving engines to measure throughput, TTFT, and quality.

## Quick Start

1. **Install dependencies:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install vllm lmcache # need version >v0.3.0
# This nightly install will work
# uv pip install -i https://test.pypi.org/simple/ lmcache==0.3.1.dev75 
```

2. **Start your serving engine:**
```bash
# vLLM
VLLM_USE_V1=1 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests --port 56789

# LMCache
LMCACHE_CONFIG_FILE="example_blending.yaml" \
VLLM_USE_V1=1 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests --port 45678 \
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

3. **Run benchmark:**
```bash
# Standard mode (vLLM, Ray Serve, etc.)
./rag_bench.sh standard --base-url http://localhost:56789/v1

# CacheBlend mode (LMCache with precomputation)
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1
```

## Modes

| Mode | Use Case | Request Count | Precomputation |
|------|----------|---------------|----------------|
| `standard` | vLLM, Ray Serve, etc. | Controlled by `--end-index` | None |
| `cacheblend` | LMCache with blending | Determined by `--kv-storage-size` | Automatic |

## Examples

```bash
# Basic usage
./rag_bench.sh standard --base-url http://localhost:56789/v1 # rag_bench.sh standard defaults to --baseline-name "standard"
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 # rag_bench.sh cacheblend defaults to default --baseline-name "cacheblend"

# Custom settings
./rag_bench.sh standard --base-url http://localhost:8000/v1 --baseline-name "rayserve" --qps 5.0 --end-index 100
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --kv-storage-size 100GB --qps 5.0

# Test CacheBlend without precomputation
./rag_bench.sh standard --base-url http://localhost:45678/v1 --end-index 100 --baseline-name "lmcache_no_precompute"

# Test without document shuffling (shuffling is enabled by default)
./rag_bench.sh cacheblend --base-url http://localhost:45678/v1 --no-shuffle-docs
```

## Key Parameters

- `--base-url`: **REQUIRED** - Server endpoint (e.g., `http://localhost:56789/v1`)
- `--baseline-name`: Output file suffix (default: mode name i.e. "standard" or "cacheblend")
- `--qps`: Queries per second (default: 3.5)
- `--end-index`: Request count for standard mode (default: 32)
- `--kv-storage-size`: Cache size for cacheblend mode (default: 30GB)
- `--kv-chunk-size`: Chunk size for cacheblend mode (default: 256)
- `--no-shuffle-docs`: Disable document shuffling (enabled by default to trigger CacheBlend)

## Output

Two files are generated:
- **CSV**: `{dataset}_{model}_{baseline}.csv` - Per-request details
- **JSON**: `{dataset}_{model}_{baseline}_summary.json` - Statistics (mean, median, p99)

Example: `musique_s_Mistral-7B-Instruct-v0.2_standard.csv`

## Important Notes

**CacheBlend Mode:**
- Request count is determined by `--kv-storage-size`, NOT `--end-index`
- Larger storage size = more requests processed
- Smaller chunk sizes may increase request count through better packing

**Standard Mode:**
- Request count controlled by `--end-index`
- Can be used on any OpenAI-compatible endpoint
- No precomputation overhead

## Help

```bash
./rag_bench.sh --help
```