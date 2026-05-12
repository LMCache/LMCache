# KV Cache SDK Examples

These examples show how to use the Python SDK to store, look up, and retrieve
KV cache tensors through the LMCache MP HTTP API. The SDK path is memory-first:
applications can retrieve a tensor package, edit its metadata, and store it
again without writing the tensor through a local storage format.

## End-to-end vLLM flow

`run_e2e_kv_edit.sh` starts an LMCache MP server and vLLM, then runs an
end-to-end KV remapping experiment:

1. Send a source prompt to vLLM so the normal connector stores KV in LMCache.
2. Retrieve the source KV cache into an in-memory `KVCachePackage` with
   `lmcache.sdk.retrieve`.
3. Build a target token-ID prompt with different synthetic leading tokens, the
   same length as the source prompt, and identical cache-covered trailing
   tokens.
4. Store the source KV under the target prefix with `lmcache.sdk.store`.
5. Send the target token IDs to vLLM so the target prefix hits the remapped KV.
6. Print lookup counts, hit counts, latencies, response previews, and whether
   the source and target outputs match.

The target prompt starts with different token IDs, so it does not rely on a
serving-engine local prefix match. Because the prompts have identical trailing
tokens after the remapped prefix, the target request reconstructs the same final
context as the source request and should produce the same deterministic output.

Run:

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
examples/kvcache_sdk/run_e2e_kv_edit.sh
```

Useful overrides:

```bash
MODEL=/path/to/model \
TOKENIZER=/path/to/tokenizer \
GPU_DEVICE=0 \
LMCACHE_PORT=6555 \
LMCACHE_HTTP_PORT=8080 \
VLLM_PORT=8000 \
CHUNK_SIZE=256 \
MIN_PROMPT_TOKENS=512 \
FAKE_PREFIX_TOKENS=32 \
MAX_TOKENS=32 \
VLLM_BATCH_INVARIANT=1 \
examples/kvcache_sdk/run_e2e_kv_edit.sh
```

Logs are written under `/tmp/lmcache_kvcache_sdk_e2e` by default. Override with
`TMP_DIR=...`.

The core SDK pattern used by the end-to-end example is:

```python
import lmcache.sdk as lmc_sdk

result = lmc_sdk.retrieve(
    "http://localhost:8080",
    model_name="...",
    tokens=source_tokens,
)

lmc_sdk.store(
    result.package,
    "http://localhost:8080",
    tokens=target_tokens,
)
```

## Memory-first standalone flow

The standalone example can also generate a toy tensor in memory and store it
without writing a KV package file. Set the shape fields to match `/api/status`
for your server:

```bash
python examples/kvcache_sdk/store_retrieve.py store-generated \
  --url http://localhost:8080 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --chunk-size 256 \
  --num-chunks 1 \
  --num-layers 32 \
  --hidden-dim 4096 \
  --dtype bfloat16
```

Probe the cached prefix:

```bash
python examples/kvcache_sdk/store_retrieve.py lookup \
  --url http://localhost:8080 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --num-tokens 256
```

Retrieve the hit prefix into memory and print only metadata about the returned
tensor. The SDK intentionally avoids file helpers; applications that want
durability can serialize their own `KVCachePackage` fields outside
`lmcache.sdk`.

```bash
python examples/kvcache_sdk/store_retrieve.py retrieve \
  --url http://localhost:8080 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --num-tokens 256
```

### Requirements

- An LMCache MP server running with HTTP enabled.
- A model already registered with that server. Check `/api/status` for the
  registered `model_name`, `chunk_size`, layer count, dtype, and hidden dim.
- A homogeneous `KV_2LTD` layout. Hybrid attention is rejected by the v1 API.

Use `--cache-salt` on all commands when storing and retrieving from a
non-default namespace.
