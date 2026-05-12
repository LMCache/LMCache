# KV Cache SDK Examples

These examples show how to use the Python SDK to store, look up, and retrieve
KV cache packages through the LMCache MP HTTP API.

## End-to-end vLLM flow

`run_e2e_kv_edit.sh` starts an LMCache MP server and vLLM, then runs an
end-to-end KV remapping experiment:

1. Send a source prompt to vLLM so the normal connector stores KV in LMCache.
2. Retrieve the source KV cache with `lmcache.sdk.retrieve`.
3. Store that same KV cache under a different target token prefix with
   `lmcache.sdk.store`.
4. Send the target prompt to vLLM so the target token IDs hit the remapped KV.
5. Print lookup counts, hit counts, latencies, and response previews.

This intentionally associates KV from one prompt with different token IDs. It
is meant to demonstrate the SDK and editing workflow, not to produce a
semantically correct generation.

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
MAX_TOKENS=32 \
examples/kvcache_sdk/run_e2e_kv_edit.sh
```

Logs and the retrieved KV package are written under
`/tmp/lmcache_kvcache_sdk_e2e` by default. Override with `TMP_DIR=...`.

## SDK-only file flow

The example script can create a small `.pt` package with the metadata expected
by `lmcache.sdk.store`, then call the public SDK helpers:

```python
import lmcache.sdk as lmc_sdk

lmc_sdk.store("kv.pt", "http://localhost:8080")
lmc_sdk.lookup("http://localhost:8080", model_name="...", tokens=[...])
lmc_sdk.retrieve("kv-hit.pt", "http://localhost:8080", model_name="...", tokens=[...])
```

### Requirements

- An LMCache MP server running with HTTP enabled.
- A model already registered with that server. Check `/api/status` for the
  registered `model_name`, `chunk_size`, layer count, dtype, and hidden dim.
- A homogeneous `KV_2LTD` layout. Hybrid attention is rejected by the v1 API.

The toy package generated below is useful for checking the SDK path, but its
shape and dtype must match the registered server model.

### Run

Create a toy package. Set the shape fields to match `/api/status` for your
server:

```bash
python examples/kvcache_sdk/store_retrieve.py make-package \
  --output /tmp/lmcache-kv.pt \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --chunk-size 256 \
  --num-chunks 1 \
  --num-layers 32 \
  --hidden-dim 4096 \
  --dtype bfloat16
```

Store it:

```bash
python examples/kvcache_sdk/store_retrieve.py store \
  --url http://localhost:8080 \
  --input /tmp/lmcache-kv.pt
```

Probe the cached prefix:

```bash
python examples/kvcache_sdk/store_retrieve.py lookup \
  --url http://localhost:8080 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --num-tokens 256
```

Retrieve the hit prefix:

```bash
python examples/kvcache_sdk/store_retrieve.py retrieve \
  --url http://localhost:8080 \
  --output /tmp/lmcache-kv-hit.pt \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --num-tokens 256
```

Use `--cache-salt` on all commands when storing and retrieving from a
non-default namespace.
