# Issue 3640 CPU vLLM reproduction

This note records the CPU-only reproduction setup for
[LMCache issue #3640](https://github.com/LMCache/LMCache/issues/3640).

The issue is specifically about vLLM chunked prefill using
`LMCacheMPConnector`. A faithful reproduction must exercise vLLM's scheduler
and vLLM's built-in connector resolution. A direct `lmcache bench server` run is
useful for checking the LMCache MP server, but it bypasses vLLM and cannot
reproduce this issue.

## Required connector path

Use the built-in vLLM connector:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "lmcache.mp.host": "tcp://localhost",
    "lmcache.mp.port": 5660,
    "lmcache.mp.mp_transfer_mode": "engine_driven"
  }
}
```

Do not set `kv_connector_module_path` for this reproduction. This form is not
faithful to the issue because it forces vLLM to import the connector from the
LMCache checkout:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector"
}
```

If the vLLM package is installed from `/home/hanqiu/vllm`, the built-in
connector should resolve to:

```text
/home/hanqiu/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py
```

If vLLM is installed from a CPU wheel, it should resolve to the same module
under that environment's `site-packages/vllm/` directory.

Verify the imported connector before running the workload:

```bash
.venv-vllm-cpu/bin/python - <<'PY'
import inspect

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector import (
    LMCacheMPConnector,
)

print(inspect.getfile(LMCacheMPConnector))
PY
```

The printed path must be a vLLM path, not:

```text
/home/hanqiu/LMCache/lmcache/integration/vllm/lmcache_mp_connector.py
```

## CPU vLLM environment

The normal PyPI CUDA vLLM wheel is not enough for this reproduction. Install a
CPU vLLM wheel or install a CPU build of the local vLLM checkout.

The official vLLM CPU documentation recommends `uv`, a CPU wheel index, explicit
CPU KV-cache space, OpenMP CPU binding, and `LD_PRELOAD` for the CPU runtime
libraries:

```text
https://docs.vllm.ai/en/stable/getting_started/installation/cpu/
```

Create a repo-local environment:

```bash
uv venv --python 3.12 --seed --managed-python .venv-vllm-cpu
```

Install the latest CPU vLLM wheel:

```bash
uv pip install \
  --python .venv-vllm-cpu/bin/python \
  vllm \
  --extra-index-url https://wheels.vllm.ai/nightly/cpu \
  --index-strategy first-index \
  --torch-backend cpu
```

To reproduce against a specific upstream vLLM commit, use the commit-specific
CPU wheel index instead:

```bash
VLLM_COMMIT=<full-vllm-commit>

uv pip install \
  --python .venv-vllm-cpu/bin/python \
  vllm \
  --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu \
  --index-strategy first-index \
  --torch-backend cpu
```

To reproduce against the local `/home/hanqiu/vllm` checkout instead of a wheel,
install that checkout in CPU mode inside the same environment:

```bash
cd /home/hanqiu/vllm

VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_VARIANT=cpu \
VLLM_TARGET_DEVICE=cpu \
uv pip install \
  --python /home/hanqiu/LMCache/.venv-vllm-cpu/bin/python \
  --editable .
```

Return to the LMCache checkout and install LMCache with CPU-only native
extensions:

```bash
cd /home/hanqiu/LMCache

NO_GPU_EXT=1 uv pip install \
  --python .venv-vllm-cpu/bin/python \
  -e . \
  --no-build-isolation
```

Find the CPU runtime libraries that should be preloaded for vLLM CPU:

```bash
TCMALLOC_PATH="$(
  find "$PWD/.venv-vllm-cpu" -name 'libtcmalloc_minimal.so.4' -print -quit
)"
IOMP_PATH="$(
  find "$PWD/.venv-vllm-cpu" -name 'libiomp5.so' -print -quit
)"

printf 'TCMALLOC_PATH=%s\n' "$TCMALLOC_PATH"
printf 'IOMP_PATH=%s\n' "$IOMP_PATH"
```

## Start LMCache MP server

Use `engine_driven` transfer mode and align the LMCache chunk size with the
issue parameters. The issue body used `chunk_size=16`; a later comment used
`chunk_size=128`. Start with `16` unless intentionally reproducing the later
comment.

```bash
LMCACHE_DISABLE_BANNER=1 NO_GPU_EXT=1 \
  .venv-vllm-cpu/bin/lmcache server \
  --port 5660 \
  --http-port 18090 \
  --chunk-size 16 \
  --max-workers 4 \
  --l1-size-gb 1 \
  --eviction-policy LRU \
  --supported-transfer-mode engine_driven
```

Health-check and reset metrics before the vLLM cold pass:

```bash
curl -s http://127.0.0.1:18090/healthcheck
curl -s -X POST http://127.0.0.1:18090/metrics/reset
```

## Start CPU vLLM with built-in LMCacheMPConnector

This smoke-test command keeps the model and sequence lengths small enough for a
CPU-only machine while still exercising vLLM chunked prefill, the vLLM
connector registry, and the LMCache MP connector path.

```bash
LD_PRELOAD="${TCMALLOC_PATH}:${IOMP_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}" \
VLLM_CPU_KVCACHE_SPACE=1 \
VLLM_CPU_OMP_THREADS_BIND=0-2 \
LMCACHE_DISABLE_BANNER=1 \
LMCACHE_MP_TRANSFER_MODE=engine_driven \
NO_GPU_EXT=1 \
  .venv-vllm-cpu/bin/vllm serve facebook/opt-125m \
  --host 127.0.0.1 \
  --port 8008 \
  --dtype=float32 \
  --load-format dummy \
  --max-model-len 512 \
  --block-size 16 \
  --max-num-batched-tokens 64 \
  --max-num-seqs 2 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":5660,"lmcache.mp.mp_transfer_mode":"engine_driven"}}'
```

The important detail is that `--kv-transfer-config` contains
`"kv_connector":"LMCacheMPConnector"` but does not contain
`"kv_connector_module_path"`.

Confirm the startup logs show CPU execution, for example:

```text
device_config=cpu
```

Also confirm that LMCache registers a non-CUDA vLLM worker and that the
transfer context is CPU `engine_driven`.

## Smoke workload

Send one cold request followed by two warm requests with the same prompt:

```bash
PROMPT="$(
  .venv-vllm-cpu/bin/python - <<'PY'
print(" ".join(f"lmcache-token-{i % 64}" for i in range(320)))
PY
)"

REQUEST_BODY="$(
  PROMPT="$PROMPT" .venv-vllm-cpu/bin/python - <<'PY'
import json
import os

print(json.dumps({
    "model": "facebook/opt-125m",
    "prompt": os.environ["PROMPT"],
    "max_tokens": 1,
    "temperature": 0,
}))
PY
)"

for run in 1 2 3; do
  curl -s http://127.0.0.1:8008/v1/completions \
    -H 'Content-Type: application/json' \
    -d "$REQUEST_BODY" > /tmp/issue-3640-vllm-cpu-run-${run}.json
done
```

Check LMCache metrics:

```bash
curl -s http://127.0.0.1:18090/metrics | \
  rg 'lmcache_mp_l1_write_chunks_total|lmcache_mp_l1_read_chunks_total|lmcache_mp_lookup_requested_tokens_total|lmcache_mp_lookup_hit_tokens_total'
```

Interpretation:

- Cold pass: writes all complete prompt chunks.
- Warm pass 1: reads and hits the same complete chunks.
- Warm pass 2: reads and hits the same complete chunks again.

For example, with a 321-token prompt and `chunk_size=16`, only 320
chunk-aligned tokens count. Three identical successful requests should produce
one cold miss and two warm hits:

```text
cold:        0 / 320 hit tokens, 20 chunks written
warm pass 1: 320 / 320 hit tokens, 20 chunks read
warm pass 2: 320 / 320 hit tokens, 20 chunks read
total:       640 / 960 hit tokens
```

That `640 / 960` result is expected for a healthy smoke test. It is not the
issue signature.

## Issue-sized workload

After the smoke test confirms that CPU vLLM can exercise the built-in connector,
rerun with issue-sized parameters.

Use the same LMCache server command but choose the target chunk size:

```text
chunk_size=16   # issue body
chunk_size=128  # later issue comment
```

Use vLLM parameters that preserve the original failure surface:

```text
--enable-chunked-prefill
--no-enable-prefix-caching
--max-num-batched-tokens 8192   # issue body
--max-num-batched-tokens 32768  # larger follow-up attempt
--max-num-seqs 32
--max-model-len <larger-than-the-prompt-token-count>
```

Then send the original long prompts twice:

1. Reset LMCache metrics.
2. Send all prompts once as the cold pass.
3. Send the same prompts again as the warm pass.
4. Compare expected complete chunks, actual writes, reads, and hit tokens.

Expected complete chunks per prompt:

```text
floor(tokenized_prompt_length / lmcache_chunk_size)
```

Expected total chunks for a cold pass:

```text
sum(floor(tokenized_prompt_length_i / lmcache_chunk_size) for each prompt)
```

The issue is reproduced only if:

```text
lmcache_mp_l1_write_chunks_total < expected complete chunks from the cold pass
```

and the warm pass misses the same missing chunks. A healthy result stores every
complete chunk on the cold pass and hits every complete chunk on the warm pass.

## Previous CPU-only checks

Earlier repo-local checks used `lmcache bench server` in CPU mode. Those runs
validated the lower-level LMCache MP server path:

```text
expected stored chunks: 45625
actual stored chunks: 45625
warm missed chunks: 0
```

Those checks did not use vLLM, did not instantiate `LMCacheMPConnector`, and did
not exercise the vLLM scheduler metadata path. They are therefore useful as a
server sanity check but are not a reproduction of issue #3640.

A later CPU vLLM smoke test used:

```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector"
}
```

That proved CPU vLLM can drive the LMCache MP path, but it forced the LMCache
checkout connector and was not faithful to the issue's built-in vLLM connector
resolution.

## Cleanup

Stop the CPU vLLM process and LMCache server when testing is complete. Then
check that no local server ports are still owned:

```bash
ss -ltnp | rg '(:5660|:18090|:8008)'
```
