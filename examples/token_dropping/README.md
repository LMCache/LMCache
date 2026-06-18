# Token Dropping Example

End-to-end demo of remapping KV cache through LMCache using the 
[KV Cache SDK](../../docs/design/sdk/kvcache.md). 
A request's KV is **retrieved** from LMCache, **token-dropped / compressed**, 
and **stored** back.
This example uses CPU only.

## Topology

Four processes, each in its **own terminal**, started in this order:

| # | Process | Listens on | Talks to |
| --- | --- | --- | --- |
| 1 | LMCache MP server | MQ `6556`, HTTP `8081` | — |
| 2 | vLLM server (`LMCacheMPConnector`) | HTTP `8001` | LMCache MQ `6556` |
| 3 | Token-dropping app (`multi_req_split.py`) | HTTP `9000` | LMCache `8081`/SDK, vLLM `8001` |
| 4 | Load driver (`run_multi_req_split.py`) | — | app `9000` |

Ports `6556` (ZMQ message queue, used by the SDK + the connector) and `8081` (HTTP API)
are both exposed by the LMCache server; `mypool` is the shared-memory pool name the SDK
maps for the data plane.

---

## Terminal 1 — LMCache MP server

The standalone LMCache multiprocess server (L1 SHM pool + the SDK/connector endpoints).
Please adjust the L1 size.

```bash
lmcache server \
  --l1-size-gb 140 \
  --eviction-policy noop \
  --chunk-size 256 \
  --port 6556 \
  --http-port 8081 \
  --shm-name mypool \
  --no-l1-use-lazy
```

---

## Terminal 2 — vLLM server (LMCache MP connector)

Serves the model and offloads/loads KV through the LMCache server on MQ port `6556`.

```bash
env -u VLLM_PORT \
  CUDA_VISIBLE_DEVICES="" \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_BATCH_INVARIANT=1 \
  PYTHONHASHSEED=0 \
  vllm serve Qwen/Qwen3-8B \
  --port 8001 \
  --served-model-name Qwen/Qwen3-8B \
  --enforce-eager \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.5 \
  --kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://localhost",
      "lmcache.mp.port": 6556,
      "lmcache.mp.mq_timeout": 60
    }
  }'
```

---

## Terminal 3 — Token-dropping app

The FastAPI driver that uses the LMCache SDK to do prefill, then retrieve the KV cache
before dropping some tokens, remapping the positional embedding, and store the resulting 
edited KV. Point it at the LMCache HTTP endpoint and the vLLM server.

```bash
python multi_req_split.py \
  --model Qwen/Qwen3-8B \
  --vllm-model-name Qwen/Qwen3-8B \
  --lmcache-url http://localhost:8081 \
  --vllm-url http://localhost:8001 \
  --chunk-size 256 \
  --timeout 600 \
  --trust-remote-code \
  --app-host 0.0.0.0 \
  --app-port 9000 \
  --lmcache-mp-host tcp://localhost \
  --lmcache-mp-port 6556
```

---

## Terminal 4 — Load driver

Sends requests to the token dropping app.
Ignores the EOS, so all requests will be generating `--max-tokens` tokens.

```bash
python run_multi_req_split.py \
  --app-port 9000 \
  --num-requests 1 \
  --compression 0.0 \
  --repeat 1 \
  --separate-repeats \
  --sleep-between-repeats 0 \
  --prompt-repeats 227 \
  --max-tokens 128
```
