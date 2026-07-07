KV Cache SDK
====================

The LMCache **SDK** lets you retrieve a request's KV cache from a LMCache
server, transform it on the CPU, and store it back. This can be used for KV
cache transformations, such as token dropping. In the example: we prefill a
batch of long prompts, drop half of each request's KV chunks, and show the
decode-throughput gain. The full runnable notebook lives at
`examples/token_dropping/token_dropping.ipynb
<https://github.com/LMCache/LMCache/tree/dev/examples/token_dropping>`_.

.. contents::
   :local:
   :depth: 2


Why KV Cache SDK
----------------

- **Improving Decode Throughput** when shrinking KV cache using token dropping.
  Token dropping reduces the KV cache size, allowing more requests to fit in a
  batch, improving decode throughput.

The SDK gives you the hooks to retrieve a request's KV, supply your own 
function to edit the KV, and store the edited KV back. The SDK also provides a
batched-stream API to prefill, modify, and store the cache back before
decoding continues.


How it works
------------

A request flows through three phases on the batched-stream API:

- **prefill** — run each prompt through vLLM once (``max_tokens=1``); vLLM
  computes the KV cache and stores it in LMCache.
- **modify** — the SDK retrieves the cached KV to CPU, hands it to your edit
  function, and stores the result back.
- **decode** — continue generation against the smaller, edited cache.

The SDK runs on **CPU** and hands you KV tensors in ``HND`` order with shape
``[2, L, T, D]`` (K/V, layers, chunk-aligned tokens, ``num_kv_heads * head_dim``).


Configuration
-------------

To start the LMCache server with shared-memory transfer enabled, pass
``--shm-name`` and disable lazy L1 allocation with ``--no-l1-use-lazy``. If
shared memory is unavailable and these flags are not specified, the SDK falls 
back to pickle.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 150 \
        --eviction-policy LRU \
        --chunk-size 256 \
        --port 6555 \
        --http-port 8080 \
        --shm-name lmcache_kvcache_sdk \
        --no-l1-use-lazy

Then start vLLM with the LMCache MP connector.

.. code-block:: bash

    vllm serve Qwen/Qwen3-8B \
        --port 8000 \
        --enforce-eager \
        --gpu-memory-utilization 0.65 \
        --kv-transfer-config '{
            "kv_connector":"LMCacheMPConnector",
            "kv_role":"kv_both",
            "kv_connector_extra_config":{"lmcache.mp.port":6555}
        }' \
        --trust-remote-code \
        --return-tokens-as-token-ids

The SDK keys the KV cache by token ids: ``create_request`` takes the prompt as
token ids, and every ``post_completion`` must report a ``token_id`` for each
generated token. The example gets these ids straight from vLLM by passing
``--return-tokens-as-token-ids``. Otherwise, if vLLM returns only text, the
``post_completion`` must tokenize each generated token back into a token id.

.. code-block:: python

    import lmcache.sdk.kvcache as lmc_sdk

    ctx = lmc_sdk.connect(
        url="tcp://localhost:6555",         # must match --port
        http_url="http://localhost:8080",   # must match --http-port
        model_name="Qwen/Qwen3-8B",
        timeout=60,
    )
    ...
    lmc_sdk.close(ctx)


Writing a custom edit function
------------------------------

An edit function takes the retrieved KV tensor and its token ids and returns the
edited ``(kv, tokens)``. ``batch.modify(fn)`` applies it to every stream.

``modify`` operates only on the **chunk-aligned** prefix. A trailing partial
chunk is tracked by the SDK and re-sent on the next ``decode``, so
``tokens`` arrives already truncated to the cached length.


API reference
-------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function / method
     - Description
   * - ``lmc_sdk.connect(url, http_url, model_name, timeout=60.0)``
     - Create an SDK context and register the transfer context, pass it to
       every other call.
   * - ``lmc_sdk.close(ctx)``
     - Close the context and release resources. Called when done with the SDK.
   * - ``lmc_stream.create_request(ctx, post_completion,
       prompt_token_ids, cache_salt="")``
     - Create one request stream to add to a batch.
   * - ``lmc_batch.LMCacheBatchedStream()``
     - Create an empty batch.
   * - ``batch.add(stream)``
     - Register a stream to the batch.
   * - ``batch.prefill(sampling_params)``
     - Prefill every stream once (``max_tokens`` forced to 1). Returns a
       ``Metrics`` report.
   * - ``batch.modify(fn)``
     - Apply the edit function ``fn`` to every stream's cached KV. Returns a
       ``Metrics`` report.
   * - ``batch.decode(sampling_params)``
     - Decode every stream. Returns a ``Metrics`` report.

```Metrics`` returns ``input_tokens``, ``input_tput`` for prefill, ``duration``
for modify, and ``output_tokens``, ``output_tput`` for decode.
