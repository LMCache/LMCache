KV Cache SDK
====================

The LMCache **SDK** lets you retrieve a request's KV cache from a LMCache
server, transform it on the CPU, and store it back. This can be used for KV
cache transformations, such as token dropping. In the example: we prefill a
batch of long prompts, drop half of each request's KV chunks, and show the
decode-throughput gain. The SDK API is meant for offline setup.
The full runnable notebook lives at `examples/token_dropping
<https://github.com/LMCache/LMCache/tree/dev/examples/token_dropping>`_.

.. contents::
   :local:
   :depth: 2


Why KV Cache SDK
----------------

- **Improving Decode Throughput** when shrinking KV cache using token dropping.
  Token dropping reduces the KV cache size, allowing more requests to fit in a
  batch, improving decode throughput.
  Doing so will affect the accuracy minimally, as we demonstrated with the
  SnapKV example in the `examples/token_dropping/snapkv_token_dropping.ipynb
  <https://github.com/LMCache/LMCache/tree/dev/examples/token_dropping/snapkv_token_dropping.ipynb>`_.

The SDK gives you the hooks to retrieve a request's KV and other intermediate
tensors (currently only query tensors), supply your function to edit the KV,
and store the edited KV back. The SDK also provides a batched-stream API to 
prefill, modify, and store the cache back before decoding continues.

Since many token dropping algorithms rely on the intermediate tensors, we also
provided a flag to transfer the intermediate tensors from vLLM to LMCache.
Currently, the SDK only supports transferring query intermediate tensors.


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

The SDK is a separate process from vLLM and the LMCache server. It retrieves
and stores KV cache and query tensors through the LMCache server over the
LMCache-driven transfer path. That path needs one side of each copy in GPU
memory, so the SDK borrows free blocks of the QRingBuffer, the GPU ring in
which vLLM captures query tensors, as its staging buffer: the server copies
cached chunks' L1 objects into the ring as raw bytes, and the SDK decodes them
into contiguous CPU tensors (KV at TP 1 only; query tensors at any TP). This
needs:

- the opt-in ``qstore`` server module, loaded with ``--server-module``. It
  requires the LMCache-driven transfer path (``--supported-transfer-mode
  lmcache_driven`` or ``auto``) and advertises the ``transfer_query``
  experimental feature;
- vLLM with ``"lmcache.mp.transfer_intermediate_tensors": true``, which
  creates the Q ring;
- no requests scheduled by vLLM while the SDK transfers (the token-dropping
  workflow pauses every stream during ``modify``); a transfer that happens
  during vLLM prefill/decode will fail.

The SDK request client supports either ZMQ or gRPC. Its ``url`` scheme and the
vLLM connector endpoint must match the server's ``--transport`` setting.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 150 \
        --eviction-policy LRU \
        --chunk-size 256 \
        --port 6555 \
        --http-port 8080 \
        --supported-transfer-mode lmcache_driven \
        --server-module '{"module_path": "lmcache.v1.multiprocess.plugin_modules.qstore"}'

Then start vLLM with the LMCache MP connector and
``"lmcache.mp.transfer_intermediate_tensors": true`` in
``kv_connector_extra_config``.
By default, the QRingBuffer has the capacity to hold the query tensors of 2
forward passes (``"lmcache.q.ring_depth":2``), or an explicit number of blocks
with ``"lmcache.q.ring_blocks"``. The SDK transfers through the ring's free
blocks, so it needs at least one chunk's worth of them; a larger ring moves a
transfer in fewer batches.
Example:

.. code-block:: bash

    vllm serve Qwen/Qwen3-8B \
        --port 8000 \
        --enforce-eager \
        --gpu-memory-utilization 0.65 \
        --kv-transfer-config '{
            "kv_connector":"LMCacheMPConnector",
            "kv_role":"kv_both",
            "kv_connector_extra_config":{
                "lmcache.mp.transfer_intermediate_tensors": true,
                "lmcache.mp.port":6555,
                "lmcache.q.ring_depth":2
            }
        }' \
        --trust-remote-code \
        --return-tokens-as-token-ids

The SDK keys the KV cache by token ids: ``create_request`` takes the prompt as
token ids, and every ``post_completion`` must report a ``token_id`` for each
generated token. The example gets these ids straight from vLLM by passing
``--return-tokens-as-token-ids``. Otherwise, if vLLM returns only text, the
``post_completion`` must tokenize each generated token back into a token id.

Here's an example of creating a context and connecting to the LMCache server.
Each type of tensor (KV, query intermediate) has its own context.

.. code-block:: python

    import lmcache.sdk as lmc_sdk

    kv_ctx = lmc_sdk.kvcache.connect(
        url="tcp://localhost:6555",         # tcp:// for ZMQ; grpc:// for gRPC
        http_url="http://localhost:8080",   # must match --http-port
        model_name="Qwen/Qwen3-8B",
        timeout=60,
    )
    q_ctx = lmc_sdk.qcache.connect(
        url="tcp://localhost:6555",         # tcp:// for ZMQ; grpc:// for gRPC
        http_url="http://localhost:8080",   # must match --http-port
        model_name="Qwen/Qwen3-8B",
        timeout=60,
    )
    ...
    kv_ctx.close()
    q_ctx.close()


Writing a custom edit function
------------------------------

An edit function takes the Mapping[kind, retrieved tensor] and its token ids and returns the
edited ``(kv, tokens)``. ``batch.modify(fn)`` applies it for every requests. The
function should be implemented as if it's only for single request, and the SDK
will call it in parallel for every requests in the batch.

``modify`` operates only on the **chunk-aligned** prefix. A trailing partial
chunk is tracked by the SDK and re-sent on the next ``decode``, so
``tokens`` arrives already truncated to the cached length.


API reference
-------------

The SDK lives under ``lmcache.sdk``. The examples above alias
``import lmcache.sdk as lmc_sdk`` (and ``lmcache.sdk.request`` / ``.batch`` as
``lmc_request`` / ``lmc_batch``).

Modules
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Module
     - Purpose
   * - ``lmcache.sdk``
     - Package entry point; ``connect(kind, ...)`` dispatcher and re-exports of
       the submodules below.
   * - ``lmcache.sdk.kvcache``
     - ``connect()`` for the **KV** cache.
   * - ``lmcache.sdk.qcache``
     - ``connect()`` for the **query (Q)** cache (model name is ``<model>##query``).
   * - ``lmcache.sdk.context``
     - The server-connection context plus the shared cache-kind enum and error type.
   * - ``lmcache.sdk.request``
     - Per-request streaming: ``create_request()`` and ``LMCacheRequestStream``.
   * - ``lmcache.sdk.batch``
     - Orchestrates many request streams together via ``LMCacheBatchedStream``.
   * - ``lmcache.sdk.wrapper.ring_transfer``
     - ``RingSDKTransferWrapper`` — the transfer helper the context holds,
       moving tensors through free blocks of vLLM's Q ring (used internally;
       exposed via ``ctx.transfer_ctx``).
   * - ``lmcache.sdk.wrapper.ring_mapping``
     - ``RingMapping`` — opens the Q ring's IPC handles and checks that vLLM
       took no ring blocks during a transfer.

Classes
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Purpose
   * - ``cache_kind.LMCacheSDKCacheKind``
     - Enum selecting the cache: ``KV`` or ``QUERY``.
   * - ``context.LMCacheSDKContext``
     - A connection to one LMCache server for one model + kind; returned by
       ``connect`` and passed to every other call.
   * - ``request.LMCacheRequestStream``
     - One request's lifecycle (prefill / decode / retrieve / modify).
   * - ``batch.LMCacheBatchedStream``
     - Runs a set of ``LMCacheRequestStream`` objects together and reports
       metrics.
   * - ``request.StreamPerfMetrics``
     - Throughput / latency report for a single ``generate()`` call; the batch
       keeps the latest one per request stream in ``batch.perf_metrics``.
   * - ``request.TokenEvent``
     - One generated-token event passed back through the request.
   * - ``request.PostCompletion``
     - Protocol you implement: a callable that submits a request to your engine.
   * - ``context.ModifyFnType``
     - Type alias for the edit function you pass to ``modify`` / ``modify_kv``:
       ``Callable[[Mapping[LMCacheSDKCacheKind, torch.Tensor], Sequence[int]],
       tuple[torch.Tensor, Sequence[int]]]``.
   * - ``lmcache.v1.common.metrics.Metrics``
     - Aggregated report returned by ``batch.prefill`` / ``modify`` / ``decode``
       (not an ``lmcache.sdk`` type). Emit it or convert it to a dict.
   * - ``context.LMCacheSDKError`` / ``request.LMCacheRequestStreamError`` /
       ``batch.LMCacheBatchedStreamError``
     - Error types raised by the SDK, streams, and batches respectively.

Functions and methods
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function / method
     - Description
   * - ``ctx = lmc_sdk.connect(kind, url, http_url, model_name, timeout=60.0)``
     - Connect and register caches for ``kind`` (dispatches to ``kvcache`` /
       ``qcache``). Returns an ``LMCacheSDKContext``.
   * - ``kv_ctx = lmc_sdk.kvcache.connect(url, http_url, model_name, timeout=60.0)``
     - Connect for the KV cache directly. Alternatively, use 
       ``connect(kind=LMCacheSDKCacheKind.KV, ...)``.
   * - ``q_ctx = lmc_sdk.qcache.connect(url, http_url, model_name, timeout=60.0)``
     - Connect for the query cache directly (model name is ``<model>##query``).
       Alternatively, use ``connect(kind=LMCacheSDKCacheKind.QUERY, ...)``.
   * - ``ctx.register_caches()``
     - Fetch the server-registered layout for this model + kind (called by
       ``connect``).
   * - ``ctx.retrieve(tokens, cache_salt="")``
     - Pull the cached tensor for a token sequence (``None`` on miss).
   * - ``ctx.store(kv, tokens, cache_salt="")``
     - Push a tensor of shape ``[2, L, T, D]`` back to the server for
       ``tokens`` (``len(tokens)`` must equal ``kv.shape[2]``). Returns ``False``
       if the server already had it cached.
   * - ``ctx.close()``
     - Release the context's resources when done.
   * - ``request = lmc_request.create_request(contexts, post_completion,
       prompt_token_ids, cache_salt="")``
     - Create one request stream bound to one or more contexts (e.g. KV and Q).
   * - ``batch = lmc_batch.LMCacheBatchedStream()``
     - Create an empty batch.
   * - ``batch.add(request)``
     - Register a request stream to the batch.
   * - ``batch.get_request_stream(stream_id)``
     - Fetch a registered request stream by its ``request_stream_id``. Raises
       ``LMCacheBatchedStreamError`` if the id is unknown.
   * - ``batch.prefill(sampling_params)``
     - Prefill every request stream once (``max_tokens`` forced to 1). Returns
       ``Metrics``.
   * - ``batch.modify(fn)``
     - Apply the edit function ``fn`` to every request stream's cached KV. Returns
       ``Metrics``.
   * - ``batch.decode(sampling_params)``
     - Decode every request stream. Returns ``Metrics``.
   * - ``metrics.emit()``
     - Print the ``Metrics`` report through its registered handlers (a
       terminal table by default).
   * - ``metrics.to_dict()``
     - Return the ``Metrics`` report as a JSON-serialisable dict, keyed by
       ``"title"`` and ``"metrics"``.

``Metrics`` reports ``input_tokens`` / ``input_tput`` for prefill,
``duration`` for modify, and ``output_tokens`` / ``output_tput`` for decode.

Attributes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Description
   * - ``batch.request_streams``
     - ``dict`` of every registered ``LMCacheRequestStream``, keyed by
       ``request_stream_id``. Iterate it to read each request's
       ``output_text`` / ``output_tokens`` after decode.
   * - ``batch.perf_metrics``
     - ``dict`` of the latest ``StreamPerfMetrics`` per ``request_stream_id``.
       Cleared and repopulated by each ``prefill`` / ``decode``.
