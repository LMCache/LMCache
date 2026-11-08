KV Cache Events
===============

When LMCache runs with vLLM (`LMCacheConnectorV1`), vLLM can publish per-block KV cache events (`BlockStored`, `BlockRemoved`, `AllBlocksCleared`) over ZeroMQ using `--kv-events-config`.

Downstream routers and diagnostic tools can subscribe to this ZMQ stream to track which prefix blocks reside in GPU memory or were restored by LMCache.

Enabling ZMQ KV Cache Events
----------------------------

Pass `--kv-events-config` and `--enable-prompt-tokens-details` when starting vLLM with LMCache:

.. code-block:: bash

   LMCACHE_USE_EXPERIMENTAL=True \
   vllm serve <model> \
     --enable-prefix-caching \
     --enable-prompt-tokens-details \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
     --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

When running with data parallelism (`--data-parallel-size > 1`), vLLM offsets the ZMQ port for each data-parallel rank starting from the base port in `endpoint`.

Attribute Cache Misses and Offload Hits with Prefixwise
-------------------------------------------------------

When ``cached_tokens`` drops on a multi-turn agent workload, the cause is either a prompt change in the client or a KV eviction in the engine. Also, because vLLM includes LMCache restores in ``cached_tokens`` (``PrefillStats``), standard response counters do not separate GPU HBM hits from LMCache restores.

You can connect `Prefixwise <https://github.com/tejachavali/prefixwise>`_ (MIT) to the ZMQ KV-event stream. Prefixwise stores block hashes and token counts, never prompt text.

.. code-block:: bash

   pip install "prefixwise[engines]"

   prefixwise serve \
     --upstream http://127.0.0.1:8000 \
     --engine vllm \
     --kv-events tcp://127.0.0.1:5557

Send requests to ``http://127.0.0.1:8080/v1``, then run ``prefixwise report --detail``:

* **If the prefix stayed in GPU HBM:** The verdict is ``none``.
* **If GPU HBM evicted the prefix and LMCache restored it:** The verdict is ``none`` with detail ``host_restored``.
* **If both GPU HBM and LMCache evicted the prefix:** The verdict is ``evicted, confirmed``.
* **If the client changed the prompt:** The verdict names the first changed part (``volatile_system``, ``tool_reserialize``, ``history_edit``, ``tool_result_mutation``).

For multi-replica setups and benchmark records, see `docs/LMCACHE.md <https://github.com/tejachavali/prefixwise/blob/main/docs/LMCACHE.md>`_.
