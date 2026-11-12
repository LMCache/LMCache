KV Cache Events
===============

When vLLM runs with LMCache in multiprocess mode (``LMCacheMPConnector``), vLLM can publish
KV cache events (``BlockStored``, ``BlockRemoved``, ``AllBlocksCleared``) over ZeroMQ using
``--kv-events-config``. The stream carries vLLM's own GPU block events and, converted by
``LMCacheMPConnector``, the LMCache server's CPU-tier events (``medium`` is ``CPU``; LMCache's
events are chunk-sized and keyed by LMCache chunk hashes).

Downstream routers and diagnostic tools can subscribe to this ZMQ stream to track which prefix
blocks reside in GPU memory and whether LMCache holds a copy.

Enabling ZMQ KV Cache Events
----------------------------

Start the LMCache server:

.. code-block:: bash

   lmcache server --l1-size-gb 20 --eviction-policy LRU

Then start vLLM with the LMCache MP connector, KV events, and per-request cache counts:

.. code-block:: bash

   vllm serve <model> \
     --enable-prefix-caching \
     --enable-prompt-tokens-details \
     --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
     --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

With data parallelism (``--data-parallel-size > 1``), vLLM offsets the ZMQ port for each
data-parallel rank from the base port in ``endpoint``. LMCache's events are published when vLLM
drains them from the connector, so they can arrive a step after the store they describe. The
LMCache server's events come from its event log; ``--disable-observability`` on ``lmcache server``
turns them off.

Attribute Cache Misses and Offload Hits with Prefixwise
-------------------------------------------------------

When ``cached_tokens`` drops on a multi-turn agent workload, the cause is either a prompt change
in the client or a KV eviction in the engine. And because vLLM counts LMCache restores in
``cached_tokens``, response counters alone do not separate GPU hits from LMCache restores.

`Prefixwise <https://github.com/tejachavali/prefixwise>`_ (MIT) can subscribe to the ZMQ
KV-event stream. It stores block hashes and token counts, never prompt text.

.. code-block:: bash

   pip install "prefixwise[engines]"

   prefixwise serve \
     --upstream http://127.0.0.1:8000 \
     --engine vllm \
     --kv-events tcp://127.0.0.1:5557

Send requests to ``http://127.0.0.1:8080/v1``, then run ``prefixwise report --detail``:

* **The prefix stayed in GPU memory:** the verdict is ``none``.
* **GPU memory evicted the prefix and LMCache restored it:** ``none``, with detail
  ``host_restored``. This is claimed only once the stream has shown LMCache's own CPU events.
* **GPU memory evicted the prefix and LMCache did not restore it:** ``evicted``, confirmed from
  the events.
* **The client changed the prompt:** the verdict names the first changed part
  (``volatile_system``, ``tool_reserialize``, ``history_edit``, ``tool_result_mutation``).

Measured with this setup (``lmcache server`` and ``LMCacheMPConnector``; LMCache 0.5.7, vLLM 0.33.0,
Qwen3-4B, GPU cache flooded 3x): every prefix LMCache restored was a hit with ``host_restored``,
5 of 5 (`record <https://github.com/tejachavali/prefixwise/blob/main/experiments/e_evict/runs/2026-11-08-colab-l4-host-restored.md>`_).
For multi-replica setups, see
`docs/LMCACHE.md <https://github.com/tejachavali/prefixwise/blob/main/docs/LMCACHE.md>`_.
