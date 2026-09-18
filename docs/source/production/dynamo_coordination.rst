Dynamo Integration
==================

`NVIDIA Dynamo <https://github.com/ai-dynamo/dynamo>`_ routes each request to
the worker that already holds the longest matching KV prefix. Its KV-aware
router learns what every worker holds from vLLM's KV event stream. With
LMCache MP mode, a worker also serves prefixes from the MP server's host
cache (L1) and L2 storage, so LMCache reports those placements through the
same stream:

- ``BlockStored`` with ``medium="CPU"`` when a chunk lands in the host cache
  (the worker's own completed stores, and stores by other engines sharing
  the MP server), and with ``medium="STORAGE"`` when it lands in L2;
- ``BlockRemoved`` with the same media when the host cache evicts a chunk or
  L2 deletes it.

Dynamo keeps a separate index tier per medium (``CPU`` is its
``HostPinned`` tier, ``STORAGE`` its ``Disk`` tier), so LMCache events never
disturb its view of the GPU cache, and a request whose prefix is only in one
worker's host cache is routed there.

.. contents::
   :local:
   :depth: 2

How it works
------------

.. code-block:: text

   MP server                           vLLM worker                     Dynamo frontend
   ---------                           -----------                     ---------------
   storage layer ─► event bus ─► log   connector polls the log ─► BlockStored / BlockRemoved ─► ZMQ ─► KV router
   (L1/L2 key events)   (POLL_KV_EVENTS, one non-blocking poll per step)

1. The MP server records the storage layer's key events (host-cache store
   completions and evictions, L2 stores and deletes) in a bounded,
   sequenced log.
2. Each vLLM worker's LMCache connector polls that log without blocking the
   model-runner step and translates the records for the worker: a removal is
   published only for chunks the worker had announced, and a store only once.
   While the log is polled, the server's write-finished records are the only
   source of store announcements: they name exactly the chunks written,
   whereas a store's completion only says it finished without a fatal error
   (chunks the server could not reserve are skipped).
3. vLLM's KV event publisher sends the events on the same ZMQ stream as its
   GPU block events; Dynamo applies them to the worker's host and disk tiers.

If the log cannot be followed exactly (the MP server restarted, or events
were discarded because a worker fell behind), the worker withdraws every
placement it announced and re-announces chunks as they are stored again, so
the router is never left with stale host-cache entries.

Requirements
------------

- Enable vLLM KV events (``--kv-events-config``) with the LMCache MP
  connector.
- Use the same block size for vLLM, Dynamo (``--kv-cache-block-size``), and
  the LMCache chunk size (``--chunk-size``): a chunk becomes one event
  block.
- Set ``lmcache.mp.hash_algorithm`` in the connector's extra config to the
  MP server's ``--hash-algorithm`` (for example ``sha256_cbor``, which vLLM's
  ``--prefix-caching-hash-algo sha256_cbor`` also uses), and give the MP
  server, the coordinator, and the workers the same ``PYTHONHASHSEED``.
- Keep the MP server's observability event bus on (it is on by default;
  ``--disable-observability`` also disables the KV event channel).

Example
-------

MP server (one per node; ``--kv-event-log-size`` defaults to 32768 records,
0 disables the channel):

.. code-block:: bash

   lmcache server \
     --host 0.0.0.0 --port 6555 --http-port 8100 \
     --chunk-size 16 --hash-algorithm sha256_cbor \
     --l1-size-gb 4 --eviction-policy LRU

vLLM worker under Dynamo:

.. code-block:: bash

   python -m dynamo.vllm \
     --model Qwen/Qwen2.5-7B-Instruct --block-size 16 \
     --enable-prefix-caching --prefix-caching-hash-algo sha256_cbor \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557","enable_kv_cache_events":true}' \
     --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":6555,"lmcache.mp.hash_algorithm":"sha256_cbor"}}'

Dynamo frontend with the KV router weighing host-cache hits:

.. code-block:: bash

   python -m dynamo.frontend \
     --router-mode kv --router-kv-events \
     --router-host-cache-hit-weight 1.0 \
     --kv-cache-block-size 16

Connector settings (``kv_connector_extra_config``):

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Key
     - Default
     - Meaning
   * - ``lmcache.mp.kv_event_poll_interval``
     - ``0.1``
     - Seconds between polls of the MP server's event log. ``0`` disables
       polling: the worker then publishes its own completed stores (which
       may include chunks the server skipped) and evictions never reach the
       router.
   * - ``lmcache.mp.kv_event_poll_max_events``
     - ``1024``
     - Records fetched per poll; a full page is followed by an immediate
       poll.
   * - ``lmcache.mp.hash_algorithm``
     - ``blake3``
     - Must match the MP server's ``--hash-algorithm``.

Verifying
---------

- The MP server's ``/status`` endpoint reports ``kv_events`` (``enabled``,
  ``log_depth``, ``lost_markers``, ``unbound_stores``).
- The worker's Prometheus registry exports
  ``vllm:lmcache_mp_kv_events_generated_total``,
  ``vllm:lmcache_mp_kv_events_drained_total``,
  ``vllm:lmcache_mp_kv_events_buffered``,
  ``vllm:lmcache_mp_kv_event_polls_total``, and
  ``vllm:lmcache_mp_kv_event_resyncs_total{reason}`` (a resync means the
  worker withdrew and will re-announce its placements).
- Dynamo's ``dynamo_component_kv_cache_events_applied`` counter shows
  ``event_type="stored"`` and ``event_type="removed"`` with
  ``status="ok"``; any other status means an event the router could not
  apply.
- Subscribing to the worker's ``kv-events`` ZMQ topic (see vLLM's
  ``kv_events_subscriber`` example) shows ``BlockStored`` and
  ``BlockRemoved`` events with ``medium='CPU'`` next to the GPU ones.

Limitations
-----------

- Access (LRU touch) events are not published: vLLM's event vocabulary has
  no such event.
- Requests with a ``cache_salt`` hash differently in vLLM and in LMCache, so
  their host-cache placements do not match the router's lookups.
- A shared L2 backend is reported as each worker's ``STORAGE`` tier rather
  than as a shared pool.
- A store whose token binding the server no longer holds (for example an L2
  prefetch of a chunk stored long ago) is not announced; the server counts
  it in ``unbound_stores``.
