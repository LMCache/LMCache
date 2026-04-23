Observability
=============

LMCache multiprocess mode provides three complementary observability modes:
**metrics** (Prometheus counters via OTel), **logging** (Python logging with
optional OTel log forwarding), and **tracing** (OTel spans for per-request
latency).

All three modes are powered by an internal **EventBus** that decouples
producers (L1Manager, StorageManager, MPCacheEngine) from subscribers.

.. contents::
   :local:
   :depth: 2

Quick Start
-----------

By default, **metrics** and **logging** are enabled; **tracing** is disabled.
No extra flags are needed:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU

To enable tracing, supply an OTLP endpoint:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --enable-tracing --otlp-endpoint http://localhost:4317

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--disable-observability``
     - off
     - Master switch: disable the EventBus entirely (no metrics, logging, or
       tracing subscribers are registered).
   * - ``--disable-metrics``
     - off
     - Skip metrics subscribers (Prometheus endpoint is not started).
   * - ``--disable-logging``
     - off
     - Skip logging subscribers.
   * - ``--enable-tracing``
     - off
     - Register tracing subscribers. Requires ``--otlp-endpoint``.
   * - ``--event-bus-queue-size``
     - ``10000``
     - Maximum events in the EventBus queue before tail-drop.
   * - ``--otlp-endpoint``
     - *(none)*
     - OTLP gRPC endpoint (e.g. ``http://localhost:4317``). Used for
       exporting metrics (push mode) and traces.
   * - ``--prometheus-port``
     - ``9090``
     - Port for the Prometheus ``/metrics`` HTTP endpoint.
   * - ``--service-instance-id``
     - *(unset, default UUID v4)*
     - Identifier for this MP server instance. Attached as the OTel
       Resource attribute ``service.instance.id`` on every metric and
       span. When the flag is not passed, defaults to a random UUID v4
       minted at startup. Pass ``--service-instance-id=""`` to force an
       explicit empty value. See :ref:`mp-observability-resource`.
   * - ``--metrics-sample-rate``
     - ``0.01``
     - Fraction of chunks/blocks to track for lifecycle histograms
       (0, 1.0]. Counters always count all events. Default is 1%.
   * - ``--trace-level``
     - *(none)*
     - Enable trace recording at the given level. Currently only
       ``storage`` is supported (records ``StorageManager`` public-API
       calls for offline replay). When unset, trace recording is off.
       See :ref:`trace-recording` for details.
   * - ``--trace-output``
     - *(none)*
     - Path to write the trace file. If omitted while ``--trace-level``
       is set, a timestamped file under ``$TMPDIR`` is minted
       (``lmcache-trace-<pid>-<UTC>.lct``) and its path is logged at INFO.

**Environment variables:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Description
   * - ``LMCACHE_LOG_LEVEL``
     - ``INFO``
     - Controls the log level for all LMCache loggers. Valid values:
       ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.

Metrics
-------

Metrics are collected via OpenTelemetry counters and exported through an
in-process **Prometheus** ``/metrics`` HTTP endpoint (default port 9090).
When ``--otlp-endpoint`` is set, metrics are also pushed to the OTel
collector.

All metrics use the ``lmcache_mp.`` prefix (multiprocess). On Prometheus,
dots are converted to underscores and counters get a ``_total`` suffix
(e.g. ``lmcache_mp_l1_read_keys_total``).

.. _mp-observability-resource:

Global Resource Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every metric and span exported by an MP server carries Resource-level
attributes built at startup. These identify the process producing the
telemetry and are orthogonal to per-metric attributes such as
``cache_salt``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Attribute
     - CLI flag / config
     - Default when unset
   * - ``service.instance.id``
     - ``--service-instance-id`` / ``ObservabilityConfig.service_instance_id``
     - Random UUID v4 minted at startup.

Resource attributes attach to the ``MeterProvider`` / ``TracerProvider``
and propagate to every exported datapoint via OTLP. On Prometheus, SDK
resource attributes surface on the ``target_info`` series rather than
on each time-series — this is standard OTel behavior.

StorageManager Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.sm_read_requests``
     - Counter
     - Number of read (prefetch) requests received by the StorageManager.
   * - ``lmcache_mp.sm_read_succeed_keys``
     - Counter
     - Number of keys successfully read from LMCache.
   * - ``lmcache_mp.sm_read_failed_keys``
     - Counter
     - Number of keys that failed to read.
   * - ``lmcache_mp.sm_write_requests``
     - Counter
     - Number of write (reserve) requests.
   * - ``lmcache_mp.sm_write_succeed_keys``
     - Counter
     - Number of keys successfully reserved for write.
   * - ``lmcache_mp.sm_write_failed_keys``
     - Counter
     - Number of keys that failed to reserve (OOM, write conflict).

L1 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_read_keys``
     - Counter
     - Number of keys read from L1.
   * - ``lmcache_mp.l1_write_keys``
     - Counter
     - Number of keys written to L1.
   * - ``lmcache_mp.l1_evicted_keys``
     - Counter
     - Number of keys evicted by the EvictionController.

L1 Chunk Lifecycle Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampled (default 1%) chunk-level lifecycle tracking via
``L1LifecycleSubscriber``. Only sampled chunks contribute to histograms;
counters above always count all events. Sampling is deterministic
(hash-based), so the same key always gets the same decision with zero
memory overhead.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_chunk_lifetime_seconds``
     - Histogram
     - Time from allocation to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_idle_before_evict_seconds``
     - Histogram
     - Time from last access to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_reuse_gap_seconds``
     - Histogram
     - Time gap between consecutive touches (read or write) of the same chunk.
   * - ``lmcache_mp.l1_chunk_evict_reuse_gap_seconds``
     - Histogram
     - Time from eviction to next reuse (capped at 300 s).

L2 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l2_store_tasks``
     - Counter
     - Number of L2 store tasks submitted.
   * - ``lmcache_mp.l2_store_keys``
     - Counter
     - Number of keys submitted for L2 store.
   * - ``lmcache_mp.l2_store_completed``
     - Counter
     - Number of L2 store tasks completed.
   * - ``lmcache_mp.l2_store_succeeded_keys``
     - Counter
     - Number of keys successfully stored to L2.
   * - ``lmcache_mp.l2_store_failed_keys``
     - Counter
     - Number of keys that failed to store to L2.
   * - ``lmcache_mp.l2_prefetch_lookups``
     - Counter
     - Number of L2 prefetch lookup requests.
   * - ``lmcache_mp.l2_prefetch_lookup_keys``
     - Counter
     - Number of keys submitted for L2 prefetch lookup.
   * - ``lmcache_mp.l2_prefetch_hit_keys``
     - Counter
     - Number of prefix keys found in L2 lookup.
   * - ``lmcache_mp.l2_prefetch_load_tasks``
     - Counter
     - Number of L2 prefetch load tasks submitted.
   * - ``lmcache_mp.l2_prefetch_load_keys``
     - Counter
     - Number of keys submitted for L2 load.
   * - ``lmcache_mp.l2_prefetch_loaded_keys``
     - Counter
     - Number of keys successfully loaded from L2.
   * - ``lmcache_mp.l2_prefetch_failed_keys``
     - Counter
     - Number of keys that failed to load from L2.

L0 (GPU) Block Lifecycle Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampled (default 1%) GPU KV cache block lifecycle tracking via
``L0LifecycleSubscriber``. Eviction is detected at reallocation time
(when a block is assigned different tokens). Sampling uses random
selection with a ``_skipped`` set (bounded by the finite number of
physical GPU blocks).

All L0 histograms are emitted with ``instance_id`` and ``model_name``
OTel attributes, enabling per-instance and per-model metric slicing
in Prometheus (e.g.
``lmcache_mp_l0_block_lifetime_seconds{instance_id="12345",model_name="llama-7b"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l0_block_lifetime_seconds``
     - Histogram
     - Time from allocation to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_idle_before_evict_seconds``
     - Histogram
     - Time from last access to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_reuse_gap_seconds``
     - Histogram
     - Time gaps between consecutive accesses of the same GPU block.

L0 ↔ L1 Throughput Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sampled (default 1%) per-request throughput of GPU↔CPU copies via
``L0L1ThroughputSubscriber``. Each sampled request contributes one sample
to the appropriate histogram: ``total_bytes / (end_ts - start_ts)`` in
GB/s. Timestamps come from ``MP_{STORE,RETRIEVE}_{START,END}`` events
published on the GPU cupy stream, so they reflect true GPU-stream copy
time — not Python/lock overhead.

All throughput histograms are emitted with ``engine_id`` (vLLM worker
instance id) and ``gpu_id`` OTel attributes, enabling per-worker and
per-GPU slicing in Prometheus (e.g.
``lmcache_mp_l0_l1_store_throughput_gbs{engine_id="0",gpu_id="3"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l0_l1_store_throughput_gbs``
     - Histogram
     - GPU→CPU (L0→L1) store throughput in GB/s per sampled request.
   * - ``lmcache_mp.l0_l1_load_throughput_gbs``
     - Histogram
     - CPU→GPU (L1→L0) load throughput in GB/s per sampled request.

Observable Gauges
~~~~~~~~~~~~~~~~~

Point-in-time state snapshots registered via ``register_gauge``
(pull-based OTel observable gauges).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.active_prefetch_jobs``
     - ObservableGauge
     - Number of prefetch jobs currently in-flight. A sustained high
       value may indicate slow L2 backends or polling delays.

Prometheus Scrape Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the LMCache server as a Prometheus scrape target:

.. code-block:: yaml

    scrape_configs:
      - job_name: "lmcache-mp"
        static_configs:
          - targets: ["<lmcache-host>:9090"]

Logging
-------

Logging subscribers emit debug-level messages for store, retrieve, lookup,
L1, and StorageManager events via Python's standard ``logging`` module.

When OpenTelemetry is installed, ``init_logger`` automatically attaches an
OTel ``LoggingHandler`` so that log records are forwarded to any configured
OTel ``LoggerProvider``. The handler respects the ``LMCACHE_LOG_LEVEL``
environment variable.

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server ...

Key log messages:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Level
     - Message
   * - INFO
     - ``Stored N tokens in X seconds``
   * - INFO
     - ``Retrieved N tokens in X seconds``
   * - INFO
     - ``Prefetch request completed (L1+L2): N/M prefix hits``
   * - DEBUG
     - ``MP store start: session=... device=...``
   * - DEBUG
     - ``MP retrieve end: session=... retrieved_count=...``

Tracing
-------

.. note::

   ``--enable-tracing`` **requires** ``--otlp-endpoint`` to be set.
   The server will refuse to start if tracing is enabled without an
   OTLP endpoint, since there is no local fallback for trace export.

When tracing is enabled (``--enable-tracing --otlp-endpoint <URL>``),
the tracing subscriber creates OTel spans from START/END event pairs:

- ``mp.store`` — from ``MP_STORE_START`` to ``MP_STORE_END``
- ``mp.retrieve`` — from ``MP_RETRIEVE_START`` to ``MP_RETRIEVE_END``
- ``mp.lookup_prefetch`` — from ``MP_LOOKUP_PREFETCH_START`` to ``MP_LOOKUP_PREFETCH_END``

Each span carries event metadata as span attributes (e.g. ``device``,
``stored_count``, ``found_count``).

View traces in any OTel-compatible backend such as **Jaeger** or
**Grafana Tempo**.

.. code-block:: bash

    # Start Jaeger all-in-one (OTLP gRPC on 4317)
    docker run -d --name jaeger \
        -p 16686:16686 -p 4317:4317 \
        jaegertracing/all-in-one:latest

    # Start LMCache with tracing
    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --enable-tracing --otlp-endpoint http://localhost:4317

.. _trace-recording:

Trace Recording
---------------

.. note::

   Trace recording is **distinct from** ``--enable-tracing`` (OTel
   spans). Trace recording captures every ``StorageManager`` public-API
   call to a binary file so the same workload can be **replayed** later
   for testing, regression hunting, and benchmarking — without needing
   vLLM and (eventually) without a GPU. ``--enable-tracing`` exports
   live OTel spans to an OTLP endpoint for online observability.
   The two features are independent and can be used together.

When ``--trace-level storage`` is set, LMCache records every call to
``StorageManager.{reserve_write, finish_write, submit_prefetch_task,
read_prefetched_results, finish_read_prefetched}`` to a binary file
for later replay.

Recording is **off by default** and adds near-zero overhead when off
(a single boolean check per ``StorageManager`` call). When on,
recording happens on the EventBus drain thread, off the request path.

Capturing a trace
~~~~~~~~~~~~~~~~~

With an explicit output path:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage --trace-output /tmp/run.lct

With an implicit timestamped output path under ``$TMPDIR``:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --trace-level storage
    # → INFO log: "trace recording enabled (level=storage); no
    #   --trace-output given, writing to
    #   /tmp/lmcache-trace-<pid>-<UTC>.lct"

The trace file is closed cleanly on shutdown (SIGTERM is handled by
the EventBus stop path).

Replay
~~~~~~

Replaying a recorded trace, plus the full set of CLI flags for
driving, monitoring, and exporting replay results, is covered in
its own page: :doc:`tracing_and_debugging`.

What is captured (and what is not)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Captured:**

- The fully-qualified name of every decorated ``StorageManager`` call.
- Each call's input arguments (e.g. ``keys``, ``layout_desc``,
  ``mode``, ``extra_count``, ``external_request_id``).
- Wall-clock and monotonic timestamps of each call.
- A header carrying a trace schema version, start times, and a
  SHA-256 digest of the active ``StorageManagerConfig`` so replay can
  detect mismatched configurations.

**Not captured:**

- KV tensor bytes. Replay exercises bookkeeping and controller logic;
  payloads at replay time are zeros.
- Calls inside the ``MPCacheEngine``, the message queue, or any
  GPU-copy code. These layers are **out of scope** for the storage
  trace level.

File format
~~~~~~~~~~~

A length-prefixed `msgpack <https://msgpack.org/>`_ stream:

::

    [4-byte big-endian length][msgpack Header]
    [4-byte big-endian length][msgpack Record]
    [4-byte big-endian length][msgpack Record]
    ...

The ``Header`` carries a magic prefix (``LMCT``), a format version,
the trace level (``storage`` today), a trace schema version, start
timestamps, and the StorageManagerConfig digest. Each ``Record``
carries a relative timestamp, a wall-clock timestamp, the
fully-qualified call site (``qualname``), and an argument dict.

The format is deliberately extensible: future trace **levels**
(``mq``, ``gpu``) will share this layout and use the ``level`` header
field to discriminate. Additional captured ops add new ``qualname``
strings without bumping the format version.

For the full design rationale see
``docs/design/v1/mp_observability/trace.md`` in the source tree.
