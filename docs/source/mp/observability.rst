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
(e.g. ``lmcache_mp_l1_read_chunks_total``).

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

L1 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_read``
     - Counter
     - Number of chunks read from L1.
   * - ``lmcache_mp.l1_write``
     - Counter
     - Number of chunks written to L1.
   * - ``lmcache_mp.l1_evicted``
     - Counter
     - Number of chunks evicted by the EvictionController.
   * - ``lmcache_mp.l1_eviction_loop_ticks``
     - Counter
     - L1 eviction-loop iterations (every cycle, regardless of whether
       the watermark was crossed). Driven by ``L1_EVICTION_LOOP_TICK``.
   * - ``lmcache_mp.l1_eviction_loop_triggered``
     - Counter
     - L1 eviction-loop iterations where ``usage >= watermark`` and the
       eviction policy actually ran. The two counters distinguish "loop
       is alive" from "eviction fired" — important when debugging
       short-lived benchmarks that complete faster than the 1 Hz
       polling cycle.

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
   * - ``lmcache_mp.l1_chunk_lifetime``
     - Histogram
     - Time from allocation to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_idle_before_evict``
     - Histogram
     - Time from last access to eviction per sampled chunk.
   * - ``lmcache_mp.l1_chunk_reuse_gap``
     - Histogram
     - Time gap between consecutive touches (read or write) of the same chunk.
   * - ``lmcache_mp.l1_chunk_evict_reuse_gap``
     - Histogram
     - Time from eviction to next reuse (capped at 300 s).

StorageManager Real-Reuse Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workload-level reuse histograms emitted by ``SMLifecycleSubscriber``,
driven by caller-facing StorageManager events
(``SM_READ_PREFETCHED_FINISHED``, ``SM_WRITE_FINISHED``).  Internal
read-lock releases by the store/prefetch controllers are excluded so
the signal reflects user-driven access only.

Both histograms are tagged with ``cache_salt`` for per-tenant
isolation.  The per-salt access counter advances on every read and
write of every chunk (regardless of sampling) so the chunks-gap
reflects true storage volume; the histogram itself records gaps only
for chunks that pass the (deterministic, hash-based) sampling gate.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.real_reuse_gap``
     - Histogram (tag: ``cache_salt``)
     - Time gap between a chunk's last access (read or write) and its
       next read.  Captures storage cost — how long a stored chunk sat
       between accesses.  Emitted only on read events.
   * - ``lmcache_mp.real_reuse_gap_objects``
     - Histogram (tag: ``cache_salt``)
     - Per-``cache_salt`` access-counter gap between two reads of the
       same chunk.  Captures storage volume — how many chunk-accesses
       occurred while this chunk waited for its next read.  Emitted on
       read events for sampled chunks.

L2 Metrics
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l2_store_submitted``
     - Counter
     - Number of L2 store requests submitted.
   * - ``lmcache_mp.l2_store_submitted_objects``
     - Counter
     - Number of chunks submitted for L2 store.
   * - ``lmcache_mp.l2_store_completed``
     - Counter (attr: ``l2_name``)
     - Number of L2 store requests completed, labeled by adapter type.
   * - ``lmcache_mp.l2_store_completed_objects``
     - Counter
     - Number of chunks successfully stored to L2.
   * - ``lmcache_mp.l2_prefetch_lookup``
     - Counter
     - Number of L2 prefetch lookup requests.
   * - ``lmcache_mp.l2_prefetch_lookup_objects``
     - Counter
     - Number of chunks submitted for L2 prefetch lookup.
   * - ``lmcache_mp.l2_prefetch_hit``
     - Counter
     - Number of prefix chunks found in L2 lookup.
   * - ``lmcache_mp.l2_prefetch_load_submitted``
     - Counter
     - Number of L2 prefetch load requests submitted.
   * - ``lmcache_mp.l2_prefetch_load_submitted_objects``
     - Counter
     - Number of chunks submitted for L2 load.
   * - ``lmcache_mp.l2_prefetch_load_completed``
     - Counter
     - Number of chunks successfully loaded from L2.
   * - ``lmcache_mp.l2_load_completed``
     - Counter (attr: ``l2_name``)
     - Number of per-adapter L2 load requests completed, labeled by adapter type.

The ``l2_name``-labeled counters (``l2_store_completed`` and
``l2_load_completed``) exist so dashboards can compute per-backend IOPS on
demand via ``rate(lmcache_mp_l2_store_completed_requests_total{l2_name="..."}[1m])``
(and the equivalent for loads).  No separate ``*_iops`` metric is exported;
keeping the raw counter lets dashboard users pick their own window.

Failure & Health Counters
~~~~~~~~~~~~~~~~~~~~~~~~~

Health-monitoring counters emitted on the dedicated ``lmcache_mp.health``
OTel meter. Driven by the ``L1FailureMetricsSubscriber`` and
``L2FailureMetricsSubscriber``, which are registered automatically when
metrics are enabled. All three counters carry ``model_name`` (extracted
from each ``ObjectKey``) so operators can slice per-model on the
Prometheus ``/metrics`` endpoint.

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l1_allocation_failure``
     - Counter
     - L1 memory allocation failures (OOM) during ``reserve_write``.
       Tagged by ``during`` ∈ {``l1_store``, ``l2_prefetch``} to
       distinguish user-initiated stores from prefetch-triggered
       allocations, plus ``model_name``.
   * - ``lmcache_mp.l1_read_failure``
     - Counter
     - L1 ``reserve_read`` failures. Tagged by ``during`` ∈
       {``l2_store``, ``l1_retrieve``}, ``reason`` ∈ {``not_found``,
       ``write_locked``}, plus ``model_name``. **Post-lookup anomaly
       counter**, not a cache-miss counter — in MP mode ``reserve_read``
       is only called after a successful lookup, so any non-zero value
       indicates a lookup/reserve race or unexpected eviction and should
       stay near zero in healthy operation.
   * - ``lmcache_mp.l2_prefetch_failure``
     - Counter
     - Chunks that L2 reported present at lookup but failed to land in L1.
       Tagged by ``reason`` ∈ {``l1_oom``, ``not_found``} plus
       ``model_name``. ``l1_oom`` means L1 had no room to receive the
       prefetched object; ``not_found`` means the adapter returned no
       data despite a positive lookup (e.g. concurrent delete).

A ``reason=serde_failure`` value will be added to ``l2_prefetch_failure``
as an additive, non-breaking extension once L2 adapters distinguish
deserialization errors from missing objects — no dashboard migration
needed when that lands.

For the full design rationale (including which event types drive each
counter and why ``lmcache_instance_id`` is deferred), see
``docs/design/v1/mp_observability/METRICS.md`` in the source tree.

Lookup Hit-Rate Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Token-level counters whose ratio gives the fraction of tokens requested by
a lookup that were served from either L1 or L2. L0 (GPU prefix cache) is
intentionally excluded — it is vLLM-owned and not observable from LMCache.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.lookup_requested``
     - Counter (attrs: ``model_name``, ``cache_salt``)
     - Total tokens submitted for lookup (denominator of the L1+L2
       token-level hit rate). Only chunk-aligned tokens are counted.
   * - ``lmcache_mp.lookup_hit``
     - Counter (attrs: ``model_name``, ``cache_salt``)
     - Total tokens found in L1 or L2 during lookup (numerator of the
       L1+L2 token-level hit rate). Counts the contiguous prefix hit only.

Both counters are driven by the same event (``MP_LOOKUP_PREFETCH_END``),
so they always advance together per completed lookup. Early-exit lookups
contribute ``0`` to both, and abandoned lookups contribute to neither.

The ``model_name`` and ``cache_salt`` attributes are captured at lookup
time from ``IPCCacheEngineKey`` so dashboards can compute per-model or
per-tenant hit rate. ``cache_salt`` can be high-cardinality (one entry
per tenant or isolation domain); drop it at scrape time with
``metric_relabel_configs`` if storage cost matters.

**PromQL for hit rate:**

.. code-block:: promql

    # Aggregate (all models, all salts):
    rate(lmcache_mp_lookup_hit_tokens_total[5m])
    / rate(lmcache_mp_lookup_requested_tokens_total[5m])

    # Per-model:
    sum(rate(lmcache_mp_lookup_hit_tokens_total[5m])) by (model_name)
    / sum(rate(lmcache_mp_lookup_requested_tokens_total[5m])) by (model_name)

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
   * - ``lmcache_mp.l0_block_lifetime``
     - Histogram
     - Time from allocation to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_idle_before_evict``
     - Histogram
     - Time from last access to eviction per sampled GPU block.
   * - ``lmcache_mp.l0_block_reuse_gap``
     - Histogram
     - Time gaps between consecutive accesses of the same GPU block.

L0 ↔ L1 Throughput Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-request throughput of GPU↔CPU copies via
``L0L1ThroughputSubscriber``. Every store/retrieve request contributes
one sample to the appropriate histogram:
``total_bytes / (end_ts - start_ts)`` in GB/s. Timestamps come from
``MP_{STORE,RETRIEVE}_{START,END}`` events published on the GPU cupy
stream, so they reflect true GPU-stream copy time — not Python/lock
overhead.

All throughput histograms are emitted with ``engine_id`` (vLLM worker
instance id), ``device`` (e.g. ``"cuda:3"``), and ``model_name`` OTel
attributes, enabling per-worker, per-device, and per-model slicing in
Prometheus (e.g.
``lmcache_mp_l0_l1_store_throughput_GB_per_second{engine_id="0",device="cuda:3",model_name="meta-llama/Llama-3.1-8B"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l0_l1_store_throughput``
     - Histogram
     - GPU→CPU (L0→L1) store throughput in GB/s per request.
   * - ``lmcache_mp.l0_l1_load_throughput``
     - Histogram
     - CPU→GPU (L1→L0) load throughput in GB/s per request.

L1 ↔ L2 Throughput Histograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-request throughput of L1↔L2 transfers via
``L2ThroughputSubscriber``. The store path correlates
``L2_STORE_SUBMITTED`` → ``L2_STORE_COMPLETED`` by
``(adapter_index, task_id)``. The load path correlates the per-adapter
``L2_LOAD_TASK_SUBMITTED`` → ``L2_LOAD_TASK_COMPLETED`` events by
``(request_id, adapter_index)``; the request-level
``L2_PREFETCH_LOAD_*`` events used by the chunk-count counters aggregate
across adapters and cannot be attributed to a specific ``l2_name``.

Timestamps span **submit → complete**, so the duration includes adapter
queue, network, and disk I/O — the value is *bytes / end-to-end
latency*, not raw transfer rate. Use these histograms to compare
adapter types and catch regressions; use the L0↔L1 histograms when you
need pure copy-time throughput.

All L1↔L2 throughput histograms carry a single ``l2_name`` OTel
attribute — the registered adapter type (e.g. ``"fs"``, ``"nixl_store"``,
``"mooncake_store"``) — enabling per-backend slicing in Prometheus (e.g.
``lmcache_mp_l2_store_throughput_GB_per_second{l2_name="nixl_store"}``).

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.l2_store_throughput``
     - Histogram
     - L1→L2 store throughput in GB/s per request.
   * - ``lmcache_mp.l2_load_throughput``
     - Histogram
     - L2→L1 load throughput in GB/s per (request, adapter) pair.

Engine Counters
~~~~~~~~~~~~~~~

Worker-scoped counters tied to what the MP server delivers back to each
vLLM worker via ``retrieve()``.  Labeled by ``worker_id`` (the vLLM
worker instance id) — distinct from any scheduler-scoped id that may
appear on other metrics.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.num_chunks_loaded``
     - Counter (attrs: ``worker_id``, ``model_name``, ``cache_salt``)
     - Total number of LMCache chunks loaded into the engine, summed
       over all ``retrieve()`` completions.  Sliceable per worker, per
       model, and per tenant / isolation domain (``cache_salt``).
       ``cache_salt`` may be high-cardinality; drop it at scrape time
       with ``metric_relabel_configs`` if storage cost matters.

Observable Gauges
~~~~~~~~~~~~~~~~~

Point-in-time state snapshots registered via ``register_gauge``
(pull-based OTel observable gauges).

The three in-flight metrics carry two attributes that distinguish
adapters even when more than one is registered with the same backend
type — same shape as ``lmcache_mp.l2_store_completed``:

- ``l2_name`` — the registered adapter type (e.g. ``"fs"``,
  ``"nixl_store"``, ``"mooncake_store"``).
- ``adapter_index`` — position in the controller's adapter list.

Adapters with no in-flight work emit no datapoint for that scrape.

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
   * - ``lmcache_mp.l1_memory_usage_bytes``
     - ObservableGauge
     - Bytes currently held in L1.  Rising without plateauing typically
       indicates a leak; saturating at the configured ``--l1-size-gb``
       indicates working set exceeds capacity.
   * - ``lmcache_mp.l1_usage_ratio``
     - ObservableGauge
     - L1 used/total ratio (``0.0``–``1.0``), sampled at scrape time
       from ``L1Manager.get_memory_usage()``. Returns ``0.0`` when the
       gauge target is not yet wired up or ``total_bytes`` is zero, so
       the callback never raises during a scrape. Compare against the
       eviction watermark (default ``0.8``) to read whether the
       eviction loop is below or above its trigger threshold.
   * - ``lmcache_mp.l2_usage_bytes``
     - ObservableGauge (attr: ``l2_name``)
     - Bytes currently held in each L2 adapter, sampled at scrape time
       from ``adapter.get_usage()``.  One observation per configured
       adapter, tagged by ``l2_name`` (the adapter type, e.g. ``"fs"``,
       ``"nixl_store"``, ``"mooncake_store"``).  Parallel to
       ``l1_memory_usage_bytes`` for the L2 tier — use it to see how
       much each L2 backend currently holds.  Adapters whose
       ``get_usage()`` raises are skipped silently rather than poisoning
       the observation, so a missing datapoint for one ``l2_name`` can
       mean either "not configured" or "adapter errored on this
       scrape" — cross-check with the L2 store/load counters.
   * - ``lmcache_mp.num_inflight_l2_stores``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L2 store tasks currently executing, per adapter.  Sustained
       non-zero values indicate the adapter cannot keep up with the
       L1 → L2 write rate.
   * - ``lmcache_mp.num_inflight_l2_loads``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L2 → L1 prefetch load tasks currently executing, per adapter.
       Pair with ``num_inflight_l2_stores`` to see whether read or write
       traffic dominates a given backend.
   * - ``lmcache_mp.inflight_load_memory_usage_bytes``
     - ObservableGauge (attrs: ``l2_name``, ``adapter_index``)
     - L1 bytes reserved by in-flight L2 → L1 prefetch loads, per
       adapter.  Rising in-flight bytes alongside rising
       ``l1_memory_usage_bytes`` is a signal that prefetch reservations
       are crowding out cacheable data.  Per-adapter byte attribution
       follows each request's ``load_plan`` bitmap, so summing across
       adapters never double-counts.

EventBus Self-Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

Health metrics for the EventBus itself, registered by
``EventBusSelfMetricsSubscriber`` on the ``lmcache.event_bus`` OTel
meter.  These metrics observe bus state directly via the ``EventBus``
accessors and report on every OTel scrape — they are not driven by
events, so dropping or failing subscribers cannot silence them.

Use them to answer: is the EventBus keeping up with publishers, is
anything being dropped, and are any subscriber callbacks raising?
A non-zero ``dropped_events_total`` or a sustained non-zero
``drain_lag_seconds`` indicates the bus is at ``--event-bus-queue-size``
and tail-dropping; raise that flag or investigate slow subscribers.

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp.event_bus.queue_depth``
     - ObservableGauge
     - Events currently queued in the EventBus (``len(_queue)`` at
       scrape time).
   * - ``lmcache_mp.event_bus.drain_lag_seconds``
     - ObservableGauge
     - Seconds since the oldest queued event was published; ``0.0``
       when empty.  Rising values mean the drain thread is falling
       behind.
   * - ``lmcache_mp.event_bus.dropped_events_total``
     - ObservableCounter
     - Cumulative events dropped because the EventBus queue was at
       ``--event-bus-queue-size``.
   * - ``lmcache_mp.event_bus.subscriber_exceptions``
     - ObservableCounter (attr: ``subscriber_name``)
     - Cumulative exceptions raised by subscriber callbacks during
       EventBus dispatch, tagged by ``subscriber_name`` (the failing
       callback's owning class for bound methods, or ``__qualname__``
       for free functions).

For the full design rationale and the in-process accessors that back
each metric see ``docs/design/v1/mp_observability/METRICS.md`` and
``docs/design/v1/mp_observability/event-bus.md`` in the source tree.

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

Per-Request Hit-Rate Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each session is wrapped in a per-request root span — ``request`` for the
standard MP path and ``cb.request`` for the CacheBlend path — that nests
all child spans (``mp.store``, ``mp.retrieve``, ``mp.lookup_prefetch``)
beneath it.  When the lookup phase ends, the root span is annotated with
three OTel attributes that summarise the request-level cache hit rate:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - OTel type
     - Description
   * - ``hit_tokens``
     - ``int``
     - Tokens served from L1+L2 (numerator).
   * - ``requested_tokens``
     - ``int``
     - Chunk-aligned tokens submitted for lookup (denominator).
   * - ``hit_rate``
     - ``float``
     - ``hit_tokens / requested_tokens``; ``0.0`` when the denominator is
       zero.  Stored as a precomputed float because trace UIs (Tempo,
       Jaeger) cannot derive it from two integer attributes at query time.

The attributes are written when ``MP_LOOKUP_PREFETCH_END`` (standard MP
path) or ``CB_LOOKUP_END`` (CacheBlend path) is processed — while the
root span is still open.  **Store-only requests** that never call
``lookup_prefetch_start()`` emit no end event for the lookup phase, so
their root span will not carry these attributes.

Example TraceQL queries (Grafana Tempo):

.. code-block:: text

    # Requests with less than 50% cache hit rate
    { name = "request" && span.hit_rate < 0.5 }

    # Full cache hits only
    { name = "request" && span.hit_rate = 1.0 }

    # Complete misses (lookup ran but nothing was cached)
    { name = "request" && span.requested_tokens > 0 && span.hit_tokens = 0 }

For the full event-to-span mapping and the registry pattern that links
child spans back to the root see
``docs/design/observability/request-event-span.md`` in the source tree.

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
