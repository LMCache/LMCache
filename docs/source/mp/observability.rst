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
