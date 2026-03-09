Observability
=============

LMCache multiprocess mode includes two complementary observability systems:
**Prometheus metrics** for aggregate counters and **telemetry events** for
per-request tracing.

.. contents::
   :local:
   :depth: 2

Prometheus Metrics
------------------

Prometheus metrics are **enabled by default** (port 9090).  Disable with
``--disable-prometheus``.

All metrics use the ``lmcache_mp:`` prefix to distinguish them from the
single-process ``lmcache:`` namespace.

StorageManager Read Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp:sm_read_requests``
     - Counter
     - Number of read (prefetch) requests received by the StorageManager.
   * - ``lmcache_mp:sm_read_succeed_keys``
     - Counter
     - Number of keys successfully found in L1 during read.
   * - ``lmcache_mp:sm_read_failed_keys``
     - Counter
     - Number of keys not found in L1 during read.

StorageManager Write Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Metric
     - Type
     - Description
   * - ``lmcache_mp:sm_write_requests``
     - Counter
     - Number of write (reserve) requests.
   * - ``lmcache_mp:sm_write_succeed_keys``
     - Counter
     - Number of keys successfully reserved for write.
   * - ``lmcache_mp:sm_write_failed_keys``
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
   * - ``lmcache_mp:l1_read_keys``
     - Counter
     - Number of keys successfully read from L1.
   * - ``lmcache_mp:l1_write_keys``
     - Counter
     - Number of keys successfully written to L1.
   * - ``lmcache_mp:l1_evicted_keys``
     - Counter
     - Number of keys evicted from L1 by the EvictionController.

.. note::
   L2 metrics are not yet finalized and will be added in a future release.

Grafana / Prometheus Scrape Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the LMCache server as a Prometheus scrape target:

.. code-block:: yaml

    scrape_configs:
      - job_name: "lmcache-mp"
        static_configs:
          - targets: ["<lmcache-host>:9090"]

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--disable-prometheus``
     - ``False``
     - Disable Prometheus metrics.
   * - ``--prometheus-port``
     - ``9090``
     - Port for the ``/metrics`` endpoint.
   * - ``--prometheus-log-interval``
     - ``10.0``
     - Flush interval (seconds) from internal stats to Prometheus counters.

Telemetry Event System
----------------------

The telemetry system produces structured **START/END event pairs** for each
server operation (lookup, store, retrieve).  It is **disabled by default**
and must be explicitly enabled.

Enabling Telemetry
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.server \
        --l1-size-gb 100 --eviction-policy LRU \
        --enable-telemetry \
        --telemetry-processor '{"type": "logging", "log_level": "DEBUG"}'

Event Model
~~~~~~~~~~~

Each telemetry event contains:

- **name**: Operation name (e.g., ``lookup``, ``retrieve``, ``store``).
- **event_type**: ``START`` or ``END``.
- **session_id**: Request ID for correlating start/end pairs.
- **timestamp**: High-resolution monotonic timestamp.
- **metadata**: Operation-specific data (e.g., ``found_count``,
  ``retrieved_count``, ``device``).

Processors
~~~~~~~~~~

Telemetry events are dispatched to one or more **processors** configured via
``--telemetry-processor <JSON>``.

**Built-in: ``logging`` processor**

Logs each event via LMCache's logger at the specified level.

.. code-block:: bash

    --telemetry-processor '{"type": "logging", "log_level": "DEBUG"}'

Sample output:

.. code-block:: text

    LMCache DEBUG: Telemetry: lookup START session=req-001 ts=12345.678 metadata={}
    LMCache DEBUG: Telemetry: lookup END session=req-001 ts=12345.680 metadata={'found_count': 3}

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--enable-telemetry``
     - ``False``
     - Enable telemetry.
   * - ``--telemetry-max-queue-size``
     - ``10000``
     - Maximum events in queue before tail-drop.
   * - ``--telemetry-processor``
     - *(none)*
     - Processor spec JSON (repeatable).

Logging
-------

LMCache uses Python's ``logging`` module.  Control the log level with the
``LMCACHE_LOG_LEVEL`` environment variable:

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG python3 -m lmcache.v1.multiprocess.server ...

Key log messages to look for:

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
     - ``Submitted store task ...``
   * - DEBUG
     - ``L2 store task N completed ...``
   * - DEBUG
     - ``Prefetch request submitted: X total keys, Y L1 prefix hits, Z remaining for L2``
