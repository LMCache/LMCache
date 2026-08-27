lmcache coordinator
===================

The ``lmcache coordinator`` command launches the LMCache MP **coordinator**, a
standalone HTTP service that tracks the MP server instances in a deployment. MP
servers register with it and send periodic heartbeats; the coordinator evicts
any instance whose heartbeat lapses past ``--instance-timeout``.

It is the preferred form of ``python -m lmcache.v1.mp_coordinator``, which still
works and accepts the same flags. The process runs in the foreground; stop it
with ``Ctrl-C``.

.. code-block:: bash

   lmcache coordinator [options]

Quick start
-----------

.. code-block:: bash

   lmcache coordinator \
       --host 0.0.0.0 --port 9300 \
       --instance-timeout 30 \
       --health-check-interval 10

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Description
   * - ``--host HOST``
     - Bind address for the coordinator's HTTP server (default: ``0.0.0.0``).
   * - ``--port PORT``
     - HTTP port (default: ``9300``).
   * - ``--instance-timeout SECS``
     - Seconds without a heartbeat after which an instance is evicted
       (default: ``30``).
   * - ``--health-check-interval SECS``
     - Seconds between health-check sweeps; ``0`` disables the loop
       (default: ``10``).
   * - ``--eviction-check-interval SECS``
     - Seconds between L2 eviction sweeps; ``0`` disables the loop
       (default: ``5``).
   * - ``--eviction-ratio RATIO``
     - Fraction of tracked keys (by count) to evict per cycle, ``0.0`` to
       ``1.0`` (default: ``0.2``).
   * - ``--trigger-watermark RATIO``
     - Eviction fires when usage reaches this fraction of the quota, ``0.0``
       (exclusive) to ``1.0`` (default: ``1.0``).
   * - ``--chunk-size N``
     - Tokens per KV chunk: the CacheBlend match unit and the unit used to
       resolve pin ``token_ids`` to keys. Must equal the MP servers'
       ``--chunk-size`` (default: ``256``).
   * - ``--hash-algorithm NAME``
     - Token hash algorithm for pin key resolution; must equal the MP servers'
       ``--hash-algorithm``. ``blake3`` (default) is self-contained; other
       algorithms require vLLM importable in the coordinator.
   * - ``--enable-blend-lookup``
     - Index stored chunk content so ``POST /directory/blend-lookup`` can serve
       fleet CacheBlend reuse. Off by default: hashing content costs CPU on
       every store and is useless without CacheBlend. Also requires the MP
       servers' ``--coordinator-event-reporting``, which feeds the index.
   * - ``--blend-probe-stride N``
     - Positions between CacheBlend match probes; ``1`` probes every offset
       for full recall (default: ``1``). Ignored unless blend lookup is on.
   * - ``--checkpoint-path FILE``
     - Checkpoint the coordinator's directory, usage view and stream cursors
       to this file, so a restart resumes instead of starting cold. Unset
       disables checkpointing.
   * - ``--checkpoint-interval SECS``
     - Seconds between checkpoint writes; ``0`` writes only on a clean stop
       (default: ``60``). Ignored unless ``--checkpoint-path`` is set.
   * - ``--metadata-path FILE``
     - Store operator-set state -- L2 pins and per-``cache_salt`` quotas -- in
       this file, written whenever it changes. Unset means that state is lost
       on restart.
   * - ``--extra-config JSON``
     - JSON object of settings the core flags do not name, read by whichever
       view or controller looks for them. Lets a new one ship with its own
       settings without a flag here.
   * - ``--timeout-keep-alive SECS``
     - Seconds the HTTP server keeps idle connections open before closing
       them. Must be greater than the MP servers' heartbeat interval
       (default ``5``), otherwise heartbeat requests may hit a closing
       connection and fail with ``Server disconnected without sending a
       response`` (default: ``10``).
   * - ``--disable-metrics``
     - Disable OpenTelemetry metrics. Metrics are enabled by default.
   * - ``--otlp-endpoint URL``
     - Push metrics to the specified OTLP gRPC endpoint. When unset, Prometheus
       pull mode exposes ``/metrics`` on the coordinator HTTP port.

Configuration
-------------

Every flag is optional; an unset flag keeps the built-in default listed above.

Prometheus pull mode reuses the coordinator's existing HTTP server; it does not
start a second server or reserve a separate Prometheus port. Metrics-disabled
and OTLP push modes both return HTTP 404 from the local ``/metrics`` route.

See :doc:`/mp/coordinator` for the active eviction loop.

The coordinator drives fleet-wide L2 eviction by calling each MP
server's ``DELETE /l2`` endpoint, documented at
:ref:`mp-http-l2-keys-api`.

See :doc:`/mp/coordinator` for the coordinator's architecture, registration
protocol, and HTTP API.
