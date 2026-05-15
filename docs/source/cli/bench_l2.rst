.. _lmcache-bench-l2:

lmcache bench l2
================

The ``lmcache bench l2`` command benchmarks an L2 cache adapter
(e.g. the local-filesystem adapter) end-to-end through the same
``parse_args_to_l2_adapters_config`` + ``create_l2_adapter`` pipeline that
LMCache uses in production. Any registered adapter type can be tested
without code changes: you describe the adapter with a single JSON spec
and pick the operations to exercise.

.. code-block:: bash

   lmcache bench l2 [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command
does **not** require an inference engine or an LMCache MP server. It
only needs the adapter's own backing storage to be reachable (for the
``fs`` adapter, that simply means a writable directory).


What it does
------------

For each measured operation the tool drives the adapter directly via
its public submit/wait API:

* ``Store``  -- ``submit_store_task`` writes ``num_keys`` MemoryObjs per
  submit and waits for the store eventfd.
* ``Lookup`` -- ``submit_lookup_and_lock_task`` checks key existence
  (no payload transfer) and waits for the lookup eventfd.
* ``Load``   -- ``submit_load_task`` reads ``num_keys`` MemoryObjs per
  submit and waits for the load eventfd.

Each measured **round** issues ``--in-flight`` submits sequentially from
a single producer thread and then waits for all of them to complete; the
round duration is the wall-clock time from the first submit until the
last completion. Warmup rounds run before measurement and their results
are discarded from the final summary.

All three operations share the same key idx universe, so running
``--only store`` followed by ``--only load`` (or ``--only lookup``) with
identical other flags hits exactly the same keys. This makes the
benchmark useful as a quick regression test for adapters that should
support a clean store -> load round-trip.

.. note::

   When ``--only`` is not given, the three operations are run **in a
   single process in the order** ``store -> lookup -> load``. For
   adapters whose backing storage sits behind an OS-level cache --
   most notably the local-filesystem (``fs``) adapter, which is
   subject to the Linux **page cache** -- this means ``lookup`` and
   ``load`` will almost always observe the data that ``store`` just
   wrote still hot in RAM, and the reported numbers reflect
   page-cache throughput rather than the underlying device.

   To benchmark each operation against a cold cache, run them
   separately with ``--only`` and drop the OS caches in between, for
   example::

      lmcache bench l2 --l2-adapter '...' --only store
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only lookup
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only load

   For adapters that bypass the page cache (e.g. ``fs`` with
   ``"use_odirect": true``) or that talk to a remote service without
   a local cache, the default combined run is usually fine.
-----------

Benchmark the local filesystem adapter with default parameters:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}'

This runs all three operations (store, lookup, load) with one warmup
round and one measurement round.

Stress the adapter with more in-flight submits and larger payloads:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}' \
       --num-keys 32 --in-flight 4 \
       --data-size-kb 512 \
       --rounds 5 --warmup-rounds 1

Run only one operation (useful to isolate store vs. load throughput):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only store

Lookup with a controlled hit rate (the benchmark splits the lookup keys
between a potentially-existing range and a guaranteed-non-existent
range):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only lookup --lookup-max-hit-rate 0.5

Enable a store -> load round-trip data integrity check on the last
measured round:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --no-skip-verify

If you prefer to keep the JSON spec out of the command line, set the
``L2_ADAPTER_JSON`` environment variable instead of passing
``--l2-adapter``:

.. code-block:: bash

   export L2_ADAPTER_JSON='{"type":"fs","base_path":"/tmp/lmcache-bench"}'
   lmcache bench l2 --num-keys 32 --in-flight 2


Options
-------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-adapter JSON``
     - *(unset)*
     - L2 adapter spec as JSON with a ``"type"`` field plus
       adapter-specific configs, e.g.
       ``'{"type":"fs","base_path":"/tmp/bench"}'``. May be passed
       multiple times; only the first spec is benchmarked. If not
       provided, falls back to the ``L2_ADAPTER_JSON`` environment
       variable. Either the flag or the env var is **required**.
   * - ``--num-keys N``
     - ``32``
     - Number of keys per submit.
   * - ``--in-flight N``
     - ``1``
     - In-flight submits per round. Each round issues this many
       submits sequentially from a single producer thread, then waits
       for all of them.
   * - ``--data-size-kb N``
     - ``256``
     - Data size per key, in KiB.
   * - ``--rounds N``
     - ``1``
     - Measurement rounds per operation.
   * - ``--warmup-rounds N``
     - ``1``
     - Warmup rounds run before measurement; their results are
       discarded.
   * - ``--lookup-max-hit-rate F``
     - ``0.0``
     - Upper bound on the lookup hit rate, in ``[0, 1]``. The benchmark
       requests ``floor(N * rate)`` keys from the
       potentially-existing range and ``N - hit`` keys from a
       guaranteed-non-existent range, where ``N`` is the total number
       of lookup keys. The actual hit rate may be lower if those keys
       were never stored in this run.
   * - ``--skip-verify`` / ``--no-skip-verify``
     - ``--skip-verify``
     - Skip the store -> load round-trip data integrity check (the
       default). Pass ``--no-skip-verify`` to enable verification on
       the last measured round; this requires both ``store`` and
       ``load`` to be exercised.
   * - ``--only {lookup,store,load}``
     - *(unset)*
     - Run only the specified operation. When omitted, all three
       operations are run in the order ``store -> lookup -> load``.


Adapter JSON spec
-----------------

The ``--l2-adapter`` JSON is parsed by
``lmcache.v1.distributed.l2_adapters.config.parse_args_to_l2_adapters_config``,
the same entry point LMCache uses everywhere else. The minimum required
field is ``type``; all remaining fields are forwarded to the adapter
implementation as keyword arguments.

Example for the local-filesystem adapter:

.. code-block:: json

   {
     "type": "fs",
     "base_path": "/data/lmcache-bench",
     "relative_tmp_dir": "tmp",
     "read_ahead_size": null,
     "use_odirect": false
   }

See the source under ``lmcache/v1/distributed/l2_adapters/`` for the
full list of adapter types and their accepted fields.


Example output
--------------

Per-round progress (suppressed by ``-q`` if you wire it through):

.. code-block:: text

   ============================================================
   L2 Adapter Benchmark
   ============================================================
     Adapter config         : FSL2AdapterConfig
     L2 adapter JSON        : {"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}
     Keys / submit          : 32
     In-flight / round      : 3
     Keys / round           : 96
     Data size / key        : 256 KB
     Data / round           : 24.00 MB
     Rounds                 : 1 (+ 1 warmup)
     Lookup max hit rate    : 0.00%
   ============================================================

   [Init] Creating adapter...
   [Init] Adapter created successfully (FSL2Adapter).

   [Store] Running 1 warmup + 1 measurement rounds...
     [Store] Round 1: 47.83 ms, success_keys=96/96
     [Store] Round 2: 46.19 ms, success_keys=96/96

   [Lookup] Running 1 warmup + 1 measurement rounds...
     [Lookup] Round 1:  5.36 ms, found=96/96
     [Lookup] Round 2:  5.03 ms, found=96/96

   [Load] Running 1 warmup + 1 measurement rounds...
     [Load] Round 1: 18.15 ms, loaded=96/96
     [Load] Round 2: 17.63 ms, loaded=96/96

Final summary (one section per exercised operation):

.. code-block:: text

   ====== L2 Adapter Benchmark Result (FSL2Adapter) =======
   ----------------------- Configuration -------------------
   Adapter:                          FSL2Adapter
   Keys / submit:                    32
   In-flight / round:                3
   Data size / key (KB):             256
   Measurement rounds:               1
   Warmup rounds:                    1
   Lookup max hit rate:              0.0
   --------------------------- Store -----------------------
   Operation:                        Store
   Rounds:                           1
   Keys / round:                     96
   Total keys:                       96
   Total success:                    96
   Duration avg (ms):                46.19
   ...
   Throughput avg (MB/s):            519.62
   Avg ops/s:                        2078.50
   Avg latency / key (ms):           0.481
   --------------------------- Lookup ----------------------
   ...
   ---------------------------- Load -----------------------
   ...
   =========================================================

Each operation section reports per-round duration statistics
(avg / min / max / p50 / p99 / std), aggregate throughput
(``avg_throughput_mbps`` -- 0 for ``Lookup`` since it has no payload),
average key-rate (``avg_ops_per_sec``), and a per-key latency.

For ``Lookup``, three additional fields are reported when
``--lookup-max-hit-rate`` is non-zero or some keys were found:

* ``Expected max hit rate`` -- the configured upper bound.
* ``Expected hit keys`` -- ``floor(total_keys * rate)``, scaled for
  the measured rounds only.
* ``Actual hit rate`` -- the measured hit rate over the kept rounds.


Round-trip verification
-----------------------

When ``--no-skip-verify`` is passed and both ``store`` and ``load`` were
run, the benchmark compares the load buffers from the last measured
round against the byte pattern that ``store`` wrote (see
``make_memory_objects`` in
``lmcache/cli/commands/bench/l2_adapter_bench/data.py``):

.. code-block:: text

   [Verify] Checking store -> load data integrity for last measured round...
   [Verify] OK

Verification is **off** by default because the stricter byte pattern
also forces every key to allocate its own ``data_size`` buffer
(otherwise the runner is free to reuse a single shared buffer across
keys to keep the memory footprint small).


Exit codes
----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requested operations completed and (when enabled) the
       round-trip verification passed.
   * - ``1``
     - Adapter creation failed, round-trip verification failed, or
       an operation hit a fatal error (e.g. all rounds timed out).
   * - ``2``
     - The ``--l2-adapter`` JSON / ``L2_ADAPTER_JSON`` env var was
       missing or could not be parsed.


See also
--------

* :doc:`bench` -- ``lmcache bench engine`` for engine-side workload
  benchmarks.
* :doc:`bench_kvcache` -- end-to-end sanity test against an LMCache MP
  server.
* :doc:`kvcache` -- ``lmcache kvcache`` for managing KV cache state on
  a running server.
