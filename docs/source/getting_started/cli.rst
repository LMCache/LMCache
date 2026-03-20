CLI Reference
=============

LMCache provides a unified ``lmcache`` command-line interface for interacting
with KV cache servers, running benchmarks, and inspecting cache state.

.. code-block:: bash

   lmcache <command> [options]

Quick Start
-----------

After installing LMCache, the ``lmcache`` command is available:

.. code-block:: bash

   # Show available commands
   lmcache -h

   # Start the LMCache server
   lmcache server --l1-size-gb 60 --eviction-policy LRU

   # Run the example mock command
   lmcache mock --name my-run --num-items 5

   # JSON on stdout (for scripts)
   lmcache mock --name my-run --format json

   # Save metrics to a file (format follows --format, default: terminal)
   lmcache mock --name my-run --num-items 5 --format json --output result.json


Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``server``
     - Start the LMCache cache server (ZMQ + HTTP). See :ref:`cli-server`
       below for details.
   * - ``mock``
     - Example command that outputs fake metrics. Useful for testing the CLI
       framework and as a reference for new commands.


.. _cli-server:

``lmcache server``
------------------

Start the LMCache cache server. This replaces the standalone
``lmcache_server`` and ``python3 -m lmcache.v1.multiprocess.http_server``
entry points.

By default the server starts both the ZMQ backend and an HTTP frontend.
Use ``--no-http`` for ZMQ-only mode.

.. code-block:: bash

   # HTTP + ZMQ (default)
   lmcache server \
       --engine-type blend --host 0.0.0.0 --port 5555 \
       --l1-size-gb 60 --eviction-policy LRU

   # ZMQ-only (no HTTP frontend)
   lmcache server --no-http \
       --host 0.0.0.0 --port 5555 \
       --l1-size-gb 60 --eviction-policy LRU

Server Arguments
~~~~~~~~~~~~~~~~

The server command accepts arguments from several groups:

**Core server (ZMQ)**

- ``--host`` — ZMQ server host (default: ``localhost``)
- ``--port`` — ZMQ server port (default: ``5555``)
- ``--chunk-size`` — Chunk size for KV cache operations (default: ``256``)
- ``--max-workers`` — Maximum worker threads (default: ``1``)
- ``--hash-algorithm`` — Hash algorithm: ``builtin``, ``sha256_cbor``, ``blake3`` (default: ``blake3``)
- ``--engine-type`` — Cache engine type: ``default``, ``blend`` (default: ``default``)

**HTTP frontend**

- ``--http-host`` — HTTP server host (default: ``0.0.0.0``)
- ``--http-port`` — HTTP server port (default: ``8000``)
- ``--no-http`` — Disable the HTTP frontend entirely

**L1 memory**

- ``--l1-size-gb`` — Size of L1 memory in GB (**required**)
- ``--l1-use-lazy`` — Use lazy allocation (default: ``True``)
- ``--l1-init-size-gb`` — Initial allocation size in GB when lazy (default: ``20``)
- ``--l1-align-bytes`` — Alignment in bytes (default: ``4096``)

**Eviction**

- ``--eviction-policy`` — Eviction policy: ``LRU``, ``noop`` (**required**)
- ``--eviction-trigger-watermark`` — Memory usage fraction to trigger eviction (default: ``0.8``)
- ``--eviction-ratio`` — Fraction of memory to evict (default: ``0.2``)

**Observability**

- ``--disable-prometheus`` — Disable Prometheus metrics
- ``--prometheus-port`` — Prometheus metrics port (default: ``9090``)
- ``--enable-telemetry`` — Enable the telemetry event system

.. note::

   The standalone ``lmcache_server`` entry point is deprecated.
   Use ``lmcache server`` instead.


Metrics Output
--------------

All commands that produce metrics support two output formats:

Terminal Output
~~~~~~~~~~~~~~~

Human-readable ASCII table matching the ``vllm bench serve`` style:

.. code-block:: text

   ============= Mock Result ==============
   ----------- Input Parameters -----------
   Name:                           test-run
   Num items:                             5
   ------------- Mock Metrics -------------
   Items processed:                      42
   Total time (ms):                   12.34
   Throughput (items/s):            3403.73
   -------------- Validation --------------
   Status:                               OK
   ========================================

JSON Output
~~~~~~~~~~~

Machine-readable output with structured keys, available via ``--format json``
(stdout) or ``--output`` (file):

.. code-block:: bash

   lmcache mock --name test-run --output result.json

.. code-block:: json

   {
     "title": "Mock Result",
     "metrics": {
       "input": {
         "name": "test-run",
         "num_items": 5
       },
       "mock": {
         "items_processed": 42,
         "total_time_ms": 12.34,
         "throughput": 3403.73
       },
       "validation": {
         "status": "OK"
       }
     }
   }

The terminal output uses human-readable labels (e.g., ``"Total time (ms)"``),
while the JSON output uses machine-readable keys (e.g., ``"total_time_ms"``).


Adding New Commands
-------------------

New CLI subcommands can be added by creating a ``BaseCommand`` subclass and
registering it. See :doc:`/developer_guide/cli` for details.
