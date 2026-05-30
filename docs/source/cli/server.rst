lmcache server
==============

The ``lmcache server`` command launches the standalone LMCache
Multi-Process (MP) server, which exposes a ZMQ control plane and an HTTP
frontend (status, healthcheck, cache-clear, checksum APIs). It is the server
that ``lmcache describe``, ``lmcache ping kvcache``, ``lmcache kvcache``, and
``lmcache bench server`` talk to.

.. note::

   This command requires the full ``lmcache`` installation with CUDA
   extensions. It is **not** available in the lightweight ``lmcache-cli``
   package.

.. code-block:: bash

   lmcache server [options]

Quick start
-----------

.. code-block:: bash

   lmcache server \
       --host 0.0.0.0 --port 5555 \
       --l1-size-gb 100 \
       --eviction-policy LRU

Options
-------

The server composes its arguments from several configuration modules — the
multiprocess server, the storage manager (L1 / L2 adapters / eviction), the
HTTP frontend, and the Prometheus / telemetry observability layer. The full,
authoritative list is large and evolves with the runtime, so consult:

.. code-block:: bash

   lmcache server --help

Commonly used flags include:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Description
   * - ``--host HOST``
     - Bind address for the server.
   * - ``--port PORT``
     - ZMQ control-plane port.
   * - ``--chunk-size N``
     - KV cache chunk size in tokens.
   * - ``--l1-size-gb GB``
     - L1 (CPU/DRAM) cache capacity in GB.
   * - ``--eviction-policy POLICY``
     - L1 eviction policy (e.g. ``LRU``).
   * - ``--eviction-trigger-watermark RATIO``
     - L1 fill ratio at which eviction begins.
   * - ``--eviction-ratio RATIO``
     - Fraction of L1 cleared per eviction cycle.
   * - ``--max-workers N``
     - Number of server worker processes.
   * - ``--trace-level {storage}``
     - Enable storage-level trace recording (see :doc:`trace`).
   * - ``--trace-output PATH``
     - Destination for recorded ``.lct`` trace files.

L2 adapters, observability, and Prometheus exporters are configured through
their own flag groups; see ``lmcache server --help`` for the complete set.
