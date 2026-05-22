CLI Reference
=============

The ``lmcache`` command-line interface provides tools for managing and
inspecting LMCache servers.

.. code-block:: bash

   lmcache <command> [options]

After installing LMCache, the ``lmcache`` command is available globally.
Run ``lmcache -h`` to see all commands.

Available Commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``describe``
     - Show detailed status of a running LMCache service.
   * - ``query``
     - Single-shot query interface for the serving engine.
   * - ``conf``
     - Fetch the active MP server configuration as JSON and optionally
       persist it to a file.
   * - ``ping``
     - Liveness check for LMCache or vLLM servers.
   * - ``bench``
     - Run sustained performance benchmarks against an inference engine,
       an end-to-end sanity test against an LMCache MP server, or a
       throughput/latency benchmark against an L2 cache adapter.
   * - ``kvcache``
     - Manage KV cache state (e.g. clear L1 cache).
   * - ``server``
     - Launch the LMCache server (ZMQ + HTTP).

For a comprehensive guide with examples, see :doc:`/getting_started/cli`.

Configuration Snapshot
----------------------

Use ``lmcache conf`` to fetch the active MP server configuration from
``GET /conf``. The command prints formatted JSON to stdout. Pass ``--output`` or ``-o`` to
also save the same JSON for debugging or issue reports.

.. code-block:: bash

   lmcache conf --url http://localhost:8080 -o lmcache-config.json

For a local server, the port shorthand is also accepted:

.. code-block:: bash

   lmcache conf --url 8080 -o lmcache-config.json

Passing the full endpoint URL is also valid:

.. code-block:: bash

   lmcache conf --url http://localhost:8080/conf

The output includes the server's active MP, HTTP, storage-manager, L1/L2,
policy, and observability configuration values. Sensitive fields such as
passwords and secrets are redacted.

.. toctree::
   :maxdepth: 2

   bench
   bench_kvcache
   bench_l2
   kvcache
