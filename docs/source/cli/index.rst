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
   * - ``ping``
     - Liveness check for LMCache or vLLM servers.
   * - ``bench``
     - Run sustained performance benchmarks against an inference engine.
   * - ``kvcache``
     - Manage KV cache state (e.g. clear L1 cache).
   * - ``server``
     - Launch the LMCache server (ZMQ + HTTP).

For a comprehensive guide with examples, see :doc:`/getting_started/cli`.

.. toctree::
   :maxdepth: 2

   bench
   kvcache
