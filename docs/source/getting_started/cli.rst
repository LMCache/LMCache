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
   * - ``mock``
     - Example command that outputs fake metrics. Useful for testing the CLI
       framework and as a reference for new commands.


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
