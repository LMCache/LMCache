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
   * - ``describe``
     - Show detailed status of a running LMCache service, including cache
       health, L1 storage, registered models, and L2 adapters.
   * - ``mock``
     - Example command that outputs fake metrics. Useful for testing the CLI
       framework and as a reference for new commands.


``describe`` — Service Status Dashboard
----------------------------------------

Inspect the state of a running LMCache KV cache server:

.. code-block:: bash

   lmcache describe kvcache --url http://localhost:8000

.. code-block:: text

   ============ LMCache KV Cache Service ============
   Health:                                  OK
   URL:                            http://localhost:8000
   Engine type:                         BlendEngine
   Chunk size:                              256
   L1 capacity (GB):                       60.00
   L1 used (GB):                    42.30 (70.5%)
   Eviction policy:                         LRU
   Cached objects:                          1024
   Active sessions:                            3
   --- Model: meta-llama/Llama-3.1-70B-Instruct ----
   Model:          meta-llama/Llama-3.1-70B-Instruct
   World size:                                4
   GPU IDs:                          0, 1, 2, 3
   Attention backend:  vLLM non-MLA flash attention
   GPU KV shape:          NL x [2, NB, BS, NH, HS]
   GPU KV tensor shape: 80 x [2, 2048, 128, 8, 128]
   Num layers:                               80
   Block size:                              128
   Hidden dim size:                        1024
   Dtype:                          torch.float16
   MLA:                                   False
   Num blocks:                             2048
   --------- L2: NixlStoreL2Adapter ------------
   Type:                      NixlStoreL2Adapter
   Health:                                  OK
   Backend:                           nixl_rdma
   Stored objects:                          512
   Pool used:                 480 / 512 (93.8%)
   ==================================================

The output shows:

- **Overview** — health status, engine type, chunk size.
- **L1 storage** — capacity, usage, eviction policy, cached object count.
- **Registered models** — per-model KV cache layout including the GPU KV
  tensor shape (symbolic and concrete), attention backend, and layer details.
- **L2 adapters** — type, health, backend, stored objects, and utilization.

Arguments
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``kvcache``
     - Target to describe (currently only ``kvcache`` is supported).
   * - ``--url``
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - Save metrics to a file (format follows ``--format``).

JSON Output
~~~~~~~~~~~

Use ``--format json`` for machine-readable output. Models and L2 adapters
are collected into lists for easy programmatic access:

.. code-block:: bash

   lmcache describe kvcache --url http://localhost:8000 --format json

.. code-block:: json

   {
     "title": "LMCache KV Cache Service",
     "metrics": {
       "health": "OK",
       "url": "http://localhost:8000",
       "engine_type": "BlendEngine",
       "chunk_size": 256,
       "l1_capacity_gb": 60.0,
       "l1_used_gb": "42.30 (70.5%)",
       "eviction_policy": "LRU",
       "cached_objects": 1024,
       "active_sessions": 3,
       "models": [
         {
           "model": "meta-llama/Llama-3.1-70B-Instruct",
           "world_size": 4,
           "gpu_ids": "0, 1, 2, 3",
           "attention_backend": "vLLM non-MLA flash attention",
           "gpu_kv_shape": "NL x [2, NB, BS, NH, HS]",
           "gpu_kv_concrete_shape": "80 x [2, 2048, 128, 8, 128]",
           "num_layers": 80,
           "block_size": 128,
           "hidden_dim_size": 1024,
           "dtype": "torch.float16",
           "is_mla": false,
           "num_blocks": 2048
         }
       ],
       "l2_adapters": [
         {
           "type": "NixlStoreL2Adapter",
           "health": "OK",
           "backend": "nixl_rdma",
           "stored_object_count": 512,
           "pool_used": "480 / 512 (93.8%)"
         }
       ]
     }
   }

GPU KV Shape Abbreviations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``gpu_kv_shape`` field uses short names from the ``GPUKVFormat`` enum:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Abbrev
     - Meaning
   * - NB
     - num_blocks
   * - NL
     - num_layers
   * - BS
     - block_size
   * - NH
     - num_heads
   * - HS
     - head_size
   * - PBS
     - page_buffer_size (NB × BS)


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
