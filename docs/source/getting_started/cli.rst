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

   # Check if the KV cache server is alive
   lmcache ping kvcache

   # Launch the LMCache server (ZMQ + HTTP)
   lmcache server --host 0.0.0.0 --port 5555 --l1-size-gb 100 --eviction-policy LRU

   # JSON on stdout (for scripts)
   lmcache ping kvcache --format json

   # Save metrics to a file (format follows --format, default: terminal)
   lmcache describe kvcache --format json --output status.json


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
   * - ``query``
     - Single-shot query interface for both the serving engine and KV cache worker.
   * - ``server``
     - Launch the LMCache server (ZMQ + HTTP).
   * - ``ping``
     - Liveness check for LMCache or vLLM servers.
   * - ``kvcache``
     - Manage KV cache state (e.g. clear L1 cache) on a running server.


``describe`` — Service Status Dashboard
----------------------------------------

Inspect the state of a running LMCache KV cache server:

.. code-block:: bash

   lmcache describe kvcache --url http://localhost:8000

.. code-block:: text

   ============ LMCache KV Cache Service ============
   Health:                                         OK
   URL:                         http://localhost:8000
   Engine type:                           BlendEngine
   Chunk size:                                    256
   L1 capacity (GB):                            60.00
   L1 used (GB):                        42.30 (70.5%)
   Eviction policy:                               LRU
   Cached objects:                               1024
   Active sessions:                                 3
   ---- Model: meta-llama/Llama-3.1-70B-Instruct ----
   Model:           meta-llama/Llama-3.1-70B-Instruct
   World size:                                      4
   GPU IDs:                                0, 1, 2, 3
   Attention backend:    vLLM non-MLA flash attention
   GPU KV shape:             NL x [2, NB, BS, NH, HS]
   GPU KV tensor shape:   80 x [2, 2048, 128, 8, 128]
   Num layers:                                     80
   Block size:                                    128
   Hidden dim size:                              1024
   Dtype:                               torch.float16
   MLA:                                         False
   Num blocks:                                   2048
   ------------- L2: NixlStoreL2Adapter -------------
   Type:                           NixlStoreL2Adapter
   Health:                                         OK
   Backend:                                 nixl_rdma
   Stored objects:                                512
   Pool used:                       480 / 512 (93.8%)
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

``query``
---------

The ``query engine`` subcommand sends one request to the engine API and reports metrics.
``--prompt`` supports placeholders: ``{lmcache}`` loads
``lmcache/cli/documents/lmcache.txt``, and custom documents can be passed with
``--documents NAME=PATH``.

.. code-block:: bash

   lmcache query engine --url http://localhost:8000/v1 \
     --prompt "{lmcache} Summarize LMCache usage." \
     --format terminal \
     --max-tokens 128
    
  ================= Query Engine =================
  Model:                         facebook/opt-125m
  Prompt documents lmcache:                    608
  Prompt query:                                  9
  --------------- Latency Metrics ----------------
  Input tokens:                             618.00
  Output tokens:                              9.00
  TTFT (ms):                                 26.88
  TPOT (ms/token):                            0.91
  Total latency (ms):                        35.05
  Throughput (tokens/s):                   1100.64
  ================================================

  

``ping`` — Liveness Check
--------------------------

Check whether an LMCache KV cache server or a vLLM serving engine is reachable:

.. code-block:: bash

   # Ping the KV cache server (default: http://localhost:8080)
   lmcache ping kvcache

   # Ping the serving engine (default: http://localhost:8000)
   lmcache ping engine --url http://localhost:8000

.. code-block:: text

   ======= Ping KV Cache ========
   Status:                   OK
   Round trip time (ms):     3.42
   ==============================

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``kvcache`` | ``engine``
     - Target to ping (positional, required).
   * - ``--url``
     - Server URL. Defaults to ``http://localhost:8080`` for ``kvcache``,
       ``http://localhost:8000`` for ``engine``.
   * - ``--format``
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - Suppress stdout output. Exit code only.

Exit Codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Server is reachable (HTTP 200).
   * - ``1``
     - Connection failure or non-200 response.

``kvcache`` — KV Cache Management
-----------------------------------

Manage KV cache state on a running LMCache server. See :doc:`/cli/kvcache`
for full documentation including examples, options, and common patterns.

Quick example:

.. code-block:: bash

   # Clear all L1 (CPU) cache
   lmcache kvcache clear --url http://localhost:8000


Metrics Output
--------------

All commands that produce metrics support two output formats:

Terminal Output
~~~~~~~~~~~~~~~

Human-readable ASCII table:

.. code-block:: text

   ======= Ping KV Cache ========
   Status:                   OK
   Round trip time (ms):     3.42
   ==============================

JSON Output
~~~~~~~~~~~

Machine-readable output with structured keys, available via ``--format json``
(stdout) or ``--output`` (file):

.. code-block:: bash

   lmcache ping kvcache --format json

.. code-block:: json

   {
     "title": "Ping KV Cache",
     "metrics": {
       "status": "OK",
       "round_trip_time_ms": 3.42
     }
   }

The terminal output uses human-readable labels (e.g., ``"Round trip time (ms)"``),
while the JSON output uses machine-readable keys (e.g., ``"round_trip_time_ms"``).


Adding New Commands
-------------------

New CLI subcommands can be added by creating a ``BaseCommand`` subclass and
registering it. See :doc:`/developer_guide/cli` for details.
