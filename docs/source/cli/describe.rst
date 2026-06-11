lmcache describe
================

The ``lmcache describe`` command shows the detailed status of a running
LMCache service, including cache health, L1 storage, registered models, and
L2 adapters.

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
   Num layers:                                     80
   Num blocks:                                   2048
   Cache size per token (bytes):               327680
   --- Kernel group 0 (meta-llama/Llama-3.1-70B-Instruct) ---
   Kernel group index:                              0
   Engine group index:                              0
   Object group index:                              0
   Num layers:                                     80
   Slots per block:                               128
   Dtype:                               torch.float16
   MLA:                                         False
   Attention backend:    vLLM non-MLA flash attention
   GPU KV shape:             NL x [2, NB, BS, NH, HS]
   GPU KV tensor shape:   80 x [2, 2048, 128, 8, 128]
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
- **Registered models** — per-model KV cache layout: a context-wide summary
  followed by one kernel group section per kernel group, each with the GPU KV
  tensor shape (symbolic and concrete), attention backend, and group geometry.
- **L2 adapters** — type, health, backend, stored objects, and utilization.

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``kvcache``
     - Target to describe (positional, required; currently only
       ``kvcache`` is supported).
   * - ``--url``
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output PATH``
     - Save metrics to a file (format follows ``--format``).
   * - ``-q`` / ``--quiet``
     - Suppress stdout output. Exit code only.

JSON Output
-----------

Use ``--format json`` for machine-readable output. Models, kernel groups, and
L2 adapters are collected into lists for easy programmatic access:

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
           "num_layers": 80,
           "num_blocks": 2048,
           "cache_size_per_token": 327680
         }
       ],
       "kernel_groups": [
         {
           "model": "meta-llama/Llama-3.1-70B-Instruct",
           "kernel_group_idx": 0,
           "engine_group_idx": 0,
           "object_group_idx": 0,
           "num_layers": 80,
           "slots_per_block": 128,
           "dtype": "torch.float16",
           "is_mla": false,
           "attention_backend": "vLLM non-MLA flash attention",
           "gpu_kv_shape": "NL x [2, NB, BS, NH, HS]",
           "gpu_kv_concrete_shape": "80 x [2, 2048, 128, 8, 128]"
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
--------------------------

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
