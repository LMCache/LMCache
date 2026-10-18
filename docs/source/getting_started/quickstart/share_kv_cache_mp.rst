.. _share_kv_cache_mp:

Example: Share KV cache across engines (MP mode)
================================================

This is the recommended way to share KV cache across multiple serving
engines on one node. A single ``lmcache server`` holds the cache; two
vLLM instances connect to it with ``LMCacheMPConnector``. A prefix
computed on the first engine is reused on the second without recomputing
prefill.

The older in-process walkthrough is on :doc:`share_kv_cache`.

Why the prompt is long
----------------------

``lmcache server`` defaults to ``--chunk-size 256``. The hasher keeps
only complete chunks and discards the remainder, so a 12-token prompt
never writes anything and the second engine always misses.

The ``curl`` commands below repeat one sentence until the prompt exceeds
a full chunk — the same pattern as the P2P example on
:doc:`share_kv_cache`. The main :doc:`../quickstart` instead lowers
``--chunk-size`` to 16 for a short-prompt demo; either approach works,
but production should keep the default 256.

Prerequisites
-------------

- Two GPUs on the same machine.
- Ports **5555** (LMCache ZMQ), **8000** and **8001** (vLLM).
- ``lmcache`` and ``vllm`` installed. See :doc:`../quickstart`.

Hashing runs inside the single ``lmcache server`` process, so you do
**not** need ``PYTHONHASHSEED`` (that variable is only required for
in-process cross-engine sharing).

Start the cache server
----------------------

.. code-block:: bash

   lmcache server \
       --host localhost --port 5555 \
       --l1-size-gb 20 --eviction-policy LRU \
       --max-gpu-workers 2

``--max-gpu-workers 2`` gives each vLLM instance its own GPU worker
thread so the two engines do not serialize on one store/retrieve
worker. ``--chunk-size`` stays at the default 256.

Start two vLLM engines
----------------------

In a second terminal, bind engine A to GPU 0:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   vllm serve Qwen/Qwen3-8B \
       --port 8000 --gpu-memory-utilization 0.8 \
       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "localhost", "lmcache.mp.port": 5555}}'

In a third terminal, bind engine B to GPU 1:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=1 \
   vllm serve Qwen/Qwen3-8B \
       --port 8001 --gpu-memory-utilization 0.8 \
       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "localhost", "lmcache.mp.port": 5555}}'

Wait until both engines report that they are ready. On vLLM 0.20.0 or
newer you can point ``kv_connector_module_path`` at the
LMCache-shipped connector; see :doc:`../quickstart`.

Populate the cache on engine A
------------------------------

.. code-block:: bash

   curl -X POST http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d "{
           \"model\": \"Qwen/Qwen3-8B\",
           \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
           \"max_tokens\": 10
       }"

Reuse the cache on engine B
---------------------------

Send the **same** prompt to the other engine:

.. code-block:: bash

   curl -X POST http://localhost:8001/v1/completions \
       -H "Content-Type: application/json" \
       -d "{
           \"model\": \"Qwen/Qwen3-8B\",
           \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
           \"max_tokens\": 10
       }"

What you should see
-------------------

Store and retrieve logs come from the **``lmcache server``** process,
not from vLLM.

**First request** — the server offloads aligned chunks:

.. code-block:: text

   LMCache INFO: Stored 256 tokens in ... (server.py:...)

**Second request** — the shared prefix is retrieved from the same L1
pool:

.. code-block:: text

   LMCache INFO: Retrieved 256 tokens in ... (server.py:...)

If the second request only shows ``Stored`` lines and no ``Retrieved``
line, the prompt did not fill a chunk, the two engines pointed at
different ``lmcache.mp.host`` / ``lmcache.mp.port`` values, or they
served different models (chunk identity includes the model).

For request-level hit ratios see :doc:`/mp/observability/index`. For
Kubernetes (one LMCache DaemonSet per node, many vLLM pods) see
:doc:`/mp/deployment`.
