.. devstral_2_123b_instruct_2512:

Devstral-2-123B-Instruct-2512 Usage Guide
==========================================

Installing vLLM and LMCache
----------------------------

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install lmcache vllm


Launching Devstral-2-123B-Instruct-2512 on 8xH200
---------------------------------------------------

.. code-block:: bash

   PYTHONHASHSEED=0 \
   vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
       --tool-call-parser mistral --enable-auto-tool-choice \
       --tensor-parallel-size 8 \
       --no-enable-prefix-caching \
       --port 8000 --kv-transfer-config \
       '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


.. note::
   ``Prefix caching`` is disabled so that all reuse comes from LMCache, making cache hits and TTFT deltas easier to interpret. For real deployments, keep vLLM prefix caching enabled and size the CPU cache appropriately.
   Setting ``PYTHONHASHSEED=0`` is recommended for deterministic chunk hashing, especially when scaling to multiple processes or instances.


Test LMCache in Action
-----------------------

Cold request (first run)
~~~~~~~~~~~~~~~~~~~~~~~~

Send a long prompt (≥256 tokens) to force full chunk creation:

.. code-block:: bash

   python - <<'PY' | curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d @-
   import json
   prompt = "You are helpful.\n" + ("LMCache reuse test. " * 400)
   payload = {
       "model": "mistralai/Devstral-2-123B-Instruct-2512",
       "prompt": prompt,
       "max_tokens": 32,
   }
   print(json.dumps(payload))
   PY

Expected LMCache logs (cold):

.. code-block:: bash

   (EngineCore_DP0 pid=81349) [2026-01-29 05:21:50,195] LMCache INFO: Reqid: cmpl-a55f69ca5cd7f7a1-0-9b2e34a2, Total tokens 2406, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP1 pid=81548) [2026-01-29 05:21:50,353] LMCache INFO: Stored 2304 out of total 2304 tokens. size: 0.0967 gb, cost 6.9944 ms, throughput: 13.8224 GB/s; offload_time: 6.8512 ms, put_time: 0.1433 ms (cache_engine.py:446:lmcache.v1.cache_engine)
   (APIServer pid=81132) INFO:     127.0.0.1:58222 - "POST /v1/completions HTTP/1.1" 200 OK


Warm request (second run)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the same command again.

Expected LMCache logs (warm):

.. code-block:: bash

   (EngineCore_DP0 pid=81349) [2026-01-29 05:21:55,292] LMCache INFO: Reqid: cmpl-af0af9aa0c0ec3b3-0-af5d6ca7, Total tokens 2406, LMCache hit tokens: 2304, need to load: 2304 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP2 pid=81549) [2026-01-29 05:21:55,299] LMCache INFO: Retrieved 2304 out of 2304 required tokens (from 2304 total tokens). size: 0.0967 gb, cost 3.5766 ms, throughput: 27.0309 GB/s; (cache_engine.py:742:lmcache.v1.cache_engine)
   (APIServer pid=81132) INFO 01-29 05:21:55 [loggers.py:257] Engine 000: Avg prompt throughput: 481.2 tokens/s, Avg generation throughput: 5.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 47.9%


.. code-block:: text

   External prefix cache hit rate: 47.9%

This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.
