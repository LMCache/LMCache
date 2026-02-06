.. kimi_k2_think:

Kimi-K2-Thinking Usage Guide
=============================

Installing vLLM and LMCache
----------------------------

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install lmcache vllm


Launching Kimi-K2-Thinking on 8xH200
-------------------------------------

.. code-block:: bash

   PYTHONHASHSEED=0 \
   vllm serve moonshotai/Kimi-K2-Thinking \
     --tensor-parallel-size 8 \
     --enable-auto-tool-choice \
     --tool-call-parser kimi_k2 \
     --reasoning-parser kimi_k2 \
     --trust-remote-code \
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
       "model": "moonshotai/Kimi-K2-Thinking",
       "prompt": prompt,
       "max_tokens": 32,
   }
   print(json.dumps(payload))
   PY

Expected LMCache logs (cold):

.. code-block:: bash

   (EngineCore_DP0 pid=39720) [2026-01-28 06:26:35,512] LMCache INFO: Reqid: cmpl-a52220e7cfe4a4a7-0-bf005dc0, Total tokens 2005, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP0 pid=39937) [2026-01-28 06:26:35,686] LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.1173 gb, cost 3.8560 ms, throughput: 30.4150 GB/s; offload_time: 3.8075 ms, put_time: 0.0485 ms (cache_engine.py:446:lmcache.v1.cache_engine)
   (APIServer pid=39503) INFO:     127.0.0.1:38840 - "POST /v1/completions HTTP/1.1" 200 OK


Warm request (second run)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the same command again.

Expected LMCache logs (warm):

.. code-block:: bash

   (EngineCore_DP0 pid=39720) [2026-01-28 06:26:43,758] LMCache INFO: Reqid: cmpl-b65f9fb259c1af7b-0-be16654c, Total tokens 2005, LMCache hit tokens: 1792, need to load: 1792 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP0 pid=39937) [2026-01-28 06:26:43,791] LMCache INFO: Retrieved 1792 out of 1792 required tokens (from 1792 total tokens). size: 0.1173 gb, cost 28.8691 ms, throughput: 4.0624 GB/s; (cache_engine.py:742:lmcache.v1.cache_engine)
   (APIServer pid=39503) INFO:     127.0.0.1:34028 - "POST /v1/completions HTTP/1.1" 200 OK
   (APIServer pid=39503) INFO 01-28 06:26:43 [loggers.py:257] Engine 000: Avg prompt throughput: 481.2 tokens/s, Avg generation throughput: 5.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 47.9%



.. code-block:: text

   External prefix cache hit rate: 47.9%

This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.
