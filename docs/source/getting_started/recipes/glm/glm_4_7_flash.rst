.. glm_4_7_flash:

GLM-4.7-Flash Usage Guide
==========================

Installing vLLM, Transformer, LMCache
--------------------------------------

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
   uv pip install git+https://github.com/huggingface/transformers.git
   uv pip install lmcache


Launching GLM-4.7-Flash on 4xH200
----------------------------------

.. code-block:: bash

   PYTHONHASHSEED=0 \
   vllm serve zai-org/GLM-4.7-Flash \
        --tensor-parallel-size 4 \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        --served-model-name glm-4.7-flash \
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
       "model": "glm-4.7-flash",
       "prompt": prompt,
       "max_tokens": 32,
   }
   print(json.dumps(payload))
   PY

Expected LMCache logs (cold):

.. code-block:: bash

   (EngineCore_DP0 pid=78692) [2026-01-29 04:52:51,044] LMCache INFO: Reqid: cmpl-854d703d593e0f6c-0-96b17ffd, Total tokens 2005, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP0 pid=78890) [2026-01-29 04:52:51,168] LMCache INFO: Stored 1792 out of total 1792 tokens. size: 0.0904 gb, cost 3.6901 ms, throughput: 24.4881 GB/s; offload_time: 3.5892 ms, put_time: 0.1009 ms (cache_engine.py:446:lmcache.v1.cache_engine)
   (APIServer pid=78475) INFO:     127.0.0.1:43184 - "POST /v1/completions HTTP/1.1" 200 OK


Warm request (second run)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the same command again.

Expected LMCache logs (warm):

.. code-block:: bash

   (EngineCore_DP0 pid=78692) [2026-01-29 04:52:53,222] LMCache INFO: Reqid: cmpl-b35278010c36ebdf-0-b9e7c528, Total tokens 2005, LMCache hit tokens: 1792, need to load: 1792 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP1 pid=78891) [2026-01-29 04:52:53,236] LMCache INFO: Retrieved 1792 out of 1792 required tokens (from 1792 total tokens). size: 0.0000 gb, cost 9.9369 ms, throughput: 0.0000 GB/s; (cache_engine.py:742:lmcache.v1.cache_engine)
   (APIServer pid=78475) INFO:     127.0.0.1:43200 - "POST /v1/completions HTTP/1.1" 200 OK
   (APIServer pid=78475) INFO 01-29 04:52:53 [loggers.py:257] Engine 000: Avg prompt throughput: 481.2 tokens/s, Avg generation throughput: 5.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 47.9%


.. code-block:: text

   External prefix cache hit rate: 47.9%

This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.
