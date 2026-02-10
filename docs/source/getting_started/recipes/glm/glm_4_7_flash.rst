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
        --no-enable-prefix-caching \
        --port 8000 --kv-transfer-config \
       '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


.. note::
   ``Prefix caching`` is disabled so that all reuse comes from LMCache, making cache hits and TTFT deltas easier to interpret. For real deployments, keep vLLM prefix caching enabled and size the CPU cache appropriately.
   Setting ``PYTHONHASHSEED=0`` is recommended for deterministic chunk hashing, especially when scaling to multiple processes or instances. You can remove it in production.


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
       "model": "zai-org/GLM-4.7-Flash",
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
   (APIServer pid=78475) INFO 01-29 04:52:53 [loggers.py:257] Engine 000: Avg prompt throughput: 401.0 tokens/s, Avg generation throughput: 6.4 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 44.7%


.. code-block:: text

   External prefix cache hit rate: 44.7%

This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.


Benchmark  
----------------------------------

.. code-block:: bash

  vllm bench serve \
   --model zai-org/GLM-4.7-Flash \
   --dataset-name random \
   --random-input 2048 \
   --random-output 1024 \
   --request-rate 10 \
   --num-prompt 100 \
   --trust-remote-code


Vllm + LMCache 
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ---------------Time to First Token----------------
   Mean TTFT (ms):                          219.01
   Median TTFT (ms):                        186.22
   P99 TTFT (ms):                           757.41
   -----Time per Output Token (excl. 1st token)------
   Mean TPOT (ms):                          16.20
   Median TPOT (ms):                        16.27
   P99 TPOT (ms):                           18.39
   ---------------Inter-token Latency----------------
   Mean ITL (ms):                           16.20
   Median ITL (ms):                         13.99
   P99 ITL (ms):                            100.29