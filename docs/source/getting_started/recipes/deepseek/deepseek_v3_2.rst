.. deepseek_v3_2:

DeepSeek-V3.2 Usage Guide
==========================

Installing DeepGEMM
--------------------

.. code-block:: bash

   uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation

Note:
DeepGEMM is used in two places: MoE and MQA logits computation. It is necessary for MQA logits computation. If you want to disable the MoE part, you can set ``VLLM_USE_DEEP_GEMM=0`` in the environment variable. Some users reported that the performance is better with ``VLLM_USE_DEEP_GEMM=0``, e.g. on H20 GPUs. It might also be beneficial to disable DeepGEMM if you want to skip the long warmup.


Installing vLLM and LMCache
----------------------------

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
   uv pip install lmcache


LMCache Configuration
----------------------

Create ``deepseek_v3_2.yaml``:

.. code-block:: yaml

   chunk_size: 1024
   use_gpu_connector_v3: True
   local_cpu: True
   max_local_cpu_size: 5
   save_unfull_chunk: False
   extra_config:
     first_rank_max_local_cpu_size: 10
     save_only_first_rank: True


Launching DeepSeek-V3.2 on 8xH200
----------------------------------

The chat-template changes in the DeepSeek-V3.2 are quite significant. vLLM adapts to this through ``--tokenizer-mode deepseek_v32``.

.. code-block:: bash

   PYTHONHASHSEED=0 \
   LMCACHE_CONFIG_FILE=deepseek_v3_2.yaml \
   vllm serve deepseek-ai/DeepSeek-V3.2 \
     --tensor-parallel-size 8 \
     --tokenizer-mode deepseek_v32 \
     --tool-call-parser deepseek_v32 \
     --enable-auto-tool-choice \
     --reasoning-parser deepseek_v3 \
     --no-enable-prefix-caching \
     --port 8000 --kv-transfer-config \
     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'


.. note::
   ``Prefix caching`` is disabled so that all reuse comes from LMCache, making cache hits and TTFT deltas easier to interpret. For real deployments, keep vLLM prefix caching enabled and size the CPU cache appropriately.
   Setting ``PYTHONHASHSEED=0`` is recommended for deterministic chunk hashing, especially when scaling to multiple processes or instances.

 

Test LMCache in Action
-----------------------

Cold request (first run)
~~~~~~~~~~~~~~~~~~~~~~~~~

Send a long prompt (≥256 tokens) to force full chunk creation:

.. code-block:: bash

   python - <<'PY' | curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d @-
   import json
   prompt = "You are helpful.\n" + ("LMCache reuse test. " * 400)
   payload = {
       "model": "deepseek-ai/DeepSeek-V3.2",
       "prompt": prompt,
       "max_tokens": 32,
   }
   print(json.dumps(payload))
   PY

Expected LMCache logs (cold):

.. code-block:: bash

   (EngineCore_DP0 pid=609358) [2026-02-04 08:15:39,462] LMCache INFO: Reqid: cmpl-bfeea05d780f0266-0-9356f16d, Total tokens 2405, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP0 pid=609556) [2026-02-04 08:15:40,638] LMCache INFO: Stored 2048 out of total 2048 tokens. size: 0.1494 gb, cost 4.7811 ms, throughput: 31.2462 GB/s; offload_time: 4.6914 ms, put_time: 0.0897 ms (cache_engine.py:446:lmcache.v1.cache_engine)
   (APIServer pid=609141) INFO 02-04 08:15:42 [loggers.py:257] Engine 000: Avg prompt throughput: 240.5 tokens/s, Avg generation throughput: 2.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.5%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 0.0%
   (APIServer pid=609141) INFO:     127.0.0.1:41250 - "POST /v1/completions HTTP/1.1" 200 OK


Warm request (second run)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the same command again.

Expected LMCache logs (warm):

.. code-block:: bash

   (EngineCore_DP0 pid=609358) [2026-02-04 08:15:45,166] LMCache INFO: Reqid: cmpl-9d6484609001f713-0-9a817515, Total tokens 2405, LMCache hit tokens: 2048, need to load: 2048 (vllm_v1_adapter.py:1602:lmcache.integration.vllm.vllm_v1_adapter)
   (Worker_TP7 pid=609563) [2026-02-04 08:15:45,185] LMCache INFO: Retrieved 2048 out of 2048 required tokens (from 2048 total tokens). size: 0.0000 gb, cost 14.7022 ms, throughput: 0.0000 GB/s; (cache_engine.py:742:lmcache.v1.cache_engine)
   (APIServer pid=609141) INFO:     127.0.0.1:39256 - "POST /v1/completions HTTP/1.1" 200 OK
   (APIServer pid=609141) INFO 02-04 08:16:02 [loggers.py:257] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%, External prefix cache hit rate: 42.6%


.. code-block:: text

   External prefix cache hit rate: 42.6%

This is expected: vLLM labels KV supplied by external connectors (LMCache) as **external prefix cache**.
