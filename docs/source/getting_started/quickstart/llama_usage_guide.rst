Cookbook: Llama-3.1-8B
====================================

Llama Usage Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Meta Llama** series is an open-source family of large language models (LLMs) developed by **Meta**, offering industry-leading performance.  
These models provide developers and researchers with diverse options for efficient deployment and large-scale applications.

**LMCache** provides extensive support for the Llama family of models — including `meta-llama/Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` — and continuously optimizes for performance.

------------------------------------
Llama 3.1 Model Highlights
------------------------------------

`Llama 3.1 <https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md>`_ is a major iteration in the Llama series, featuring multilingual capabilities, up to **128K context length**, and built-in **tool usage** support.  
The **8B model** is designed specifically for efficient development and deployment on consumer-grade GPUs.

------------------------------------
Starting Llama 3.1 with LMCache
------------------------------------

You can start the Llama 3.1 8B model server in **LMCache** with the following command:

.. code-block:: bash

   # The chunk size below is for demonstration purposes only.
   # It is recommended to use the default value (256) in production.
   # You may replace the model with a local path if preferred.
   LMCACHE_CHUNK_SIZE=8 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct        --port 8000 --kv-transfer-config        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

When you see the following INFO logs, it means the server has successfully started:

.. code-block:: bash

   (APIServer pid=459096) INFO:     Started server process [459096]
   (APIServer pid=459096) INFO:     Waiting for application startup.
   (APIServer pid=459096) INFO:     Application startup complete.

The server is **OpenAI API-compatible**, so you can interact with it using either the `openai` Python library or `curl`.  
For example, send your first test request in a new terminal:

.. code-block:: bash

   curl http://localhost:8000/v1/completions      -H "Content-Type: application/json"      -d '{
       "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
       "prompt": "Llama3.1 is the latest generation of large language models in the Llama series, offering a comprehensive suite of dense and mixture-of-experts",
       "max_tokens": 100,
       "temperature": 0.7
     }'

You should see the following logs on the server side:

.. code-block:: bash

   (EngineCore_DP0 pid=3085876) [2025-10-20 03:54:02,823] LMCache INFO: Storing KV cache for 30 out of 30 tokens (skip_leading_tokens=0) for request cmpl-7ee35d8834cb4c0aa346fa2c3ed60fa7-0 (vllm_v1_adapter.py:1077:lmcache.integration.vllm.vllm_v1_adapter)
   (EngineCore_DP0 pid=3085876) [2025-10-20 03:54:02,825] LMCache INFO: Stored 30 out of total 30 tokens. size: 0.0037 gb, cost 1.5041 ms, throughput: 2.4348 GB/s; offload_time: 1.4613 ms, put_time: 0.0428 ms (cache_engine.py:294:lmcache.v1.cache_engine)
   (EngineCore_DP0 pid=3085876) [2025-10-20 03:54:05,709] LMCache INFO: Calculated bytes per chunk per rank: 1048576 (local_cpu_backend.py:606:lmcache.v1.storage_backend.local_cpu_backend)
   (APIServer pid=3085709) INFO:     127.0.0.1:33864 - "POST /v1/completions HTTP/1.1" 200 OK

------------------------------------
Benchmark Results
------------------------------------

We conducted benchmarking on **Meta-Llama-3.1-8B-Instruct** to demonstrate the performance benefits of **LMCache** compared to **vLLM** with one Nvidia A40 GPU.

`````````````````````````````````````
Long Document QA Workload Generator
`````````````````````````````````````

**Long Doc QA** (located in ``benchmarks/long_doc_qa/``) is a flexible traffic simulator that sends long-context “document” queries to an inference service.  
Configurable parameters include:

- Document token length (default: 10,000)  
- Number of documents (default: 20)  
- Output tokens per request (default: 100)  
- Cache hit/miss ratio (e.g., ``2:2`` means two hits followed by two misses)  

You can also specify the number of prompt repetitions and repetition mode: ``random``, ``tile``, or ``interleave``.

LMCache provides a **Long Doc QA recommender** tool to help you deploy LMCache and generate realistic workloads.  
It can automatically suggest appropriate **tensor parallelism** and **CPU memory size** based on your hardware configuration.

.. code-block:: bash

   python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model <YOUR_MODEL_NAME>


`````````````````````````````````````
Example Usage
`````````````````````````````````````

.. code-block:: bash

   python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model meta-llama/Llama-3.1-8B-Instruct

   # The output depends on your hardware environment.
   # Directly copying it may not yield optimal performance.
   # We recommend running the recommender script on your own system.

   1. vLLM Deployment:
   ---------------------
   PYTHONHASHSEED=0 vllm serve meta-llama/Llama-3.1-8B-Instruct        --tensor-parallel-size 1 --load-format dummy

   2. LMCache Deployment:
   --------------------------
   PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=66        vllm serve meta-llama/Llama-3.1-8B-Instruct        --tensor-parallel-size 1 --load-format dummy        --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

   3. Multi-turn QA Workload Generation:
   ----------------------------------------
   python benchmarks/long_doc_qa/long_doc_qa.py        --model meta-llama/Llama-3.1-8B-Instruct        --num-documents 46        --document-length 10000        --output-len 100        --repeat-count 1        --repeat-mode tile        --max-inflight-requests 4

`````````````````````````````````````
Llama-3.1-8B-Instruct vLLM Metrics
`````````````````````````````````````

.. code-block:: bash

   Warm-up Avg TTFT: 2.767 s
   Warm-up Duration: 121.347 s
   Warm-up Prompts: 46

   === Benchmark Results ===
   Query Avg TTFT: 2.783 s
   Query Duration: 121.741 s
   Query Prompts: 46


```````````````````````````````````````
Llama-3.1-8B-Instruct LMCache Metrics
```````````````````````````````````````

.. code-block:: bash

   Warm-up Avg TTFT: 2.862 s
   Warm-up Duration: 124.216 s
   Warm-up Prompts: 46

   === Benchmark Results ===
   Query Avg TTFT: 0.353 s
   Query Duration: 50.966 s
   Query Prompts: 46

`````````````````````````````````````
Performance Summary
`````````````````````````````````````

With **LMCache offloading**, the **TTFT (Time to First Token)** was reduced by **87%** (from 2.783 s → 0.353 s),  
and total inference time decreased by **58%** (from 121.741 s → 50.966 s).
