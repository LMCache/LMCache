Cookbook: Qwen-3-8B
====================================

Qwen Usage Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Qwen-3** is the next-generation open-source large language model (LLM) released by **Alibaba Group**, renowned for its exceptional performance, robust multilingual understanding, and wide applicability.  
The **Qwen-3-8B** model strikes an excellent balance between computational efficiency and inference quality, making it ideal for medium-to-large-scale applications such as inference services, conversational systems, and intelligent assistants.

**LMCache** provides deep integration with the Qwen series and other mainstream models, optimizing their KV Cache management and context reuse mechanisms.  
This significantly reduces **Time to First Token (TTFT)** and overall inference latency.

------------------------------------
Model Highlights
------------------------------------

- **Efficient inference**: Achieves both high precision and low latency with 8B parameters.  
- **Long-context capability**: Supports input lengths of tens of thousands of tokens.  
- **High compatibility**: Fully compatible with the **OpenAI API** format.  
- **Multi-task adaptability**: Well-suited for Q&A, summarization, code generation, and multi-turn dialogue.  
- **LMCache optimization**: Greatly reduces redundant computation through cache reuse.

------------------------------------
Deploying Qwen-3-8B with LMCache
------------------------------------

The following example demonstrates how to quickly deploy the **Qwen-3-8B** model using **LMCache**:

.. code-block:: bash

   # Note: The chunk size below is for demonstration only.
   # It is recommended to use the default value (256) in production.
   # The model parameter may refer to either a local path or a Hugging Face model name.
   LMCACHE_CHUNK_SIZE=8 vllm serve qwen/Qwen-3-8B-Instruct --port 8000 --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

This service is fully **OpenAI API-compatible**, meaning you can directly send requests using `curl` or the `openai` SDK:

.. code-block:: bash

   curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
       "model": "qwen/Qwen-3-8B-Instruct",
       "prompt": "Briefly introduce the architecture and use cases of the Qwen model.",
       "max_tokens": 100,
       "temperature": 0.7
   }'

Upon successful deployment, the server will output LMCache logs similar to the following:

.. code-block:: bash

   (EngineCore_DP0 pid=3284044) [2025-10-20 08:56:03,487] LMCache INFO: Reqid: cmpl-128628cf86ee43788cc7f2599c855e2b-0, Total tokens 14, LMCache hit tokens: 0, need to load: 0 (vllm_v1_adapter.py:1189:lmcache.integration.vllm.vllm_v1_adapter)
   (EngineCore_DP0 pid=3284044) [2025-10-20 08:56:03,519] LMCache INFO: Storing KV cache for 14 out of 14 tokens (skip_leading_tokens=0) for request cmpl-128628cf86ee43788cc7f2599c855e2b-0 (vllm_v1_adapter.py:1077:lmcache.integration.vllm.vllm_v1_adapter)
   (EngineCore_DP0 pid=3284044) [2025-10-20 08:56:03,520] LMCache INFO: Stored 14 out of total 14 tokens. size: 0.0019 gb, cost 1.0830 ms, throughput: 1.7753 GB/s; offload_time: 1.0501 ms, put_time: 0.0328 ms (cache_engine.py:294:lmcache.v1.cache_engine)

**Log Explanation:**

- **Total tokens 14**: The request contains 14 tokenized segments.  
- **LMCache hit tokens: 0**: No tokens were served from cache.  
- **Need to load: 0**: No tokens need to be loaded from LMCache.  
- **Storing KV cache for 14 out of 14 tokens**: The newly processed 14 tokens are hashed and stored in CPU memory for reuse in future requests.  

🎉 **At this point, LMCache is automatically caching and reusing KV states, dramatically cutting redundant computation.**

------------------------------------
Benchmarking
------------------------------------

We conducted standardized performance evaluations on **Qwen-3-8B-Instruct**, comparing **vLLM baseline** and **LMCache-accelerated** modes under long-context workloads with one Nvidia A40 GPU.

`````````````````````````````````````
Long Doc QA Benchmark
`````````````````````````````````````

**Long Doc QA** is a benchmarking tool designed to simulate long-context inference workloads, testing model responsiveness with extended inputs.  
Supported parameters include:

- ``--document-length``: Document length (default: 10,000 tokens)  
- ``--num-documents``: Number of documents (default: 20)  
- ``--output-len``: Output length per request (default: 100 tokens)  
- ``--repeat-mode``: Repetition mode (``random``, ``tile``, or ``interleave``)  
- ``--repeat-count``: Number of repetitions  

LMCache provides an automatic recommender script to optimize settings based on hardware configuration:

.. code-block:: bash

   python benchmarks/long_doc_qa/long_doc_qa_recommender.py --model qwen/Qwen-3-8B-Instruct

------------------------------------
Example Commands
------------------------------------

.. code-block:: bash

   # 1️⃣ vLLM Deployment
   PYTHONHASHSEED=0 vllm serve qwen/Qwen-3-8B-Instruct --tensor-parallel-size 1 --load-format dummy

   # 2️⃣ LMCache Deployment
   PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=66 vllm serve qwen/Qwen-3-8B-Instruct        --tensor-parallel-size 1 --load-format dummy        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

   # 3️⃣ Long Document QA Workload
   python benchmarks/long_doc_qa/long_doc_qa.py        --model qwen/Qwen-3-8B-Instruct        --num-documents 46        --document-length 10000        --output-len 100        --repeat-count 1        --repeat-mode tile        --max-inflight-requests 4

------------------------------------
Performance Metrics
------------------------------------

`````````````````````````````````````
vLLM Mode
`````````````````````````````````````

.. code-block:: bash

   Warmup round mean TTFT: 2.933s
   Warmup round time: 124.902s
   Warmup round prompt count: 46

   === BENCHMARK RESULTS ===
   Query round mean TTFT: 2.956s
   Query round time: 125.415s
   Query round prompt count: 46

`````````````````````````````````````
LMCache Mode
`````````````````````````````````````

.. code-block:: bash

   Warmup round mean TTFT: 3.073s
   Warmup round time: 128.198s
   Warmup round prompt count: 46

   === BENCHMARK RESULTS ===
   Query round mean TTFT: 0.393s
   Query round time: 52.351s
   Query round prompt count: 46

------------------------------------------
Performance Comparison and Conclusion
------------------------------------------

+-------------------+-------------+---------------+--------------+
| Metric            | vLLM Mode   | LMCache Mode  | Improvement  |
+===================+=============+===============+==============+
| Warmup Avg TTFT   | 2.933s      | 3.073s        | —            |
+-------------------+-------------+---------------+--------------+
| Query Avg TTFT    | 2.956s      | **0.393s**    | ✅ **86.7%** |
+-------------------+-------------+---------------+--------------+
| Total Query Time  | 125.415s    | **52.351s**   | ✅ **58.2%** |
+-------------------+-------------+---------------+--------------+

The results clearly show that **LMCache** effectively reduces the **Time to First Token (TTFT)** and cuts overall inference latency by more than **50%**.  
Thanks to its efficient KV Cache reuse and multi-tier caching design, **Qwen-3-8B** delivers outstanding performance for long-context scenarios.

------------------------------------
Summary
------------------------------------

With LMCache acceleration, **Qwen-3-8B** achieves remarkable inference efficiency while maintaining high-quality generation.  
It is particularly well-suited for high-frequency interaction, long-document processing, and context-heavy workloads.
