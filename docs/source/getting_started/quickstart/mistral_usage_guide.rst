Cookbook: Mistral 7B
====================================

Mistral-7B-Instruct-v0.2 User Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mistral-7B-Instruct-v0.2** is a high-performance, open-source instruction-tuned large language model (LLM) developed by **Mistral AI**.  
Renowned for its efficient inference speed, strong multilingual capabilities, and robust instruction-following behavior, it is a popular choice for mid-scale conversational and reasoning tasks.

**LMCache** provides deep optimization support for mainstream large models, including the Mistral family. By efficiently managing KV caches and reusing context across requests, it significantly reduces both the *Time to First Token (TTFT)* and overall inference latency.

------------------------------------
Model Highlights
------------------------------------

- ⚡ **High Inference Efficiency**: Balanced performance and compute cost at the 7B parameter scale.  
- 🧠 **Strong Instruction Understanding**: Optimized for various instruction-following tasks, improving response accuracy.  
- 🌍 **Multilingual Support**: Natively supports multiple languages, including mixed Chinese-English input.  
- 🔄 **Long Context Handling**: Supports inputs of tens of thousands of tokens, suitable for document-level reasoning.  
- 🧰 **LMCache Acceleration**: Dramatically reduces computation overhead for repeated contexts.

-----------------------------------------------
Launching Mistral-7B-Instruct with LMCache
-----------------------------------------------

The following example demonstrates how to deploy **Mistral-7B-Instruct** with **LMCache** acceleration:

.. code-block:: bash

    PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=66 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --tensor-parallel-size 1 --load-format dummy --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'


This service is fully **OpenAI API compatible**, allowing inference requests via `curl` or the `openai` SDK.

-----------------------------------------------
Example Inference Request
-----------------------------------------------

.. code-block:: bash

    curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Introduce LLM and its usage",
    "max_tokens": 100,
    "temperature": 0.7
    }'


-----------------------------------------------
Log Example and Explanation
-----------------------------------------------

During runtime, LMCache automatically prints cache hit and store logs for KV data:

.. code-block:: bash

    (EngineCore_DP0 pid=2583371) [2025-10-17 13:26:31,345] LMCache INFO: Reqid: cmpl-d6f1552507e040cdbc0cf99ee82dcb89-0, Total tokens 9, LMCache hit tokens: 0, need to load: 0
    (EngineCore_DP0 pid=2583371) [2025-10-17 13:26:31,347] LMCache INFO: Post-initializing LMCacheEngine
    (EngineCore_DP0 pid=2583371) [2025-10-17 13:26:31,377] LMCache INFO: Storing KV cache for 9 out of 9 tokens (skip_leading_tokens=0) for request cmpl-d6f1552507e040cdbc0cf99ee82dcb89-0
    (EngineCore_DP0 pid=2583371) [2025-10-17 13:26:31,378] LMCache INFO: Stored 9 out of total 9 tokens. size: 0.0011 gb, cost 0.7749 ms, throughput: 1.4178 GB/s; offload_time: 0.7468 ms, put_time: 0.0281 ms

**Log Explanation:**

- **Total tokens**: Total token count in the current request.  
- **LMCache hit tokens**: Number of tokens retrieved from cache.  
- **Need to load**: Tokens that must be recomputed.  
- **Stored X out of total Y tokens**: KV data newly cached for reuse in later requests.  

By leveraging chunked caching and fast indexing, LMCache efficiently reuses context across multi-turn tasks, drastically reducing first-token latency.

---------------------
Benchmark Results
---------------------

We conducted standardized performance evaluations on **Mistral-7B-Instruct-v0.2**, comparing **vLLM baseline** and **LMCache-accelerated** modes under long-context workloads with one Nvidia A40 GPU.


`````````````````````````````````````
Workload Configuration
`````````````````````````````````````

- Document length: 10,000 tokens  
- Number of documents: 46  
- Output length: 100 tokens  
- Repeat count: 1  
- Mode: tile  
- Max inflight requests: 4  


`````````````````````````````````````
vLLM Mode
`````````````````````````````````````

.. code-block:: bash

    PYTHONHASHSEED=0 vllm serve /mnt/sda1/Mistral-7B-Instruct-v0.2 --tensor-parallel-size 1 --load-format dummy

    python benchmarks/long_doc_qa/long_doc_qa.py --model /mnt/sda1/Mistral-7B-Instruct-v0.2 --num-documents 46 --document-length 10000 --output-len 100 --repeat-count 1 --repeat-mode tile --max-inflight-requests 4


Results:

.. code-block:: bash

    Warmup round mean TTFT: 2.479s
    Warmup round time: 118.520s
    Warmup round prompt count: 46

    === BENCHMARK RESULTS ===
    Query round mean TTFT: 2.518s
    Query round time: 119.677s
    Query round prompt count: 46



`````````````````````````````````````
LMCache Mode
`````````````````````````````````````

.. code-block:: bash

    PYTHONHASHSEED=0 LMCACHE_MAX_LOCAL_CPU_SIZE=66 vllm serve /mnt/sda1/Mistral-7B-Instruct-v0.2 --tensor-parallel-size 1 --load-format dummy --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'

    python benchmarks/long_doc_qa/long_doc_qa.py --model /mnt/sda1/Mistral-7B-Instruct-v0.2 --num-documents 46 --document-length 10000 --output-len 100 --repeat-count 1 --repeat-mode tile --max-inflight-requests 4


Results:

.. code-block:: bash

    Warmup round mean TTFT: 2.616s
    Warmup round time: 122.226s
    Warmup round prompt count: 46

    === BENCHMARK RESULTS ===
    Query round mean TTFT: 0.322s
    Query round time: 48.761s
    Query round prompt count: 46



------------------------------------------
Performance Comparison and Conclusion
------------------------------------------

+-------------------+-------------+---------------+--------------+
| Metric            | vLLM Mode   | LMCache Mode  | Improvement  |
+===================+=============+===============+==============+
| Warmup Mean TTFT  | 2.479s      | 2.616s        | —            |
+-------------------+-------------+---------------+--------------+
| Query Mean TTFT   | 2.518s      | **0.322s**    | ✅ **87.2%** |
+-------------------+-------------+---------------+--------------+
| Query Total Time  | 119.677s    | **48.761s**   | ✅ **59.2%** |
+-------------------+-------------+---------------+--------------+

The results clearly show that **LMCache** dramatically reduces the *Time to First Token (TTFT)*, cutting total inference time by over **59%**.  
Thanks to intelligent KV cache reuse and hierarchical cache management, **Mistral-7B-Instruct-v0.2** achieves substantial performance gains in long-context reasoning scenarios.


------------------------------------------
📘 **Summary**  
------------------------------------------
With LMCache acceleration, Mistral-7B-Instruct-v0.2 reduces first-token latency to less than one-eighth of the native vLLM mode while maintaining identical output quality.  
This optimization delivers a markedly smoother user experience and significantly lowers inference costs in multi-turn and long-context workloads.
