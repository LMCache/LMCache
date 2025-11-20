.. _quickstart:

Quickstart
==========

This guide will help you get LMCache up and running quickly within 2 minutes. You'll see LMCache in action with a complete end-to-end example.

(Terminal 1) Install LMCache
----------------------------

First, install LMCache with these three commands:

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install lmcache vllm

Start vLLM with LMCache using a single command:

.. code-block:: bash

   # The chunk size here is only for illustration purpose, use default one (256) later
   LMCACHE_CHUNK_SIZE=8 \
   vllm serve Qwen/Qwen3-8B \
       --port 8000 --kv-transfer-config \
       '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

.. note::
   If you want to customize configurations further, you can create a configuration file. See the :doc:`../api_reference/configurations` page to learn about all available options.

Alternatively, you can start LMCache with vLLM using a simpler command:

.. code-block:: bash

   vllm serve <MODEL NAME> \
       --kv-offloading-backend lmcache \
       --kv-offloading-size <SIZE IN GB> \
       --disable-hybrid-kv-cache-manager

The ``--disable-hybrid-kv-cache-manager`` flag is mandatory. All configuration options from the :doc:`../api_reference/configurations` page still apply.

(Terminal 2) Test LMCache in Action
-----------------------------------

Now let's see LMCache working! Open a new terminal and send your first request:

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
       "max_tokens": 100,
       "temperature": 0.7
     }'

You should see LMCache logs like this:

.. code-block:: text

   (EngineCore_DP0 pid=458469) [2025-09-30 00:08:43,982] LMCache INFO: Stored 27 out of total 27 tokens. size: 0.0037 gb, cost 1.8470 ms, throughput: 2.0075 GB/s; offload_time: 1.7962 ms, put_time: 0.0509 ms

**What this means:** The 27 tokens from your prompt are being stored in CPU RAM because this is the first time the system processes this text. LMCache is caching the KV cache for future reuse.

Now send a second request with a prefix that overlaps with the first:

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
       "max_tokens": 100,
       "temperature": 0.7
     }'

You should see logs like this:

.. code-block:: text

   Reqid: cmpl-6709d8795d3c4464b01999c9f3fffede-0, Total tokens 32, LMCache hit tokens: 24, need to load: 8
   (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,502] LMCache INFO: Retrieved 8 out of total 8 out of total 24 tokens. size: 0.0011 gb, cost 0.5547 ms, throughput: 1.9808 GB/s;
   (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,509] LMCache INFO: Storing KV cache for 8 out of 32 tokens (skip_leading_tokens=24)
   (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,510] LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.4274 ms, throughput: 2.5702 GB/s; offload_time: 0.4013 ms, put_time: 0.0262 ms

**What this means:**

- **Total tokens 32**: The new prompt has 32 tokens after tokenization
- **LMCache hit tokens: 24**: 24 tokens were found in the cache (24 is a multiple of 8, our chunk size in this example)
- **Need to load: 8**: vLLM has automatic prefix caching enabled with block size 16. Although there are 24 hit tokens, 16 are already in GPU RAM managed by vLLM, so LMCache only needs to load 24-16=8 tokens
- **Why 24 hit tokens instead of 27?** LMCache hashes every 8 tokens incrementally (8, 16, 24, 27). When the new request comes in, it checks every 8-token chunk, so it uses the 24-token hash instead of checking the 27-token hash
- **Stored another 8 tokens**: The new 8 tokens form a complete chunk that gets hashed and stored in CPU RAM for future use

🎉 **Congratulations!** You've just seen LMCache automatically cache and reuse KV caches, reducing computation for overlapping text.

Next Steps
----------

- **Performance Testing**: Try the :doc:`benchmarking` section to experience LMCache's performance benefits with more comprehensive examples
- **More Examples**: Explore the :doc:`quickstart/index` section for detailed examples including KV cache sharing across instances and disaggregated prefill
