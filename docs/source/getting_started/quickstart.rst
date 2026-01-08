.. _quickstart:

Quickstart
==========

This guide helps you get LMCache running end-to-end in a couple of minutes. Use the tabs below to switch the engine (similar to an environment toggle). Steps are the same; only the libraries and launch commands change.

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **(Terminal 1) Install LMCache**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install lmcache vllm

      **Start vLLM with LMCache:**

      .. code-block:: bash

         # The chunk size here is only for illustration purpose, use default one (256) later
         LMCACHE_CHUNK_SIZE=8 \
         vllm serve Qwen/Qwen3-8B-Instruct \
             --port 8000 --kv-transfer-config \
             '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

      .. note::
         To customize further, create a config file. See :doc:`../api_reference/configurations` for all options.

      **Alternative simpler command:**

      .. code-block:: bash

         vllm serve <MODEL NAME> \
             --kv-offloading-backend lmcache \
             --kv-offloading-size <SIZE IN GB> \
             --disable-hybrid-kv-cache-manager

      The ``--disable-hybrid-kv-cache-manager`` flag is mandatory. All configuration options from the :doc:`../api_reference/configurations` page still apply.

   .. tab-item:: SGLang

      **(Terminal 1) Install SGLang**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install --prerelease=allow lmcache "sglang"

      **Start SGLang with LMCache**

      .. code-block:: bash

         cat > lmc_config.yaml <<'EOF'
         chunk_size: 8  # demo only; use 256 for production
         local_cpu: true
         use_layerwise: true
         max_local_cpu_size: 10  # GB
         EOF

         export LMCACHE_CONFIG_FILE=$PWD/lmc_config.yaml

         python -m sglang.launch_server \
           --model-path Qwen/Qwen3-14B-Instruct \
           --host 0.0.0.0 \
           --port 30000 \
           --enable-lmcache

      .. note::
         Configure LMCache via the config file. See :doc:`../api_reference/configurations` for the full list.

(Terminal 2) Test LMCache in Action
-----------------------------------

Open a new terminal. Pick your engine tab, send the first request, then an overlapping second request:

.. tab-set::

   .. tab-item:: vLLM

      **First request**

      .. code-block:: bash

         curl http://localhost:8000/v1/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B-Instruct",
             "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **Second request (overlap)**

      .. code-block:: bash

         curl http://localhost:8000/v1/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B-Instruct",
             "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
             "max_tokens": 100,
             "temperature": 0.7
           }'

   .. tab-item:: SGLang

      **First request**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-14B-Instruct",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **Second request (overlap)**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-14B-Instruct",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

You should see LMCache logs like this (examples for each engine):

.. tab-set::

   .. tab-item:: vLLM

      .. code-block:: text

         (EngineCore_DP0 pid=458469) [2025-09-30 00:08:43,982] LMCache INFO: Stored 31 out of total 31 tokens. size: 0.0040 gb, cost 1.95 ms, throughput: 1.98 GB/s; offload_time: 1.88 ms, put_time: 0.07 ms

   .. tab-item:: SGLang

      .. code-block:: text

         LMCache INFO: Stored 35 out of total 35 tokens. size: 0.0045 gb, cost 2.10 ms, throughput: 2.14 GB/s; offload_time: 2.00 ms, put_time: 0.10 ms

**What this means:** The first request caches the prompt. The second reuses the cached prefix and only loads the missing chunk. You should see logs like this:

.. tab-set::

   .. tab-item:: vLLM

      .. code-block:: text

         Reqid: cmpl-6709d8795d3c4464b01999c9f3fffede-0, Total tokens 32, LMCache hit tokens: 24, need to load: 8
         (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,502] LMCache INFO: Retrieved 8 out of 24 required tokens (from 32 total tokens). size: 0.0011 gb, cost 0.55 ms, throughput: 1.98 GB/s;
         (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,509] LMCache INFO: Storing KV cache for 8 out of 32 tokens (skip_leading_tokens=24)
         (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,510] LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.43 ms, throughput: 2.57 GB/s; offload_time: 0.40 ms, put_time: 0.03 ms

   .. tab-item:: SGLang

      .. code-block:: text

         Prefill batch: new-token=1, cached-token=34
         LMCache INFO: Retrieved 8 out of 24 required tokens (from 32 total tokens). size: 0.0011 gb, cost 0.55 ms, throughput: 1.98 GB/s
         LMCache INFO: Storing KV cache for 8 out of 32 tokens (skip_leading_tokens=24)
         LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.43 ms, throughput: 2.57 GB/s; offload_time: 0.40 ms, put_time: 0.03 ms

**What this means (per engine):**

.. tab-set::

   .. tab-item:: vLLM

      - **Total tokens 32**: The new prompt has 32 tokens after tokenization.
      - **LMCache hit tokens: 24**: 24 tokens (full 8-token chunks) were found in the cache from the first request that stored 31 tokens.
      - **Need to load: 8**: vLLM auto prefix caching uses block size 16; 16 tokens already sit in GPU RAM, so LMCache only loads 24-16=8.
      - **Why 24 hit tokens instead of 31?** LMCache hashes every 8 tokens (8, 16, 24, 31). It matches page-aligned chunks, so it uses the 24-token hash.
      - **Stored another 8 tokens**: The new 8 tokens form a full chunk and are stored for future reuse.

   .. tab-item:: SGLang

      - **Prefill batch line**: Shows new-token=1, cached-token=34, meaning most of the prefix is reused.
      - **Retrieved 8 of 24**: With chunk size 8, LMCache pulled the missing chunk(s) for the overlapping prefix.
      - **Store 8 of 32 tokens**: The new tail chunk is written back for future requests.
      - Chunk alignment and reuse follow the same 8-token hashing; counts differ slightly because SGLang reports prefill/cache stats explicitly.

🎉 **You now have LMCache caching and reusing KV caches for both engines.**

Next Steps
----------

- **Performance Testing**: Try the :doc:`benchmarking` section to experience LMCache's performance benefits with more comprehensive examples
- **More Examples**: Explore the :doc:`quickstart/index` section for detailed examples including KV cache sharing across instances and disaggregated prefill
