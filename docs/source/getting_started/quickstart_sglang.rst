.. _sglang_quickstart:

SGLang Quickstart
=================

This guide shows how to run LMCache with SGLang end to end in a couple of minutes. It mirrors the vLLM quickstart but uses the SGLang runtime.

(Terminal 1) Install LMCache and SGLang
---------------------------------------

Create a fresh environment and install the two packages:

.. code-block:: bash

   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install --prerelease=allow lmcache "sglang"

(Terminal 1) Start SGLang with LMCache
--------------------------------------

Set up a minimal LMCache config (chunk size 8 for demo; use the default 256 in real runs), then launch SGLang with LMCache enabled:

.. code-block:: bash

   cat > lmc_config.yaml <<'EOF'
   chunk_size: 8  # demo only; use 256 for production
   local_cpu: true
   use_layerwise: true
   max_local_cpu_size: "auto"
   EOF

   export LMCACHE_USE_EXPERIMENTAL=True
   export LMCACHE_CONFIG_FILE=$PWD/lmc_config.yaml

   python -m sglang.launch_server \
     --model-path Qwen/Qwen3-8B-Instruct \
     --host 0.0.0.0 \
     --port 30000 \
     --enable-lmcache

If you want to customize LMCache further, set the options in the config file. See the :doc:`../api_reference/configurations` page for all parameters.

(Terminal 2) Test LMCache in Action
-----------------------------------

Send a request through SGLang's OpenAI-compatible endpoint:

.. code-block:: bash

   curl http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B-Instruct",
       "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts"}],
       "max_tokens": 100,
       "temperature": 0.7
     }'

LMCache logs will show the first prompt being cached, for example:

.. code-block:: text

   [2025-09-30 00:08:43,982] LMCache INFO: Stored 27 out of total 27 tokens. size: 0.0037 gb, cost 1.8470 ms, throughput: 2.0075 GB/s; offload_time: 1.7962 ms, put_time: 0.0509 ms

Send another request with an overlapping prefix to see cache hits:

.. code-block:: bash

   curl http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B-Instruct",
       "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models"}],
       "max_tokens": 100,
       "temperature": 0.7
     }'

You should see hits and reloads similar to:

.. code-block:: text

   [2025-09-30 01:12:36,502] LMCache INFO: Retrieved 8 out of 24 required tokens (from 32 total tokens). size: 0.0011 gb, cost 0.5547 ms, throughput: 1.9808 GB/s;
   [2025-09-30 01:12:36,510] LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.4274 ms, throughput: 2.5702 GB/s; offload_time: 0.4013 ms, put_time: 0.0262 ms

**What to look for**

- The first request stores all tokens in LMCache.
- The second request shows cache hits (aligned to the 8-token chunk in this demo).
- LMCache only reloads the missing chunk and then stores the new part for future reuse.

Next Steps
----------

- Try the :doc:`benchmarking` guide for larger-scale measurements.
- Explore more setups in :doc:`quickstart/index`, including shared KV cache and disaggregated prefill.
