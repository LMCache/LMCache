.. _sharing_kv_cache:

Sharing
===============================================

LMCache can share the KV cache across multiple vLLM instances. LMCache supports sharing KV using the ``lmcache.server`` module.
Here is a quick example:

.. code-block:: console

   # Start lmcache server
   $ lmcache_server localhost 65432
   
   # Then, start two vLLM instances with the LMCache config file

   $ wget https://raw.githubusercontent.com/LMCache/LMCache/refs/heads/dev/examples/example.yaml
   
   # start the first vLLM instance
   $ LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8000
   
   # start the second vLLM instance
   $ LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8001


