.. _p2p:

P2P KV Cache Sharing
====================

This is an example to demonstrate P2P KV cache sharing.

Prerequisites
-------------

Your server should have at least 2 GPUs.

This will use the following ports:

- 8000 and 8001 for 2 vLLMs
- 8200 and 8201 for 2 distributed cache servers
- 8100 for the lookup server

Steps
-----

1. Pull Redis Docker and Start Lookup Server at Port 8100

.. code-block:: bash

    docker pull redis
    docker run --name some-redis -d -p 8100:6379 redis

2. Start Two vLLM Engines

   - Start vLLM engine 1 at port 8000:

   .. code-block:: bash

       CUDA_VISIBLE_DEVICES=0 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example1.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'

   - Start vLLM engine 2 at port 8001:

   .. code-block:: bash

       CUDA_VISIBLE_DEVICES=1 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example2.yaml vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'  

   Note that the two distributed cache servers will start at ports 8200 and 8201.

3. Send Request to vLLM Engine 1 

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 10
      }'

4. Send Request to vLLM Engine 2 

.. code-block:: bash

    curl -X POST http://localhost:8001/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 10
      }'

The cache will be automatically retrieved from vLLM engine 1.
