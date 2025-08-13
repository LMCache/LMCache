.. _share_kv_cache:

Example: Share KV cache across multiple LLMs
============================================

LMCache should be able to reduce the generation time of the second and following calls.

We have examples for the following types of across-instance KV cache sharing:

- KV cache sharing through a centralized cache server: ``centralized_sharing``
- KV cache sharing through p2p cache transfer: ``p2p_sharing``

Prerequisites
-------------

Your server should have at least 2 GPUs.

For Centralized sharing, this will use the port 8000 and 8001 (for vLLM) and port 65432 (for LMCache).  

For P2P sharing, this will use the port 8000 and 8001 for 2 vllms,
And will use port 8200 and 8201 for 2 distributed cache servers,
And will use port 8100 for lookup server.

Centralized KV cache sharing
----------------------------

This section demonstrates how to share KV cache across multiple vLLM instances using a centralized LMCache server.

Setup centralized sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create a configuration file named ``lmcache_config.yaml`` with the following content:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    remote_url: "lm://localhost:65432"
    remote_serde: "cachegen"

Run centralized sharing example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Start the LMCache centralized server,

.. code-block:: bash

    lmcache_server localhost 65432

2. In a different terminal,

.. code-block:: bash

    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
    CUDA_VISIBLE_DEVICES=0 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8000 --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

In another terminal,

.. code-block:: bash

    LMCACHE_CONFIG_FILE=lmcache_config.yaml \
    CUDA_VISIBLE_DEVICES=1 \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8001 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Wait until both engines are ready.

3.  Send one request to the engine at port 8000,

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models.",
            "max_tokens": 10
        }'

4. Send the same request to the engine at port 8001,

.. code-block:: bash

    curl -X POST http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "prompt": "Explain the significance of KV cache in language models.",
            "max_tokens": 10
        }'

The second request will automatically retrieve and reuse the KV cache from the first instance, significantly reducing generation time.

P2P KV cache sharing
--------------------

This section demonstrates how to share KV cache across multiple vLLM instances using peer-to-peer transfer.

Setup P2P sharing
~~~~~~~~~~~~~~~~~~

Create two configuration files for the P2P sharing setup:

Instance 1 configuration (``lmcache_config1.yaml``):

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    
    # P2P configuration
    enable_p2p: true
    lookup_url: "localhost:8100"
    distributed_url: "localhost:8200"

Instance 2 configuration (``lmcache_config2.yaml``):

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    
    # P2P configuration
    enable_p2p: true
    lookup_url: "localhost:8100"
    distributed_url: "localhost:8201"

Run P2P sharing example
~~~~~~~~~~~~~~~~~~~~~~~

1. Pull redis docker and start lookup server at port 8100:

.. code-block:: bash

    docker pull redis
    docker run --name lmcache-redis -d -p 8100:6379 redis

2. Start two vllm engines:
   
Start vllm engine 1 at port 8000:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 \
    LMCACHE_CONFIG_FILE=lmcache_config1.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8 \
        --port 8000 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start vllm engine 2 at port 8001:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=1 \
    LMCACHE_CONFIG_FILE=lmcache_config2.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8 \
        --port 8001 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Note that the two distributed cache servers will start at port 8200 and 8201.

3. Send request to vllm engine 1:  

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 100
        }'

4. Send request to vllm engine 2:  

.. code-block:: bash

    curl -X POST http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "prompt": "Explain the significance of KV cache in language models.",
        "max_tokens": 100
        }'

The cache will be automatically retrieved from vllm engine 1.
