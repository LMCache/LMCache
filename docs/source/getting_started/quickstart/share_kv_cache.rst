.. _share_kv_cache:

Example: Share KV cache across multiple LLMs
============================================

In this example, we will show you how to share KV cache across multiple LLM instances using LMCache.

.. note::
    LMCache supports two main approaches for sharing KV cache across instances:
    
    - **Centralized sharing**: Using a centralized cache server
    - **P2P sharing**: Using peer-to-peer cache transfer

Prerequisites
-------------

Before you begin, make sure you have:

- vLLM v1 with LMCache installed (see :doc:`Installation <../installation>`)
- At least 2 GPUs for running multiple instances
- `Logged into HuggingFace <https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login>`_ using a token with gated access permission (required for model downloads)
- Redis server (for P2P sharing) or LMCache server (for centralized sharing)

Centralized KV cache sharing
----------------------------

This section demonstrates how to share KV cache across multiple vLLM instances using a centralized LMCache server.

Setup centralized sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~

First, create a configuration file named ``lmcache_config.yaml`` with the following content:

.. code-block:: yaml

    # Basic configurations
    chunk_size: 256
    local_cpu: true
    
    # Centralized sharing configurations
    remote_url: "lm://localhost:65432"
    remote_serde: "cachegen"
    
    # Whether retrieve() is pipelined or not
    pipelined_backend: false

Run centralized sharing example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start the LMCache centralized server:**

   .. code-block:: bash

       lmcache_server localhost 65432

2. **Launch the first vLLM instance on GPU 0:**

   .. code-block:: bash

       LMCACHE_CONFIG_FILE=lmcache_config.yaml \
       CUDA_VISIBLE_DEVICES=0 \
       vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
           --gpu-memory-utilization 0.8 \
           --port 8000 \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

3. **Launch the second vLLM instance on GPU 1:**

   .. code-block:: bash

       LMCACHE_CONFIG_FILE=lmcache_config.yaml \
       CUDA_VISIBLE_DEVICES=1 \
       vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
           --gpu-memory-utilization 0.8 \
           --port 8001 \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

   Wait until both engines are ready.

4. **Send a request to the first instance:**

   .. code-block:: bash

       curl -X POST http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
           "prompt": "Explain the significance of KV cache in language models.",
           "max_tokens": 100
         }'

5. **Send the same request to the second instance:**

   .. code-block:: bash

       curl -X POST http://localhost:8001/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
           "prompt": "Explain the significance of KV cache in language models.",
           "max_tokens": 100
         }'

The second request will automatically retrieve and reuse the KV cache from the first instance, significantly reducing generation time.

P2P KV cache sharing
--------------------

This section demonstrates how to share KV cache across multiple vLLM instances using peer-to-peer transfer.

Setup P2P sharing
~~~~~~~~~~~~~~~~~~

Create two configuration files for the P2P sharing setup:

**Instance 1 configuration (``lmcache_config1.yaml``):**

.. code-block:: yaml

    # Basic configurations
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    
    # P2P configuration
    enable_p2p: true
    lookup_url: "localhost:8100"
    distributed_url: "localhost:8200"

**Instance 2 configuration (``lmcache_config2.yaml``):**

.. code-block:: yaml

    # Basic configurations
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5
    
    # P2P configuration
    enable_p2p: true
    lookup_url: "localhost:8100"
    distributed_url: "localhost:8201"

Run P2P sharing example
~~~~~~~~~~~~~~~~~~~~~~~

1. **Start Redis server as lookup service:**

   .. code-block:: bash

       docker pull redis
       docker run --name lmcache-redis -d -p 8100:6379 redis

2. **Launch the first vLLM instance on GPU 0:**

   .. code-block:: bash

       CUDA_VISIBLE_DEVICES=0 \
       LMCACHE_CONFIG_FILE=lmcache_config1.yaml \
       vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
           --max-model-len 4096 \
           --gpu-memory-utilization 0.8 \
           --port 8000 \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

3. **Launch the second vLLM instance on GPU 1:**

   .. code-block:: bash

       CUDA_VISIBLE_DEVICES=1 \
       LMCACHE_CONFIG_FILE=lmcache_config2.yaml \
       vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
           --max-model-len 4096 \
           --gpu-memory-utilization 0.8 \
           --port 8001 \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

   .. note::
       The two distributed cache servers will automatically start at ports 8200 and 8201.

4. **Send a request to the first instance:**

   .. code-block:: bash

       curl -X POST http://localhost:8000/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
           "prompt": "Explain the significance of KV cache in language models.",
           "max_tokens": 100
         }'

5. **Send the same request to the second instance:**

   .. code-block:: bash

       curl -X POST http://localhost:8001/v1/completions \
         -H "Content-Type: application/json" \
         -d '{
           "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
           "prompt": "Explain the significance of KV cache in language models.",
           "max_tokens": 100
         }'

The cache will be automatically retrieved from the first vLLM instance through P2P transfer.

Understanding the benefits
--------------------------

Sharing KV cache across multiple LLM instances provides several advantages:

**Performance Benefits:**
- Significantly reduces generation time for requests with shared prefixes
- Eliminates redundant computation across instances
- Enables efficient resource utilization in multi-instance deployments

**Use Cases:**
- Multi-tenant serving environments
- Load balancing across multiple GPU instances  
- Distributed inference setups
- Development and testing with multiple model instances

**Cache Hit Indicators:**

When KV cache sharing is working correctly, you'll see log messages similar to:

.. code-block:: text

    LMCache INFO: Storing KV cache for 31 out of 31 tokens for request cmpl-xxx
    LMCache INFO: Total tokens 31, LMCache hit tokens: 30, need to load: 14

The second line indicates successful cache retrieval from another instance.

Centralized vs P2P sharing
---------------------------

Choose the appropriate sharing method based on your deployment needs:

**Centralized Sharing:**
- Best for: Stable, long-running deployments
- Pros: Simple setup, centralized management, persistent cache
- Cons: Single point of failure, additional server overhead

**P2P Sharing:**
- Best for: Dynamic, distributed deployments
- Pros: No single point of failure, direct peer communication
- Cons: More complex setup, requires service discovery (Redis)

Configuration Parameters
-------------------------

**Common Parameters:**
- ``chunk_size``: Size of cache chunks (default: 256)
- ``local_cpu``: Enable local CPU caching (recommended: true)
- ``max_local_cpu_size``: CPU memory limit in GB

**Centralized Sharing:**
- ``remote_url``: LMCache server URL (format: ``lm://host:port``)
- ``remote_serde``: Serialization format (recommended: ``cachegen``)
- ``pipelined_backend``: Enable pipelined retrieval (default: false)

**P2P Sharing:**
- ``enable_p2p``: Enable P2P mode (required: true)
- ``lookup_url``: Redis lookup server URL
- ``distributed_url``: Local cache server URL (unique per instance)

Troubleshooting
---------------

**Common Issues:**

1. **Cache not being shared:**
   - Verify all instances use the same model and configuration
   - Check network connectivity between instances
   - Ensure proper port accessibility

2. **Performance not improving:**
   - Confirm requests have shared prefixes
   - Check cache hit rates in logs
   - Verify sufficient CPU memory allocation

3. **Connection issues:**
   - Ensure Redis server is running (P2P mode)
   - Verify LMCache server is accessible (centralized mode)
   - Check firewall settings for required ports 