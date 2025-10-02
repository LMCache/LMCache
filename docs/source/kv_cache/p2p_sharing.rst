.. _p2p_sharing:

P2P KV Cache Sharing
====================

P2P (Peer-to-Peer) KV cache sharing enables direct cache transfer between multiple serving engine instances without requiring a centralized cache server. This approach provides high-performance cache sharing with reduced latency and improved scalability, especially beneficial in distributed inference scenarios.

LMCache supports P2P sharing through a controller-based architecture using NIXL (NVIDIA Inference Xfer Library) for optimized data transfer between instances.

Prerequisites
-------------

- **Multi-GPU Setup**: Your server should have at least 2 GPUs
- **NIXL**: Install from `NIXL <https://github.com/ai-dynamo/nixl>`_
- **LMCache**: Install from :ref:`installation_guide`

Configuration
-------------

Create two configuration files for the P2P sharing setup.

The only difference between the two configurations is the ``lmcache_instance_id`` and the ``p2p_init_ports`` and ``p2p_lookup_ports`` and ``lmcache_worker_ports``.

**Instance 1 Configuration (example1.yaml)**:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5
    enable_async_loading: True

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8200
    p2p_lookup_ports: 8201
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_1"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8500

    extra_config:
      lookup_backoff_time: 0.001

**Instance 2 Configuration (example2.yaml)**:

.. code-block:: yaml

    chunk_size: 256
    local_cpu: True
    max_local_cpu_size: 5
    enable_async_loading: True

    # P2P configurations
    enable_p2p: True
    p2p_host: "localhost"
    p2p_init_ports: 8202
    p2p_lookup_ports: 8203
    transfer_channel: "nixl"

    # Controller configurations
    enable_controller: True
    lmcache_instance_id: "lmcache_instance_2"
    controller_pull_url: "localhost:8300"
    controller_reply_url: "localhost:8400"
    lmcache_worker_ports: 8501

    extra_config:
      lookup_backoff_time: 0.001

Setup and Usage
---------------

**Step 1: Start the LMCache Controller**

.. code-block:: bash

    PYTHONHASHSEED=123 lmcache_controller --host localhost --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'

Make sure that the 8300 and 8400 ports are set up in **controller_pull_url** and **controller_reply_url** in the configuration files.
Port 9000 is the controller main port, which is arbitrary and can be changed.

**Step 2: Start vLLM Engines with LMCache Workers**

Start vLLM engine 1 at port 8010:

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=example1.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8010 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

Start vLLM engine 2 at port 8011:

.. code-block:: bash

    PYTHONHASHSEED=123 UCX_TLS=rc CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=example2.yaml \
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --gpu-memory-utilization 0.8 \
        --port 8011 \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3: Test P2P Cache Sharing**

Send a request to vLLM engine 1 to populate the cache:

.. code-block:: bash

    curl -X POST http://localhost:8010/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Send the same request to vLLM engine 2 to demonstrate cache retrieval from **engine 1**:

.. code-block:: bash

    curl -X POST http://localhost:8011/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"meta-llama/Meta-Llama-3.1-8B-Instruct\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

Expected Output
---------------

When the second request successfully retrieves cache from the first instance, you should see logs similar to:

.. code-block:: bash

    (EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,706] LMCache INFO:[0m Established connection to peer_init_url localhost:8200. The peer_lookup_url: localhost:8201 (p2p_backend.py:278:lmcache.v1.storage_backend.p2p_backend)
    (EngineCore_DP0 pid=2577584)[2025-09-21 00:00:11,792] LMCache INFO: Retrieved 1002 out of total 1002 out of total 1002 tokens. size: 0.1223 gb, cost 60.3595 ms, throughput: 2.0264 GB/s; (cache_engine.py:496:lmcache.v1.cache_engine)

These logs indicate successful P2P connection establishment and high-throughput cache retrieval.