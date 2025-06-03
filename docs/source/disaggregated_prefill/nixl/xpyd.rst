XpYd
====

X Prefiller, Y Decoder (XpYd) Example
--------------------------------------

This example demonstrates how to run LMCache with disaggregated prefill using NIXL on a single node with multiple prefiller and decoder instances. This configuration allows for horizontal scaling of both the compute-intensive prefill operations and the decode operations, enabling better resource utilization and higher throughput.

Architecture Overview
~~~~~~~~~~~~~~~~~~~~~

The XpYd setup consists of multiple components that can be scaled independently:

1. **Multiple Prefiller Servers** - Handle the prefill phase of inference (initial prompt processing)
2. **Multiple Decoder Servers** - Handle the decode phase of inference (token generation) 
3. **Proxy Server** - Coordinates requests between prefillers and decoders using round-robin load balancing

Example 2p1d Architecture:

.. code-block::

                ┌─────────────┐
                │   Client    │
                └─────┬───────┘
                      │
              ┌───────▼───────┐
              │ Proxy Server  │
              │   Port 9000   │--------------|
              │ (Round-Robin) │              |
              └───┬───────┬───┘              |
                  │       │                  |
         ┌────────▼──┐  ┌─▼────────┐         |
         │Prefiller1 │  │Prefiller2│         |
         │Port 8100  │  │Port 8101 │         |
         │  GPU 0    │  │  GPU 1   │         |
         └─────┬─────┘  └─────┬────┘         |
               │              │              |
               └──────┬───────┘              |
                      │ NIXL Transfer        |
                ┌─────▼─────┐                |
                │  Decoder  │                |
                │Port 8200  │<---------------|
                │  GPU 2    │                  
                └───────────┘

Prerequisites
~~~~~~~~~~~~~

- **LMCache**: Install with ``pip install lmcache``
- **NIXL**: Install from `NIXL GitHub repository <https://github.com/ai-dynamo/nixl>`_
- **Hardware**: At least 3 GPUs (2 for prefillers + 1 for decoder in 2p1d setup)
- **Model Access**: Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct

Quick Start (2p1d Example)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Set your Hugging Face token**:

   .. code-block:: bash

      export HF_TOKEN=hf_your_token_here

2. **Navigate to the example directory**:

   .. code-block:: bash

      cd examples/disagg_prefill/xp1d

3. **Run the example**:

   .. code-block:: bash

      bash disagg_example_xp1d.sh

The script will automatically:

- Launch a decoder instance on port 8200 (GPU 2)
- Launch two prefiller instances on ports 8100 and 8101 (GPU 0 and GPU 1)
- Launch a proxy server on port 9000 with round-robin load balancing
- Wait for all servers to be ready

Press ``Ctrl+C`` to stop all servers.

Configuration
~~~~~~~~~~~~~

Prefiller Configuration
^^^^^^^^^^^^^^^^^^^^^^^

All prefillers share the same configuration via ``configs/lmcache-prefiller-config.yaml``:

.. code-block:: yaml

   local_cpu: False
   max_local_cpu_size: 0
   max_local_disk_size: 0
   remote_serde: NULL

   enable_nixl: True
   nixl_role: "sender"
   nixl_peer_host: "localhost"
   nixl_peer_port: 55555
   nixl_buffer_size: 1073741824 # 1GB
   nixl_buffer_device: "cuda"
   nixl_enable_gc: True

Key settings:
- ``nixl_role: "sender"`` - Configures these instances to send KV cache data
- ``nixl_buffer_size: 1GB`` - Buffer size for NIXL transfers
- ``nixl_buffer_device: "cuda"`` - Uses GPU memory for buffering

Decoder Configuration
^^^^^^^^^^^^^^^^^^^^^

The decoder(s) are configured via ``configs/lmcache-decoder-config.yaml``:

.. code-block:: yaml

   local_cpu: False
   max_local_cpu_size: 0
   max_local_disk_size: 0
   remote_serde: NULL

   enable_nixl: True
   nixl_role: "receiver"
   nixl_peer_host: "localhost"
   nixl_peer_port: 55555
   nixl_buffer_size: 1073741824 # 1GB
   nixl_buffer_device: "cuda"
   nixl_enable_gc: True

Key settings:
- ``nixl_role: "receiver"`` - Configures these instances to receive KV cache data
- Same buffer configuration as the prefillers for compatibility

Components Deep Dive
~~~~~~~~~~~~~~~~~~~~

Proxy Server (disagg_proxy_server.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The proxy server coordinates the multi-prefiller disaggregated workflow:

1. **Request Handling**: Receives client requests on port 9000
2. **Load Balancing**: Distributes requests across multiple prefillers using round-robin
3. **Prefill Coordination**: Sends requests to prefillers with ``max_tokens=1``
4. **Response Streaming**: Streams the full response from the decoder
5. **Performance Monitoring**: Tracks Time-To-First-Token (TTFT) statistics

Key features:
- **Round-robin distribution**: Balances load across ``--num-prefillers`` instances
- **Fault tolerance**: Handles prefiller failures gracefully
- **Monitoring**: Provides detailed TTFT statistics for each prefiller

Supported endpoints:
- ``/v1/completions``
- ``/v1/chat/completions``

vLLM Server Launcher (disagg_vllm_launcher.sh)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This script launches individual vLLM servers with appropriate configurations:

**Prefiller1 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
   LMCACHE_CONFIG_FILE=configs/lmcache-prefiller-config.yaml \
   VLLM_ENABLE_V1_MULTIPROCESSING=1 \
   VLLM_WORKER_MULTIPROC_METHOD=spawn \
   CUDA_VISIBLE_DEVICES=0 \
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --port 8100 \
       --disable-log-requests \
       --enforce-eager \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'

**Prefiller2 Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
   LMCACHE_CONFIG_FILE=configs/lmcache-prefiller-config.yaml \
   VLLM_ENABLE_V1_MULTIPROCESSING=1 \
   VLLM_WORKER_MULTIPROC_METHOD=spawn \
   CUDA_VISIBLE_DEVICES=1 \
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --port 8101 \
       --disable-log-requests \
       --enforce-eager \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer2"}}'

**Decoder Launch Command**:

.. code-block:: bash

   UCX_TLS=cuda_ipc,cuda_copy,tcp \
   LMCACHE_CONFIG_FILE=configs/lmcache-decoder-config.yaml \
   VLLM_ENABLE_V1_MULTIPROCESSING=1 \
   VLLM_WORKER_MULTIPROC_METHOD=spawn \
   CUDA_VISIBLE_DEVICES=2 \
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
       --port 8200 \
       --disable-log-requests \
       --enforce-eager \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'

Key differences from 1p1d:
- Each prefiller gets a unique ``lmcache_rpc_port`` (producer1, producer2, etc.)
- Each prefiller runs on a different GPU (CUDA_VISIBLE_DEVICES)
- Different ports for each prefiller (8100, 8101, etc.)

Basic Test
~~~~~~~~~~

Once all servers are running, you can test with a simple curl command:

.. code-block:: bash

   curl -X POST http://localhost:9000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Llama-3.1-8B-Instruct",
       "prompt": "The future of AI is",
       "max_tokens": 50,
       "temperature": 0.7
     }'

Performance Benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^

For comprehensive performance testing, use vLLM's benchmark tool:

.. code-block:: bash

   python benchmark_serving.py --port 9000 --seed $(date +%s) \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --dataset-name random --random-input-len 7500 --random-output-len 200 \
       --num-prompts 30 --burstiness 100 --request-rate 1 --ignore-eos

Expected performance improvements with 2p1d:
- **Higher throughput**: Multiple prefillers can handle more concurrent requests
- **Better TTFT**: Load balancing reduces queuing delays
- **Improved utilization**: Better GPU utilization across multiple devices

Sample benchmark results:

.. code-block::

   ============ Serving Benchmark Result ============
   Successful requests:                     30
   Benchmark duration (s):                  31.34
   Total input tokens:                      224970
   Total generated tokens:                  6000
   Request throughput (req/s):              0.96
   Output token throughput (tok/s):         191.44
   Total Token throughput (tok/s):          7369.36
   ---------------Time to First Token----------------
   Mean TTFT (ms):                          313.41
   Median TTFT (ms):                        272.83
   P99 TTFT (ms):                           837.32
   ===============================================

Log Files and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

The example generates multiple log files for comprehensive monitoring:

- ``prefiller1.log`` - First prefiller server logs and errors
- ``prefiller2.log`` - Second prefiller server logs and errors  
- ``decoder.log`` - Decoder server logs and errors
- ``proxy.log`` - Proxy server logs and TTFT statistics

The proxy server provides detailed statistics for each prefiller:

.. code-block::

   ===============================
   Num requests: 20
   Prefiller 1 TTFT stats:
    - Average (ms): 42.3
    - Median (ms): 40.1
    - 99th Percentile (ms): 48.7
   Prefiller 2 TTFT stats:
    - Average (ms): 43.8
    - Median (ms): 41.5
    - 99th Percentile (ms): 52.1
   ===============================

This helps identify performance differences between prefiller instances and optimize load balancing.

Troubleshooting
~~~~~~~~~~~~~~~

Common Issues
^^^^^^^^^^^^^

1. **GPU Memory**: Ensure each GPU has sufficient memory for the model
2. **NIXL Installation**: Verify NIXL is properly installed and accessible
3. **Port Conflicts**: Check that all required ports are available
4. **HF Token**: Ensure your Hugging Face token has access to Llama models
5. **GPU Assignment**: Verify CUDA_VISIBLE_DEVICES assignments don't conflict

Multi-Instance Specific Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Uneven Load**: Monitor prefiller statistics to ensure balanced distribution
2. **Resource Contention**: Watch for GPU memory pressure with multiple instances
3. **Network Bottlenecks**: Monitor NIXL transfer performance between instances
4. **Startup Timing**: Stagger prefiller launches to avoid resource conflicts



