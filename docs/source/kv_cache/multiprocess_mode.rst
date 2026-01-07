(Experimental) LMCache Multi-process Mode
=========================================

LMCache multi-process mode allows you to run LMCache as a separate service in a different process.

During runtime, vLLM instances will establish a connection to the LMCache process and send requests to it.

In the future, each GPU node will have a single LMCache process running in multi-process mode. These LMCache processes
will be interconnected to form a distributed KV cache service.

.. note::
   This is an experimental feature and is under active development. Please expect breaking changes in the future.

.. note::
    Currently, the multi-process mode only supports CPU offloading without eviction. It is not recommended for production use.


Prerequisites
-------------

- vLLM version >= 0.11.1
- LMCache latest dev branch

Quick Start
-----------

**Step 1: Start the LMCache server**

Run the following command to start the LMCache server with a 100 GB CPU buffer:

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.server --cpu-buffer-size 100

You should see the following log output:

.. code-block:: text

    [2025-11-19 21:20:58,901] LMCache INFO: LMCache cache server is running... (server.py:483:__main__)

.. note::
    The default port for LMCache is 5555. It will accept connections from vLLM instances on this port.

**Step 2: Start vLLM with LMCacheMPConnector**

In a new terminal window, start vLLM with the LMCache connector:

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

You should see the following logs on the vLLM side:

.. code-block:: text

    (EngineCore_DP0 pid=3086423) [2025-11-19 23:10:25,072] LMCache INFO: Registering kv caches! (lmcache_mp_connector.py:405:vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector)
    (EngineCore_DP0 pid=3086423) [2025-11-19 23:10:25,072] LMCache INFO: Registering kv caches (multi_process_adapter.py:205:vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.multi_process_adapter)

You should also see the following logs on the LMCache side:

.. code-block:: text

    [2025-11-19 23:10:25,084] LMCache INFO: Registered KV cache for GPU ID 3086423 with 40 layers (server.py:215:__main__)

**Step 3: Send requests to the vLLM instance**

Send a test request with a repeated prompt to demonstrate caching:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"Qwen/Qwen3-14B\",
        \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
        \"max_tokens\": 10
    }"

On the first request, you should see the following logs on the LMCache side, indicating that tokens were stored in the cache:

.. code-block:: text

    [2025-11-19 23:24:39,547] LMCache INFO: Stored 768 tokens in 0.001 seconds (server.py:299:__main__)

If you send the same request again, you should see the following logs, indicating that tokens were retrieved from the cache:

.. code-block:: text

    [2025-11-19 23:24:47,312] LMCache INFO: Retrieved 768 tokens in 0.001 seconds (server.py:370:__main__)

Docker Deployment
-----------------

You can also run LMCache and vLLM in separate Docker containers. This approach is useful for deployment scenarios where you want to isolate the LMCache server from vLLM instances.

**Step 1: Start the LMCache standalone container**

Run the LMCache server in a Docker container:

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/standalone:nightly \
        /opt/venv/bin/python3 -m lmcache.v1.multiprocess.server \
        --cpu-buffer-size 60 --max-workers 4 --port 6555

.. note::
    We use ``--network host`` to allow the vLLM container to connect to the LMCache server on localhost. The ``--ipc host`` flag is needed for shared memory access.

**Step 2: Start the vLLM container with LMCache connector**

In a new terminal, start vLLM with the LMCache multi-process connector:

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/vllm-openai:latest-nightly \
        Qwen/Qwen3-14B \
        --kv-transfer-config '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.port": 6555}}'

.. note::
    It is recommended to use the nightly builds (``lmcache/standalone:nightly`` and ``lmcache/vllm-openai:latest-nightly``) as the multi-process mode interfaces are actively evolving.

**Step 3: Send requests to the vLLM instance**

Once both containers are running, you can send requests to vLLM the same way as in the local deployment:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"Qwen/Qwen3-14B\",
        \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
        \"max_tokens\": 10
    }"

Kubernetes Deployment
---------------------

You can deploy LMCache and vLLM on Kubernetes using a DaemonSet pattern. This approach runs one LMCache server per node that can be shared by multiple vLLM pods on the same node, making it suitable for production deployments.

.. note::
    The DaemonSet deployment does not request GPU resources, allowing GPUs to remain exclusively allocated to vLLM pods. The NVIDIA container runtime automatically provides GPU access to the LMCache server for IPC-based memory transfers.

Prerequisites
~~~~~~~~~~~~~

- Kubernetes cluster with GPU support (NVIDIA GPU Operator installed)
- At least 4 GPUs per node
- ``kubectl`` configured to access your cluster

**Step 1: Create Namespace**

.. code-block:: bash

    kubectl create namespace multi-process

**Step 2: Deploy LMCache DaemonSet**

First, deploy the LMCache server as a DaemonSet (one instance per node):

.. code-block:: bash

    kubectl apply -f examples/multi_process/lmcache-daemonset.yaml

**Step 3: Deploy vLLM Application**

Deploy one or more vLLM instances that will connect to the LMCache DaemonSet:

.. code-block:: bash

    kubectl apply -f examples/multi_process/vllm-deployment.yaml

.. note::
    The default model is ``Qwen/Qwen3-14B``, which does not require a Hugging Face token. If you want to use a gated model (like Llama), you need to:
    
    1. Add a Secret with your Hugging Face token:
    
    .. code-block:: bash
    
        kubectl create secret generic vllm-secrets \
          --from-literal=hf_token=your_hf_token_here \
          -n multi-process
    
    2. Add the HF_TOKEN environment variable to the vLLM container in the YAML:
    
    .. code-block:: yaml
    
        env:
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                key: hf_token
                name: vllm-secrets
    
    3. Update the model name in the args section to your desired gated model.

.. note::
    Multiple vLLM pods on the same node will automatically connect to the same LMCache DaemonSet instance via ``status.hostIP``.

**Step 4: Monitor Deployment**

Check DaemonSet status:

.. code-block:: bash

    kubectl get daemonset -n multi-process
    kubectl get pods -n multi-process -l app=lmcache-server

Check vLLM deployment status:

.. code-block:: bash

    kubectl get pods -n multi-process -l app=vllm-deployment -w

Check vLLM server logs:

.. code-block:: bash

    kubectl logs -n multi-process -l app=vllm-deployment -f

Check LMCache server logs (on the same node as a vLLM pod):

.. code-block:: bash

    # Get the node where a vLLM pod is running
    VLLM_NODE=$(kubectl get pod -n multi-process -l app=vllm-deployment -o jsonpath='{.items[0].spec.nodeName}')
    
    # Get the LMCache pod on that node and view its logs
    LMCACHE_POD=$(kubectl get pod -n multi-process -l app=lmcache-server --field-selector spec.nodeName=$VLLM_NODE -o jsonpath='{.items[0].metadata.name}')
    kubectl logs -n multi-process $LMCACHE_POD -f

Wait for the pods to be ready (this may take several minutes for model loading).

**Step 5: Send Test Requests**

Forward the port to your local machine:

.. code-block:: bash

    kubectl port-forward -n multi-process deployment/vllm-deployment 8000:8000

Send a test request with repeated prompts:

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"Qwen/Qwen3-14B\",
        \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
        \"max_tokens\": 10
      }"

On the first request, check LMCache server logs for:

.. code-block:: text

    [LMCache INFO] Stored X tokens in Y seconds

On subsequent identical requests, you should see:

.. code-block:: text

    [LMCache INFO] Retrieved X tokens in Y seconds

Kubernetes Configuration Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DaemonSet Architecture:**

- One LMCache server runs per node as a DaemonSet
- Multiple vLLM pods on the same node can share the same LMCache instance
- LMCache uses ``hostNetwork: true`` so vLLM pods can connect via node IP
- vLLM pods use ``status.hostIP`` to discover the LMCache server on their node
- Both containers mount the host's ``/dev/shm`` to enable CUDA IPC memory sharing

**GPU Access:**

- GPUs are NOT requested in the DaemonSet resources section
- This allows GPUs to remain exclusively allocated to vLLM pods
- LMCache can still access GPUs for IPC-based memory transfers via the NVIDIA container runtime

.. note::
    LMCache pods on nodes without GPUs will crash with CUDA initialization errors. This is expected behavior - LMCache only needs to run successfully on GPU nodes where vLLM pods are scheduled.

**Minimal Configuration Requirements:**

**vLLM Deployment:**

- ``/dev/shm`` hostPath mount (required for CUDA IPC shared memory)

**LMCache DaemonSet:**

- ``hostNetwork: true`` (required for vLLM to connect via node IP using ``status.hostIP``)
- ``/dev/shm`` hostPath mount (required for CUDA IPC shared memory)
- ``LMCACHE_LOG_LEVEL=DEBUG`` environment variable (optional, for observability)

**Key Insight:**

Mounting the same ``/dev/shm`` from the host in both containers provides the shared memory space needed for CUDA IPC communication. The NVIDIA container runtime (installed via NVIDIA GPU Operator) automatically provides GPU access to the LMCache server.

**Cleanup:**

.. code-block:: bash

    kubectl delete -f examples/multi_process/vllm-deployment.yaml
    kubectl delete -f examples/multi_process/lmcache-daemonset.yaml
    kubectl delete namespace multi-process

Detailed Configuration
----------------------

Server Configuration
~~~~~~~~~~~~~~~~~~~~

The LMCache multi-process server supports the following command-line arguments:

- ``--host``: Host address to bind the server (default: ``localhost``)
- ``--port``: Port number to bind the server (default: ``5555``)
- ``--chunk-size``: Chunk size for KV cache operations in tokens (default: ``256``)
- ``--cpu-buffer-size``: CPU buffer size in GB for caching (default: ``5.0``)
- ``--max-workers``: Maximum number of worker threads for handling requests (default: ``1``)

Example with custom configuration:

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.server \
        --host 0.0.0.0 \
        --port 6000 \
        --chunk-size 512 \
        --cpu-buffer-size 50.0 \
        --max-workers 4

vLLM Client Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

On the vLLM side, you can specify the host and port of the LMCache server through the ``kv_connector_extra_config`` parameter:

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 6000}}'

Future Work
-----------

- Thread-safe memory allocator and storage manager.
- Eviction policy.
- Plugin the current storage backends.
- Potential performance improvements (double buffering, new kernels, etc.).
- Lock and unlock semantics in new storage manager.
- Distributed mode with sharding.