Quick Start
===========

This page walks through the fastest ways to get LMCache multiprocess mode
running -- locally, in Docker, and with the HTTP server variant.

Local Quick Start
-----------------

**Step 1: Start the LMCache server**

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.server \
        --l1-size-gb 100 --eviction-policy LRU

Expected log output:

.. code-block:: text

    LMCache INFO: LMCache cache server is running...

.. note::
   The default ZMQ port is **5555**.  Use ``--port`` to change it.

**Step 2: Start vLLM with the LMCache connector**

In a new terminal:

.. code-block:: bash

    vllm serve Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

.. note::
   This connects to the default LMCache port (5555) on localhost.  If you
   changed the server port with ``--port``, pass it on the vLLM side via
   ``kv_connector_extra_config``:

   .. code-block:: bash

       vllm serve Qwen/Qwen3-14B \
           --kv-transfer-config \
           '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.port": 6555}}'

   To connect to a remote host, also set ``lmcache.mp.host``:

   .. code-block:: bash

       --kv-transfer-config \
       '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.host": "10.0.0.1", "lmcache.mp.port": 6555}}'

You should see on the **vLLM** side:

.. code-block:: text

    LMCache INFO: Registering kv caches!

And on the **LMCache** side:

.. code-block:: text

    LMCache INFO: Registered KV cache for GPU ID <pid> with 40 layers

**Step 3: Send a request**

.. code-block:: bash

    curl -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Qwen/Qwen3-14B\",
            \"prompt\": \"$(printf 'Explain the significance of KV cache in language models.%.0s' {1..100})\",
            \"max_tokens\": 10
        }"

First request -- tokens are **stored**:

.. code-block:: text

    LMCache INFO: Stored 768 tokens in 0.001 seconds

Second identical request -- tokens are **retrieved** from cache:

.. code-block:: text

    LMCache INFO: Retrieved 768 tokens in 0.001 seconds

Docker Quick Start
------------------

**Step 1: Start the LMCache container**

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/standalone:nightly \
        /opt/venv/bin/python3 -m lmcache.v1.multiprocess.server \
        --l1-size-gb 60 --eviction-policy LRU --max-workers 4 --port 6555

.. note::
   ``--network host`` lets the vLLM container reach the LMCache server on
   localhost.  ``--ipc host`` is required for CUDA IPC shared memory.

**Step 2: Start the vLLM container**

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
        --network host \
        --ipc host \
        lmcache/vllm-openai:latest-nightly \
        Qwen/Qwen3-14B \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both", "kv_connector_extra_config": {"lmcache.mp.port": 6555}}'

.. note::
   Use the nightly images (``lmcache/standalone:nightly`` and
   ``lmcache/vllm-openai:latest-nightly``) as the MP-mode interfaces are
   actively evolving.

**Step 3: Send requests** the same way as in the local quick start.

HTTP Server Quick Start
-----------------------

The HTTP server wraps the ZMQ server with a FastAPI frontend, adding HTTP
management endpoints for health checking and cache administration.

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU

The HTTP server listens on ``0.0.0.0:8000`` by default (configurable with
``--http-host`` and ``--http-port``).

**Endpoints:**

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Method
     - Path
     - Description
   * - GET
     - ``/api/healthcheck``
     - Returns ``{"status": "healthy"}`` when the engine is initialized and
       memory checks pass. Suitable for Kubernetes liveness/readiness probes.
   * - POST
     - ``/api/clear-cache``
     - Force-clears all KV cache data stored in L1 (CPU) memory, including
       objects with active read/write locks. Returns ``{"status": "ok"}`` on
       success.
   * - GET
     - ``/api/status``
     - Returns detailed internal state of all MP components including L1 cache,
       L2 adapters, controllers, registered GPUs, and active sessions.

Examples:

.. code-block:: bash

    # Health check
    curl http://localhost:8000/api/healthcheck
    # {"status": "healthy"}

    # Clear all KV cache data in L1 (CPU) memory
    curl -X POST http://localhost:8000/api/clear-cache
    # {"status": "ok"}

    # Inspect detailed internal state
    curl http://localhost:8000/api/status

The ZMQ server runs on the same default port (5555) and accepts vLLM
connections exactly as in the local quick start.
