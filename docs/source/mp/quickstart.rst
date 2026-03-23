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

Tested SGLang Example
---------------------

The same LMCache MP server can also be used from SGLang.

**Step 1: Start the LMCache server**

.. code-block:: bash

    python3 -m lmcache.v1.multiprocess.server \
        --host 127.0.0.1 \
        --port 65000 \
        --l1-size-gb 10 \
        --eviction-policy LRU

**Step 2: Start SGLang with LMCache MP**

In a new terminal:

.. code-block:: bash

    export LMCACHE_CONFIG_FILE=$PWD/lmc_config.yaml

    python -m sglang.launch_server \
        --model-path Qwen/Qwen3-0.6B \
        --host 127.0.0.1 \
        --port 30000 \
        --enable-lmcache \
        --lmcache-mp-host 127.0.0.1 \
        --lmcache-mp-port 65000

.. note::
   During local validation, SGLang needed ``--disable-piecewise-cuda-graph``
   and ``--disable-cuda-graph`` in this environment to avoid an unrelated
   startup CUDA issue. That was not required by LMCache MP itself.

**Step 3: Send the same deterministic request twice**

.. code-block:: bash

    curl http://127.0.0.1:30000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
          "model": "Qwen/Qwen3-0.6B",
          "messages": [{"role": "user", "content": "tok1 tok2 tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10 tok11 tok12 tok13 tok14 tok15 tok16 tok17 tok18 tok19 tok20 tok21 tok22 tok23 tok24 tok25 tok26 tok27 tok28 tok29 tok30 tok31 tok32 tok33 tok34 tok35 tok36 tok37 tok38 tok39 tok40 tok41 tok42 tok43 tok44 tok45 tok46 tok47 tok48 tok49 tok50 tok51 tok52 tok53 tok54 tok55 tok56 tok57 tok58 tok59 tok60 tok61 tok62 tok63 tok64 tok65 tok66 tok67 tok68 tok69 tok70 tok71 tok72 tok73 tok74 tok75 tok76 tok77 tok78 tok79 tok80 tok81 tok82 tok83 tok84 tok85 tok86 tok87 tok88 tok89 tok90 tok91 tok92 tok93 tok94 tok95 tok96 tok97 tok98 tok99 tok100 tok101 tok102 tok103 tok104 tok105 tok106 tok107 tok108 tok109 tok110 tok111 tok112 tok113 tok114 tok115 tok116 tok117 tok118 tok119 tok120 tok121 tok122 tok123 tok124 tok125 tok126 tok127 tok128 tok129 tok130 tok131 tok132 tok133 tok134 tok135 tok136 tok137 tok138 tok139 tok140 tok141 tok142 tok143 tok144 tok145 tok146 tok147 tok148 tok149 tok150 tok151 tok152 tok153 tok154 tok155 tok156 tok157 tok158 tok159 tok160 tok161 tok162 tok163 tok164 tok165 tok166 tok167 tok168 tok169 tok170 tok171 tok172 tok173 tok174 tok175 tok176 tok177 tok178 tok179 tok180 tok181 tok182 tok183 tok184 tok185 tok186 tok187 tok188 tok189 tok190 tok191 tok192 tok193 tok194 tok195 tok196 tok197 tok198 tok199 tok200"}],
          "temperature": 0,
          "max_tokens": 16
        }'

Example result observed during validation:

.. code-block:: text

    first request:  cached_tokens=0
    second request: cached_tokens=768
    output text: " tok200 tok201 tok202 tok203"

We also validated deterministic in-process vs MP output parity on:

- ``Qwen3-0.6B``
- ``facebook/opt-125m``
- ``Qwen3-8B``
- ``Qwen2.5-14B-Instruct``

and confirmed that two independent SGLang servers can connect to the same
LMCache MP daemon and reuse cached KV across servers.

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

    python3 -m lmcache.v1.multiprocess.http_server \
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
