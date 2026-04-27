HTTP API
========

When the MP server is started via ``lmcache server`` (the recommended entry
point), a FastAPI-based HTTP frontend is exposed alongside the ZMQ socket
used by vLLM. This HTTP API is intended for operators, orchestrators
(e.g. Kubernetes), and debugging tools — it is **not** on the inference
data path.

New endpoints are registered automatically from
``lmcache/v1/multiprocess/http_apis/``: any module named ``*_api.py`` that
exposes a module-level ``router`` (a :class:`fastapi.APIRouter`) is
discovered at startup.

.. contents::
   :local:
   :depth: 2

Server Configuration
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Argument
     - Default
     - Description
   * - ``--http-host``
     - ``0.0.0.0``
     - Host to bind the HTTP server.
   * - ``--http-port``
     - ``8080``
     - Port to bind the HTTP server.

Example:

.. code-block:: bash

    lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --http-host 0.0.0.0 --http-port 8080

All examples below assume the server is reachable at
``http://localhost:8080``.

Endpoints
---------

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Method
     - Path
     - Purpose
   * - GET
     - ``/``
     - Basic liveness ping.
   * - GET
     - ``/api/healthcheck``
     - K8s liveness/readiness probe.
   * - GET
     - ``/api/status``
     - Detailed engine status for inspection and debugging.
   * - POST
     - ``/api/clear-cache``
     - Force-clear all KV data in L1 (CPU) memory.

``GET /``
~~~~~~~~~

Basic liveness check. Returns a static payload indicating the HTTP server
is running. Use ``/api/healthcheck`` instead for probes that also verify
the cache engine is initialized.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "ok",
      "service": "LMCache HTTP API"
    }

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/

``GET /api/healthcheck``
~~~~~~~~~~~~~~~~~~~~~~~~

Health check endpoint suitable for Kubernetes liveness and readiness
probes. A ``200`` response implies the HTTP server is alive **and** the
MP cache engine is initialized. A ``503`` response indicates the engine
is not yet ready (still initializing, or failed to initialize).

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "healthy"
    }

**Response** (``503 Service Unavailable``):

.. code-block:: json

    {
      "status": "unhealthy",
      "reason": "engine not initialized"
    }

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/api/healthcheck

**Kubernetes probe snippet:**

.. code-block:: yaml

    livenessProbe:
      httpGet:
        path: /api/healthcheck
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /api/healthcheck
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5

``GET /api/status``
~~~~~~~~~~~~~~~~~~~

Returns a detailed snapshot of the MP engine's internal state: L1 cache,
L2 adapters, registered GPU contexts, active sessions, and in-flight
prefetch jobs. Intended for operators and debugging, not for monitoring
(use Prometheus metrics for time-series data — see
:doc:`observability`).

**Response** (``200 OK``):

.. code-block:: json

    {
      "is_healthy": true,
      "engine_type": "MPCacheEngine",
      "chunk_size": 256,
      "hash_algorithm": "builtin-hash",
      "registered_gpu_ids": [0, 1],
      "gpu_context_meta": {
        "0": {
          "model_name": "meta-llama/Llama-3.1-8B-Instruct",
          "world_size": 1,
          "kv_cache_layout": {
            "num_layers": 32,
            "block_size": 16,
            "hidden_dim_sizes": "...",
            "dtype": "torch.bfloat16",
            "is_mla": false,
            "num_blocks": 12345,
            "gpu_kv_format": "...",
            "gpu_kv_shape": "...",
            "gpu_kv_concrete_shape": "...",
            "attention_backend": "...",
            "cache_size_per_token": 131072
          }
        }
      },
      "active_sessions": 2,
      "active_prefetch_jobs": 0,
      "storage_manager": {
        "is_healthy": true,
        "...": "backend-specific fields"
      }
    }

**Response** (``503 Service Unavailable``) when the engine has not yet
been initialized:

.. code-block:: json

    {
      "error": "engine not initialized"
    }

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/api/status | jq

``POST /api/clear-cache``
~~~~~~~~~~~~~~~~~~~~~~~~~

Force-clears **all** KV cache data currently held in L1 (CPU) memory.

.. warning::

   This endpoint is destructive and bypasses read/write locks. In-flight
   store or prefetch operations may be corrupted. Use only when the
   server is idle, or when recovering from a known-bad cache state.

The request body is ignored.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "ok"
    }

**Response** (``503 Service Unavailable``):

.. code-block:: json

    {
      "status": "error",
      "reason": "engine not initialized"
    }

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/api/clear-cache

Adding New Endpoints
--------------------

Endpoints are auto-discovered from
``lmcache/v1/multiprocess/http_apis/``. To add a new endpoint:

1. Create a new module in that directory named ``<name>_api.py``.
2. Define a module-level ``router = APIRouter()``.
3. Register handlers on ``router`` using FastAPI decorators.
4. Access the engine via ``request.app.state.engine`` and guard for the
   ``None`` case (engine not yet initialized).

The :class:`~lmcache.v1.multiprocess.http_api_registry.HTTPAPIRegistry`
will pick the module up automatically at startup — no central
registration list to edit.

When adding a new endpoint, please also add a matching section to this
page documenting the endpoint's purpose, request/response schema, and
an example ``curl`` invocation.
