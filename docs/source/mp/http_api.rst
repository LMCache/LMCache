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

A subset of routes defined under
``lmcache/v1/internal_api_server/common/`` is also exposed on this HTTP
server. The module
``lmcache/v1/multiprocess/http_apis/common_api.py`` aggregates those
routers (skipping modules listed in ``_MP_INCOMPATIBLE_MODULES``, such as
``run_script_api``) and forwards them to the auto-discovery pipeline.
Adding a new compatible module under ``internal_api_server/common``
therefore requires no wiring changes on the MP side.

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

The table below groups the routes by purpose. Paths under ``/api/`` are
the operational surface (health, status, cache control). Routes without
the ``/api/`` prefix are inherited from the shared
``internal_api_server`` package and kept at their original paths for
compatibility with the vLLM-embedded API server.

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

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
   * - POST
     - ``/api/kv/store``
     - Store opaque KV cache bytes for a token sequence.
   * - POST
     - ``/api/kv/retrieve``
     - Retrieve KV cache bytes for a token sequence's longest cached prefix.
   * - POST
     - ``/api/kv/lookup``
     - Probe the cached-prefix length for a token sequence (no payload).
   * - GET
     - ``/conf``
     - Dump merged server configurations (mp, storage_manager,
       observability).
   * - GET
     - ``/version``
     - Full version descriptor (package version + commit id).
   * - GET
     - ``/lmc_version``
     - LMCache package version string.
   * - GET
     - ``/commit_id``
     - Current build commit id.
   * - GET
     - ``/env``
     - Dump process environment variables (JSON, plain text).
   * - GET
     - ``/loglevel``
     - List or inspect logger levels; also accepts ``level`` to mutate.
   * - GET
     - ``/metrics``
     - Prometheus exposition format.
   * - POST
     - ``/metrics/reset``
     - Reset all observability metrics to their initial state.
   * - GET
     - ``/threads``
     - Enumerate active Python threads and their stack traces.
   * - GET
     - ``/periodic-threads``
     - List registered periodic threads with summary counts.
   * - GET
     - ``/periodic-threads/{thread_name}``
     - Detailed status for a single periodic thread.
   * - GET
     - ``/periodic-threads-health``
     - Quick health check for critical/high-level periodic threads.

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

Bytes-Level KV Cache Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``/api/kv/*`` endpoints expose direct read and write access to the
KV cache bytes addressed by a token sequence. They are intended for ML
developers, orchestration layers, and debugging — for example, priming
the cache from an offline source, snapshotting a sequence's KV cache for
inspection, or building higher-level editing workflows on top.

These endpoints are **not** on the inference data path. Inference traffic
continues to flow over the ZMQ protocol; vLLM never calls these routes.

Wire format
^^^^^^^^^^^

The KV cache payload is a contiguous ``KV_2LTD`` tensor with shape:

.. code-block:: text

    [2, num_layers, num_tokens, hidden_dim]

serialized as raw bytes in row-major order. Notes:

- ``num_tokens`` is the number of complete chunks times the engine's
  ``chunk_size``. Trailing tokens that do not form a full chunk are
  ignored (the engine only stores chunk-aligned data).
- ``hidden_dim`` is the **full** hidden dimension of the model — the
  un-sharded value. The server transparently splits the bytes across
  TP workers along ``D``; clients do not see TP topology. For MLA
  models, ``hidden_dim`` corresponds to the single shared shard
  (registered ``world_size`` is ``1``).
- ``dtype`` is whatever the registered model uses (commonly
  ``bfloat16`` or ``fp16``); the payload byte length must match exactly.

Routing metadata (``model_name``, ``tokens``, optional ``cache_salt``)
identifies which model and token sequence the payload belongs to.
``model_name`` must match a model that has registered its KV cache with
the MP server; you can confirm this via ``GET /api/status``.

``POST /api/kv/store``
^^^^^^^^^^^^^^^^^^^^^^

Store KV cache bytes for the given token sequence.

The request body is the raw KV payload in the wire format above.
Routing metadata travels in **headers** so that the body remains a
single binary blob — this avoids multipart parsing for large (hundreds
of MB) payloads.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Header
     - Required
     - Description
   * - ``X-LMCache-Model-Name``
     - yes
     - Registered model name. Must match a model in ``/api/status``.
   * - ``X-LMCache-Tokens``
     - yes
     - Comma-separated token IDs covering the payload (whole chunks).
   * - ``X-LMCache-Cache-Salt``
     - no
     - Per-namespace isolation salt (forwarded to ``ObjectKey.cache_salt``).
       Use this to run parallel cache-priming experiments without collision.
   * - ``Content-Type``
     - recommended
     - ``application/x-lmcache-kv; v=1``.

**Response** (``200 OK``):

.. code-block:: json

    {
      "status": "ok",
      "total_tokens": 768,
      "total_chunks": 3,
      "stored_tokens": 768,
      "stored_chunks": 3
    }

``stored_chunks`` may be less than ``total_chunks`` if the L1 backend
could not reserve some keys (e.g. capacity pressure). Trailing tokens
that do not form a full chunk are excluded from ``total_chunks``.

**Errors:**

- ``400`` if a required header is missing/malformed, the model is not
  registered, or the payload length does not match the expected
  ``total_chunks * world_size * per_shard_byte_size``.
- ``503`` if the engine is not yet initialized.

**Example (curl, small file):**

.. code-block:: bash

    # Assume kv_payload.bin holds [2, L, T, D] bytes for tokens 0..T-1.
    TOKENS=$(seq -s, 0 767)  # 768 tokens = 3 chunks @ chunk_size=256

    curl -s -X POST http://localhost:8080/api/kv/store \
        -H "Content-Type: application/x-lmcache-kv; v=1" \
        -H "X-LMCache-Model-Name: meta-llama/Llama-3.1-8B-Instruct" \
        -H "X-LMCache-Tokens: $TOKENS" \
        --data-binary @kv_payload.bin

**Example (Python helper):**

.. code-block:: python

    import torch
    import requests

    def store_kv(
        base_url: str,
        model_name: str,
        tokens: list[int],
        kv_2ltd: torch.Tensor,        # shape [2, num_layers, T, hidden_dim]
        *,
        cache_salt: str = "",
    ) -> dict:
        """Upload a KV_2LTD tensor for ``tokens`` to LMCache.

        ``kv_2ltd`` must be CPU, contiguous, and its T dim must equal
        ``len(tokens) // chunk_size * chunk_size``.
        """
        payload = bytes(kv_2ltd.contiguous().view(torch.uint8).numpy())
        headers = {
            "Content-Type": "application/x-lmcache-kv; v=1",
            "X-LMCache-Model-Name": model_name,
            "X-LMCache-Tokens": ",".join(str(t) for t in tokens),
        }
        if cache_salt:
            headers["X-LMCache-Cache-Salt"] = cache_salt
        r = requests.post(f"{base_url}/api/kv/store", data=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    # Example call:
    tokens = list(range(768))
    kv = torch.randn(2, 32, 768, 4096, dtype=torch.bfloat16)
    print(store_kv("http://localhost:8080",
                   "meta-llama/Llama-3.1-8B-Instruct", tokens, kv))

``POST /api/kv/retrieve``
^^^^^^^^^^^^^^^^^^^^^^^^^

Retrieve the bytes for the longest cached prefix of ``tokens``.

The request body is JSON; the response body is the raw KV payload in
the wire format above. Hit metadata is exposed in response headers
because the body is binary.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Required
     - Description
   * - ``model_name``
     - yes
     - Registered model name.
   * - ``tokens``
     - yes
     - List of token IDs to address.
   * - ``cache_salt``
     - no
     - Per-namespace isolation salt; must match the salt used at store.

**Response headers (always present):**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Header
     - Meaning
   * - ``X-LMCache-Hit-Tokens``
     - Number of tokens covered by the returned bytes.
   * - ``X-LMCache-Hit-Chunks``
     - Number of chunks present in the response.
   * - ``X-LMCache-Total-Tokens``
     - Token count for the whole-chunk prefix of the input.
   * - ``X-LMCache-Total-Chunks``
     - Total whole-chunk count in the input.

**Response codes:**

- ``200`` with binary body: at least one chunk hit. The body shape is
  ``[2, num_layers, hit_tokens, hidden_dim]``.
- ``404`` with empty body: nothing in the requested sequence is cached.
- ``400`` if ``model_name`` is not registered.
- ``503`` if the engine is not yet initialized.

This endpoint is **non-destructive**: the cache is unchanged regardless
of any ``remove_after_retrieve`` engine setting.

**Example (curl):**

.. code-block:: bash

    cat <<'EOF' > /tmp/req.json
    {
      "model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "tokens": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    }
    EOF

    curl -sS -X POST http://localhost:8080/api/kv/retrieve \
        -H "Content-Type: application/json" \
        -D /tmp/headers.txt \
        -o /tmp/kv_payload.bin \
        --data-binary @/tmp/req.json

    grep -i '^X-LMCache-' /tmp/headers.txt
    ls -l /tmp/kv_payload.bin

**Example (Python helper):**

.. code-block:: python

    import torch
    import requests

    def retrieve_kv(
        base_url: str,
        model_name: str,
        tokens: list[int],
        *,
        num_layers: int,
        hidden_dim: int,
        dtype: torch.dtype,
        cache_salt: str = "",
    ) -> tuple[torch.Tensor | None, dict[str, int]]:
        """Fetch the cached KV_2LTD prefix for ``tokens``.

        Returns ``(tensor, meta)``: ``tensor`` has shape
        ``[2, num_layers, meta['hit_tokens'], hidden_dim]`` when there
        was a hit, ``None`` on a clean miss.
        """
        body = {"model_name": model_name, "tokens": tokens, "cache_salt": cache_salt}
        r = requests.post(f"{base_url}/api/kv/retrieve", json=body)
        meta = {
            "hit_tokens": int(r.headers.get("X-LMCache-Hit-Tokens", "0")),
            "hit_chunks": int(r.headers.get("X-LMCache-Hit-Chunks", "0")),
            "total_tokens": int(r.headers.get("X-LMCache-Total-Tokens", "0")),
            "total_chunks": int(r.headers.get("X-LMCache-Total-Chunks", "0")),
        }
        if r.status_code == 404 or not r.content:
            return None, meta
        r.raise_for_status()
        flat_u8 = torch.frombuffer(bytearray(r.content), dtype=torch.uint8)
        tensor = flat_u8.view(dtype).reshape(
            2, num_layers, meta["hit_tokens"], hidden_dim
        )
        return tensor, meta

    # Example call:
    kv, meta = retrieve_kv(
        "http://localhost:8080",
        "meta-llama/Llama-3.1-8B-Instruct",
        tokens=list(range(768)),
        num_layers=32, hidden_dim=4096, dtype=torch.bfloat16,
    )
    if kv is None:
        print("miss", meta)
    else:
        print("hit", meta, "tensor shape", tuple(kv.shape))

``POST /api/kv/lookup``
^^^^^^^^^^^^^^^^^^^^^^^

Probe how much of a token sequence is currently cached, without
materializing the payload. Useful for clients that want to decide
whether the bandwidth cost of a full retrieve is worth it, or to
confirm that a prior store landed.

The request body matches ``/api/kv/retrieve`` (JSON with ``model_name``,
``tokens``, optional ``cache_salt``).

**Response** (``200 OK``):

.. code-block:: json

    {
      "total_tokens": 768,
      "total_chunks": 3,
      "hit_tokens": 512,
      "hit_chunks": 2
    }

**Errors:**

- ``400`` if ``model_name`` is not registered.
- ``503`` if the engine is not yet initialized.

**Example (curl):**

.. code-block:: bash

    curl -sS -X POST http://localhost:8080/api/kv/lookup \
        -H "Content-Type: application/json" \
        --data '{"model_name":"meta-llama/Llama-3.1-8B-Instruct","tokens":[0,1,2,3,4,5,6,7]}' \
      | jq

**Example (Python):**

.. code-block:: python

    import requests

    r = requests.post(
        "http://localhost:8080/api/kv/lookup",
        json={
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "tokens": list(range(768)),
        },
    )
    r.raise_for_status()
    print(r.json())   # {'total_tokens': 768, 'total_chunks': 3, 'hit_tokens': 512, 'hit_chunks': 2}

``GET /conf``
~~~~~~~~~~~~~

Returns every server-side configuration object registered on
``app.state.configs`` (typically ``mp``, ``storage_manager`` and
``observability``) as a single indented JSON document. Dataclasses are
serialized via ``safe_asdict``; other values go through ``make_json_safe``.
Useful for confirming what the process actually loaded — including
environment overrides — without restarting.

**Response** (``200 OK``):

.. code-block:: json

    {
      "mp": {
        "http_host": "0.0.0.0",
        "http_port": 8080,
        "...": "..."
      },
      "storage_manager": {
        "...": "..."
      },
      "observability": {
        "...": "..."
      }
    }

**Response** (``503 Service Unavailable``) when configs are not wired
onto ``app.state`` yet:

.. code-block:: json

    {
      "error": "configs not initialized"
    }

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/conf | jq

``GET /version``
~~~~~~~~~~~~~~~~

Returns the full version descriptor (package version combined with the
current commit id), formatted by ``lmcache.utils.get_version()``.

**Response** (``200 OK``):

.. code-block:: json

    "0.3.x+<commit-id>"

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/version

``GET /lmc_version``
~~~~~~~~~~~~~~~~~~~~

Returns the raw LMCache package version string (``lmcache.utils.VERSION``).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/lmc_version

``GET /commit_id``
~~~~~~~~~~~~~~~~~~

Returns the git commit id baked into the build (``lmcache.utils.COMMIT_ID``).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/commit_id

``GET /env``
~~~~~~~~~~~~

Dumps the process environment variables as a sorted, pretty-printed
JSON document. Response ``Content-Type`` is ``text/plain`` so it can be
piped directly to a terminal.

.. warning::

   The payload may contain secrets injected via environment
   variables. Restrict network access to this endpoint in production.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/env

``GET /loglevel``
~~~~~~~~~~~~~~~~~

Inspect or mutate Python logger levels at runtime. All responses are
``text/plain``. The endpoint has three modes driven by query parameters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - (no params)
     - List every logger registered with :mod:`logging` and its level.
   * - ``?logger_name=<name>``
     - Return the effective level of the named logger.
   * - ``?logger_name=<name>&level=<LEVEL>``
     - Set the named logger (and its handlers) to ``LEVEL``
       (``DEBUG``/``INFO``/``WARNING``/``ERROR``/``CRITICAL``).
       Returns ``400`` on an unknown level.

**Examples:**

.. code-block:: bash

    # list everything
    curl -s http://localhost:8080/loglevel

    # read one
    curl -s 'http://localhost:8080/loglevel?logger_name=lmcache'

    # elevate to DEBUG
    curl -s 'http://localhost:8080/loglevel?logger_name=lmcache&level=DEBUG'

``GET /metrics``
~~~~~~~~~~~~~~~~

Prometheus exposition format for every metric registered on the default
``prometheus_client`` registry. Scrape this directly from Prometheus.
See :doc:`observability` for the list of exported metrics.

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/metrics

``POST /metrics/reset``
~~~~~~~~~~~~~~~~~~~~~~~

Resets all LMCache observability metrics to their initial state
(``reset_observability_metrics``). Intended for test harnesses and
benchmarks — not for production.

**Response** (``200 OK``):

.. code-block:: text

    ok

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:8080/metrics/reset

``GET /threads``
~~~~~~~~~~~~~~~~

Enumerate active Python threads in the server process along with their
stack traces, plus a total-count summary. Useful for live debugging of
hangs or runaway workers.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - ``?name=<substr>``
     - Keep only threads whose name contains ``<substr>``
       (case-insensitive).
   * - ``?thread_id=<int>``
     - Keep only the thread with the matching ``ident``.

**Example:**

.. code-block:: bash

    curl -s 'http://localhost:8080/threads?name=periodic'

``GET /periodic-threads``
~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a JSON snapshot of the
:class:`~lmcache.v1.periodic_thread.PeriodicThreadRegistry`: counts by
level plus per-thread status (last run timestamp, latest summary, etc.).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Query
     - Behavior
   * - ``?level=critical|high|medium|low``
     - Only include threads at the given level. ``400`` on unknown.
   * - ``?running_only=true``
     - Only include threads currently running.
   * - ``?active_only=true``
     - Only include threads considered active (recent tick).

**Response** (``200 OK``):

.. code-block:: json

    {
      "summary": {
        "total_count": 4,
        "running_count": 4,
        "active_count": 4,
        "by_level": {"critical": 1, "high": 2, "medium": 1, "low": 0}
      },
      "threads": [
        {"name": "...", "level": "high", "is_running": true, "...": "..."}
      ]
    }

**Example:**

.. code-block:: bash

    curl -s 'http://localhost:8080/periodic-threads?level=critical' | jq

``GET /periodic-threads/{thread_name}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed status for a single periodic thread (``404`` if not found).

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/periodic-threads/storage-flush | jq

``GET /periodic-threads-health``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast health check covering only ``critical`` and ``high`` level periodic
threads. A thread is flagged unhealthy when it is marked running but has
not ticked within its expected interval.

**Response** (``200 OK``):

.. code-block:: json

    {
      "healthy": true,
      "unhealthy_count": 0,
      "unhealthy_threads": []
    }

When something is lagging:

.. code-block:: json

    {
      "healthy": false,
      "unhealthy_count": 1,
      "unhealthy_threads": [
        {
          "name": "storage-flush",
          "level": "critical",
          "last_run_ago": 42.5,
          "interval": 5.0
        }
      ]
    }

**Example:**

.. code-block:: bash

    curl -s http://localhost:8080/periodic-threads-health

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

If the route is generic enough to be shared with the vLLM-embedded API
server, add it under ``lmcache/v1/internal_api_server/common/`` instead.
It will be picked up on the MP side via ``common_api.py`` unless its
module name is listed in ``_MP_INCOMPATIBLE_MODULES`` there (used for
modules that require vLLM-specific ``app.state`` attributes, e.g.
``run_script_api``).

When adding a new endpoint, please also add a matching section to this
page documenting the endpoint's purpose, request/response schema, and
an example ``curl`` invocation.