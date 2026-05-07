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
   * - GET
     - ``/api/quota``
     - List every registered ``cache_salt`` quota with live usage.
   * - PUT
     - ``/api/quota/{cache_salt}``
     - Set or update the quota (in GB) for a ``cache_salt``.
   * - GET
     - ``/api/quota/{cache_salt}``
     - Read the quota and live usage for a single ``cache_salt``.
   * - DELETE
     - ``/api/quota/{cache_salt}``
     - Remove a ``cache_salt``'s quota entry (its data is evicted next
       cycle).
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

.. _mp-http-quota-api:

``/api/quota`` — per-``cache_salt`` quota management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These endpoints manage the per-``cache_salt`` storage budgets consumed by
the ``IsolatedLRU`` eviction policy (selected via
``--eviction-policy IsolatedLRU``). Quotas are **soft**: setting a limit
does not reject writes — any over-budget ``cache_salt`` is evicted at
the next eviction cycle (~1 s).
A ``cache_salt`` with no registered quota has an effective limit of
``0`` bytes, so its data is cleared next cycle (allowlist semantics).

These endpoints are no-ops on engines that did not start with
``--eviction-policy IsolatedLRU``: the ``QuotaManager`` is still
present, but the LRU policy ignores the registered quotas.

**URL escaping for the empty salt.** ``cache_salt=""`` (un-salted /
anonymous traffic) cannot appear in a URL path parameter, so the API
accepts the sentinel ``_default`` in its place. ``PUT /api/quota/_default``
sets the quota for ``cache_salt=""``. A user that legitimately stores
data with ``cache_salt="_default"`` cannot be managed via this HTTP API
distinctly from anonymous traffic — both map to the same path parameter;
pick any other value (e.g. ``"default"``) to disambiguate.

``PUT /api/quota/{cache_salt}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create or update a quota.

**Body:** ``{"limit_gb": <float>}`` (required, finite, non-negative).

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "alice", "limit_gb": 10.0, "status": "ok"}

**Errors:** ``400`` for malformed JSON, missing ``limit_gb``, non-numeric
``limit_gb``, ``nan`` / ``inf``, or negative values; ``503`` if the
engine is not initialized.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:8080/api/quota/alice \
        -H 'Content-Type: application/json' \
        -d '{"limit_gb": 10.0}'

``GET /api/quota/{cache_salt}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read the current quota and live usage for one ``cache_salt``.

**Response** (``200 OK``):

.. code-block:: json

    {
      "cache_salt": "alice",
      "limit_gb": 10.0,
      "current_usage_gb": 2.137,
      "exists": true
    }

``exists`` is ``false`` when no quota was ever registered for this
``cache_salt`` (``limit_gb`` is then ``0.0`` and ``current_usage_gb``
reflects whatever bytes are currently cached for that salt — those bytes
will evict next cycle under ``IsolatedLRU``).

``DELETE /api/quota/{cache_salt}``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove a ``cache_salt``'s quota entry. Any bytes still cached under this
``cache_salt`` become over-budget on the next eviction cycle (effective
limit drops to ``0``) and will be evicted.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "alice", "status": "removed"}

When no quota was registered for the given ``cache_salt``, the response
is ``{"cache_salt": "...", "status": "not_found"}`` (still ``200 OK``).

``GET /api/quota``
^^^^^^^^^^^^^^^^^^

List every registered quota alongside its live usage.

**Response** (``200 OK``):

.. code-block:: json

    {
      "users": {
        "alice": {"limit_gb": 10.0, "current_usage_gb": 2.137},
        "bob":   {"limit_gb":  4.0, "current_usage_gb": 0.812}
      }
    }

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