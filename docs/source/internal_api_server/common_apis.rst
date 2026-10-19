.. _common_apis:

Common APIs
===========

These diagnostics are shared by the MP HTTP frontend and the deprecated
in-process scheduler, worker, and controller API servers. The MP server imports
the common routers from ``lmcache.v1.internal_api_server.common``.

Examples use the MP HTTP address ``http://localhost:8080``. Start it with
``lmcache server --l1-size-gb 10 --eviction-policy LRU``; see
:doc:`/mp/http_api` for authentication and deployment configuration. For an
in-process worker, substitute its configured API port (normally 7000).

.. contents:: Endpoints
   :local:
   :depth: 2


``GET /env`` — Environment Variables
-------------------------------------

Get environment variables of the running process. Secret values such as the
HTTP API key are redacted by the endpoint.

- **Method**: ``GET``
- **Path**: ``/env``
- **Parameters**: None
- **Response**: ``application/json`` — JSON object of all environment variables (sorted by key).

.. code-block:: bash

    curl http://localhost:8080/env

**Example Response**:

.. code-block:: json

    {
      "HOME": "/root",
      "PATH": "/usr/local/bin:/usr/bin",
      "PYTHONPATH": "/app"
    }


``GET /loglevel`` — Log Level Management
------------------------------------------

Get or set the log level for Python loggers. Behavior depends on query parameters:

- **No parameters**: List all loggers and their levels.
- **``logger_name`` only**: Get the level of the specified logger.
- **``logger_name`` and ``level``**: Set the level of the specified logger (including all its handlers).

- **Method**: ``GET``
- **Path**: ``/loglevel``
- **Parameters**:

  =============== ======== ============================================
  Name            Type     Description
  =============== ======== ============================================
  ``logger_name`` str      (Optional) Logger name to query or set
  ``level``       str      (Optional) Log level to set (e.g. ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``)
  =============== ======== ============================================

- **Response**: ``text/plain``

.. code-block:: bash

    # List all loggers
    curl http://localhost:8080/loglevel

    # Get a specific logger level
    curl "http://localhost:8080/loglevel?logger_name=lmcache.v1.cache_engine"

    # Set a specific logger level
    curl "http://localhost:8080/loglevel?logger_name=lmcache.v1.cache_engine&level=DEBUG"

**Example Response** (list all):

.. code-block:: text

    === Loggers and Levels ===
    lmcache.v1.cache_engine: WARNING
    lmcache.v1.storage_backend: INFO

**Example Response** (get):

.. code-block:: text

    lmcache.v1.cache_engine: WARNING

**Example Response** (set):

.. code-block:: text

    Set lmcache.v1.cache_engine level to DEBUG (including all handlers)

**Error Response** (invalid level, HTTP 400):

.. code-block:: text

    Invalid log level: INVALID_LEVEL


``GET /metrics`` — Prometheus Metrics
--------------------------------------

Get Prometheus metrics data in the standard exposition format.

- **Method**: ``GET``
- **Path**: ``/metrics``
- **Parameters**: None
- **Response**: ``text/plain`` — Prometheus text-based exposition format.

.. code-block:: bash

    curl http://localhost:8080/metrics


``POST /metrics/reset`` — Reset Prometheus Metrics
----------------------------------------------------

Reset the legacy ``lmcache`` observability metrics. This endpoint calls
``reset_observability_metrics()``; it does not reset MP OpenTelemetry counters.
For MP metric export and lifecycle, see :doc:`/mp/observability/metrics`.

- **Method**: ``POST``
- **Path**: ``/metrics/reset``
- **Parameters**: None
- **Response**: ``text/plain`` — ``"ok"`` on success.

.. code-block:: bash

    curl -X POST http://localhost:8080/metrics/reset


``GET /threads`` — Thread Information
--------------------------------------

Get information about active threads with optional filtering.

- **Method**: ``GET``
- **Path**: ``/threads``
- **Parameters**:

  ============== ======= ===========================================
  Name           Type    Description
  ============== ======= ===========================================
  ``name``       str     (Optional) Filter by thread name (fuzzy match, case-insensitive)
  ``thread_id``  int     (Optional) Filter by thread ID
  ============== ======= ===========================================

- **Response**: ``text/plain`` — Thread info with stack traces and summary.

.. code-block:: bash

    # Get all threads
    curl http://localhost:8080/threads

    # Filter by name
    curl "http://localhost:8080/threads?name=api-server"

    # Filter by thread ID
    curl "http://localhost:8080/threads?thread_id=12345"


``GET /periodic-threads`` — Periodic Thread Status
----------------------------------------------------

Get information about registered periodic threads.

- **Method**: ``GET``
- **Path**: ``/periodic-threads``
- **Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``level``        str     (Optional) Filter by thread level: ``critical``, ``high``, ``medium``, ``low``
  ``running_only`` bool    (Optional) Only show running threads (default: ``false``)
  ``active_only``  bool    (Optional) Only show active threads (default: ``false``)
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    # Get all periodic threads
    curl http://localhost:8080/periodic-threads

    # Filter by level
    curl "http://localhost:8080/periodic-threads?level=critical"

    # Only running threads
    curl "http://localhost:8080/periodic-threads?running_only=true"

**Example Response**:

.. code-block:: json

    {
      "summary": {
        "total_count": 5,
        "running_count": 3,
        "active_count": 3,
        "by_level": {
          "critical": {"total": 1, "running": 1, "active": 1},
          "high": {"total": 2, "running": 1, "active": 1}
        }
      },
      "threads": [
        {
          "name": "heartbeat",
          "level": "critical",
          "is_running": true,
          "is_active": true,
          "last_run_time": "2025-01-01T00:00:00",
          "interval": 10.0
        }
      ]
    }

**Error Response** (invalid level, HTTP 400):

.. code-block:: json

    {
      "error": "Invalid level: unknown. Valid values: critical, high, medium, low"
    }


``GET /periodic-threads/{thread_name}`` — Single Periodic Thread
------------------------------------------------------------------

Get detailed information about a specific periodic thread by name.

- **Method**: ``GET``
- **Path**: ``/periodic-threads/{thread_name}``
- **Path Parameters**:

  ================ ======= =============================================
  Name             Type    Description
  ================ ======= =============================================
  ``thread_name``  str     Name of the periodic thread
  ================ ======= =============================================

- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:8080/periodic-threads/heartbeat

**Error Response** (not found, HTTP 404):

.. code-block:: json

    {
      "error": "Thread not found: heartbeat"
    }


``GET /periodic-threads-health`` — Periodic Thread Health Check
-----------------------------------------------------------------

Quick health check for periodic threads. Returns healthy status if all
``critical`` and ``high`` level threads are active.

- **Method**: ``GET``
- **Path**: ``/periodic-threads-health``
- **Parameters**: None
- **Response**: ``application/json``

.. code-block:: bash

    curl http://localhost:8080/periodic-threads-health

**Example Response** (healthy):

.. code-block:: json

    {
      "healthy": true,
      "unhealthy_count": 0,
      "unhealthy_threads": []
    }

**Example Response** (unhealthy):

.. code-block:: json

    {
      "healthy": false,
      "unhealthy_count": 1,
      "unhealthy_threads": [
        {
          "name": "heartbeat",
          "level": "critical",
          "last_run_ago": 120.5,
          "interval": 10.0
        }
      ]
    }


``POST /run_script`` — Run Script
-----------------------------------

.. danger::

   This endpoint runs caller-supplied code in-process. The restricted
   builtins are **not** a security sandbox — combined with the injected
   ``app`` object and any allowed imports, treat it as full remote code
   execution. It is therefore **disabled by default** (requests return
   ``404``); enable it explicitly with ``run_script_api_enabled``
   (``LMCACHE_RUN_SCRIPT_API_ENABLED=true``) for in-process mode, or
   ``--run-script-api-enabled`` on the MP server, and only on a trusted network.

Upload and execute a Python script with restricted builtins.
The script has access to ``app`` (the FastAPI application instance) and
a limited set of builtins. Import is restricted to modules configured
in ``script_allowed_imports`` (in-process) or ``--script-allowed-imports`` (MP).

- **Method**: ``POST``
- **Path**: ``/run_script``
- **Content-Type**: ``multipart/form-data``
- **Parameters**:

  ============ ======== =============================================
  Name         Type     Description
  ============ ======== =============================================
  ``script``   file     Python script file to execute
  ============ ======== =============================================

- **Response**: ``text/plain`` — The ``result`` variable from the script, or
  ``"Script executed successfully"`` if no ``result`` is set.

.. code-block:: bash

    curl -X POST http://localhost:8080/run_script \
      -F "script=@/path/to/scratch.py"

**Example Script** (``scratch.py``):

.. code-block:: python

    # MP server: report the standalone engine state.
    result = app.state.engine.report_status()

On an in-process worker, the engine instead lives at
``app.state.lmcache_adapter.lmcache_engine``. Scripts accessing that adapter
are specific to the deprecated embedded API server.

**Error Response** (no script, HTTP 400):

.. code-block:: text

    No script file provided

**Error Response** (execution error, HTTP 500):

.. code-block:: text

    Error executing script: Import of 'os' is not allowed


Version Information
-------------------

These routes are also shared with the MP server via its info router.

``GET /lmc_version`` — LMCache Version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the LMCache library version string.

- **Method**: ``GET``
- **Path**: ``/lmc_version``
- **Parameters**: None
- **Response**: Plain version string.

.. code-block:: bash

    curl http://localhost:8080/lmc_version


``GET /commit_id`` — LMCache Commit ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the LMCache git commit ID.

- **Method**: ``GET``
- **Path**: ``/commit_id``
- **Parameters**: None
- **Response**: Plain commit ID string.

.. code-block:: bash

    curl http://localhost:8080/commit_id


``GET /version`` — Full Version Info
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get full version info (version + commit ID).

- **Method**: ``GET``
- **Path**: ``/version``
- **Parameters**: None
- **Response**: Combined version string.

.. code-block:: bash

    curl http://localhost:8080/version
