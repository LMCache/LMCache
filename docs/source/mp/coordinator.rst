Multi-Server Coordination
=========================

When you run more than one LMCache multiprocess (MP) server, the **MP
Coordinator** is a standalone service they register with, giving you a single,
fleet-wide view of every running server. Each MP server caches independently;
the coordinator ties them together into one coordinated fleet.

Running the coordinator
-----------------------

The coordinator is a FastAPI service. Start it with:

.. code-block:: bash

    lmcache coordinator

Expected log output:

.. code-block:: text

    LMCache INFO: MP coordinator listening on http://0.0.0.0:9300

The CLI accepts ``--host``, ``--port``, ``--instance-timeout``,
``--health-check-interval``, ``--eviction-check-interval``,
``--eviction-ratio``, and ``--trigger-watermark``; any flag overrides the
matching environment variable below. See :doc:`/cli/coordinator` for details.
Equivalently, the coordinator can still be launched as a module with
``python3 -m lmcache.v1.mp_coordinator``.

Configuration
-------------

The coordinator is configured through ``LMCACHE_MP_COORDINATOR_*`` environment
variables:

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - Environment variable
     - Default
     - Description
   * - ``LMCACHE_MP_COORDINATOR_HOST``
     - ``0.0.0.0``
     - Host the HTTP server binds to.
   * - ``LMCACHE_MP_COORDINATOR_PORT``
     - ``9300``
     - Port the HTTP server binds to.
   * - ``LMCACHE_MP_COORDINATOR_INSTANCE_TIMEOUT``
     - ``30``
     - Seconds without a heartbeat after which a server is dropped from the
       fleet.
   * - ``LMCACHE_MP_COORDINATOR_HEALTH_CHECK_INTERVAL``
     - ``10``
     - Seconds between health-check sweeps. ``0`` disables eviction.
   * - ``LMCACHE_MP_COORDINATOR_EVICTION_CHECK_INTERVAL``
     - ``5``
     - Seconds between L2 eviction sweeps. ``0`` disables the loop.
   * - ``LMCACHE_MP_COORDINATOR_EVICTION_RATIO``
     - ``0.2``
     - Fraction of tracked keys (by count) to evict per cycle (0.0 to 1.0).
   * - ``LMCACHE_MP_COORDINATOR_TRIGGER_WATERMARK``
     - ``1.0``
     - Eviction fires when usage reaches this fraction of the quota
       (0.0 exclusive to 1.0).

Connecting MP servers
---------------------

An MP server (``lmcache server``) joins the coordinator when you point it at one
with ``--coordinator-url``. It registers on startup, heartbeats while running,
and deregisters on shutdown -- all on the server's own event loop. This is
opt-in: with no URL set, the server runs exactly as before. Each flag falls back
to a matching ``LMCACHE_COORDINATOR_*`` environment variable (handy for the
Kubernetes downward API); an explicit flag wins over the env var.

.. list-table::
   :header-rows: 1
   :widths: 38 24 38

   * - Flag (on the MP server)
     - Env fallback
     - Description
   * - ``--coordinator-url``
     - ``LMCACHE_COORDINATOR_URL``
     - Coordinator base URL, e.g. ``http://coordinator:9300``. Enables
       registration when set.
   * - ``--coordinator-advertise-ip``
     - ``LMCACHE_COORDINATOR_ADVERTISE_IP``
     - IP the coordinator should reach this server at (defaults to the server's
       outbound IP).
   * - ``--coordinator-heartbeat-interval``
     - ``LMCACHE_COORDINATOR_HEARTBEAT_INTERVAL``
     - Seconds between heartbeats (must be ``> 0``, default ``5``). Keep it well
       below the coordinator's ``INSTANCE_TIMEOUT``.
   * - ``--coordinator-l2-event-reporting``
     - ``LMCACHE_COORDINATOR_L2_EVENT_REPORTING``
     - Enable reporting L2 store/lookup events to the coordinator for
       fleet-wide usage tracking and quota-based eviction.
   * - ``--coordinator-l2-event-flush-interval``
     - ``LMCACHE_COORDINATOR_L2_EVENT_FLUSH_INTERVAL``
     - Seconds between L2 event batch flushes (must be ``> 0``, default ``1``).

The server registers under its stable identity (``--instance-id`` / OTel
``service.instance.id``); if the flag is not passed, the server mints a
random UUID v4 at startup and registers under that.

Registration is best-effort: if the coordinator is unreachable, the MP server
logs a warning, keeps retrying, and continues serving. A malformed
heartbeat-interval value is rejected at startup.

Inspecting the fleet
--------------------

Two read-only endpoints let you observe the coordinator:

- ``GET /instances`` -- list every registered MP server.
- ``GET /healthz`` -- coordinator liveness probe (for Kubernetes).

.. code-block:: bash

    curl -s http://localhost:9300/instances
    # -> {"instances": [{"instance_id": "...", "ip": "10.0.0.5", "http_port": 8080, ...}]}

    curl -s http://localhost:9300/healthz
    # -> {"status": "healthy"}

L2 usage tracking and eviction
------------------------------

When MP servers enable ``--coordinator-l2-event-reporting``, they stream L2
store and lookup events to the coordinator. The coordinator aggregates
per-``cache_salt`` usage, enforces quotas, and selects LRU keys to evict.

Each event batch carries the server's ``instance_id`` and a monotonically
increasing sequence number (``seq``) scoped to that instance. These fields
enable future gap detection to identify lost batches.

**Quota management** -- set per-``cache_salt`` byte budgets. Salts without a
quota default to a 0-byte limit (allowlist semantics).

.. code-block:: bash

    # Set a 10 GiB quota for tenant "user-a"
    curl -s -X PUT http://localhost:9300/l2/quota/user-a \
        -H 'Content-Type: application/json' \
        -d '{"limit_gb": 10.0}'
    # -> {"cache_salt": "user-a", "limit_gb": 10.0, "status": "ok"}

    # Remove the quota
    curl -s -X DELETE http://localhost:9300/l2/quota/user-a
    # -> {"cache_salt": "user-a", "limit_gb": 0.0, "status": "removed"}

Use ``_default`` as the path parameter to target the empty-string salt.

**Event ingestion** -- MP servers POST batched events; this is handled
automatically by the event listener and is not typically called manually.

.. code-block:: bash

    curl -s -X POST http://localhost:9300/l2/events \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "seq": 1,
            "events": [
                {"type": "store", "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 1024}
            ]
        }'
    # -> {"recorded": 1}

**Status queries** -- inspect usage and quota info.

.. code-block:: bash

    # Single salt
    curl -s http://localhost:9300/l2/status/user-a
    # -> {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}

    # All salts
    curl -s http://localhost:9300/l2/status
    # -> {"total_gb": 0.005, "by_cache_salt": [...]}

L2 endpoint summary
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 12 38 50

   * - Method
     - Path
     - Description
   * - ``PUT``
     - ``/l2/quota/{cache_salt}``
     - Create or update a quota (body: ``{"limit_gb": N}``).
   * - ``DELETE``
     - ``/l2/quota/{cache_salt}``
     - Remove a salt's quota entry.
   * - ``POST``
     - ``/l2/events``
     - Ingest a batch of L2 store/lookup events.
   * - ``GET``
     - ``/l2/status/{cache_salt}``
     - Quota and usage for a single salt.
   * - ``GET``
     - ``/l2/status``
     - Total usage and per-salt breakdown.
