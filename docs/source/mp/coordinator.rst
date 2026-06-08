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

The CLI accepts ``--host``, ``--port``, ``--instance-timeout``, and
``--health-check-interval``; any flag overrides the matching environment
variable below. See :doc:`/cli/coordinator` for details. Equivalently, the
coordinator can still be launched as a module with
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

The server registers under its telemetry identity (``--service-instance-id`` /
OTel ``service.instance.id``); if that is unset, the coordinator assigns an id.

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
    