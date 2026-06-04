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

    python3 -m lmcache.v1.mp_coordinator

Expected log output:

.. code-block:: text

    LMCache INFO: MP coordinator listening on http://0.0.0.0:9300

.. note::
   A first-class ``lmcache`` CLI subcommand is planned; for now the coordinator
   runs as the module above and is configured via environment variables.

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
    