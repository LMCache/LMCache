lmcache coordinator
===================

The ``lmcache coordinator`` command launches the LMCache MP **coordinator**, a
standalone HTTP service that tracks the MP server instances in a deployment. MP
servers register with it and send periodic heartbeats; the coordinator evicts
any instance whose heartbeat lapses past ``--instance-timeout``.

It replaces ``python -m lmcache.v1.mp_coordinator``. The process runs in the
foreground; stop it with ``Ctrl-C``.

.. code-block:: bash

   lmcache coordinator [options]

Quick start
-----------

.. code-block:: bash

   lmcache coordinator \
       --host 0.0.0.0 --port 9300 \
       --instance-timeout 30 \
       --health-check-interval 10

Options
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Flag
     - Description
   * - ``--host HOST``
     - Bind address for the coordinator's HTTP server (default: ``0.0.0.0``).
   * - ``--port PORT``
     - HTTP port (default: ``9300``).
   * - ``--instance-timeout SECS``
     - Seconds without a heartbeat after which an instance is evicted
       (default: ``30``).
   * - ``--health-check-interval SECS``
     - Seconds between eviction sweeps; ``0`` disables the loop
       (default: ``10``).

Configuration
-------------

Every flag is optional. Unset flags fall back to the
``LMCACHE_MP_COORDINATOR_*`` environment variables (``HOST``, ``PORT``,
``INSTANCE_TIMEOUT``, ``HEALTH_CHECK_INTERVAL``), and then to the built-in
defaults. A supplied flag always overrides the matching env-derived value, so
env-only deployments keep working unchanged.

See :doc:`/mp/coordinator` for the coordinator's architecture, registration
protocol, and HTTP API.
