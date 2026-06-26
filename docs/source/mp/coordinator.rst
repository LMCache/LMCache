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
``--eviction-ratio``, ``--trigger-watermark``, ``--blend-chunk-size``,
``--blend-probe-stride``, and ``--timeout-keep-alive``; any flag overrides the
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
   * - ``LMCACHE_MP_COORDINATOR_BLEND_CHUNK_SIZE``
     - ``256``
     - Tokens per chunk for the global CacheBlend directory. Must equal the
       LMCache chunk size the blend servers use.
   * - ``LMCACHE_MP_COORDINATOR_BLEND_PROBE_STRIDE``
     - ``1``
     - Positions between CacheBlend match probes. ``1`` probes every offset
       for full recall.
   * - ``LMCACHE_MP_COORDINATOR_TIMEOUT_KEEP_ALIVE``
     - ``10``
     - Seconds the HTTP server keeps idle connections open before closing
       them. Must be greater than the MP servers' heartbeat interval
       (default ``5``), otherwise heartbeat requests may hit a closing
       connection and fail with ``Server disconnected without sending a
       response``.
   * - ``LMCACHE_MP_COORDINATOR_ENABLE_STARTUP_RESYNC``
     - ``True``
     - When ``True``, the coordinator runs a one-shot L2 resync on
       startup that paginates an MP server's ``GET /l2/keys`` and
       backfills usage + eviction trackers from existing L2 contents.
       Disable to start from empty trackers (handy for tests, or
       deployments that start the coordinator before any MP server).
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_POLL_INTERVAL``
     - ``1``
     - Seconds between registry checks while waiting for the first
       MP server to register so startup resync can begin.
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_MAX_WAIT``
     - ``60``
     - Maximum seconds startup resync waits for an MP server before
       giving up. The coordinator keeps running with empty trackers
       until normal usage events fill them in.
   * - ``LMCACHE_MP_COORDINATOR_RESYNC_PAGE_SIZE``
     - ``1000``
     - ``page_size`` forwarded to the MP server's ``GET /l2/keys``
       during resync. Larger values reduce RTT count; the server
       clamps to its own ceiling.

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
``store``, ``lookup``, and ``delete`` events to the coordinator. The coordinator
aggregates per-``cache_salt`` usage, enforces quotas, and selects LRU keys
to evict.

Each event batch carries the server's ``instance_id`` and a monotonically
increasing sequence number (``seq``) scoped to that instance. These fields
enable future gap detection to identify lost batches.

**Active eviction loop.** Every
``LMCACHE_MP_COORDINATOR_EVICTION_CHECK_INTERVAL`` seconds, the
coordinator inspects per-salt usage against the registered quotas and,
for any salt over the trigger watermark, picks LRU victims and
dispatches a single ``DELETE /l2`` to a uniformly random registered MP
server. Because all MP servers share the same backing L2 (e.g. one S3
bucket), one dispatch evicts the keys for the whole fleet. The MP
server's L2 adapter fires ``on_l2_keys_deleted`` listeners after the
delete completes; those listeners ship ``delete`` events back through
``POST /l2/events``, which is what updates the coordinator's LRU +
per-salt totals. Dispatch failures or no-instances-registered fall
through to the next cycle — at-least-once semantics, safe because the
S3 delete is idempotent.

**Startup resync.** On boot, the coordinator waits up to
``LMCACHE_MP_COORDINATOR_RESYNC_MAX_WAIT`` seconds for the first MP
server to register, then paginates its
``GET /l2/keys`` and seeds the in-memory usage + eviction trackers
with whatever is already resident in L2 — so a fresh coordinator
does not start from zero usage. Set
``LMCACHE_MP_COORDINATOR_ENABLE_STARTUP_RESYNC=False`` to skip this
phase. Best-effort: resync failures are logged and the manager gives
up; the ongoing usage-event stream from MP servers eventually corrects
any initial blind spots.

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
Supported event types are ``store``, ``lookup``, and ``delete``. A
``delete`` event subtracts the key's previously-recorded bytes from the
per-salt totals (the wire ``bytes`` field is ignored for ``delete``;
the coordinator already knows the size from the original ``store``).

.. code-block:: bash

    curl -s -X POST http://localhost:9300/l2/events \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "seq": 1,
            "events": [
                {"type": "store",  "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 1024},
                {"type": "lookup", "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 0},
                {"type": "delete", "key": {"chunk_hash_hex": "aa", "model_name": "m", "kv_rank": 0, "cache_salt": "user-a"}, "bytes": 0}
            ]
        }'
    # -> {"recorded": 3}

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
     - Ingest a batch of L2 ``store`` / ``lookup`` / ``delete`` events.
   * - ``GET``
     - ``/l2/status/{cache_salt}``
     - Quota and usage for a single salt.
   * - ``GET``
     - ``/l2/status``
     - Total usage and per-salt breakdown.
   * - ``POST``
     - ``/l2/prefetch``
     - Submit a warm prefetch of a token sequence on one server; returns a ``request_id``.
   * - ``GET``
     - ``/l2/prefetch/{instance_id}/{request_id}``
     - Poll a submitted warm prefetch (``pending`` / ``completed``).

Warm prefetch (pre-loading L1 from L2)
--------------------------------------

Pre-warm one MP server's L1 with the KV for a known prompt **before** the
requests arrive, so the first request hits L1 instead of paying the L2 fetch
inline -- useful when you know a workload is about to be routed to a node (a
traffic shift, a hot shared system prompt).

You describe the content by **token ids** -- the unit the cache speaks -- never
by internal cache keys, which you cannot construct (a key is a content hash
plus a per-rank layout bitmap). The coordinator forwards the request to the
named server, which hashes the tokens, expands them across the node's ranks,
loads the chunks from L2 into L1, and **retains** them so a later lookup hits.

The submit returns a ``request_id``; poll the status endpoint until
``completed``. The warm acquires no lock -- the poll simply reports progress
and clears the server-side job once the load finishes.

.. code-block:: bash

    # Submit: warm tenant "user-a"'s prompt on server "server-1"
    curl -s -X POST http://localhost:9300/l2/prefetch \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"instance_id": "server-1", "request_id": "abc123", "chunks": 12, "status": "submitted"}

    # Poll until completed
    curl -s http://localhost:9300/l2/prefetch/server-1/abc123
    # -> {"status": "pending"}      # load still running
    # -> {"status": "completed", "found_keys": 12, "total_keys": 12}

A few rules of thumb:

- ``token_ids`` must match what was stored (same tokenizer / special tokens)
  and contain at least one full ``chunk_size`` of tokens -- only complete
  chunks are warmed (a shorter sequence returns ``status: "noop"``).
- ``world_size`` selects the server's KV layout and the per-rank fan-out
  (``1`` for a single-GPU, TP=1 deployment).
- **Single-node scope**: one ``instance_id`` warms only that node's shards.
  For a model sharded across multiple nodes, warm each node's instance.
