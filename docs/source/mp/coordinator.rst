Multi-Server Coordination
=========================

When you run more than one LMCache multiprocess (MP) server, the **MP
Coordinator** is a standalone service they register with, giving you a single,
fleet-wide view of every running server. Each MP server caches independently;
the coordinator ties them together into one coordinated fleet.

.. contents::
   :local:
   :depth: 2

Running the coordinator
-----------------------

The coordinator is a FastAPI service. Start it with:

.. code-block:: bash

    lmcache coordinator

Expected log output:

.. code-block:: text

    LMCache INFO: MP coordinator listening on http://0.0.0.0:9300

See :doc:`/cli/coordinator` for the full flag list. Equivalently, the
coordinator can be launched as a module with
``python3 -m lmcache.v1.mp_coordinator``, which accepts the same flags.

Configuration
-------------

The coordinator is configured through CLI flags only; every flag left unset
keeps the default below.

.. list-table::
   :header-rows: 1
   :widths: 38 14 48

   * - Flag
     - Default
     - Description
   * - ``--host``
     - ``0.0.0.0``
     - Host the HTTP server binds to.
   * - ``--port``
     - ``9300``
     - Port the HTTP server binds to.
   * - ``--instance-timeout``
     - ``30``
     - Seconds without a heartbeat after which a server is dropped from the
       fleet.
   * - ``--health-check-interval``
     - ``10``
     - Seconds between health-check sweeps that expire stale MP-server
       registrations. ``0`` disables the stale-instance eviction loop; it does
       **not** affect the ``/quota`` L2 eviction loop (see
       ``--eviction-check-interval`` below).
   * - ``--eviction-check-interval``
     - ``5``
     - Seconds between L2 eviction sweeps. ``0`` disables the loop.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of tracked keys (by count) to evict per cycle (0.0 to 1.0).
   * - ``--trigger-watermark``
     - ``1.0``
     - Eviction fires when usage reaches this fraction of the quota
       (0.0 exclusive to 1.0).
   * - ``--chunk-size``
     - ``256``
     - Tokens per KV chunk: the CacheBlend match unit and the unit used to
       resolve pin ``token_ids`` to keys. Must equal the MP servers'
       ``--chunk-size``.
   * - ``--hash-algorithm``
     - ``blake3``
     - Token hash algorithm for pin key resolution. Must equal the MP servers'
       ``--hash-algorithm``. ``blake3`` is self-contained; other algorithms
       require vLLM importable in the coordinator process.
   * - ``--enable-blend-lookup``
     - off
     - Index stored chunk content so ``POST /directory/blend-lookup`` can serve
       fleet CacheBlend reuse. Off by default: hashing content costs CPU on
       every store. Also requires the MP servers'
       ``--coordinator-event-reporting``.
   * - ``--blend-probe-stride``
     - ``1``
     - Positions between CacheBlend match probes. ``1`` probes every offset
       for full recall. Ignored unless blend lookup is on.
   * - ``--checkpoint-path``
     - (empty)
     - File the coordinator's derived state is checkpointed to. Empty
       disables it and the coordinator starts cold after every restart.
   * - ``--checkpoint-interval``
     - ``60``
     - Seconds between checkpoint writes; ``0`` writes only on a clean stop.
       Ignored without a path. Does not affect the metadata file, which is
       written whenever pins or quotas change.
   * - ``--metadata-path``
     - (empty)
     - File the operator-set state (L2 pins and per-``cache_salt`` quotas)
       is stored in. Empty means that state is lost on restart.
   * - ``--extra-config``
     - (empty)
     - JSON object of settings the core flags do not name, read by whichever
       view or controller looks for them.
   * - ``--timeout-keep-alive``
     - ``10``
     - Seconds the HTTP server keeps idle connections open before closing
       them. Must be greater than the MP servers' heartbeat interval
       (default ``5``), otherwise heartbeat requests may hit a closing
       connection and fail with ``Server disconnected without sending a
       response``.
   * - ``--disable-metrics``
     - off
     - Skip OpenTelemetry metrics initialization. Metrics are on by default;
       pass this flag and the local ``/metrics`` endpoint returns 404.
   * - ``--otlp-endpoint``
     - unset
     - OTLP gRPC endpoint for metrics push mode. When unset, Prometheus pull
       mode exposes ``/metrics`` on the coordinator HTTP port. When set, the
       local ``/metrics`` endpoint returns 404.

Coordinator metrics export
--------------------------

Metrics are enabled by default. Without an OTLP endpoint, Prometheus scrapes
the coordinator's existing FastAPI port; no separate metrics server or port is
created:

.. code-block:: bash

   curl http://localhost:9300/metrics

Set ``--otlp-endpoint http://collector:4317`` to push metrics to an
OpenTelemetry Collector instead. In OTLP push mode, and when
``--disable-metrics`` is set, ``GET /metrics`` returns 404. This infrastructure
does not itself define coordinator business metrics; instruments register with
the shared OpenTelemetry provider as coordinator capabilities add them.

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
   * - ``--coordinator-event-reporting``
     - ``LMCACHE_COORDINATOR_EVENT_REPORTING``
     - Stream cache store/access/delete events to the coordinator,
       feeding the key directory (fleet-wide placement tracking) and,
       for L2 events, usage/quota tracking and eviction.
   * - ``--coordinator-event-flush-interval``
     - ``LMCACHE_COORDINATOR_EVENT_FLUSH_INTERVAL``
     - Seconds between cache-event batch flushes (must be ``> 0``, default
       ``1``).

The server registers under its stable identity (``--instance-id`` / OTel
``service.instance.id``); if the flag is not passed, the server mints a
random UUID v4 at startup and registers under that.

Registration is best-effort: if the coordinator is unreachable, the MP server
logs a warning, keeps retrying, and continues serving. A malformed
heartbeat-interval value is rejected at startup.

HTTP endpoints
--------------

The coordinator's HTTP surface (base URL ``http://localhost:9300``) groups into:

- **Fleet membership and health** -- registration and liveness
  (``/instances``, ``/healthz``).
- **Quota, usage, and eviction** -- the ``/quota`` group: per-tenant byte
  budgets, usage accounting, and the usage-event ingest that drives fleet-wide
  eviction.
- **Cache control** -- the ``/cache`` group: cache operations dispatched to a
  named server (warm prefetch, pin/unpin, and delete, with more to come).
- **Fleet memory** -- the ``/instances/usage`` endpoints: how full each
  server's memory compartments are, joining event-derived usage against the
  capacity each server declares on the same event stream. Read-only.
- **CacheBlend fragment lookup** -- ``POST /directory/blend-lookup``: finds
  cached chunk content anywhere inside a query sequence, using the blend index
  derived from the key directory's token bindings. Server-to-coordinator only;
  not usually called by hand.

Each endpoint is documented below. Success is ``200`` unless noted, and
``{cache_salt}`` uses the ``_default`` sentinel for the empty salt. The wire
types live in ``lmcache/v1/mp_coordinator/schemas.py``.

.. tip::

   The read-only endpoints below can be read without ``curl`` and ``jq``.
   ``lmcache query coordinator --api NAME`` fetches one of them — ``usage``,
   ``instances``, ``health``, ``directory``, ``keys``, ``quota``,
   ``quota-config``, ``prefetch``, or ``metrics`` — and prints it as an
   aligned table with byte counts and ratios already formatted. See
   :doc:`/cli/query`.

Fleet membership and health
---------------------------

MP servers register, heartbeat, and deregister automatically (see
`Connecting MP servers`_); ``GET /instances`` and ``GET /healthz`` are read-only
operator views.

``POST /instances``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register (or re-register) an MP server. Called automatically by each server on
startup.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``ip``
     - string
     - IP/host of the server's HTTP API; the coordinator dials this address, so
       it must be non-empty.
   * - ``http_port``
     - int
     - Port of the server's HTTP API.
   * - ``instance_id``
     - string
     - Optional. Server identifier; if omitted (or blank) the coordinator
       generates one and returns it.
   * - ``metadata``
     - object
     - Optional. Free-form ``string -> string`` registration hints.
   * - ``p2p_advertised_url``
     - string
     - Optional. URL the server advertises for peer-to-peer transfers; empty
       when it is not in P2P.
   * - ``mq_port``
     - int
     - Optional (default ``0``). ZMQ message-queue port P2P peers send
       lookup/unlock RPCs to; ``0`` when P2P is disabled.
**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "re_registered": false}

``instance_id`` is the registered id (the generated one when the request omitted
it); ``re_registered`` is ``true`` when this replaced an existing registration.

**HTTP status codes:**

- ``200``: registered.
- ``422``: request body fails field-level validation (e.g. blank ``ip`` or
  out-of-range ``http_port``).

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/instances \
        -H 'Content-Type: application/json' \
        -d '{"ip": "10.0.0.5", "http_port": 8080}'
    # -> {"instance_id": "mp-3f2c9d...", "re_registered": false}

``PUT /instances/{instance_id}/heartbeat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Record a liveness heartbeat. Called automatically while the server runs.

**Path parameters:** ``instance_id`` — the instance recording the heartbeat.

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1"}

**HTTP status codes:**

- ``200``: heartbeat recorded.
- ``404``: unknown instance — the caller should re-register via
  ``POST /instances``.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:9300/instances/server-1/heartbeat
    # -> {"instance_id": "server-1"}

``DELETE /instances/{instance_id}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deregister an MP server. Called automatically on shutdown.

**Path parameters:** ``instance_id`` — the server to deregister.

**Response:** ``204 No Content`` with an empty body, returned whether or not the
instance was registered (idempotent).

**HTTP status codes:**

- ``204``: deregistered (also returned for an unknown instance).

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/instances/server-1 -o /dev/null -w '%{http_code}\n'
    # -> 204

``GET /instances``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List every registered MP server.

**Response** (``200 OK``):

.. code-block:: json

    {
      "instances": [
        {
          "instance_id": "server-1",
          "ip": "10.0.0.5",
          "http_port": 8080,
          "registration_time": 1719000000.0,
          "metadata": {},
          "p2p_advertised_url": "",
          "mq_port": 0
        }
      ]
    }

Each entry reports the server's ``instance_id``, the ``ip`` / ``http_port`` the
coordinator reaches it at, the wall-clock ``registration_time`` (epoch seconds),
any ``metadata`` supplied at registration, and the ``p2p_advertised_url`` /
``mq_port`` used for peer-to-peer transfers (empty / ``0`` when P2P is disabled).

**HTTP status codes:**

- ``200``: fleet listed (an empty fleet returns ``{"instances": []}``).

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/instances

``GET /healthz``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coordinator liveness probe (for Kubernetes).

**Response** (``200 OK``):

.. code-block:: json

    {"status": "healthy"}

**HTTP status codes:**

- ``200``: the coordinator is up.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/healthz
    # -> {"status": "healthy"}

Quota, usage, and eviction
--------------------------

The ``/quota`` group owns per-``cache_salt`` byte budgets, the live usage
accounting behind them, and the usage-event stream that drives fleet-wide
eviction. (The MP server exposes a node-local ``/quota`` with the same shape;
this is its fleet-wide counterpart.) Use ``_default`` as the path parameter to
target the empty-string salt.

.. warning::

   Do **not** use the MP server's node-local ``/quota`` API together with the
   coordinator's. The two are independent, unsynchronized quota registries
   enforcing eviction on the **same shared L2**: the server-side enforcer
   (active when the server runs a per-salt eviction policy) uses strict
   allowlist semantics — any salt missing from *its own* table is fully
   evicted — and it never sees quotas registered on the coordinator, and vice
   versa. Mixing the two produces competing eviction decisions: the server can
   wipe data the coordinator considers within quota (or still exempt before
   the default limit is armed). Pick one owner per deployment — in
   coordinator-managed deployments, register quotas **only** through the
   coordinator's ``/quota`` API and leave the servers' node-local quota tables
   untouched.

Salts without an explicit quota are governed by the registry's **default
limit** (``PUT /quota/config``). On boot the default is unset, and unquota'd
salts are **exempt** from eviction — quotas live in memory, so a freshly
(re)started coordinator has an empty quota table until the external quota
controller re-syncs it, and the exempt default keeps that window from
mass-evicting unknown tenants. After re-registering every per-salt quota, the
controller sets the default to ``0`` — the signal that arms strict allowlist
enforcement (all bytes under unquota'd salts become evictable on the next
cycle):

.. code-block:: bash

    # 1. re-register every tenant quota
    curl -s -X PUT http://localhost:9300/quota/user-a \
        -H 'Content-Type: application/json' -d '{"limit_gb": 10.0}'
    # ... one PUT per tenant ...

    # 2. arm eviction of everything else
    curl -s -X PUT http://localhost:9300/quota/config \
        -H 'Content-Type: application/json' -d '{"default_limit_gb": 0}'
    # -> {"default_limit_gb": 0.0}

When MP servers enable ``--coordinator-event-reporting``, they stream cache
``store``, ``access``, and ``delete`` events to the coordinator's
``POST /events``. Applied ``l2`` batches also feed the quota side:
the coordinator aggregates per-``cache_salt`` usage, enforces quotas, and
selects LRU keys to evict. Each batch carries the server's ``instance_id``,
``incarnation``, and a monotonically increasing sequence number (``seq``)
scoped to that instance, so replays are deduplicated and lost batches are
detected.

**Active eviction loop.** Every ``--eviction-check-interval`` seconds, the
coordinator inspects per-salt usage against the registered quotas and,
for any salt over the trigger watermark, picks LRU victims and
dispatches a single ``DELETE /cache/objects`` to a uniformly random registered MP
server. Because all MP servers share the same backing L2 (e.g. one S3
bucket), one dispatch evicts the keys for the whole fleet. The MP
server's L2 adapter fires ``on_l2_keys_deleted`` listeners after the
delete completes; those listeners ship ``delete`` events back through
``POST /events``, which is what updates the coordinator's LRU +
per-salt totals. Dispatch failures or no-instances-registered fall
through to the next cycle — at-least-once semantics, safe because the
S3 delete is idempotent.

**Cold start.** The coordinator's trackers are in-memory and are built
only from the cache-event stream, so after a restart per-salt usage
starts at zero even though the bytes are still resident in L2. Quotas
under-report until enough events accumulate. Unquota'd salts are exempt
from eviction until the quota controller sets a default limit, so a cold
coordinator cannot mass-evict — but it can under-evict, and an operator
re-arming quotas right after a restart should expect the usage numbers
to climb toward the true value rather than start at it.

``PUT /quota/config`` / ``GET /quota/config``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set / read the default limit applied to salts with no explicit quota entry.

**Request body** (``PUT``):

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``default_limit_gb``
     - float or null
     - ``null`` (the boot default) exempts unquota'd salts from eviction;
       ``0`` arms strict allowlist enforcement (all unquota'd bytes become
       evictable next cycle); a positive value grants every unquota'd salt
       that byte budget.
   * - ``tier``
     - string
     - Optional (default ``l2``). Only ``l2`` is supported today.

**Response** (``200 OK``):

.. code-block:: json

    {"default_limit_gb": 0.0}

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota/config
    # -> {"default_limit_gb": null}          (boot state: unquota'd exempt)

    curl -s -X PUT http://localhost:9300/quota/config \
        -H 'Content-Type: application/json' -d '{"default_limit_gb": 0}'
    # -> {"default_limit_gb": 0.0}           (allowlist enforcement armed)

``PUT /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create or update a tenant's byte budget.

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Type
     - Description
   * - ``limit_gb``
     - float
     - Byte budget in GiB; must be ``>= 0`` (``0`` clears the tenant's data on
       the next eviction cycle).
   * - ``tier``
     - string
     - Optional (default ``l2``). Cache tier the quota applies to; only ``l2`` is
       supported today.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "limit_gb": 10.0, "status": "ok"}

**HTTP status codes:**

- ``200``: quota applied.
- ``400``: invalid limit (negative or non-finite).
- ``422``: request body fails field-level validation.

**Example:**

.. code-block:: bash

    curl -s -X PUT http://localhost:9300/quota/user-a \
        -H 'Content-Type: application/json' \
        -d '{"limit_gb": 10.0}'
    # -> {"cache_salt": "user-a", "limit_gb": 10.0, "status": "ok"}

``DELETE /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove a salt's quota entry. Any bytes still cached under it become over-budget
on the next eviction cycle (effective limit drops to ``0``).

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Query parameters:** ``tier`` — optional (default ``l2``); cache tier the quota
applies to.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "limit_gb": 0.0, "status": "removed"}

When no quota was registered for the salt, ``status`` is ``"not_found"`` (still
``200 OK``).

**HTTP status codes:**

- ``200``: removed, or ``not_found`` if no quota existed.

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/quota/user-a
    # -> {"cache_salt": "user-a", "limit_gb": 0.0, "status": "removed"}

``GET /quota/{cache_salt}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read the quota and live usage for a single salt.

**Path parameters:** ``cache_salt`` — tenant identifier (``_default`` for the
empty salt).

**Query parameters:** ``tier`` — ``l1`` or ``l2`` (default ``l2``). Every field
describes the requested tier. Quotas are enforced on L2 only, so an ``l1`` read
reports L1 usage with ``quota_exists: false`` and ``quota_limit_gb: 0.0`` --
never the L2 budget, which governs different bytes.

**Response** (``200 OK``):

.. code-block:: json

    {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}

``quota_limit_gb`` is the configured limit in GiB (``0.0`` when no quota is set),
``quota_exists`` whether an explicit quota is registered, and ``usage_gb`` the
current aggregate usage. This endpoint never returns ``404`` for an unknown salt.

**HTTP status codes:**

- ``200``: quota and usage reported.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota/user-a
    # -> {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}

``GET /quota``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List total usage and a per-salt breakdown.

**Query parameters:** ``tier`` — ``l1`` or ``l2`` (default ``l2``). Every field
describes the requested tier, and rows come from that tier's usage plus the
quotas that apply to it -- so an ``l1`` listing holds only salts with L1 bytes,
each with ``quota_exists: false``. ``all`` is rejected with ``400``, because a
key resident in both tiers holds bytes in both and a cross-tier total would
count it twice.

**Response** (``200 OK``):

.. code-block:: json

    {
      "total_gb": 0.005,
      "by_cache_salt": [
        {"cache_salt": "user-a", "quota_limit_gb": 10.0, "quota_exists": true, "usage_gb": 0.001}
      ]
    }

``total_gb`` is aggregate usage across all salts in GiB; each ``by_cache_salt``
entry has the same fields as the ``GET /quota/{cache_salt}`` response.

**HTTP status codes:**

- ``200``: usage reported.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/quota
    # -> {"total_gb": 0.005, "by_cache_salt": [...]}

Usage events arrive on the fleet cache-event stream
(``POST /events``); there is no separate quota ingestion
endpoint. See ``docs/design/v1/mp_coordinator/cache_events.md`` for the
batch format and routing semantics.

Key directory
-------------

The ``/directory`` group is a read-only operator view of the fleet-wide key
directory: which keys are cached, where (instance / tier / backend), and what
token ids each chunk holds. The directory is **eventually consistent soft
state** built from the servers' cache-event stream -- every answer is a hint to
be validated at the owning server, never a guarantee.

``GET /directory/keys``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

List cached keys and their placements, one page at a time.

**Query parameters:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Parameter
     - Default
     - Description
   * - ``tier``
     - ``all``
     - Keep placements on this tier (``l1`` / ``l2``; ``all`` keeps every
       tier).
   * - ``instance_id``
     - *(empty)*
     - Keep placements reported by this MP server (empty keeps every
       instance).
   * - ``backend``
     - *(empty)*
     - Keep placements on this backend, e.g. ``dram`` / ``fs`` (empty keeps
       every backend).
   * - ``offset``
     - ``0``
     - Matching keys to skip (pagination).
   * - ``limit``
     - ``1000``
     - Maximum keys to return (1 to 10000).

**Response** (``200 OK``):

.. code-block:: json

    {
      "total": 2,
      "keys": [
        {
          "key": {
            "chunk_hash_hex": "aa12...",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "kv_rank": 0,
            "object_group_id": 0,
            "cache_salt": ""
          },
          "placements": [
            {
              "instance_id": "server-1",
              "incarnation": 1719000000,
              "tier": "l1",
              "backend": "dram",
              "size_bytes": 1048576,
              "shared": false
            }
          ],
          "num_tokens": 256
        }
      ]
    }

``total`` counts every key with at least one placement matching the filters;
``keys`` is the requested page of them, each with **only its matching
placements**. ``num_tokens`` reports how many token ids the directory knows for
the key's chunk (``0`` = unknown) -- fetch the actual tokens via
``POST /directory/lookup``, which exists precisely so listing pages stay
small. Pages of a changing directory may skip or repeat keys (snapshot
semantics).

**HTTP status codes:**

- ``200``: page returned (an empty directory returns ``{"total": 0, "keys": []}``).
- ``422``: invalid parameter (negative ``offset``, ``limit`` out of range,
  unknown ``tier``).

**Example:**

.. code-block:: bash

    # Everything on server-1's L1:
    curl -s "http://localhost:9300/directory/keys?tier=l1&instance_id=server-1&limit=100"

``POST /directory/lookup``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resolve cache content to its placements and token ids. One endpoint, two
forms -- supply exactly one:

* **keys form** -- ``{"keys": [...]}``: resolve keys you already have (e.g.
  from ``GET /directory/keys``).
* **tokens form** -- ``{"token_ids": [...], "model_name": ..., "world_size":
  ..., "cache_salt": ...}``: resolve a request's tokens to the keys of its
  complete chunks (the same fan-out the pin APIs use).

.. important::

   ``token_ids`` must be the request's **whole** token sequence from position
   0, not one chunk's worth. Chunk hashes are prefix-chained -- each chunk's
   key depends on every token before it -- so a mid-request slice resolves to
   different (nonexistent) keys. Trailing tokens that do not fill a chunk are
   ignored.

**Request body** (tokens form):

.. code-block:: json

    {
      "token_ids": [15496, 11, 995, 314],
      "model_name": "meta-llama/Llama-3.1-8B-Instruct",
      "world_size": 1,
      "cache_salt": ""
    }

**Request body** (keys form):

.. code-block:: json

    {
      "keys": [
        {
          "chunk_hash_hex": "aa12...",
          "model_name": "meta-llama/Llama-3.1-8B-Instruct",
          "kv_rank": 0,
          "object_group_id": 0,
          "cache_salt": ""
        }
      ]
    }

**Response** (``200 OK``):

.. code-block:: json

    {
      "chunks": 1,
      "results": [
        {
          "key": {"chunk_hash_hex": "aa12...", "model_name": "...", "kv_rank": 0,
                  "object_group_id": 0, "cache_salt": ""},
          "placements": [
            {"instance_id": "server-1", "incarnation": 1770000000, "tier": "l1",
             "backend": "dram", "size_bytes": 8388608, "shared": false}
          ],
          "token_ids": [15496, 11, 995]
        }
      ]
    }

``chunks`` is the number of complete chunks the tokens resolved to (keys form:
the number of keys requested); ``results`` has one entry per resolved key, in
request order (tokens form: ``chunks`` x the per-rank fan-out). ``placements``
is empty for keys the directory does not know; ``token_ids`` is empty when the
directory has no tokens for the key's chunk (never stored with token reporting
on, or not yet re-reported after an event gap).

**HTTP status codes:**

- ``200``: results returned.
- ``400``: the token sequence exceeds the per-request cap, or a resolution
  parameter is invalid (e.g. ``model_name`` contains ``@``).
- ``422``: neither or both forms supplied, ``model_name`` missing with
  ``token_ids``, or a key is malformed (e.g. ``chunk_hash_hex`` is not hex).

**Examples:**

.. code-block:: bash

    # Tokens form -- where is this prompt cached, and what does each chunk hold?
    curl -s -X POST http://localhost:9300/directory/lookup \
      -H 'Content-Type: application/json' \
      -d '{"token_ids": [15496, 11, 995], "model_name": "m", "world_size": 1}'

    # Keys form -- keys taken from GET /directory/keys:
    curl -s -X POST http://localhost:9300/directory/lookup \
      -H 'Content-Type: application/json' \
      -d '{"keys": [{"chunk_hash_hex": "aa12...", "model_name": "m", "kv_rank": 0}]}'

Cache control
-------------

The ``/cache`` group dispatches cache operations to a named MP server. It covers
**warm prefetch**, **pin/unpin**, and **delete**; further cache-control
operations will be documented as endpoints here as they land.

**Warm prefetch (pre-loading L1 from L2).** Pre-warm one MP server's L1 with the
KV for a known prompt **before** the requests arrive, so the first request hits
L1 instead of paying the L2 fetch inline -- useful when you know a workload is
about to be routed to a node (a traffic shift, a hot shared system prompt).

You describe the content by **token ids** -- the unit the cache speaks -- never
by internal cache keys, which you cannot construct (a key is a content hash
plus a per-rank layout bitmap). The coordinator forwards the request to the
named server, which hashes the tokens, expands them across the node's ranks,
loads the chunks from L2 into L1, and **retains** them so a later lookup hits.
The submit returns a ``request_id``; poll the status endpoint until
``completed``. The warm acquires no lock -- the poll simply reports progress and
clears the server-side job once the load finishes.

``POST /cache/prefetches``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit a warm prefetch of a token sequence on one named server.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Field
     - Type
     - Description
   * - ``instance_id``
     - string
     - Target MP server; must be registered.
   * - ``model_name``
     - string
     - Model whose layout sizes the target's L1 buffers.
   * - ``world_size``
     - int
     - World size (``>= 1``) selecting the KV layout and the per-rank fan-out
       (``1`` for a single-GPU, TP=1 deployment).
   * - ``token_ids``
     - list[int]
     - Prompt tokens whose complete ``chunk_size`` chunks are warmed; must match
       what was stored (same tokenizer / special tokens). A sub-chunk sequence
       is a ``noop``.
   * - ``cache_salt``
     - string
     - Optional (default ``""``). Per-tenant isolation salt applied to the
       produced keys.

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "request_id": "abc123", "chunks": 12, "status": "submitted"}

When the sequence is shorter than one chunk, nothing is submitted and
``request_id`` is empty:

.. code-block:: json

    {"instance_id": "server-1", "request_id": "", "chunks": 0, "status": "noop"}

``request_id`` is the id to poll; ``chunks`` is the number of whole chunks
submitted to warm.

**HTTP status codes:**

- ``200``: submitted (or a ``noop`` as above).
- ``404``: unknown ``instance_id`` (not registered).
- ``502``: the target server was unreachable or rejected the submit.
- ``422``: request body fails field-level validation.

.. note::

   **Single-node scope:** one ``instance_id`` warms only that node's shards. For
   a model sharded across multiple nodes, submit one request per node's instance.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/prefetches \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"instance_id": "server-1", "request_id": "abc123", "chunks": 12, "status": "submitted"}

``GET /cache/prefetches/{instance_id}/{request_id}``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poll a submitted warm prefetch; the response relays the owning server's status
verbatim with its code.

**Path parameters:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``instance_id``
     - string
     - The server the prefetch was submitted to.
   * - ``request_id``
     - string
     - The id returned by ``POST /cache/prefetches``.

**Response** (``200 OK``) while the load runs:

.. code-block:: json

    {"status": "pending"}

…and once complete:

.. code-block:: json

    {"status": "completed", "found_keys": 12, "total_keys": 12}

``found_keys`` of ``total_keys`` requested chunks were resident.

**HTTP status codes:**

- ``200``: status reported (``pending`` or ``completed``).
- ``404``: unknown ``instance_id``, or unknown ``request_id`` relayed from the
  server.
- ``502``: the target server was unreachable.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/cache/prefetches/server-1/abc123
    # -> {"status": "completed", "found_keys": 12, "total_keys": 12}

**Pin/unpin (protecting cache from eviction).** Pin a token sequence's cache so
it is not evicted from L2 until unpinned. The coordinator resolves the token
sequence to its object keys **locally** (no MP-server round-trip) and records
them in its L2 eviction plan (``POST``) or releases them (``DELETE``), excluding
pinned keys from quota-based eviction. L2 pins are fleet-wide (per
``cache_salt``), so no target instance is named.

Local resolution requires the coordinator's ``chunk_size`` and
``hash_algorithm`` (see `Configuration`_) to match the MP servers' ``--chunk-size``
/ ``--hash-algorithm``; otherwise the resolved keys will not match what was
stored and the pin protects nothing. It also requires the MP servers to be
launched with ``--no-separate-object-groups`` (the coordinator resolves keys in
a single object group).

``POST /cache/pins``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pin a token sequence's keys in the L2 eviction plan.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 18 16 66

   * - Field
     - Type
     - Description
   * - ``model_name``
     - string
     - Model whose rank fan-out to use when resolving keys.
   * - ``world_size``
     - int
     - World size (``>= 1``) selecting the per-rank fan-out.
   * - ``token_ids``
     - list[int]
     - Prompt tokens whose complete chunks are pinned; must match what was
       stored. A sub-chunk sequence pins nothing (``affected`` 0).
   * - ``cache_salt``
     - string
     - Optional (default ``""``). Per-tenant isolation salt.

**Response** (``200 OK``):

.. code-block:: json

    {"requested": 12, "affected": 12, "status": "pinned"}

``requested`` is the number of whole chunks resolved; ``affected`` is the number
of L2 keys pinned (chunks times the per-rank fan-out).

**HTTP status codes:**

- ``200``: pinned.
- ``400``: ``token_ids`` exceeds the per-request cap, or ``cache_salt`` violates
  its invariants.
- ``422``: request body fails field-level validation.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/pins \
        -H 'Content-Type: application/json' \
        -d '{
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"requested": 12, "affected": 12, "status": "pinned"}

.. note::

   **Requires event reporting.** The coordinator can only exclude keys from
   eviction for a salt it is tracking, which requires the MP servers started with
   ``--coordinator-event-reporting`` (see `Connecting MP servers`_).

``DELETE /cache/pins``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unpin a token sequence's keys from the L2 eviction plan. Same request body as
``POST /cache/pins``. The response mirrors the pin (``affected`` is the number
of keys unpinned), with ``status`` ``"unpinned"``. Pins are reference-counted: a
chunk pinned *N* times needs *N* unpins before it can be evicted.

**HTTP status codes:** same as ``POST /cache/pins``.

**Example:**

.. code-block:: bash

    curl -s -X DELETE http://localhost:9300/cache/pins \
        -H 'Content-Type: application/json' \
        -d '{
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a"
        }'
    # -> {"requested": 12, "affected": 12, "status": "unpinned"}

**Delete (removing cache by token sequence).** Delete a token sequence's cache
on one named server, addressed by token ids. The coordinator resolves the tokens
to object keys locally (like pin) and issues a single key-addressed
``DELETE /cache/objects`` to the named server, which removes them from the
requested tier(s). The ``tier`` field selects the tier(s): ``l1`` deletes only
the named server's L1, ``l2`` only L2, ``all`` both. When the tier includes L2,
the coordinator first drops any key it is protecting with an L2 pin from the
delete set unless ``force`` is set — so a pinned key is retained in every tier
the delete would have touched; ``force`` deletes them and drops those pins.

``POST /cache/delete``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Delete a token sequence on one named server.

**Request body:** (``model_name``,
``world_size``, ``token_ids``, ``cache_salt``) plus ``tier`` (``l1`` / ``l2`` /
``all``) and ``force`` (bool, default ``false``). When ``force`` is ``true``,
locked keys are deleted anyway (L1 read/write locks on the node and the
coordinator's L2 pin set).

**Response** (``200 OK``):

.. code-block:: json

    {"instance_id": "server-1", "requested": 12, "affected": 24, "skipped": 0, "status": "deleted"}

``requested`` is the number of whole chunks resolved. ``affected`` and
``skipped`` are **totals across the tiers acted on**: ``affected`` counts L1 keys
removed by the node plus L2 keys removed by the coordinator, and ``skipped``
counts L1 keys the node refused plus L2 keys held back for an L2 pin (non-force
only). A chunk resident in both tiers (``tier=all``) contributes to both counts,
so ``affected`` may be up to ``2 x requested x world_size``. A sub-chunk
sequence returns ``status`` ``"noop"``.

**HTTP status codes:**

- ``200``: deleted (or a ``noop``).
- ``404``: no server is registered under ``instance_id``.
- ``502``: the target server was unreachable or rejected the delete.

**Example:**

.. code-block:: bash

    curl -s -X POST http://localhost:9300/cache/delete \
        -H 'Content-Type: application/json' \
        -d '{
            "instance_id": "server-1",
            "model_name": "Qwen/Qwen3-8B",
            "world_size": 1,
            "token_ids": [101, 102, 103, "..."],
            "cache_salt": "user-a",
            "tier": "all",
            "force": false
        }'
    # -> {"instance_id": "server-1", "requested": 12, "affected": 24, "skipped": 0, "status": "deleted"}

Fleet memory
------------

The ``/instances/usage`` endpoints report how full each MP server's memory
compartments are. A **compartment** is one thing that owns bytes: the L1 pool
of a backing medium, or one L2 adapter. It is identified by
``(tier, backend)`` -- the same pair cache events tag placements with.

Two inputs are joined, and both ride the cache-event stream. **Usage** is
derived from the events the servers already publish. **Capacity** arrives as
a capacity report on the same stream -- once at startup, then whenever an
adapter is added, removed, or reconfigured. Both are automatic; there is
nothing to configure beyond pointing servers at a coordinator and leaving
event reporting enabled.

.. note::

   Capacity travels on the event stream, so disabling event reporting
   disables both halves together: every ``usage_ratio`` reads ``null``
   (*unknown*) rather than a ratio against a stale declaration.

These endpoints are read-only. The coordinator never evicts or throttles based
on them.

.. note::

   ``usage_ratio`` is ``null`` whenever the server declared no capacity for a
   compartment, and this is common: the ``fs``, ``mooncake``, ``p2p``, and
   ``sagemaker`` adapters expose no capacity setting at all, and ``s3`` /
   ``raw_block`` report one only when you set ``max_capacity_gb`` /
   ``capacity_bytes``. A ``null`` means *unknown*, never *empty* -- do not
   treat it as ``0``.

   Ratios above ``1.0`` are reported as-is rather than capped. A compartment
   holding more than its declared capacity means the declaration is wrong, and
   that is worth seeing.

``GET /instances/usage``
~~~~~~~~~~~~~~~~~~~~~~~~

The whole fleet: every server's compartments, plus the shared pools.

**Response** (``200 OK``):

.. code-block:: json

    {
      "instances": [
        {
          "instance_id": "server-1",
          "registered": true,
          "declared_capacity": true,
          "modules": [
            {"tier": "l1", "backend": "dram", "shared": false,
             "used_bytes": 10737418240, "capacity_bytes": 42949672960,
             "usage_ratio": 0.25},
            {"tier": "l2", "backend": "fs", "shared": false,
             "used_bytes": 7516192768, "capacity_bytes": 0,
             "usage_ratio": null}
          ]
        }
      ],
      "shared_modules": [
        {"tier": "l2", "backend": "s3", "shared": true,
         "used_bytes": 4398046511104, "capacity_bytes": 17592186044416,
         "usage_ratio": 0.25}
      ]
    }

``shared_modules`` holds storage several servers mount -- one S3 bucket, one
CXL region. These are counted **once for the fleet** and appear in no
instance's ``modules``. Summing them per mounting server would multiply both
the bytes and the capacity by the number of mounts.

A server appears when it is registered, when it still holds bytes, or when it
declared capacity. ``registered: false`` therefore means a departed server
whose L2 data outlived it; its L1 bytes are dropped when it goes.

**Example:**

.. code-block:: bash

    # Which servers are most heavily loaded?
    curl -s http://localhost:9300/instances/usage | jq -r '
      .instances[] | .instance_id as $i | .modules[]
      | select(.usage_ratio != null)
      | "\($i) \(.tier)/\(.backend) \((.usage_ratio*100|floor))%"'
    # -> server-1 l1/dram 25%
    # -> server-2 l1/dram 81%

``GET /instances/{instance_id}/usage``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One server's compartments, in the same shape as an entry of ``instances``
above.

**HTTP status codes:**

- ``200``: found.
- ``404``: the coordinator knows nothing about this id -- it is not
  registered, holds no bytes, and declared no capacity.

**Example:**

.. code-block:: bash

    curl -s http://localhost:9300/instances/server-1/usage

A server whose L1 pool uses the default lazy allocator grows its heap on
demand. Capacity here is the **configured** size, not the grown heap, so a
freshly started server correctly reads near ``0``\% rather than near full.

CacheBlend fragment lookup
--------------------------

``POST /directory/blend-lookup`` is the fragment counterpart to
``/directory/lookup``: the query need not be a prefix, and each match reports
where the content sits in the query and where it sat when stored, so the caller
can re-RoPE it.

It is served from the blend index, derived from the key directory's token
bindings — the coordinator learns content from the cache-event stream, not from
a separate publish call. Both feeds default to off and both are required: the
coordinator needs ``--enable-blend-lookup``, and every MP server whose chunks
should be discoverable needs ``--coordinator-event-reporting`` (a server with a
coordinator URL but no event reporting warns at startup and matches locally
only). Matching is chunked at the coordinator's ``--chunk-size`` — which must
equal the MP servers' ``--chunk-size`` — probing every
``--blend-probe-stride`` positions.

**Request body:**

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Type
     - Description
   * - ``tokens_b64``
     - string
     - Query tokens packed as base64 little-endian ``uint32`` (see
       ``encode_tokens`` / ``decode_tokens`` in ``schemas.py``).

**Response** (``200 OK``):

.. code-block:: json

    {
      "matches": [
        {"chunk_hash": "ab12...", "old_st": 0,   "cur_st": 512},
        {"chunk_hash": "cd34...", "old_st": 256, "cur_st": 768}
      ]
    }

``chunk_hash`` is the chunk's content hash, which the caller expands to
per-rank object keys with its own model, salt, and world size; ``old_st`` is
its position in the stored sequence (re-RoPE source) and ``cur_st`` its
position in the query (re-RoPE target). Matches are sorted ascending by
``cur_st``, at most one per chunk, and may overlap — a caller that scatters
them resolves overlaps itself. A query shorter than one chunk, or a coordinator
without ``--enable-blend-lookup``, returns ``{"matches": []}``.

**HTTP status codes:**

- ``200``: lookup completed (an empty match list is not an error).
- ``422``: ``tokens_b64`` is not valid base64 or not a whole number of
  ``uint32`` tokens.

Index counts are reported under the ``blend`` key of ``GET /directory/stats``.
