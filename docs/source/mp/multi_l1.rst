Multiple L1 Managers
====================

The MP server can serve KV objects from independently configured DRAM,
Device-DAX, and GDS L1 managers. Each manager has its own capacity, eviction
controller, and store controller. L2 adapters use a fixed L1 affinity tag.

Configuration
-------------

Repeat ``--l1-manager`` in write-placement order. Each JSON object requires
``type``, a unique non-empty ``tag``, and ``size_gb`` (GiB). The types are
``DRAM``, ``DEVDAX``, and ``GDS``. The ``_default`` tag is reserved for DRAM
in this interface and is the default affinity target for L2 adapters.

For example, use a dedicated Device-DAX device and a GDS-capable directory:

.. code-block:: bash

   lmcache server --chunk-size 256 --eviction-policy LRU \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":8}' \
       --l1-manager '{"type":"DEVDAX","tag":"dax","size_gb":16,"path":"/dev/dax0.0"}' \
       --l1-manager '{"type":"GDS","tag":"gds","size_gb":32,"path":"/mnt/nvme/lmcache"}' \
       --l2-adapter '{"type":"fs","base_path":"/mnt/archive/lmcache","affinity_tag":"dax"}'

A Device-DAX L1 in this interface has no implicit DRAM pool. Its size applies
to the mapped device. Use dedicated, physically disjoint regions: different
Device-DAX paths can alias the same CXL memory. Independent managers must not
write those aliases concurrently. GDS uses
an ephemeral slab that is cleared at initialization; it does not restore
cached objects after a server restart.

Use these combinations of ``--l1-manager`` arguments for the six profiles:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Profile
     - Managers in placement order
   * - DRAM only
     - ``DRAM``
   * - GDS only
     - ``GDS``
   * - Device-DAX only
     - ``DEVDAX``
   * - DRAM + GDS
     - ``DRAM``, ``GDS``
   * - DRAM + Device-DAX
     - ``DRAM``, ``DEVDAX``
   * - DRAM + Device-DAX + GDS
     - ``DRAM``, ``DEVDAX``, ``GDS``

Common optional JSON fields are ``align_bytes`` (4096),
``read_ttl_seconds`` (300), ``write_ttl_seconds`` (600), and ``eviction``.
Alignment must be a positive power of two; GDS requires at least 4096 bytes.
TTLs must be positive integers. Unknown fields are rejected.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Backend fields
   * - ``DRAM``
     - ``use_lazy`` (true), ``init_size_gb`` (20, capped at capacity),
       ``shm_name`` (empty), ``use_hugepages`` (false). Shared memory and
       hugepages require eager allocation. Shared-memory names must be unique.
   * - ``DEVDAX``
     - Required ``path``; the allocator checks mapping size and device alignment.
   * - ``GDS``
     - Required ``path``; ``backend`` (``auto``), ``direct_io`` (true).
       See :doc:`configuration` for backend requirements and location semantics.

The global eviction flags supply defaults. Override them per manager with,
for example, ``"eviction":{"eviction_policy":"LRU","trigger_watermark":0.9,
"eviction_ratio":0.1}``. Without global defaults each manager must specify
an eviction policy. ``noop`` is useful for controlled overflow tests; an LRU
manager may free space before a later write needs to overflow.

Legacy single-L1 flags, including ``--l1-size-gb``, remain supported. Do not
combine ``--l1-manager`` with legacy capacity or backend-selection flags.
Python callers can use ``DRAML1ManagerConfig``, ``DevDaxL1ManagerConfig``, and
``GDSL1ManagerConfig``; their ``from_dict`` methods implement the JSON schema.

Placement, affinity, and reads
------------------------------

Writes try managers in argument order. Only ``OUT_OF_MEMORY`` results move to
the next manager; conflicts and exceptions do not. Allocation remains atomic
within each batch, so a batch can fail even when combined free space across
managers would suffice. This does not add cross-L1 deduplication or migration.

An L2 adapter's ``affinity_tag`` selects the host-backed L1 used for both
stores and reloads. It defaults to ``_default`` and must name a configured
manager. An adapter sees completed stores from that manager only. Set the tag
explicitly for a Device-DAX-only configuration. GDS cannot be an affinity
target for the existing host-buffer L2 adapters. Affinity is fixed; runtime
adapter additions are validated against the same tags.

Lookups visit L1 managers synchronously. Prefetch selects one owner per key
and keeps that selection in the request's result. GPU completion releases
locks on those exact owners, including when another request finds a different
copy of the same key. Every manager has independent read and write locks.

Status and Prometheus
---------------------

``GET /status`` includes these fields under ``storage_manager``:

- ``l1_managers``: status keyed by L1 tag, including object/lock counts,
  staging bytes, used/allocated/configured memory, and capacity by medium.
- ``store_controllers`` and ``l1_eviction_controllers``: controller health
  and progress keyed by tag.
- ``l1_usage``: aggregate ``[used_bytes, allocated_bytes]``. For lazy DRAM,
  allocated bytes can be below configured capacity.
- ``l1_capacity_bytes_by_backend``: configured/live capacity summed by
  ``dram``, ``devdax``, and ``gds``. Draining Device-DAX arenas are excluded.

The legacy singular status fields remain available when there is one manager.
``GET /config`` includes every L1 configuration and each adapter's affinity.

Prometheus exports ``lmcache_mp_l1_memory_usage_bytes``,
``lmcache_mp_l1_usage_ratio``, and ``lmcache_mp_l1_staging_bytes`` with
``l1_tag`` and ``backend`` labels. L1 read, write, and eviction counters carry
``l1_tag`` alongside existing cache-salt labels. Optional lifecycle histograms
also use ``l1_tag`` and track each manager's copy independently. Sum across
``l1_tag`` for server totals; do not sum usage ratios. A legacy manager that combines DRAM
and Device-DAX reports ``backend="dram+devdax"`` for its combined usage.

Limits and serving validation
-----------------------------

Only **one GDS slab per process** is supported. The native GDS libraries
register CUDA streams process-wide; adding a second slab currently conflicts
with that registration. This limit permits every profile listed above.
A successful cuFile compatibility-mode transfer does not establish native
GPUDirect DMA support; check the deployment's driver and filesystem.

L1 topology is fixed at startup. The legacy single-region descriptor,
shared-memory transfer channel, P2P registration, and Device-DAX hotplug API
require a single compatible L1. Multi-L1 GPU serving uses the existing
LMCache-driven IPC transfer path. Hybrid models additionally require their
engine's supported recurrent-state connector and matching block/chunk
geometry; see :doc:`hybrid_models`.

Storage profiles alone do not qualify an engine/model/TP combination. Validate
external-cache hits after clearing engine-local prefix state, token outputs,
per-tag writes and reads, and zero remaining locks. Exercise every configured
backend under pressure, and restart the engine and server for a second run.
