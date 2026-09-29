Multiple L1 Managers
====================

An MP server can run several L1 managers, each with its own capacity, tag,
eviction policy, and lock TTLs. Configure each manager with a separate
``--l1-manager`` JSON argument. Supported types are ``DRAM`` and ``GDS``.
The managers are peers; write and prefetch policies decide where data goes.

Start with DRAM
---------------

The existing DRAM command remains supported:

.. code-block:: bash

   lmcache server --l1-size-gb 20 --eviction-policy LRU

It creates a DRAM manager tagged ``_default``. The JSON interface lets you
name the manager and configure it independently:

.. code-block:: bash

   lmcache server \
       --eviction-policy LRU \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":20}'

Tags must be unique, non-empty strings. Only a DRAM manager may use
``_default``. You can combine ``--l1-size-gb`` with additional JSON managers,
but do not also declare a JSON manager tagged ``_default``: the legacy
option already creates one.

Configure capacity and eviction
-------------------------------

Each JSON object requires ``type``, ``tag``, and ``size_gb``. Sizes use
binary gigabytes (1 GiB = 2\ :sup:`30` bytes). Unknown fields and invalid
values are rejected at startup.

.. list-table:: Common JSON fields
   :header-rows: 1
   :widths: 30 25 45

   * - Field
     - Default
     - Meaning
   * - ``size_gb``
     - Required
     - Positive capacity; must hold at least one aligned allocation.
   * - ``align_bytes``
     - ``4096``
     - Positive power-of-two allocation alignment.
   * - ``eviction.eviction_policy``
     - ``--eviction-policy``
     - ``LRU``, ``ARC``, ``IsolatedLRU``, or ``noop``. Required in JSON
       when the CLI default is omitted.
   * - ``eviction.trigger_watermark``
     - ``--eviction-trigger-watermark`` (``0.8``)
     - Used-memory ratio that triggers eviction; range 0--1.
   * - ``eviction.eviction_ratio``
     - ``--eviction-ratio`` (``0.2``)
     - Fraction of allocated memory to evict; range 0--1.
   * - ``read_ttl_seconds``
     - ``--l1-read-ttl-seconds`` (``300``)
     - Positive integer read-lock TTL in seconds.
   * - ``write_ttl_seconds``
     - ``--l1-write-ttl-seconds`` (``600``)
     - Positive integer write-lock TTL in seconds.

JSON eviction settings override the global eviction settings when
``--eviction-policy`` is supplied. Without that CLI option, provide an
eviction policy in every JSON manager; watermark and ratio then default to
``0.8`` and ``0.2``. Lock TTLs can always be overridden per manager.
These TTLs govern locks, not the lifetime of cached data.

DRAM additionally accepts ``use_lazy`` (default ``true``), ``init_size_gb``
(default ``20``, capped at ``size_gb``), ``devdax_path`` (default unset),
and ``shm_name`` (default empty). ``init_size_gb`` controls lazy allocation
only. A non-empty ``shm_name`` cannot coexist with lazy allocation or
``devdax_path``. Device-DAX requires ``use_lazy:false``. The legacy DRAM
size, alignment, and allocator flags configure the legacy manager only;
set these fields in JSON for additional managers.

Bind L2 adapters to an L1
-------------------------

An L2 adapter's ``affinity_tag`` names the L1 whose buffers it uses for
stores and loads. It defaults to ``_default`` and must match a configured
L1 tag. For example, give two filesystem adapters separate DRAM pools:

.. code-block:: bash

   lmcache server \
       --eviction-policy LRU \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":20}' \
       --l1-manager '{"type":"DRAM","tag":"archive","size_gb":8,"read_ttl_seconds":120,"eviction":{"eviction_policy":"ARC"}}' \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache/local","affinity_tag":"_default"}' \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache/archive","affinity_tag":"archive"}'

This binds each adapter to its corresponding pool; it does not automatically
distribute writes across both pools or replicate data between them.

* **Writes:** the default placement policy selects the DRAM manager tagged
  ``_default``, or the first configured DRAM manager if that tag is absent.
  A single GDS-only deployment writes to its sole manager. There is no
  automatic spillover to another L1 when the selected manager is full.
* **Stores:** each L1 has its own store controller. The default store policy
  sends its completed writes only to L2 adapters with the same affinity tag.
* **Reads:** prefetch looks up all L1 managers concurrently. The default
  prefetch policy prefers L1 over L2 and chooses the earliest configured
  source within each tier. L2 loads allocate buffers in the adapter's
  affinity L1. Overlapping reads of a duplicated key keep using the already
  selected readable copy until their locks are released.

Custom write placement is available through the Python
``StorageManager(config, write_policy=...)`` interface. Its
``select_write_targets(keys, managers)`` method returns a manager-index to
keys mapping, with at most one target per key. There is no write-policy CLI
flag. Store and prefetch policies use the existing ``--l2-store-policy``
and ``--l2-prefetch-policy`` options; see :doc:`l2_storage/index`.

Migrate GDS configuration
-------------------------

The legacy ``--gds-l1-path``, ``--gds-l1-backend``, and
``--gds-l1-use-direct-io`` / ``--no-gds-l1-use-direct-io`` flags have been
removed. Use ``"type":"GDS"`` with ``path``, ``backend``, and ``direct_io`` in
JSON. The GDS manager's ``size_gb`` replaces its previous ``--l1-size-gb``.

For a GDS-only server:

.. code-block:: bash

   lmcache server \
       --eviction-policy LRU \
       --l1-manager '{"type":"GDS","tag":"nvme","size_gb":100,"path":"/mnt/nvme","backend":"auto","direct_io":true}'

To configure DRAM and GDS together:

.. code-block:: bash

   lmcache server \
       --eviction-policy LRU \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":20}' \
       --l1-manager '{"type":"GDS","tag":"nvme","size_gb":100,"path":"/mnt/nvme"}'

The second command still writes to DRAM by default; adding GDS does not
automatically populate it. Only one GDS manager is supported per process.
GDS requires its platform-specific libraries and storage setup, described
under :ref:`mp/configuration:GDS L1 Tier`. Byte-array L2 adapters such as
``fs`` must use a DRAM affinity L1, not the GDS slab.

Inspect the configuration
-------------------------

With the :doc:`HTTP server <http_api>` enabled, ``GET /config`` lists
``l1_manager_configs`` and each L2 adapter's ``affinity_tag``. The storage
manager section of ``GET /status`` contains ``l1_managers``,
``l1_eviction_controllers``, and ``store_controllers``, each keyed by L1
tag. The old singular status fields are retained only for single-L1 servers.

The ``lmcache_mp.l1_memory_usage_bytes``, ``lmcache_mp.l1_usage_ratio``,
and ``lmcache_mp.l1_staging_bytes`` gauges carry an ``l1_tag`` attribute.
Coordinator capacity reporting sums L1 capacity by backing medium, while
usage telemetry reports total L1 occupancy across all managers.

Current limits
--------------

Multiple L1 managers disable single-region SHM transfer advertising;
the engine-driven transport falls back to its non-SHM path. P2P requires a
single registerable L1 region and rejects multi-L1 configurations.

Device-DAX remains an option on a DRAM configuration, not a separate JSON
type. A remote shared CXL manager and its RPC service are not implemented.
