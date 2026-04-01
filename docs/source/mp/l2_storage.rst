L2 Storage (Persistent Cache)
=============================

LMCache multiprocess mode supports a two-tier storage architecture:

- **L1 (in-memory)** -- Fast CPU memory managed by the L1 Manager.  All KV
  cache chunks live here during active use.
- **L2 (persistent)** -- Durable storage backends (NIXL-based or plain
  file-system).  The StoreController asynchronously pushes data from L1
  to L2, and the PrefetchController loads data from L2 back into L1 on
  cache misses.

.. contents::
   :local:
   :depth: 2

Data Flow
---------

**Write path (L1 -> L2):**

1. vLLM stores KV cache chunks into L1 via the ``STORE`` RPC.
2. The ``StoreController`` detects new objects (via eventfd) and
   asynchronously submits store tasks to each configured L2 adapter.
3. The L2 adapter writes the data to its backend (e.g., local SSD via GDS).

**Read path (L2 -> L1):**

1. A ``LOOKUP`` RPC checks L1 for prefix hits.
2. For keys not found in L1, the ``PrefetchController`` submits lookup
   requests to L2 adapters.
3. If found in L2, the data is loaded back into L1 and read-locked for the
   pending ``RETRIEVE`` RPC.

Adapter Types
-------------

``nixl_store`` -- NIXL-based persistent storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary production adapter.  Uses NIXL (NVIDIA Interconnect Library) for
high-performance storage I/O.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``, ``OBJ``.
- ``pool_size``: Number of storage descriptors to pre-allocate (must be > 0).

**Backend-specific parameters (``backend_params``):**

File-based backends (``GDS``, ``GDS_MT``, ``POSIX``, ``HF3FS``) require:

- ``file_path``: Directory path for storing L2 data.
- ``use_direct_io``: ``"true"`` or ``"false"`` -- whether to use direct I/O.

The ``OBJ`` backend (object store) does not require ``file_path``.

**Backend descriptions:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Backend
     - Description
   * - ``POSIX``
     - Standard POSIX file I/O.  Works on any file system.  No direct I/O.
   * - ``GDS``
     - NVIDIA GPU Direct Storage.  Enables direct GPU-to-storage transfers
       bypassing the CPU.  Requires NVMe SSDs with GDS support.
   * - ``GDS_MT``
     - Multi-threaded variant of GDS for higher throughput.
   * - ``HF3FS``
     - Shared file system backend (e.g., for distributed/networked storage).
   * - ``OBJ``
     - Object store backend.  No local file path required.

**Configuration examples:**

.. code-block:: bash

    # POSIX backend
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

    # GDS backend
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # GDS_MT backend
    --l2-adapter '{"type": "nixl_store", "backend": "GDS_MT", "backend_params": {"file_path": "/data/nvme/lmcache", "use_direct_io": "true"}, "pool_size": 128}'

    # HF3FS backend
    --l2-adapter '{"type": "nixl_store", "backend": "HF3FS", "backend_params": {"file_path": "/mnt/hf3fs/lmcache", "use_direct_io": "false"}, "pool_size": 64}'

    # OBJ backend
    --l2-adapter '{"type": "nixl_store", "backend": "OBJ", "backend_params": {}, "pool_size": 32}'

``fs`` -- File-system backed storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pure file-system L2 adapter using async I/O (``aiofiles``).  Each KV cache
object is stored as a raw ``.data`` file whose name encodes the full
``ObjectKey``.  Does **not** require NIXL -- works on any POSIX file system.

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``relative_tmp_dir``: Relative sub-directory for temporary files during
  writes (atomic rename on completion).
- ``read_ahead_size``: Trigger file-system read-ahead by reading this many
  bytes first (positive integer, optional).
- ``use_odirect``: ``true`` or ``false`` (default ``false``) -- bypass the
  page cache via ``O_DIRECT``.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

``mock`` -- Mock adapter for testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates L2 storage with configurable size and bandwidth.  Useful for testing
the L2 pipeline without real storage hardware.

**Fields:**

- ``max_size_gb``: Maximum size in GB (> 0).
- ``mock_bandwidth_gb``: Simulated bandwidth in GB/sec (> 0).

.. code-block:: bash

    --l2-adapter '{"type": "mock", "max_size_gb": 256, "mock_bandwidth_gb": 10}'

Multiple Adapters (Cascade)
---------------------------

You can configure multiple L2 adapters by repeating the ``--l2-adapter``
argument.  Adapters are used in the order they are specified.  The
``StoreController`` pushes data to all configured adapters, and the
``PrefetchController`` queries adapters in order during lookups.

.. code-block:: bash

    # SSD (fast, smaller) + NVMe GDS (larger capacity)
    --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/ssd/l2", "use_direct_io": "false"}, "pool_size": 64}' \
    --l2-adapter '{"type": "nixl_store", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"}, "pool_size": 128}'

Store and Prefetch Policies
----------------------------

The **store policy** controls how keys flow from L1 to L2: which adapters
receive each key and whether keys are deleted from L1 after a successful
L2 store.  The **prefetch policy** controls how keys flow from L2 back to
L1: when multiple adapters have the same key, the policy decides which
adapter loads it.

Select policies via CLI:

.. code-block:: bash

    --l2-store-policy default \
    --l2-prefetch-policy default

**Built-in policies:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Flag
     - Name
     - Behaviour
   * - ``--l2-store-policy``
     - ``default``
     - Store all keys to all adapters.  Never delete from L1.
   * - ``--l2-store-policy``
     - ``skip_l1``
     - Buffer-only mode.  Store all keys to all adapters, then
       **delete them from L1** immediately.  Pair with
       ``--eviction-policy noop`` to avoid useless LRU overhead.
   * - ``--l2-prefetch-policy``
     - ``default``
     - For each key, pick the first (lowest-indexed) adapter that has it.

Prefetch Concurrency
~~~~~~~~~~~~~~~~~~~~~

The ``--l2-prefetch-max-in-flight`` flag limits the number of concurrent
prefetch requests that the ``PrefetchController`` can have in flight at
any time.  A higher value increases L2-to-L1 throughput but also
increases L1 memory pressure from in-flight data.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-prefetch-max-in-flight``
     - ``8``
     - Maximum number of concurrent prefetch requests.

Buffer-Only Mode
~~~~~~~~~~~~~~~~~

When L1 is used purely as a write buffer (all data lives in L2), use
``--l2-store-policy skip_l1`` together with ``--eviction-policy noop``.
This combination deletes keys from L1 as soon as they are stored to L2
and disables the LRU eviction tracker entirely, reducing memory and CPU
overhead.

.. code-block:: bash

    --eviction-policy noop \
    --l2-store-policy skip_l1 \
    --l2-prefetch-policy default

Policies are extensible -- new policies can be added by creating a file
in ``storage_controllers/`` and calling ``register_store_policy()`` or
``register_prefetch_policy()`` at import time.  See the design doc
``l2_adapters/design_docs/overall.md`` for details.

Eviction
--------

LMCache supports eviction at both storage tiers so that each tier
can operate within a fixed capacity budget.

L1 Eviction
~~~~~~~~~~~

L1 eviction runs a single background thread that monitors overall L1
memory usage. When usage exceeds ``trigger_watermark``, the eviction
policy evicts a fraction of the least-recently-used keys.

**CLI flags:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--eviction-policy``
     - *(required)*
     - Policy name: ``LRU`` or ``noop``.
   * - ``--eviction-trigger-watermark``
     - ``0.8``
     - L1 usage fraction [0, 1] above which eviction is triggered.
   * - ``--eviction-ratio``
     - ``0.2``
     - Fraction of currently allocated L1 memory to evict per cycle.

**Example:**

.. code-block:: bash

    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2

L2 Eviction
~~~~~~~~~~~

L2 eviction is **per-adapter** and **opt-in**. Each adapter can
independently declare an eviction policy by adding an ``"eviction"``
sub-object to its ``--l2-adapter`` JSON spec. Adapters without an
``"eviction"`` key have no eviction controller.

When L2 eviction is enabled for an adapter, a dedicated background
thread monitors that adapter's ``get_usage()`` value. Once usage
exceeds ``trigger_watermark``, the policy evicts keys until usage
drops by ``eviction_ratio``.

**``"eviction"`` sub-object fields:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``eviction_policy``
     - *(required)*
     - Policy name: ``"LRU"`` or ``"noop"``.
   * - ``trigger_watermark``
     - ``0.8``
     - Adapter usage fraction [0, 1] above which eviction is triggered.
   * - ``eviction_ratio``
     - ``0.2``
     - Fraction of used capacity to evict per cycle.

**Example — nixl_store with LRU eviction:**

.. code-block:: bash

    --l2-adapter '{
      "type": "nixl_store",
      "backend": "POSIX",
      "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"},
      "pool_size": 128,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.8,
        "eviction_ratio": 0.2
      }
    }'

**Adapter support:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Adapter
     - L2 Eviction Support
   * - ``nixl_store``
     - Full support. ``delete`` frees pool slots; pinned keys (in-flight
       loads) are skipped and retried on the next cycle.
   * - ``mock``
     - Full support. Useful for testing eviction behaviour without
       real storage hardware.
   * - ``fs``
     - No eviction support (``delete`` and ``get_usage`` are no-ops).
   * - native connectors
     - No eviction support.

.. note::

   Each L2 adapter instance gets its own independent eviction
   controller and policy.  Two adapters of the same type can have
   different watermarks or policies.

Combined L1 + L2 Eviction Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    --l1-size-gb 100 \
    --eviction-policy LRU \
    --eviction-trigger-watermark 0.8 \
    --eviction-ratio 0.2 \
    --l2-adapter '{
      "type": "nixl_store",
      "backend": "GDS",
      "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true"},
      "pool_size": 256,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

In this setup:

- L1 evicts from memory when it is 80 % full, reclaiming 20 % of
  allocated memory per cycle.
- L2 (NIXL/GDS) evicts from the storage pool when 90 % of pool slots
  are occupied, reclaiming 10 % per cycle.
- Both tiers use independent LRU policies, so each evicts its own
  least-recently-used keys.

Verifying L2 Storage
--------------------

Set ``LMCACHE_LOG_LEVEL=DEBUG`` to see L2 activity in the server logs:

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server \
        --l1-size-gb 100 --eviction-policy LRU \
        --l2-adapter '{"type": "nixl_store", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false"}, "pool_size": 64}'

Expected log messages when L2 is active:

.. code-block:: text

    LMCache DEBUG: Submitted store task ...
    LMCache DEBUG: L2 store task N completed ...
    LMCache DEBUG: Prefetch request submitted: X total keys, Y L1 prefix hits, Z remaining for L2
