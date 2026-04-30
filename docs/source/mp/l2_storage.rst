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

``nixl_store_dynamic`` -- NIXL-based dynamic storage with persist/recover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dynamic variant of the NIXL adapter that opens and registers files
per-operation instead of pre-allocating them at init. This enables:

- **Persist/recover** -- cached KV metadata survives restarts.
- **No fd limits** -- files are opened and closed per transfer, so the
  cache can grow beyond OS open-file-descriptor limits.

.. note::

   Only file-based backends are supported (``POSIX``, ``GDS``, ``GDS_MT``,
   ``HF3FS``). The ``OBJ`` backend is not supported yet.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``.

**Backend-specific parameters (``backend_params``):**

- ``file_path``: Directory path for storing L2 data files.
- ``use_direct_io``: ``"true"`` or ``"false"``.
- ``max_capacity_gb``: Maximum storage capacity in GB. The adapter
  rejects stores when this limit is reached. Required for the eviction
  controller to compute usage.

**Optional fields (for persist):**

- ``persist_enabled`` (bool, default ``true``): If ``true``, data files
  are kept on disk at shutdown. If ``false``, all data files are deleted
  on shutdown.

Lookup always checks secondary storage (disk) on miss and lazily
populates the in-memory index when a file is found.

**Configuration examples:**

.. code-block:: bash

    # Basic dynamic POSIX backend (persist enabled by default)
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}}'

    # Explicitly disable persist
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}, "persist_enabled": false}'

    # With eviction
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true", "max_capacity_gb": "50"}, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

**Persist / secondary lookup behaviour:**

- On **shutdown**, the adapter keeps data files on disk by default
  (``persist_enabled`` defaults to ``true``). If explicitly set to
  ``false``, all data files are deleted to avoid orphaned storage.
- On **startup**, the in-memory index is empty. Every lookup miss falls
  through to a secondary lookup on disk: if the deterministic file
  exists, it is treated as a hit and the in-memory index is populated
  lazily from the file size.

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

``mooncake_store`` -- Mooncake Store native connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An L2 adapter backed by the native C++ Mooncake Store connector.  Uses
`Mooncake <https://github.com/kvcache-ai/Mooncake>`_ for high-performance
distributed KV cache storage with RDMA support.

When Mooncake is configured with ``"protocol": "rdma"``, LMCache must also
have a valid contiguous L1 memory region available.  The distributed storage
manager passes this L1 memory descriptor to the adapter factory automatically
in MP mode.  If the descriptor is missing or invalid, adapter creation fails
with ``ValueError`` instead of silently falling back to a non-RDMA path.

**Prerequisites -- Building with Mooncake support:**

The Mooncake extension is **not** built by default.  You must explicitly
enable it:

.. code-block:: bash

    BUILD_MOONCAKE=1 pip install -e . --verbose

The ``BUILD_MOONCAKE`` environment variable controls compilation:

- ``BUILD_MOONCAKE=1``: Enable the Mooncake C++ extension.
- ``BUILD_MOONCAKE=0``: Force disable (highest priority), even if
  ``MOONCAKE_INCLUDE_DIR`` is set.
- **Not set**: Falls back to checking ``MOONCAKE_INCLUDE_DIR`` for
  backward compatibility.  If ``MOONCAKE_INCLUDE_DIR`` is also unset,
  the extension is skipped.

If the Mooncake headers are not installed in the system include path
(e.g., ``/usr/local/include``), you must point to them explicitly:

.. code-block:: bash

    BUILD_MOONCAKE=1 \
    MOONCAKE_INCLUDE_DIR=/path/to/mooncake/include \
    MOONCAKE_LIB_DIR=/path/to/mooncake/lib \
    pip install -e . --verbose

**LMCache-specific fields:**

- ``num_workers``: Number of C++ worker threads (default ``4``, must
  be > 0).

**Mooncake fields:**

All other keys in the JSON config (except ``type``, ``num_workers``,
and ``eviction``) are forwarded **as-is** to Mooncake's
``setup_internal(ConfigDict)``.  Refer to the
`Mooncake documentation <https://github.com/kvcache-ai/Mooncake>`_
for available setup keys (e.g., ``local_hostname``,
``metadata_server``, ``master_server_address``, ``protocol``,
``device_name``, ``global_segment_size``).

**Configuration example:**

.. code-block:: bash

    --l2-adapter '{
      "type": "mooncake_store",
      "num_workers": 4,
      "local_hostname": "node01",
      "metadata_server": "http://localhost:8080/metadata",
      "master_server_address": "localhost:50051",
      "protocol": "tcp",
      "local_buffer_size": "3221225472"
      "global_segment_size": "3221225472"
    }'

For full Mooncake setup instructions (master service, metadata server,
etc.), see `Mooncake <https://github.com/kvcache-ai/Mooncake>`_ .

**RDMA notes:**

- ``protocol: "rdma"`` requires a valid LMCache L1 memory descriptor.
- When using ``protocol: "rdma"``, it is recommended to disable lazy L1
  allocation with ``--no-l1-use-lazy`` so the L1 buffer is fully allocated
  before Mooncake registers it.
- ``protocol: "tcp"`` does not require L1 preregistration.
- If Mooncake RDMA initialization fails at adapter creation time, verify that
  LMCache L1 memory is enabled and that the descriptor has a non-zero pointer
  and size.

``s3`` -- S3-compatible object store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An L2 adapter that stores KV cache objects as S3 objects using the AWS
Common Runtime (CRT).  Works with AWS S3, S3 Express One Zone, and any
S3-compatible endpoint (MinIO, Ceph RGW, etc.).

**Required fields:**

- ``s3_endpoint``: Bucket URL -- either ``"s3://<bucket>"`` or the bare host form
  (used for non-AWS endpoints).
- ``s3_region``: AWS region string (e.g. ``"us-west-2"``).

**Optional fields:**

- ``s3_num_io_threads`` (int, default ``64``): Number of CRT I/O threads.
- ``s3_prefer_http2`` (bool, default ``true``): Negotiate HTTP/2 via ALPN.
- ``s3_enable_s3express`` (bool, default ``false``): Enable S3 Express signing
  for S3 Express One Zone buckets.
- ``disable_tls`` (bool, default ``false``): Bypass TLS when pointing at a
  plain-HTTP endpoint (e.g. a local MinIO).
- ``aws_access_key_id`` / ``aws_secret_access_key`` (string): Static
  credentials; omit both to use the AWS default credential provider chain
  (environment, EC2 instance profile, etc.).
- ``max_capacity_gb`` (float, default ``0.0``): Aggregate capacity used by
  ``get_usage()``.  A value of ``0`` disables aggregate eviction
  (``usage_fraction == -1.0``).

**Configuration examples:**

.. code-block:: bash

    # AWS S3 with default credentials
    --l2-adapter '{"type": "s3", "s3_endpoint": "s3://my-bucket", "s3_region": "us-west-2"}'

    # Static credentials, HTTP/2 disabled
    --l2-adapter '{"type": "s3", "s3_endpoint": "s3://my-bucket", "s3_region": "us-west-2", "s3_prefer_http2": false, "aws_access_key_id": "AKIA...", "aws_secret_access_key": "..."}'

    # Local MinIO over plain HTTP
    --l2-adapter '{"type": "s3", "s3_endpoint": "minio.local:9000", "s3_region": "us-east-1", "disable_tls": true, "aws_access_key_id": "minio", "aws_secret_access_key": "minio123"}'

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
       Prefetched keys are **temporary** (deleted after the reader finishes).
   * - ``--l2-prefetch-policy``
     - ``retain``
     - Same load plan as ``default``, but prefetched keys are **retained**
       permanently in L1.  Useful when prefetched data is likely reused
       by subsequent requests (e.g. shared system-prompt chunks).

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
   * - ``nixl_store_dynamic``
     - Full support. ``delete`` removes data files from disk; pinned
       keys are skipped. ``get_usage`` is byte-based
       (``_total_bytes / max_capacity_bytes``).
   * - ``mock``
     - Full support. Useful for testing eviction behaviour without
       real storage hardware.
   * - ``s3``
     - ``delete`` removes objects from the bucket and frees aggregate
       byte accounting. ``get_usage`` reports ``usage_fraction == -1.0``
       when ``max_capacity_gb`` is ``0`` (disabled); set a non-zero
       ``max_capacity_gb`` to enable the watermark-triggered eviction
       controller.
   * - ``mooncake_store``
     - No eviction support (native connector adapter).
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
