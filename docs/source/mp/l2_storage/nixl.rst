NIXL
====

NIXL-based persistent storage — the primary production L2 backend, using NIXL
(NVIDIA Interconnect Library) for high-performance storage I/O. Two adapter
types share this backend:

- ``nixl_store`` — a fixed pool of storage descriptors pre-allocated at init.
- ``nixl_store_dynamic`` — opens and registers storage descriptors per
  operation. File backends add persist/recover across restarts and remove the
  open-file-descriptor limit; object backends use backend-managed retention.

.. note::

   Both adapters require the NIXL runtime, which ships as the optional
   ``lmcache[nixl]`` extra:

   .. code-block:: bash

       uv pip install lmcache[nixl]

Static pool — ``nixl_store``
----------------------------

The primary production adapter. Pre-allocates a pool of storage descriptors at
initialization.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``, ``OBJ``, ``AZURE_BLOB``.
- ``pool_size``: Number of storage descriptors to pre-allocate (must be > 0).

**Backend-specific parameters (``backend_params``):**

File-based backends (``GDS``, ``GDS_MT``, ``POSIX``, ``HF3FS``) require:

- ``file_path``: Directory path for storing L2 data.
- ``use_direct_io``: ``"true"`` or ``"false"`` -- whether to use direct I/O.

The ``OBJ`` and ``AZURE_BLOB`` backends (object stores) do not require ``file_path``.

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
   * - ``AZURE_BLOB``
     - Object store backend for Azure Blob Storage.  No local file path required.

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

    # AZURE_BLOB backend
    --l2-adapter '{"type": "nixl_store", "backend": "AZURE_BLOB", "backend_params": {"account_url": "https://<account_name>.blob.core.windows.net", "container_name": "<container_name>"}, "pool_size": 32}'

Dynamic (persist / recover) — ``nixl_store_dynamic``
----------------------------------------------------

A dynamic variant of the NIXL adapter that opens and registers storage
descriptors per-operation instead of pre-allocating them at init. File
backends enable:

- **Persist/recover** -- cached KV metadata survives restarts.
- **No fd limits** -- files are opened and closed per transfer, so the
  cache can grow beyond OS open-file-descriptor limits.

**Required fields:**

- ``backend``: Storage backend -- one of ``POSIX``, ``GDS``, ``GDS_MT``,
  ``HF3FS``, ``OBJ``, ``AZURE_BLOB``.

**Backend-specific parameters (``backend_params``):**

- File backends (``POSIX``, ``GDS``, ``GDS_MT``, ``HF3FS``) require:

  - ``file_path``: Directory path for storing L2 data files.
  - ``use_direct_io``: ``"true"`` or ``"false"``.
  - ``max_capacity_gb``: Maximum storage capacity in GB. The adapter
    rejects stores when this limit is reached. Required for the eviction
    controller to compute usage.
  - ``shard_dirs``: ``"true"`` or ``"false"`` (default ``"false"``). When
    ``"true"``, data files are spread across a two-level subdirectory tree
    under ``file_path`` instead of all living in one flat directory. The two
    levels are the first four hex characters of the chunk hash, matching the
    hash prefix already embedded in the filename
    (for example ``834ebc79...`` is stored as ``83/4e/<filename>``), giving a
    fanout of up to 256 × 256 subdirectories. Leaving it unset preserves the
    original flat layout.

  Large flat directories slow down metadata operations on many filesystems,
  so sharding helps once a single cache directory holds a large number of
  files. Subdirectories are created lazily on first use and cached in
  memory, so the store hot path issues at most one ``makedirs`` per bucket.

  .. note::

     ``shard_dirs`` changes where files are located, and lookup does not
     fall back to the other layout. Toggling it against an existing cache
     directory makes previously written files unreachable (they are not
     deleted, just no longer found). Choose the layout when the cache
     directory is first created, or clear it when changing the setting.

- Object backends (``OBJ``, ``AZURE_BLOB``) receive their backend-native
  NIXL parameters. They do not require file parameters or
  ``max_capacity_gb``. Object lifetime and deletion are managed by the
  backend, so dynamic object adapters do not support global eviction.

**Optional fields (for persist):**

- ``persist_enabled`` (bool, default ``true``): If ``true``, data files
  are kept on disk at shutdown. If ``false``, all data files are deleted
  on shutdown. This setting is ignored for object backends.

Lookup always checks secondary storage on miss. File backends lazily populate
the in-memory index from a matching file; object backends use a NIXL object
presence query and recover the key with an unknown (zero) size.

**Configuration examples:**

.. code-block:: bash

    # Basic dynamic POSIX backend (persist enabled by default)
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}}'

    # Explicitly disable persist
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10"}, "persist_enabled": false}'

    # With eviction
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "GDS", "backend_params": {"file_path": "/data/nvme/l2", "use_direct_io": "true", "max_capacity_gb": "50"}, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

    # Shard data files across a two-level subdirectory tree
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "POSIX", "backend_params": {"file_path": "/data/lmcache/l2", "use_direct_io": "false", "max_capacity_gb": "10", "shard_dirs": "true"}}'

    # OBJ backend (object retention is backend-managed)
    --l2-adapter '{"type": "nixl_store_dynamic", "backend": "OBJ", "backend_params": {"bucket": "<bucket_name>"}}'

**Persist / secondary lookup behaviour:**

- On **shutdown**, the adapter keeps data files on disk by default
  (``persist_enabled`` defaults to ``true``). If explicitly set to
  ``false``, all data files are deleted to avoid orphaned storage.
- On **startup**, the in-memory index is empty. Every lookup miss falls
  through to a secondary lookup on disk: if the deterministic file
  exists, it is treated as a hit and the in-memory index is populated
  lazily from the file size.
- For object backends, the same lookup derives a deterministic object key and
  uses NIXL's presence query. Object size is backend-specific, so recovered
  objects are tracked with size zero and are not globally evicted.
