``nixl_store_dynamic``
======================

A dynamic variant of the NIXL adapter that opens and registers files
per-operation instead of pre-allocating them at init. This enables:

- **Persist/recover** -- cached KV metadata survives restarts.
- **No fd limits** -- files are opened and closed per transfer, so the
  cache can grow beyond OS open-file-descriptor limits.

.. note::

   Only file-based backends are supported (``POSIX``, ``GDS``, ``GDS_MT``,
   ``HF3FS``). The ``OBJ`` and ``AZURE_BLOB`` backends are not supported yet.

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
