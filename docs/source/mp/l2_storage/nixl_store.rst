``nixl_store``
==============

The primary production adapter.  Uses NIXL (NVIDIA Interconnect Library) for
high-performance storage I/O.

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
