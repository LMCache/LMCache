FileSystem
==========

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
- ``max_capacity_gb`` (float, default ``0.0``): Aggregate filesystem capacity
  used by ``get_usage()``. A value of ``0`` keeps FS L2 aggregate eviction
  disabled.
- ``eviction``: Optional per-adapter L2 eviction config. FS L2 capacity
  governance requires both ``max_capacity_gb > 0`` and this sub-object. The
  top-level ``--eviction-policy`` flag controls L1 only.

When capacity governance is enabled, LMCache accounts file sizes and deletes
whole ``.data`` files selected by the adapter's LRU state. Existing legacy
filenames that do not include ``object_group_id`` remain readable as
``object_group_id=0`` entries, so old cache directories do not need to be
renamed before enabling the limit.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # With FS L2 capacity governance
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "max_capacity_gb": 200, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.8, "eviction_ratio": 0.2}}'
