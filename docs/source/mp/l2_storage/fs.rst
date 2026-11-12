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
- ``max_capacity_gb``: declared capacity in GiB (default ``0`` = unbounded).
  Used for usage accounting; disk usage is bounded only when the adapter spec
  also carries an ``"eviction"`` block (see :doc:`index`). A capacity without
  an ``"eviction"`` block logs a warning at startup.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # Bounded to 3840 GiB with LRU eviction
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "max_capacity_gb": 3840, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.95, "eviction_ratio": 0.01}}'
