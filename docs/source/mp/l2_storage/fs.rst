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
- ``max_capacity_gb``: Maximum aggregate capacity, in GB, used for
  ``get_usage()`` and status reporting (float, default ``0``).  A value of
  ``0`` leaves aggregate capacity disabled and reports
  ``usage_fraction == -1.0``.

``max_capacity_gb`` is a reporting capacity for the adapter. It does not reserve
filesystem space and it does not enforce a hard write quota. The pure Python
``fs`` adapter still has no delete implementation, so do not rely on this field
alone for filesystem-backed L2 eviction.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # Report usage against a 500 GB filesystem capacity
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "max_capacity_gb": 500}'
