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
- ``streaming_buffer_size_bytes``: Size reserved from the configured DRAM L1
  budget for MP streaming loads, in bytes. The size must be a positive multiple of 4096;
  the default of ``0`` disables streaming. The pool is shared by requests and
  must fit at least one object for every object-group layout in a request.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # Read cold FS objects through a 128 MiB pool
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true, "streaming_buffer_size_bytes": 134217728}'

Reserved FS streaming windows are available in the default MP engine with
``--supported-transfer-mode lmcache_driven``. The server uses ordinary prefetch
and the full resident L1 budget in ``engine_driven``, ``auto``, and blend mode.

The server carves the reservation out of its single DRAM L1 manager at
startup. For example, ``--l1-size-gb 2`` with a 128 MiB reservation leaves
1.875 GiB for resident objects. The configured streaming adapter stays attached
until shutdown. When this unwrapped ``fs`` adapter is the sole active L2
adapter and every object layout fits, lookup combines L1 and FS hits to find
the longest available prefix. Retrieve reads FS objects through bounded
windows and returns them after device transfer. The engine computes the
remaining tail. With multiple active L2 adapters or an oversized object
layout, lookup uses ordinary prefetch and the DRAM reservation stays in place.

Lookup uses the standard MP prefetch status API. Waiting for the first free
window and waiting for a batch of FS reads each use a 30-second timeout.
Windows belonging to unfinished FS reads stay reserved until that I/O
completes. Closing the reader waits for pending I/O and active device
transfers up to its deadline and reports an error if memory is still in use.

FS objects can disappear between lookup and load. In this case the connector
reports the affected GPU blocks as failed. vLLM's ``kv_load_failure_policy``
then determines whether the request fails or the missing KV is recomputed;
set it to ``recompute`` in ``kv-transfer-config`` to enable recomputation.
