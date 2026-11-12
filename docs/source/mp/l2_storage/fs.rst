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

MP lookup combines L1 hits with FS hits to find the longest available prefix.
Retrieve copies resident L1 objects and reads FS objects through bounded batches
of shared windows, returning them after device transfer completes. The engine
recomputes the remaining tail. For example, ``--l1-size-gb 2`` with a 128 MiB
streaming pool leaves 1.875 GiB for resident L1 objects. The streaming adapter is configured at
server startup and cannot be removed while the server is running.
When multiple L2 adapters are active, lookup uses ordinary prefetch across
them; the configured streaming pool remains reserved from the DRAM budget.

Streaming lookup and load operations have a 30-second timeout. A timed-out
lookup is reported as a miss. Windows belonging to an unfinished filesystem load
remain reserved until that load completes, so they cannot be reused while a disk
read is writing to them. Closing the reader waits for pending I/O and active
device transfers up to its deadline; it reports an error if memory is still in use.

FS objects can disappear between lookup and load. In this case the connector
reports the affected GPU blocks as failed. vLLM's ``kv_load_failure_policy``
then determines whether the request fails or the missing KV is recomputed;
set it to ``recompute`` in ``kv-transfer-config`` to enable recomputation.
