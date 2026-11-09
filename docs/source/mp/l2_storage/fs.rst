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
- ``streaming_buffer_size_bytes``: Size of a separate pinned-memory pool for
  MP streaming loads, in bytes. The size must be a positive multiple of 4096;
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

When all requested objects are present on FS and none is resident in L1, MP
lookup selects streaming. Retrieve loads bounded batches of objects into the
shared pool, submits each batch to the device, and returns its windows after the
device transfer completes. Requests with
L1 hits or incomplete FS coverage use the existing prefetch path. The pool is
additional to the configured L1 capacity. The streaming adapter is configured at
server startup and cannot be removed while the server is running.

Streaming lookup and load operations have a 30-second timeout. A timed-out
lookup is reported as a miss. Windows belonging to an unfinished filesystem load
remain reserved until that load completes, so they cannot be reused while a disk
read is writing to them. Closing the reader waits for pending I/O; it reports an
error instead of freeing memory that is still in use.

FS objects can disappear between lookup and load. In this case the connector
reports the affected GPU blocks as failed. vLLM's ``kv_load_failure_policy``
then determines whether the request fails or the missing KV is recomputed;
set it to ``recompute`` in ``kv-transfer-config`` to enable recomputation.
