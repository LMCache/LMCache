FS (native)
===========

A file-system L2 adapter backed by the native C++ ``LMCacheFSClient``
wrapped with ``NativeConnectorL2Adapter``.  I/O is dispatched through a
C++ worker-thread pool with eventfd-driven completions, giving a true
I/O queue depth on a single Python thread.

**Required fields:**

- ``base_path``: Directory for storing KV cache files.

**Optional fields:**

- ``num_workers`` (int, default ``4``, > 0): Number of C++ worker threads
  inside the connector.  Each one orchestrates whole batches: opening
  files, attributing results and completing futures.  With
  ``read_io_depth`` left at ``0`` this is also the read queue depth
  against the device, because each worker reads one object at a time and
  blocks; see ``read_io_depth`` for why that is usually too shallow.
- ``relative_tmp_dir`` (str, default ``""``): Relative sub-directory for
  temporary files during writes (atomic rename on completion).
- ``use_odirect`` (bool, default ``false``): Bypass the page cache via
  ``O_DIRECT``.  Required to measure real disk bandwidth.  See alignment
  caveat below.
- ``read_ahead_size`` (int, optional): Trigger filesystem readahead by
  issuing a warm-up read of this many bytes at open time.  This is skipped
  for reads that use ``O_DIRECT`` because direct I/O bypasses the page cache.
- ``max_capacity_gb`` (float, default ``0``): Maximum L2 capacity in GB
  for client-side usage tracking.  Default ``0`` disables tracking.
- ``read_io_depth`` (int, default ``0``): Number of threads dedicated to
  executing reads, and so the maximum reads in flight.  ``0`` keeps the
  legacy path, where reads run on the worker threads themselves and the
  depth against the device therefore equals ``num_workers``, which on an
  array of several devices is far below what its read bandwidth needs.
  Each reader thread holds one connection for the connector's lifetime
  and one open file at a time, so this is the ceiling on both.  Bytes in
  flight can never exceed ``read_io_depth`` x object size however large
  ``read_max_bytes_in_flight`` is: size it so the byte budget is the
  constraint that binds.
- ``read_max_bytes_in_flight`` (int, default ``0``): When
  ``read_io_depth`` is positive, the bytes this connector may keep
  outstanding against the device, shared across its workers.  Throughput
  is set by bytes in flight rather than by object count -- and object
  size here is ``chunk_size`` x bytes-per-token-per-rank, so a depth
  expressed in objects means something different at every ``chunk_size``.
  The right figure is a property of the storage, and the default ``0``
  selects **1536 MiB**, chosen for its worst case rather than its best:
  a smaller budget collapses on network-latency storage (at 32 ms per
  read, 768 MiB delivers 58% of what 1536 MiB does), while an oversized
  one costs at most 11% anywhere measured.  A deployment that knows its
  storage can do better by setting its own value; a single slow device
  is the case that gives up the most at the default, reading 89% of what
  it would at 96 MiB.  Only reachable if ``read_io_depth`` is large
  enough; see above.

.. important::

   ``O_DIRECT`` has two independent alignment requirements:

   1. **Length alignment.**  The transfer length must be a multiple of
      the filesystem's block size.  The connector queries the disk block
      size at construction time and, on each operation, checks
      ``len % disk_block_size``.  If the length is **not** a multiple,
      the connector silently falls back to a buffered open (no
      ``O_DIRECT``) for that operation -- correctness is preserved but
      you do not get true direct I/O.  To ensure ``O_DIRECT`` is
      actually used, choose ``--chunk-size`` so that the resulting
      per-chunk byte size is a multiple of the FS block size.  GPFS and
      similar parallel filesystems often use large blocks (e.g. several
      MiB).

   2. **Memory-buffer alignment.**  The I/O buffer pointer itself must
      also be aligned (typically to 4096 bytes on local disks, or to the
      FS block size on parallel filesystems).  This is controlled by
      ``--l1-align-bytes`` (default ``4096``) -- raise it to match the
      FS block size when running on a filesystem with larger blocks.  If
      the buffer is misaligned, the connector reports a runtime error instead
      of silently falling back to buffered I/O.  This protects real-disk
      benchmark runs from accidentally measuring the page cache.

   If unsure, start with ``use_odirect: false`` and confirm correctness
   before enabling ``O_DIRECT``.

**Configuration examples:**

.. code-block:: bash

    # Basic native FS adapter
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2"}'

    # Many worker threads for a parallel filesystem (e.g. GPFS, Lustre)
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32}'

    # O_DIRECT for real-disk benchmarking
    --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'

**Buffer-only mode example.**  L1 acts as a pure write buffer that
absorbs the peak burst of in-flight chunks while the C++ worker pool
drains them to disk; nothing is retained in L1 once a store completes:

.. code-block:: bash

    lmcache server \
        --host 0.0.0.0 --port 5555 \
        --max-workers 32 \
        --l1-size-gb 32 --l1-use-lazy \
        --eviction-policy noop \
        --l2-store-policy skip_l1 \
        --l2-adapter '{"type": "fs_native", "base_path": "/data/lmcache/l2", "num_workers": 32, "use_odirect": true}'
