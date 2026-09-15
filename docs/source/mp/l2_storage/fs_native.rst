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
  inside the connector.  This is the real I/O queue depth -- raise to
  push throughput on filesystems whose aggregate BW exceeds per-stream
  BW.
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

.. important::

   ``max_capacity_gb`` supplies capacity accounting; it does not delete files
   by itself. Configure an adapter-level ``eviction`` policy to enforce the
   watermark. On restart, ``fs_native`` scans the flat ``.data`` file layout
   produced by the native connector and restores sizes and a best-effort LRU
   order from ``max(atime, mtime)``. Mount options such as ``noatime`` can
   reduce the precision of this recovered order; live accesses are tracked
   normally after startup.

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

Stable multi-disk striping
---------------------------

Use one ``fs_native`` adapter per independently mounted disk and select the
``striped`` store and prefetch policies. Each mount receives a persistent
``.lmcache_disk_uuid`` file. Rendezvous hashing uses that UUID, so adapter
configuration order does not affect placement and adding one disk remaps only
approximately ``1 / (N + 1)`` of keys.

.. code-block:: bash

    lmcache server \
        --host 0.0.0.0 --port 5555 \
        --l1-size-gb 32 --l1-use-lazy \
        --l2-store-policy striped \
        --l2-prefetch-policy striped \
        --l2-adapter '{"type":"fs_native","base_path":"/mnt/nvme0/lmcache","max_capacity_gb":3500,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}' \
        --l2-adapter '{"type":"fs_native","base_path":"/mnt/nvme1/lmcache","max_capacity_gb":3500,"eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.2}}'

All striped adapters must be ``fs_native`` and expose distinct disk UUIDs.
Keep ``.lmcache_disk_uuid`` with its physical mount across restarts; never copy
one disk's UUID file to another disk. After the disk set changes, files on a
former owner are recovered into that disk's normal LRU index and reclaimed
under capacity pressure. Status output reports ``placement_id``,
``recovered_keys``, ``recovered_bytes``, and ``recovery_skipped_files`` for
each adapter.

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
