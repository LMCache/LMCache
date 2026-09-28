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
- ``checksum``: ``"none"`` (default) or ``"crc32"`` -- verify every load
  against a CRC32 written at store time. See :ref:`fs-l2-checksum`.

**Configuration examples:**

.. code-block:: bash

    # Basic FS adapter
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2"}'

    # With temp directory
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "relative_tmp_dir": ".tmp"}'

    # With O_DIRECT for bypassing page cache
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "use_odirect": true}'

    # With load-time integrity checks
    --l2-adapter '{"type": "fs", "base_path": "/data/lmcache/l2", "checksum": "crc32"}'

.. _fs-l2-checksum:

Integrity checks
----------------

Without a checksum the adapter only checks a file's length, so a file with
the right size but wrong bytes (a torn concurrent write, bit rot, or a file
left by a different writer) is served to the engine as a cache hit and
silently corrupts the model output.

With ``"checksum": "crc32"`` each file ends with an 8-byte trailer (a
4-byte format magic plus ``zlib.crc32`` of the payload). On load the payload
is re-hashed and compared. A mismatch, a missing trailer, or an unknown magic
is served as a **miss** (the engine recomputes the chunk). Loads never modify
files: a concurrent store may already have replaced the file that was read, so
deleting it could remove a valid object. Instead the adapter remembers the key,
and the next store of that key rewrites the file with an atomic rename rather
than skipping it as already stored.

- **Space:** 8 bytes per object. With ``use_odirect`` the trailer is padded to
  one file-system block (typically 4 KiB) to keep writes aligned.
- **CPU:** one CRC32 pass per store and per load, run off the adapter's event
  loop. Throughput depends on the system zlib; zlib-ng hashes at tens of
  GB/s per core, the classic zlib is several times slower.
- **Compatibility:** files written with ``"none"`` have no trailer, so after
  enabling ``"crc32"`` on an existing directory each of them misses once and is
  rewritten by its next store. Files
  written with ``"crc32"`` are still readable with ``"none"`` (the trailer is
  ignored). All writers sharing a directory should use the same setting.
