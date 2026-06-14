``dax``
=======

An L2 adapter that maps a single Device-DAX path, such as ``/dev/dax1.0``,
and stores KV cache objects in fixed-size slots.  This adapter is intended for
byte-addressable memory devices such as persistent memory or CXL memory.

The MP ``dax`` adapter is volatile in this release.  It keeps the key index in
server memory and rebuilds an empty index on restart.  Old bytes may remain on
the DAX device, but they are unreachable after the LMCache server restarts.

**Required fields:**

- ``device_path``: Path to the mmap-able DAX device or test file.
- ``max_dax_size_gb``: Number of GiB to map from ``device_path``.
- ``slot_bytes``: Fixed slot size in bytes. This must be large enough for one
  full LMCache chunk because MP memory descriptors do not expose the
  non-MP full-chunk size.

**Optional fields:**

- ``num_store_workers`` (int, default ``1``): Store worker threads.
- ``num_lookup_workers`` (int, default ``1``): Lookup worker threads.
- ``num_load_workers`` (int, default ``min(4, os.cpu_count())``): Load worker
  threads.
- ``persist_enabled`` (bool): Accepted by common L2 config parsing but has no
  effect for ``dax`` because restart recovery is not implemented.

**Configuration example:**

.. code-block:: bash

    --l2-adapter '{
      "type": "dax",
      "device_path": "/dev/dax1.0",
      "max_dax_size_gb": 100,
      "slot_bytes": 268435456,
      "num_store_workers": 1,
      "num_lookup_workers": 1,
      "num_load_workers": 4,
      "eviction": {
        "eviction_policy": "LRU",
        "trigger_watermark": 0.9,
        "eviction_ratio": 0.1
      }
    }'

**Current limits:**

- Uses one server-owned mapped DAX path. Per-TP partitions and multi-device
  striping are not implemented.
- Only single-buffer objects are supported. Multi-tensor objects are rejected.
- Capacity is slot-based, not payload-byte-based. L2 eviction and usage
  metrics count occupied slots.
- Lookups acquire DAX-side external locks. ``submit_unlock`` releases those
  locks after load/retrieve completes, making entries evictable again.
