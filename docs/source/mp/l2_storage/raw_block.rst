Raw Block (Rust)
================

A built-in L2 adapter that stores KV objects in fixed-size slots on a raw block
device or pre-sized file using the Rust raw-device I/O bindings. It reuses the
existing raw-block metadata checkpoint model and writes directly into the
caller-provided load buffers during prefetch.

**Required fields:**

- ``device_path``: Raw device path or pre-sized file path.
- ``slot_bytes``: Fixed slot size in bytes. Must be aligned to ``block_align``.

**Optional fields:**

- ``capacity_bytes``: Optional cap on the usable device bytes. Default ``0``
  means use the full device/file size.
- ``use_odirect``: ``true`` or ``false`` (default ``true``).
- ``block_align``: Device alignment in bytes (default ``4096``). Must be a
  power of two.
- ``header_bytes``: Per-slot header reservation (default ``4096``).
- ``meta_total_bytes``: Reserved metadata checkpoint region (default ``256MiB``).
- ``meta_magic`` / ``meta_version``: Metadata checkpoint identity/version knobs.
- ``meta_checkpoint_interval_sec`` / ``meta_idle_quiet_ms`` /
  ``meta_enable_periodic`` / ``meta_verify_on_load``: Checkpoint and recovery
  controls carried over from the legacy raw-block backend.
- ``load_checkpoint_on_init``: Load an existing on-device metadata checkpoint
  during startup (default ``true``). Set to ``false`` to start with an empty
  in-memory index instead.
- ``enable_zero_copy``: Try aligned direct-buffer I/O when possible.
- ``io_engine``: Rust raw-block I/O engine. Valid values are ``"posix"``
  (default synchronous ``pread``/``pwrite`` path), ``"io_uring"`` (direct Rust
  io_uring syscall path).
- ``use_uring_cmd``: Enable NVMe passthrough via io_uring command interface
  for direct device access. Requires ``io_engine="io_uring"`` and NVMe
  character device node (e.g., ``/dev/ng0n1``).
- ``iouring_queue_depth``: Queue depth for ``io_engine="io_uring"``.
- ``max_data_transfer_size``: Maximum data transfer size for
  ``use_uring_cmd=true``. Large transfers are split into smaller chunks
  that fit within device limits.
- ``fdp_enabled``: Enables NVMe Flexible Data Placement (FDP) discovery
  and non-zero placement identifier registration. The KV data placement
  policy is not active yet. Requires ``io_engine="io_uring"`` and
  ``use_uring_cmd=true``.
- ``fdp_placement_ids``: Optional non-zero placement identifier list for KV
  data placement. If omitted, the adapter uses all device-reported non-zero
  identifiers except ``meta_checkpoint_placement_id``.
- ``meta_checkpoint_placement_id``: Optional non-zero placement identifier for
  metadata checkpoint payload/header writes. Omit it to keep checkpoint writes
  on default NVMe placement.
- ``num_store_workers`` / ``num_lookup_workers`` / ``num_load_workers``:
  Worker-thread counts for each operation type.

**Notes:**

- ``raw_block`` is a server-owned MP adapter. It does **not** support
  per-TP device-path mappings in MP mode.
- ``raw_block`` remains ``"type": "raw_block"`` for all supported engines.
- ``raw_block`` owns on-device slot allocation, checkpointing, and recovery
  through ``RawBlockCore``. Slot reclamation is driven by the shared/global
  L2 eviction controller or explicit ``delete()`` calls.
- ``slot_bytes``, ``header_bytes``, and ``meta_total_bytes`` must be multiples
  of ``block_align``.
- If ``use_odirect`` is enabled, the server's ``--l1-align-bytes`` should be
  at least ``block_align``.
- With ``O_DIRECT``, raw-block I/O rejects offsets and total I/O lengths that
  are not multiples of ``block_align``. Misaligned write buffers use an aligned
  bounce buffer.
- ``persist_enabled`` must remain ``true`` for this adapter.
- For ``use_uring_cmd=true``, ``device_path`` must use the NVMe character
  device node (e.g., ``/dev/ng0n1``) instead of the block device node
  (``/dev/nvme0n1``). The character device provides direct NVMe
  command passthrough.
- ``use_uring_cmd`` requires ``io_engine="io_uring"`` to be set.
- When ``use_uring_cmd=true``, ``use_odirect`` is ignored for NVMe namespace
  character devices. FDP examples set ``use_odirect=false`` because
  ``io_uring_cmd`` uses NVMe passthrough rather than the POSIX write path.
- FDP registers only non-zero placement identifiers. If ``fdp_placement_ids`` is
  the KV data placement pool: if omitted, all discovered non-zero identifiers
  except ``meta_checkpoint_placement_id`` are used; if provided, every
  identifier must be reported by the device and must not contain 0.
- ``meta_checkpoint_placement_id`` must not overlap with ``fdp_placement_ids``.
  Keeping metadata checkpoints and KV data on separate placement identifiers
  avoids mixing long-lived raw-block metadata with cache data buckets.
- Current KV data writes still omit FDP placement identifiers until the
  placement policy is added. Metadata checkpoint writes use
  ``meta_checkpoint_placement_id`` when configured, otherwise they use default
  NVMe placement with no directive.

**Configuration examples:**

.. code-block:: bash

    # Basic raw_block with posix I/O
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "block_align": 4096, "header_bytes": 4096, "meta_total_bytes": 268435456, "use_odirect": true, "num_store_workers": 2, "num_lookup_workers": 1, "num_load_workers": 4}'

    # With io_uring
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "iouring_queue_depth": 256, "use_odirect": true}'

    # With io_uring_cmd (NVMe passthrough)
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "iouring_queue_depth": 256, "max_data_transfer_size": 131072, "use_odirect": false}'

    # With FDP discovery enabled, registering all non-zero device identifiers
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "slot_bytes": 1048576, "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "use_odirect": false}'

    # With eviction
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "slot_bytes": 1048576, "load_checkpoint_on_init": false, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

**Hardware-gated FDP status validation:**

FDP live-device validation is opt-in because it requires an FDP-capable NVMe
namespace character device. The status probe opens the character device through
the Rust raw-block binding and calls ``fetch_fdp_status()`` with a read-only file
descriptor. It does not issue writes, initialize the MP adapter layout, write KV
data, or verify a future FDP placement policy.

.. code-block:: bash

    LMCACHE_TEST_FDP_CHAR_DEVICE=/dev/ng0n1 \
      pytest -q tests/v1/storage_backend/test_raw_block_fdp_status_probe.py

When the variable is not set, the test skips. If the configured device, kernel,
or controller does not support the FDP status query, the test skips with the
underlying capability error. A passing status probe only confirms that the live
device can answer the FDP status query; full adapter initialization on hardware
and KV write placement are separate follow-ups.
