Raw Block (Rust)
================

A built-in L2 adapter that stores KV objects on a raw block device or
pre-sized file using the Rust raw-device I/O bindings. Each object gets exactly
the space it needs (a header plus its payload rounded up to ``block_align``),
so one device serves objects of any size: several models, chunk sizes, hybrid
layer groups, unfull chunks and serialized payloads, without sizing anything
at startup. It reuses the existing raw-block metadata checkpoint model and
writes directly into the caller-provided load buffers during prefetch.

**Installation:** the Rust bindings ship as a separate wheel,
``lmcache-rust-raw-block-io``, released at the same version as ``lmcache``.
Install both together through the ``raw-block`` extra. The extra does not pin
the Rust wheel's version, so pin it yourself to guarantee a matching pair:

.. code-block:: bash

    pip install 'lmcache[raw-block]'
    # exact match, e.g. for 0.5.7:
    pip install lmcache==0.5.7 lmcache-rust-raw-block-io==0.5.7

**Required fields:**

- ``device_path``: Raw device path or pre-sized file path.

**Optional fields:**

- ``capacity_bytes``: Optional cap on the usable device bytes. Default ``0``
  means use the full device/file size.
- ``use_odirect``: ``true`` or ``false`` (default ``true``).
- ``block_align``: Device alignment in bytes (default ``4096``). Must be a
  power of two.
- ``header_bytes``: Per-object header reservation (default ``4096``).
- ``slot_bytes``: Deprecated and ignored (a warning is logged). Earlier
  versions stored every object in a slot of this size.
- ``meta_total_bytes``: Reserved metadata checkpoint region (default ``256MiB``).
- ``meta_magic`` / ``meta_version``: Metadata checkpoint identity/version knobs.
- ``meta_checkpoint_interval_sec`` / ``meta_idle_quiet_ms`` /
  ``meta_enable_periodic`` / ``meta_verify_on_load``: Checkpoint and recovery
  controls carried over from the legacy raw-block backend. A periodic
  checkpoint waits for ``meta_idle_quiet_ms`` without I/O, but not for more
  than one extra interval, so steady traffic cannot hold it off. With
  ``meta_enable_periodic=false``, only a clean shutdown leaves a checkpoint
  that covers the whole cache.
- ``load_checkpoint_on_init``: Load an existing on-device metadata checkpoint
  during startup (default ``true``). Set to ``false`` to start with an empty
  in-memory index instead; the old checkpoint is then voided at startup.
- ``enable_zero_copy``: Try aligned direct-buffer I/O when possible. With
  ``io_engine="io_uring"``, also try to register eligible L1 memory as fixed
  buffers.
- ``io_engine``: Rust raw-block I/O engine. Valid values are ``"posix"``
  (default synchronous ``pread``/``pwrite`` path), ``"io_uring"`` (direct Rust
  io_uring syscall path).
- ``use_uring_cmd``: Enable NVMe passthrough via io_uring command interface
  for direct device access. Requires ``io_engine="io_uring"`` and NVMe
  character device node (e.g., ``/dev/ng0n1``).
- ``iouring_queue_depth``: Queue depth for ``io_engine="io_uring"``.
- ``max_data_transfer_size``: Maximum data transfer size for
  ``use_uring_cmd=true``. Large transfers are split into smaller chunks
  that fit within device limits. When left unset (or ``<= 0``), it is
  auto-detected from the device's sysfs queue limits, bounded by both
  ``max_hw_sectors_kb`` and ``max_segments * page_size`` (the NVMe passthrough
  path uses one scatter-gather segment per page).
- ``fdp_enabled``: Enables NVMe Flexible Data Placement (FDP) discovery
  and non-zero placement identifier registration. ``cache_salt`` values with
  ``":"`` bucket prefixes use FDP placement by default. Requires
  ``io_engine="io_uring"`` and ``use_uring_cmd=true``.
- ``fdp_placement_ids``: Optional non-zero placement identifier list for KV
  data placement. If omitted, the adapter uses all device-reported non-zero
  identifiers except ``meta_checkpoint_placement_id``.
- ``fdp_data_placement_policy``: KV data placement policy. ``"none"`` omits
  FDP directives for KV data writes. ``"cache_salt_prefix"`` assigns
  case-insensitive ``cache_salt`` prefixes to FDP placement identifiers.
  ``"cache_salt_rank"`` keeps that bucket isolation and further separates
  rank-local write streams within each bucket when identifiers are available.
  The default is ``"cache_salt_prefix"`` when ``fdp_enabled=true``.
- ``fdp_slot_reuse_policy``: Free-space reuse policy. ``"pid_affinity"``
  prefers, among free extents of exactly the needed size, one last written
  with the same FDP placement identifier, before falling back to the usual
  best fit. ``"none"`` disables affinity-based reuse. The
  default is ``"pid_affinity"`` when ``fdp_enabled=true`` and ``"none"``
  otherwise.
- ``meta_checkpoint_placement_id``: Optional non-zero placement identifier for
  metadata checkpoint payload/header writes. Omit it to keep checkpoint writes
  on default NVMe placement.
- ``num_store_workers`` / ``num_lookup_workers`` / ``num_load_workers``:
  Worker-thread counts for each operation type.

**Notes:**

- ``raw_block`` is a server-owned MP adapter. It does **not** support
  per-TP device-path mappings in MP mode.
- ``raw_block`` remains ``"type": "raw_block"`` for all supported engines.
- ``raw_block`` owns on-device space allocation, checkpointing, and recovery
  through ``RawBlockCore``. Space reclamation is driven by the shared/global
  L2 eviction controller or explicit ``delete()`` calls. Usage counts each
  object's allocated bytes (header and padded payload).
- POSIX restart recovery validates object headers with an internal pool of 8
  reader threads. Both regular ``io_uring`` and ``io_uring_cmd`` batch header
  reads up to ``iouring_queue_depth``.
- ``header_bytes`` and ``meta_total_bytes`` must be multiples of
  ``block_align``, and ``header_bytes`` at least 32.
- If ``use_odirect`` or ``use_uring_cmd`` is enabled, the server's
  ``--l1-align-bytes`` must be at least ``block_align``.
- With ``O_DIRECT``, raw-block I/O rejects offsets and total I/O lengths that
  are not multiples of ``block_align``. Misaligned write buffers use an aligned
  bounce buffer.
- ``persist_enabled`` must remain ``true`` for this adapter.
- For ``use_uring_cmd=true``, ``device_path`` must use the NVMe character
  device node (e.g., ``/dev/ng0n1``) instead of the block device node
  (``/dev/nvme0n1``). The character device provides direct NVMe
  command passthrough.
- For ``use_uring_cmd=true``, ``block_align`` must be a multiple of the NVMe
  namespace LBA size. An incompatible value is rejected when the device opens.
- ``use_uring_cmd`` requires ``io_engine="io_uring"`` to be set.
- When ``use_uring_cmd=true``, ``use_odirect`` is ignored for NVMe namespace
  character devices. FDP examples set ``use_odirect=false`` because
  ``io_uring_cmd`` uses NVMe passthrough rather than the POSIX write path.
- FDP registers only non-zero placement identifiers. ``fdp_placement_ids`` is
  the KV data placement pool: if omitted, all discovered non-zero identifiers
  except ``meta_checkpoint_placement_id`` are used; if provided, every
  identifier must be reported by the device and must not contain 0.
- ``meta_checkpoint_placement_id`` must not overlap with ``fdp_placement_ids``.
  Keeping metadata checkpoints and KV data on separate placement identifiers
  avoids mixing long-lived raw-block metadata with cache data buckets.
- With ``fdp_data_placement_policy="cache_salt_prefix"``, ``cache_salt`` values
  containing ``":"`` are bucketed by the prefix before ``":"``. The bucket name
  is case-insensitive, so ``RAG:app1``, ``rag:app2``, and ``rag:`` share the
  same bucket. Values without ``":"`` and values with an empty prefix use no
  FDP directive. Buckets are assigned exclusive FDP placement identifiers in
  first-seen order while identifiers are available.
- If the number of discovered cache-salt buckets exceeds the number of
  registered FDP placement identifiers, additional buckets continue to store
  with no FDP directive and the adapter emits one warning. ``report_status()``
  exposes a fallback count and a bounded bucket sample instead of retaining the
  complete fallback bucket set. Empty ``cache_salt`` values also use default
  NVMe placement with no directive.
- With ``fdp_data_placement_policy="cache_salt_rank"``, placement is keyed by
  the case-insensitive ``cache_salt`` bucket prefix and the local rank encoded
  in ``kv_rank``. The suffix after ``":"`` remains part of the object key but
  does not affect FDP placement. For example, ``batch:app-a`` and
  ``batch:app-b`` share the ``batch`` bucket: local rank 0 from both
  applications shares one placement identifier, local rank 1 shares another,
  and so on. Each bucket can use at most the detected visible GPU count worth
  of placement identifiers. Extra ranks, or buckets that cannot obtain an
  identifier, store with no FDP directive.
- Sharing placement identifiers within a bucket can be intentional because the
  available FDP placement-identifier pool is finite. Operators can assign the
  same prefix to applications with similar cache lifetimes, or use a distinct
  prefix for each application when placement isolation is required. For
  example, ``app01:<suffix>`` through ``app16:<suffix>`` can provide 16
  application buckets with eight rank streams each, consuming up to 128
  placement identifiers when at least 128 identifiers are available and the
  adapter detects at least eight visible GPUs. Applications using the same
  prefix share the placement identifiers for equal local ranks instead of
  receiving separate sets.
- Bucket-to-placement assignments are process-local. Restart recovery does not
  need them for correctness because ``cache_salt`` is part of the object key,
  but first-seen FDP placement assignments may change after adapter restart.
- Reuse affinity is process-local and is not stored in metadata checkpoints.
  After recovery, free space has no recorded affinity until it is reused.
- Store and retrieve requests must use the same ``cache_salt`` to address the
  same object. This is the normal LMCache key identity rule; FDP placement is a
  write directive and is not used to locate data on reads.
- Metadata checkpoint writes use ``meta_checkpoint_placement_id`` when
  configured, otherwise they use default NVMe placement with no directive.

**Space allocation:**

Free space is kept as maximal free extents. A store takes the head of the
smallest free extent that fits (best fit), so an object the size of one that
was just evicted reuses exactly that space. Freed space merges with adjacent
free space immediately. A store fails when no single free extent is long
enough, even if the free bytes add up; ``alloc_failure_count`` counts these
failures. Eviction is unchanged: it starts at ``trigger_watermark``, so at the
default 0.8 the free space left keeps such failures rare (at most 0.1 % in the
simulations and end-to-end runs of the design doc). With a higher
``trigger_watermark`` (above about 0.85) less free space is left, and the
larger objects can keep failing below the watermark. Objects that cannot be evicted
(locked keys, see ``locked_key_count``) spread across the device have the same
effect. A rising ``alloc_failure_count`` with ``largest_free_extent_bytes``
below the object size shows it.

The core status (``/status``, under the adapter's ``core``) reports
``usable_capacity_bytes``, ``allocated_bytes``, ``free_bytes``,
``free_extent_count``, ``largest_free_extent_bytes``,
``alloc_failure_count``,
``header_invalidation_count``, ``header_invalidation_failure_count``,
``quarantined_bytes`` and
``quarantined_range_count``. The fixed-slot keys ``slot_bytes``,
``max_slots``, ``next_slot``, ``free_slot_count`` and
``quarantined_slot_count`` are gone.

The first ``block_align`` bytes of the data region hold a format marker, not
objects.

**Space reuse and crash recovery:**

After a crash, recovery loads the newest on-device checkpoint and, with
``meta_verify_on_load`` (the default), drops every entry whose object header
does not match: the header holds the key, the size and a random generation of
the write that the checkpoint lists, so a header left by another write of the
same key at the same place does not match either. Because objects have
different sizes, a new object can start in the middle of a deleted one and
overwrite its payload without touching its header. A delete therefore
overwrites the object's header first and frees the space only after that write
returns, so recovery drops the deleted entry instead of returning another
object's bytes for it.

This costs one ``header_bytes`` write per deleted object (4 KiB by default:
0.4 % of the bytes a 1 MiB object took to store, 0.04 % for 10 MB), done by
the delete call itself on the eviction controller's thread: batched with
``io_engine="io_uring"``, spread over 8 threads with ``io_engine="posix"``.
In our runs (8 keys per call, ramdisk, O_DIRECT) a delete took 5.8, 8.6,
10.2 and 11.5 µs per object with io_uring and 9.7, 15.3, 19.9 and 23.7 µs
with posix for 64 KiB, 1 MiB, 4 MiB and 16 MiB objects, against 0.6 to
1.4 µs with fixed slots; with 2048 keys per call on an ext4 file on NVMe,
4 to 6 µs with io_uring and 36 µs with posix. Stores took 0.2 to 3.5 % more
with io_uring and 0.6 to 10 % more with posix at the same sizes (the most at
64 KiB); loads moved within ±4 %. If that write fails, the space is retired like
quarantined space (below) and counted in
``header_invalidation_failure_count``; a restart frees it again, which is
safe because the object's payload was never overwritten. At startup, an entry
whose header cannot be read is retired too.

- With ``meta_verify_on_load=false``, recovery trusts the checkpoint as is, so
  space reused since it was written can return wrong bytes, as with the
  fixed-slot layout. Deletes still zero headers, so a later run with
  verification on is protected.
- New checkpoints always supersede the copies of earlier runs. If no
  checkpoint is loaded (``load_checkpoint_on_init=false``, it is rejected, or
  it was written with another ``meta_magic`` or ``meta_version``), every
  metadata container is voided at startup, before any store, also the
  containers of a rejected checkpoint's own ``meta_total_bytes``; a failed
  write there stops the adapter from starting, and so does a failed read of
  the metadata region. Clear the metadata region before changing
  ``meta_total_bytes``, ``block_align`` or ``header_bytes``: a checkpoint copy
  this run cannot read stays on the device.
- Retired space returns after a restart, which frees it as unused.
- The checkpoint grows with the number of objects (about 250 bytes each with
  vLLM MP keys in our runs); a checkpoint larger than half of
  ``meta_total_bytes`` is not written, and a warning is logged. Size
  ``meta_total_bytes`` for the expected object count, for example 128 MiB for
  200,000 objects.
- Writes are not flushed to the device. These guarantees cover process
  crashes; after a power loss, a drive with a volatile write cache may keep
  any subset of the recent writes.

**Upgrading from fixed slots, and going back:**

Checkpoints written by earlier fixed-slot versions are loaded: each object
keeps only the space it needs and the rest of its slot becomes free. The
object in slot 0 is dropped, because that space now holds the format marker.
The checkpoint is then rewritten in the new format; if it no longer fits a
metadata container (entries are about 26 bytes longer), the adapter starts
empty instead.

Older versions reject the new format and start empty. Clear the metadata
region (the first ``meta_total_bytes`` of the device) before going back to an
older version and before upgrading again after it; both lose the cache. For
example, with the default ``meta_total_bytes`` of 256 MiB (check the device
name, this destroys the cache on it):

.. code-block:: bash

    dd if=/dev/zero of=/dev/nvme0n1 bs=1M count=256 oflag=direct

Without that, the format marker catches the common case: the older version
writes its first object at the start of the data region, overwriting the
marker, and this version then starts empty instead of trusting a checkpoint
whose space the older run may have reused. It does not catch an older run
that stopped before that first write reached the device while later writes
did (one io_uring batch has no order, and posix runs several store workers).

**Fixed-buffer registration and memlock:**

With ``io_engine="io_uring"`` and ``enable_zero_copy=true``, the adapter
registers the stable L1 memory range during initialization. An anonymous
``MixedMemoryAllocator`` exposes its full arena; ``LazyMemoryAllocator``
exposes only the currently pinned prefix. POSIX shared-memory arenas do not
expose an eligible range. Registration uses regions no larger than 1 GiB and
does not grow when the lazy allocator expands.

The kernel charges registered buffers against ``RLIMIT_MEMLOCK`` unless the
process has ``CAP_IPC_LOCK``. The default lazy initial prefix is 20 GiB, so
allow at least that much locked memory, plus headroom for other registrations
and locked memory. Splitting the prefix into 1 GiB regions does not reduce
the total memlock requirement. See the Linux
`io_uring_register(2) manual <https://man7.org/linux/man-pages/man2/io_uring_register.2.html>`_.

For a Bash-launched server, inspect the soft and hard limits in KiB and set
the soft limit before starting the server from the same shell:

.. code-block:: bash

    ulimit -Sl
    ulimit -Hl
    # 20 GiB in KiB; requires a hard limit of at least this value.
    ulimit -Sl 20971520

If the hard limit is too low, raise it through the service, container, or
login configuration. A systemd service can use ``LimitMEMLOCK=infinity``;
a Docker container can use ``--ulimit memlock=-1:-1``. Apply the setting to
the process running the LMCache server, then restart it. Registration is
attempted once during adapter initialization.

If registration fails, the adapter logs a warning and continues with ordinary
io_uring. The adapter can therefore remain healthy even when fixed buffers
are inactive. Check ``RawBlockL2Adapter.report_status()["core"]`` for:

- ``fixed_buffers_registered``: ``true`` only after successful kernel
  registration.
- ``fixed_buffer_registered_bytes``: Total bytes in the registered regions;
  ``0`` when no buffers are registered. This is the registered range size,
  not the final L1 capacity or the number of bytes transferred.

Both fields also appear under each raw-block adapter in
:doc:`GET /status <../http_api>`:

.. code-block:: bash

    curl -fsS http://localhost:8080/status | jq '
      .storage_manager.l2_adapters[]
      | select(.type == "RawBlockL2Adapter")
      | .core
      | {fixed_buffers_registered, fixed_buffer_registered_bytes}'

For a successfully registered 20 GiB prefix, the result is:

.. code-block:: json

    {
      "fixed_buffers_registered": true,
      "fixed_buffer_registered_bytes": 21474836480
    }

Failed or skipped registration reports ``false`` and ``0``, as does a closed
adapter. Successful registration does not mean every request uses fixed I/O:
requests outside the registered prefix or crossing a region boundary still
use ordinary io_uring. When comparing performance, verify these fields and
keep ``enable_zero_copy`` enabled in both cases to isolate registration from
direct-buffer I/O.

**Configuration examples:**

Pass these L2 options to ``lmcache server``. For a server on
``127.0.0.1:5555``, use the MP client endpoint ``grpc://127.0.0.1:5555``.
A bare ``127.0.0.1:5555`` also selects the default gRPC request transport.

.. code-block:: bash

    # Basic raw_block with posix I/O
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "block_align": 4096, "header_bytes": 4096, "meta_total_bytes": 268435456, "use_odirect": true, "num_store_workers": 2, "num_lookup_workers": 1, "num_load_workers": 4}'

    # With io_uring
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "io_engine": "io_uring", "iouring_queue_depth": 256, "use_odirect": true}'

    # With io_uring_cmd (NVMe passthrough)
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "io_engine": "io_uring", "use_uring_cmd": true, "iouring_queue_depth": 256, "max_data_transfer_size": 131072, "use_odirect": false}'

    # With FDP discovery and cache_salt prefix placement enabled
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "use_odirect": false}'

    # With FDP discovery and cache_salt/rank placement enabled
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "fdp_data_placement_policy": "cache_salt_rank", "use_odirect": false}'

    # With FDP discovery only, keeping KV data writes on default NVMe placement
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/ng0n1", "io_engine": "io_uring", "use_uring_cmd": true, "fdp_enabled": true, "fdp_data_placement_policy": "none", "use_odirect": false}'

    # With eviction
    --l2-adapter '{"type": "raw_block", "device_path": "/dev/nvme0n1", "load_checkpoint_on_init": false, "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.9, "eviction_ratio": 0.1}}'

**Failure handling and shutdown:**

Recoverable ``io_uring`` submission errors (``EAGAIN``, ``EINTR``, ``EBUSY``)
retry after completion progress or 1–100 ms exponential backoff. Permanent
errors, or 30 seconds without submission or completion progress, mark the
worker unhealthy and stop new native I/O. The core status exposes the failure
reason as ``worker_error``.

On an initialized MP adapter with ``io_engine="io_uring"`` and
``use_uring_cmd=true``, a read completion with NVMe SCT 2 / SC 0x81
confirms an Unrecovered Read Error (UECC). The status mask ignores CRD,
More and DNR. POSIX and ordinary ``io_uring`` surface Linux errors and
retain per-request handling.

Quarantine requires a matching raw-block wheel with typed completion
errors. Older wheels report failed reads as per-request misses.

On a confirmed UECC while loading a cached payload, the core quarantines
all of the object's space until the handle closes. Lookup and load miss for
its key. The worker keeps serving other objects, including successful keys
in the same load. A later store can cache the recomputed key elsewhere.
Quarantined space reduces usable capacity by its size.

For a full-attention prefix request for A, B and C, if B fails and no
other tier supplies B, the serving caller gets only A. If A fails, the
prefix is empty. The prefetch controller applies this rule after combining
the available tiers. A separate request can still use healthy C.

The core logs the failed byte range and raw NVMe status. The adapter's core
status exposes ``quarantined_range_count``, ``quarantined_bytes``,
``last_payload_uecc`` and the remaining ``usable_capacity_bytes``. A payload UECC alone leaves the worker
healthy. Checkpointing continues and records the remaining indexed keys.
The quarantine set is local to the open handle. After restart the allocator
can reuse the space, and an unreadable range can trigger quarantine again.
Checkpoint and object-header recovery reads retain their per-request handling.

After a known worker failure, new prefetch tasks report failure through their
normal task results and new stores are skipped. The adapter still publishes
completion results and notifications so controllers can release locks and
buffers; the serving engine can recompute missing data instead of receiving a
synchronous storage exception. Valid hits from L1 or other L2 adapters remain
usable. The legacy non-MP raw-block backend also skips new stores after worker
failure.

Already-accepted I/O retains its buffers until completion or successful
cancellation. Close and destructor cleanup release the Python GIL while
waiting, but can still block indefinitely if accepted I/O cannot finish or be
cancelled. The 30-second submission retry budget is not a shutdown timeout.

**Hardware-gated FDP status validation:**

FDP live-device validation is opt-in because it requires an FDP-capable NVMe
namespace character device. The status probe opens the character device through
the Rust raw-block binding and calls ``fetch_fdp_status()`` with a read-only file
descriptor. It does not issue writes, initialize the MP adapter layout, write KV
data, or verify the MP adapter's cache-salt placement policy.

.. code-block:: bash

    LMCACHE_TEST_FDP_CHAR_DEVICE=/dev/ng0n1 \
      pytest -q tests/v1/distributed/raw_block/test_fdp_status_probe.py

When the variable is not set, the test skips. If the configured device, kernel,
or controller does not support the FDP status query, the test skips with the
underlying capability error. A passing status probe only confirms that the live
device can answer the FDP status query; full adapter initialization and KV write
placement on hardware are separate validations.
