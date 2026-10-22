# CPU KV-cache POSIX shared memory (`CpuShmTensorWrapper`)

## Motivation

MP-mode registration must share paged KV-cache storage between the engine
worker and the LMCache server without copying multi-gigabyte buffers on
every request. On hosts without a GPU (or on the CPU SHM transfer path),
`CpuShmTensorWrapper` mirrors CUDA IPC: both processes `mmap` the **same**
physical pages and only exchange a segment **name** plus tensor layout
metadata on the wire.

Low-level syscalls live in `lmcache/v1/multiprocess/posix_shm.py`; this
module owns the KV-specific wrapper, migration factory, and per-process
segment registry.

## Segment naming

POSIX shared memory is **host-global**. Every `shm_open` name visible inside
one container is visible to every other process on the same kernel host,
regardless of PID namespace isolation.

Legacy naming used `os.getpid()` plus a per-process counter
(`/lmcache_kv_<pid>_<n>`). Separate containers often reuse low PIDs (both
see PID 1, 2, …), so two workers could race on the same name; the second
`shm_open(O_CREAT|O_EXCL)` fails with `EEXIST`.

Current naming:

```text
/lmcache_kv_<N lowercase hex chars>
```

The token length `N` is derived from the platform's POSIX SHM name limit:

| Platform | Effective limit¹ | Prefix `/lmcache_kv_` | Max token | Actual token |
|----------|-----------------|----------------------|-----------|--------------|
| macOS    | 30 chars        | 12 chars             | 18        | 16 (capped)  |
| Linux    | 255 chars       | 12 chars             | 243       | 16 (capped)  |

¹ macOS `PSHMNAMLEN` = 31, but XNU's `shm_open` rejects names where
`strlen(name) >= PSHMNAMLEN`, so the inclusive maximum is 30 characters (same
rule as L1 pools; see
`docs/design/v1/distributed/memory_manager/l1_pool_shm_naming.md`).

Both platforms use 16 hex digits (64 bits of randomness) because the cap of
16 keeps full names at 28 characters (≤ 30 on macOS) and provides ample
collision resistance.
Names **must not** embed process ID or any other namespace-local identifier.

`migrate_to_shm_and_wrap` retries up to `SHM_SEGMENT_NAME_ATTEMPTS` (5)
times when ``is_shm_name_collision`` in `posix_shm.py` reports a name
collision (`FileExistsError` or `OSError` with `errno.EEXIST`). Collisions
are negligible; retries are cheap insurance given `O_EXCL` creation.

## Migration unit and registry

The unit of migration is the tensor's **backing** `UntypedStorage`:

1. The first view of a storage copies the full storage into a new SHM
   segment and re-points that tensor.
2. Sibling views of the same storage hit the in-process registry and are
   re-pointed at the existing segment with their own `storage_offset` /
   stride preserved.
3. Empty tensors (`numel == 0`) use `shm_name == ""` — POSIX forbids
   zero-length `mmap`.

The registry is keyed by `id(untyped_storage)` with weak references so
entries disappear when the segment is collected; stale id reuse is treated
as a miss and triggers a fresh segment.

Concurrent first-time migration of views sharing one storage is **not**
supported; `wrap_kv_caches` wraps sequentially at registration time.

## Lifetime

- **Producer**: `weakref.finalize` on the SHM-backed storage runs
  `munmap` + `shm_unlink` when the last migrated view of that segment is
  garbage-collected.
- **Consumer**: `to_tensor()` maps the segment independently and pins the
  mmap on the returned storage until that storage is finalized.

The wire carries only `shm_name` and `nbytes` (`CpuShmIPCData`); opening the
segment is receiver-local.

## Registration

`CpuDeviceSpec.ipc_wrapper_cls` binds `CpuShmTensorWrapper`. Registration
calls `resolve_kv_wrapper_factory("cpu")` → `migrate_to_shm_and_wrap`.

See also `docs/design/v1/multiprocess/device_ipc_wrapper_design.md` for how
this wrapper fits the `DeviceIPCWrapper` hierarchy and wire format.

Distributed L1 DRAM pools use a separate `lmcache_l1_pool_<token>` naming
scheme (`docs/design/v1/distributed/memory_manager/l1_pool_shm_naming.md`).
