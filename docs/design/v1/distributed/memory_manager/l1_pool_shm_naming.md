# L1 pool POSIX shared-memory naming

## Problem

`L1MemoryManager` can back its DRAM tier with a pinned buffer in a named
POSIX SHM segment (`L1MemoryManagerConfig.shm_name`). Names are **host-global**:
processes in different containers on the same node share `/dev/shm`, and
independent PID namespaces reuse the same low PIDs. A default of
`lmcache_l1_pool_{os.getpid()}` let a new container unlink or collide with a
live segment owned by another container.

## Generated names

The default factory produces:

```text
lmcache_l1_pool_<N lowercase hex chars>
```

where `N = L1_POOL_SHM_NAME_TOKEN_HEX_LEN`, derived at import time from the
platform's POSIX SHM name-length limit:

| Platform | Effective limit¹ | Prefix (`/lmcache_l1_pool_`) | Token |
|----------|-----------------|------------------------------|-------|
| macOS    | 30 chars        | 17 chars                     | 13 hex |
| Linux    | 255 chars       | 17 chars                     | 16 hex |

¹ macOS `PSHMNAMLEN` = 31, but XNU's `shm_open` rejects names where
`strlen(name) >= PSHMNAMLEN`, so the inclusive maximum is 30 characters.
Linux resolves through `/dev/shm` with `NAME_MAX` = 255.

The prefix `lmcache_l1_pool_` is 16 bare characters; `shm_open` receives it
slash-prefixed (`/lmcache_l1_pool_<token>`), consuming 17 characters of the
limit. CPU KV segments use a separate `/lmcache_kv_<token>` scheme; see
`docs/design/v1/platform/devices/cpu/shm.md`.

Constants live on `lmcache.v1.distributed.config`:

- `L1_POOL_SHM_NAME_TOKEN_HEX_LEN` (platform-derived: 13 on macOS, 16 on Linux)
- `L1_POOL_SHM_NAME_ATTEMPTS` (5)

## Explicit (pinned) names

Operators may set `shm_name` to a fixed string (with or without the
`lmcache_l1_pool_` prefix; `create_memory_allocator` normalizes bare names).
Pinned names are for tooling that must attach to a known segment.

### Live vs stale: the advisory liveness lock

`alloc_shm_pinned_ptr` holds an exclusive `flock` on the segment for as long
as it maps the pool, and `free_shm_pinned_ptr` (or the death of the process,
which closes the fd and makes the kernel release the lock) drops it.
`_unlink_stale_shm` probes the lock via `shm_segment_is_live`
(`posix_shm.py`) before unlinking:

| Segment state | Probe | Outcome |
|---------------|-------|---------|
| No segment | — | Nothing to unlink; create proceeds |
| Live holder (lock held) | `flock` blocked | Unlink **skipped**; exclusive create raises `EEXIST`, which a pinned name surfaces as an error |
| Stale (holder crashed; kernel released the lock) | `flock` acquired | Unlinked; create proceeds |

On macOS the kernel rejects `flock` on SHM fds (`EOPNOTSUPP`), so no
liveness signal exists: the probe falls back to reclaiming the name and a
pinned name always takes over whatever holds it. Operators running multiple
live processes under one pinned name on such hosts must coordinate the names
themselves; on Linux the collision is surfaced as `FileExistsError`.

**Generated names never call `_unlink_stale_shm`.** A stale segment left by a
crashed process with the same random name is not reclaimed by name; collision
handling retries with a fresh UUID instead of unlinking.

## Creation is exclusive end to end

`alloc_shm_pinned_ptr` (in `lmcache/v1/platform/torch_ops/mem_alloc.py`)
creates its segment with `O_CREAT | O_EXCL` and **never unlinks a
pre-existing name**: a taken name raises `FileExistsError` up through
`MixedMemoryAllocator`, which is what makes the retry loop above observable.
Reclaiming a pre-existing segment is a naming-policy decision that belongs to
`create_memory_allocator`, not to the allocator.

The native CUDA implementation (`csrc/cuda/mem_alloc.cpp`, bound onto
`CudaDeviceOps` by `bind_native`) follows the same contract: exclusive
`shm_open`, `EEXIST` translated by the pybind layer into
`cuda_ops.ShmExistsError` (a `FileExistsError` subclass, so
`is_shm_name_collision` and the retry loop treat it identically), and the
same `flock` liveness guard held for the segment's lifetime and released in
`free_shm_pinned_ptr`.

## Crash / orphan segments

Each new `L1MemoryManagerConfig()` with the default factory gets a **fresh**
random name. A process that dies without `L1MemoryManager.close()` may leave
its segment in `/dev/shm` until the kernel reaps it or an operator unlinks it;
the next process does **not** reuse that name automatically. Graceful
`close()` unlinks the pool segment. A crash under a **pinned** name is
recovered automatically: the kernel's lock release marks the segment stale,
so the next creation under that name reclaims it (see the table above).

## Collision retries

`create_memory_allocator` retries up to `L1_POOL_SHM_NAME_ATTEMPTS` times when
``is_shm_name_collision`` in `posix_shm.py` reports a name collision
(`FileExistsError` or `OSError` with `errno.EEXIST`) **only for generated
names**, drawing a new UUID token on each retry. Pinned names do not retry with
alternate names.

## Pinned names matching the generated pattern

Any pinned name that happens to match the auto-generated format
(`lmcache_l1_pool_[0-9a-f]{N}` where `N = L1_POOL_SHM_NAME_TOKEN_HEX_LEN`)
is indistinguishable from a generated name and
is therefore subject to retry with a new UUID on collision, rather than surfacing
the collision as an error. This means a collision with a live segment of that
name will silently produce a different pool name instead of failing.

Operators who need strict pinned-name semantics — where a collision must be
surfaced as an error — should use a suffix that contains at least one non-hex
character, for example `lmcache_l1_pool_primary` or `lmcache_l1_pool_node0`.
Such names are recognised as pinned and never retried with alternate UUIDs.
