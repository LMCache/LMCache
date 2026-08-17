# Device-Resident L1 Memory Manager Design

This document describes the device-resident L1 tier: a backend-agnostic
framework that lets L1 entries hold GPU/device memory objects directly, so
retrieve serves via D2D instead of H2D. The framework is shipped without a
concrete backend; backends register by implementing the
`DeviceMemoryPool` protocol and adding an `_init_<backend>_pools` branch.

## Goals

- Give L1 the ability to hold device-resident `MemoryObj`s (not just CPU
  pinned-DRAM or GDS slab files).
- Keep the framework backend-agnostic: `DeviceResidentL1MemoryManager` interacts
  with device pools only through the `DeviceMemoryPool` protocol and never
  imports backend-specific types at module level.
- Add a `device_reserve_write` path that produces device objects directly,
  eliminating the CPU-placeholder-then-replace pattern used by the previous
  zero-touch adapter approach.
- Keep the existing `reserve_write` path and all existing adapters
  unchanged — device L1 is opt-in per adapter via `AdapterDescriptor.l1_tier`.
- Ship the framework first (this PR), then add backends in follow-up PRs.

## Components

`lmcache/v1/distributed/memory_manager/device_l1_memory_manager.py` defines
two public symbols:

- `DeviceMemoryPool` — a `@runtime_checkable` `Protocol` that abstracts a
  per-device DMA-registered memory pool. A backend implements five methods:
  `allocate`, `free`, `batched_free`, `wait_for_available`, `get_free_bytes`.
- `DeviceResidentL1MemoryManager` — the L1 tier object. It holds a
  `dict[int, DeviceMemoryPool]` (one pool per device id), routes allocation
  by `kv_rank`, implements backpressure, and delegates CPU-fallback objects
  to an internal `L1MemoryManager`.

`lmcache/v1/distributed/l1_manager.py` selects
`DeviceResidentL1MemoryManager` when `config.device_resident_l1_config` is set, alongside
the existing GDS and Device-DAX branches. It exposes two new methods:

- `device_reserve_write` — like `reserve_write`, but allocates from the
  device pool and creates `L1ObjectState` with a device-resident tensor.
- `has_device_l1` — returns `True` when the backing manager is a
  `DeviceResidentL1MemoryManager`.

`lmcache/v1/distributed/storage_controllers/store_policy.py` adds the
`l1_tier` field to `AdapterDescriptor` (default `"cpu"`).

`lmcache/v1/distributed/storage_controllers/prefetch_controller.py` routes
reserve calls by adapter L1 tier in `_reserve_load_buffers`.

`lmcache/v1/distributed/config.py` adds `DeviceResidentL1Config` and the
`device_resident_l1_config` field on `L1ManagerConfig`.

`lmcache/v1/distributed/api.py` adds `L1BackendType.DEVICE`.

## DeviceMemoryPool Protocol

```python
@runtime_checkable
class DeviceMemoryPool(Protocol):
    def allocate(self, *, shapes, dtypes, fmt) -> MemoryObj | None: ...
    def free(self, memory_obj: MemoryObj) -> None: ...
    def batched_free(self, memory_objs: list[MemoryObj]) -> None: ...
    def wait_for_available(self, required_bytes: int, timeout: float) -> bool: ...
    def get_free_bytes(self) -> int: ...
```

A pool owns one device's worth of pre-allocated, DMA-registered device
memory plus the backend-specific DMA handle. Backends implement this
protocol via structural typing — no inheritance is required. A backend
registers itself by adding an `_init_<backend>_pools` method to
`DeviceResidentL1MemoryManager` and a matching `elif` branch in
`_init_device_pools`.

This PR ships with no backend. The `"phx"` branch raises
`NotImplementedError` as a placeholder; a follow-up PR will implement it.

## DeviceResidentL1MemoryManager

### Pool Ownership

The manager holds a single `dict[int, DeviceMemoryPool]` keyed by device
id. Each pool internally owns its DMA handle and base pointer — there is
no separate allocator/cache/base-pointer dict triplet.

### Initialization Order

`__init__` validates the backend and creates device pools *before*
allocating CPU memory:

1. Store `device_resident_l1_config`.
2. Call `_init_device_pools(config)` — fails fast on unknown/unimplemented
   backend without wasting CPU memory.
3. Create the internal `L1MemoryManager(memory_config)` for CPU-fallback
   objects.

### Allocation

`allocate_device(layout_desc, count, kv_rank)` routes `kv_rank` to a device
id via `_kv_rank_to_device` (default: `device_ids[kv_rank %
len(device_ids)]`), then:

1. `wait_for_available(size_per_obj * count, timeout=1.0)` — backpressure.
2. Allocate one-by-one from the pool.
3. On any failure, roll back partial allocation and return
   `(L1Error.OUT_OF_MEMORY, [])` — all-or-nothing, no per-key fallback.

`allocate(layout_desc, count)` delegates to the CPU manager for
CPU-fallback objects (satisfies `L1ManagerProtocol`).

### Free Routing

`free(mem_objs)` inspects each object's `raw_tensor.device.type`:

- Device objects (`!= "cpu"`) → `obj.parent().free(obj)`. The parent is
  the `DeviceMemoryPool` that allocated the object (set by the backend
  allocator during `allocate`). The pool's `free` notifies any
  `wait_for_available` waiter.
- CPU objects → `self._cpu_manager.free(cpu_objs)`.

If a device object has no parent (should not happen), a warning is logged
and the object is leaked rather than crashing.

### kv_rank → Device Mapping

`_kv_rank_to_device(kv_rank)` returns
`device_ids[kv_rank % len(device_ids)]`. Returns `-1` when no devices are
configured (which causes `allocate_device` to return OOM). Backends or
configs may override this mapping in follow-up PRs.

## L1Manager Integration

### Tier Selection

`L1Manager.__init__` adds a new `elif` branch after GDS and Device-DAX:

```python
elif config.device_resident_l1_config is not None:
    self._memory_manager = DeviceResidentL1MemoryManager(
        config.memory_config, config.device_resident_l1_config
    )
```

Device L1 co-exists with the CPU tier (unlike GDS/Device-DAX, which are
mutually exclusive): the `DeviceResidentL1MemoryManager` holds its own device
pools but delegates CPU-fallback allocation to an internal
`L1MemoryManager`.

### device_reserve_write

```python
@l1_mgr_synchronized
def device_reserve_write(
    self,
    keys: list[ObjectKey],
    is_temporary: list[bool],
    layout_desc: MemoryLayoutDesc,
    kv_rank: int,
) -> dict[ObjectKey, L1OperationResult]
```

Like `reserve_write(mode="new")`, but allocates from the device pool:

1. Filter out keys that already exist (return `KEY_NOT_WRITABLE`).
2. Call `allocate_device(layout_desc, count, kv_rank)`.
3. On OOM: return `OUT_OF_MEMORY` for all keys, free any partial
   allocation.
4. On success: create `L1ObjectState` with the device `MemoryObj`,
   write-lock it, publish `L1_WRITE_RESERVED` event (same as
   `reserve_write`).

All-or-nothing: if the pool is exhausted after backpressure timeout, all
keys return `OUT_OF_MEMORY`. The caller (prefetch controller) abandons the
batch — no per-key POSIX fallback.

### has_device_l1

```python
def has_device_l1(self) -> bool:
    return isinstance(self._memory_manager, DeviceResidentL1MemoryManager)
```

Callers check this before calling `device_reserve_write`.

## AdapterDescriptor.l1_tier

```python
@dataclass(frozen=True)
class AdapterDescriptor:
    index: int
    config: L2AdapterConfigBase
    l1_tier: str = "cpu"   # "cpu" | "device"
```

All existing adapters default to `"cpu"` and keep their current behaviour.
A backend-specific adapter (e.g. PHX, in a follow-up PR) sets
`l1_tier="device"` at registration time.

## Prefetch Controller Routing

`_reserve_load_buffers` now takes `trimmed_plan: dict[int, Bitmap]` (the
per-adapter load plan) in addition to the merged `keys_to_reserve`. For
each group of keys (grouped by `object_group_id`), it splits by L1 tier:

- **Device path** (`desc.l1_tier == "device"` and `has_device_l1()`):
  `device_reserve_write(keys, is_temporary, layout_desc, kv_rank)`.
- **CPU path** (default): `reserve_write(keys, is_temporary, layout_desc,
  mode="new")` — the existing path, unchanged.

The two paths are fully isolated: the CPU path is identical to before,
and the device path is only reached when an adapter explicitly declares
`l1_tier="device"` *and* the L1 manager has a device tier.

This PR ships no adapter with `l1_tier="device"`, so the device branch is
never triggered — **zero behaviour change**.

## Configuration

```python
@dataclass
class DeviceResidentL1Config:
    backend: str = ""           # "" = no backend yet; "phx" in follow-up PR
    device_ids: list[int] = field(default_factory=list)
    buffer_size_mb: int = 2048
    use_direct_io: bool = True

@dataclass
class L1ManagerConfig:
    memory_config: L1MemoryManagerConfig
    gds_l1_config: GdsL1Config | None = None
    device_resident_l1_config: DeviceResidentL1Config | None = None  # new
    ...
```

Config inference (deriving `DeviceResidentL1Config` from an L2 adapter config)
is deferred to the follow-up PR that adds the first backend. This PR
provides the config dataclass and the `L1ManagerConfig` field only.

## Layering Boundary

| Layer | Visible Types | Reason |
|---|---|---|
| `DeviceResidentL1MemoryManager` (this PR) | `DeviceMemoryPool` protocol only | Backend-agnostic; adding a backend does not change this class (only adds an `_init_*_pools` branch) |
| `_init_<backend>_pools` (follow-up PRs) | Backend-specific types | Lazy import inside the method; backend details stay local |
| Backend-specific L2 adapter (follow-up PRs) | Pool's backend-specific attributes | The adapter knows its own backend; accessing `.base_pointer` / `.phx_cache` is appropriate there |
| Concrete pool implementation (follow-up PRs) | No inheritance needed | Structural typing: implement five methods |

## Failure Paths

| Scenario | Behaviour | Result |
|---|---|---|
| DMA short read (partial load failure) | Failed key's bitmap bit = 0; device obj stays in L1 (write-locked) → `finish_write` fails → framework deletes entry → `free()` routes back to device pool | No leak |
| `device_reserve_write` pool exhausted | Returns OOM → prefetch controller emits `L2_PREFETCH_FAILED` + abandons batch | Clean failure |
| Entry evicted before retrieve | Temporary entry `delete()` → `free()` → device pool回收 | No leak |
| Retrieve never happens (request abort) | Temporary device entry lingers in L1 → DMA buffer pinned | Known residual risk (same as previous approach); mitigation: periodic sweep of stale temporary entries |
| vLLM crash | Retrieve traffic → 0 → temporary entries linger → buffer exhaustion → batch abandon | Clean degradation |
| Multi-read-lock (TP/MLA) | `finish_read` unlocks all → count reaches 0 → delete entry → recycle | Equivalent |
| Mixed adapters (redis + device) | Each routes through its own reserve path | Isolated |
| Store path | `serde_wrapper` calls `reserve_write` (CPU), unaffected | Zero impact |
| This PR (no device adapter) | All `l1_tier` default `"cpu"`, `device_reserve_write` never called | **Zero behaviour change** |

## Verification

`tests/v1/distributed/test_device_l1_framework.py` unit-tests the
framework with a `MockDeviceMemoryPool`:

- `DeviceMemoryPool` protocol structural typing (mock satisfies, plain
  object does not).
- `DeviceResidentL1MemoryManager`: backend validation (empty → `ValueError`,
  `"phx"` → `NotImplementedError`), free routing (device → pool,
  CPU → CPU manager, mixed), `kv_rank`-to-device mapping.
- `DeviceResidentL1Config` defaults and construction.
- `AdapterDescriptor.l1_tier` default (`"cpu"`) and explicit
  (`"device"`).
- `L1BackendType.DEVICE` exists.

Existing distributed tests (968 passed, 41 skipped) confirm zero
regression — all existing adapters keep `l1_tier="cpu"` and follow the
unchanged `reserve_write` path.

## Backend Registration (Follow-up PRs)

A new backend registers in three steps:

1. Implement a pool class satisfying `DeviceMemoryPool` (five methods).
2. Add `_init_<backend>_pools(self, config)` to `DeviceResidentL1MemoryManager`
   with lazy imports.
3. Add an `elif` branch in `_init_device_pools`.
4. (Optional) Set `l1_tier="device"` on the adapter's
   `AdapterDescriptor` at registration.
5. (Optional) Add config inference in `config.py` to derive
   `DeviceResidentL1Config(backend="<name>")` from the L2 adapter config.

No changes to `DeviceResidentL1MemoryManager` class body, `L1Manager`,
`prefetch_controller`, or `store_policy` are needed — the framework is
extensible by construction.

## Current Limits

- No backend shipped. The `"phx"` branch raises `NotImplementedError`.
- Config inference is not wired: `device_resident_l1_config` defaults to `None`,
  so `DeviceResidentL1MemoryManager` is never instantiated unless a follow-up PR
  adds inference.
- `get_memory_usage` returns approximate values (device pools do not yet
  report total capacity through the protocol; backends should override if
  precision is needed).
- `get_l1_memory_desc` returns `None`: device pools are not a single
  registerable CPU region, so P2P/NIXL transfer registration is not
  supported for device L1 (same as Device-DAX L1).
- No periodic sweep of stale temporary device entries (retrieve never
  happens): a follow-up can add a TTL-based sweeper.
