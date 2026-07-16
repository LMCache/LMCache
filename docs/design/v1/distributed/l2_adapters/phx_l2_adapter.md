# Phoenix L2 Adapter

`lmcache/v1/distributed/l2_adapters/phx_l2_adapter.py`

A disk-based L2 adapter that uses [Phoenix](https://github.com/xPU-IO/phoenix)
GPU-direct storage I/O to load KV cache data directly from NVMe to GPU memory
via DMA, bypassing the host CPU entirely on the retrieve path.

## Motivation

Existing disk-based L2 adapters (`fs_native`, `fs`) retrieve data via POSIX
`read()`, which forces data through host CPU memory:

```
NVMe →(POSIX read)→ L1 CPU MemoryObj →(H2D scatter)→ GPU paged KV    [2× PCIe]
```

Phoenix eliminates the CPU intermediary on the load path:

```
NVMe →(phxfs DMA)→ GPU MemoryObj →(D2D scatter)→ GPU paged KV    [1× PCIe + 1× GPU-internal]
```

This halves PCIe traffic on retrieve and reduces latency, especially for
long-context workloads with large KV cache chunks.

## Asymmetric Design

The adapter uses **asymmetric store/load** because LMCache's MP-mode
architecture always hands adapters CPU `MemoryObj` buffers (L1 is CPU DRAM):

| Operation | Data Path | I/O Mechanism |
|---|---|---|
| **Store** | CPU MemoryObj → POSIX write → NVMe | `phxfs_write_batch` or POSIX `write` |
| **Load** | NVMe → `phxfs_read_batch` DMA → GPU MemoryObj | Phoenix DMA (preferred) or POSIX `read` (fallback) |

Store goes through CPU because the data is already in CPU memory (written by
the StoreController from L1). Trying to DMA from CPU to NVMe via Phoenix would
add no benefit over a standard POSIX write.

Load is the performance-critical path. The key insight: PHX value is only
realized if the DMA'd data **stays on GPU** and flows to the retrieve handler
via D2D (device-to-device copy). If data is D2H'd back to CPU, the DMA benefit
is negated.

## Device-Resident MemoryObj Flow

To let GPU-resident data flow from adapter to retrieve without going through
CPU, the adapter extends the framework with a **device-obj injection mechanism**:

```
LOOKUP phase (PrefetchController):
  1. Broadcast lookup to all adapters
  2. Collect bitmaps

LOAD phase (PrefetchController):
  3. submit_load_task → PhxL2Adapter
  4. PhxL2Adapter._process_load:
     a. Allocate GPU MemoryObj from PhxDeviceMemoryAllocator (phx pool)
     b. phxfs_read_batch DMA: NVMe → GPU MemoryObj (batched, grouped by device)
     c. Store GPU objs in _load_device_objs[task_id]
  5. PrefetchController._poll_load_results:
     a. pop_loaded_device_objs(task_id) → gets GPU MemoryObjs
     b. L1Manager.replace_memory_obj(key, gpu_obj) — swaps CPU obj for GPU obj in L1

RETRIEVE phase (retrieve handler):
  6. read_prefetched_results → gets GPU MemoryObj from L1
  7. transfer_kv_per_object_group:
     - PyTorch copy_ auto-detects device: GPU src → D2D (no PCIe!)
     - multi_layer_kv_transfer kernel scatters to vLLM paged KV
  8. release_device_objs(keys) — recycle DMA buffer (or D2H write-back to CPU obj)
```

This preserves LMCache's existing architecture: the PrefetchController still
manages the load lifecycle, L1 still serves as the cache layer, and the
retrieve handler is unchanged except for `copy_` device auto-detection.

`pop_loaded_device_objs()` returns an empty dict for keys that fell back to
POSIX read (those filled the controller-provided CPU obj directly) or for
tasks that have already been popped. The base class `L2AdapterInterface`
provides a no-op default so other adapters are unaffected.

## Fallback

When Phoenix is unavailable, the adapter transparently falls back to POSIX
read/write. This happens when:

- `phxcache` Python package is not installed (`ImportError` in `_init_devices`)
- `device_ids` is not configured (no DMA devices initialized)
- Device initialization fails (e.g., no Phoenix kernel module)
- DMA buffer pool is exhausted at load time (per-key fallback)

In fallback mode, `is_phx_available()` returns `False`, load fills the
controller-provided CPU `MemoryObj` directly via POSIX `read()`, and
`pop_loaded_device_objs()` returns an empty dict. The adapter introduces **no
hard dependency** on Phoenix hardware.

## Multi-Device Support

The adapter supports multiple GPUs via `device_ids` configuration. Keys are
routed to devices by `kv_rank` (extracted from `ObjectKey`):

```python
def _kv_rank_to_device(self, kv_rank: int) -> int:
    # kv_rank = (world_size << 24) | (global_rank << 16)
    #         | (local_world_size << 8) | local_rank
    return (kv_rank >> 16) & 0xFF  # global_rank → CUDA device id
```

One `PhxCache` + `PhxDeviceMemoryAllocator` is created per device. Batch reads
are grouped per device for concurrent I/O.

## Configuration

```json
{
  "type": "phx",
  "base_path": "/mnt/nvme/kv_cache",
  "device_ids": [4, 5, 6, 7],
  "buffer_size_mb": 2048,
  "use_direct_io": true,
  "max_capacity_bytes": 0,
  "perf_log_dir": null
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `base_path` | str | (required) | Root directory for KV cache files |
| `device_ids` | list[int] | `None` | CUDA GPU IDs for phxfs DMA (one buffer pool per device). `None` → POSIX-only mode |
| `buffer_size_mb` | int | 2048 | GPU buffer pool size per device (MiB) |
| `use_direct_io` | bool | true | Use `O_DIRECT` for I/O |
| `max_capacity_bytes` | int | 0 | Max storage, 0=unlimited |
| `perf_log_dir` | str | `None` | When set, writes perf log (hit rate + per-phase timing) to this dir |

When `device_ids` is omitted or `phxcache` is not installed, the adapter
operates in POSIX fallback mode — no GPU or Phoenix hardware is required.

## External Dependency

The `phxcache` pybind11 C++ extension (wrapping the Phoenix `phxfs_*` C API)
is maintained in the [Phoenix repository](https://github.com/xPU-IO/phoenix)
under `adapters/lmcache/phxcache/`. It is imported lazily inside
`_init_devices()`, so LMCache works without it installed (POSIX fallback).

## close()

Stops the background worker thread, closes all cached read file descriptors,
releases any device objs never popped by the controller (to avoid leaking phx
pool memory), frees device allocators, closes PhxCache instances, and closes
the three event notifiers. `close()` is idempotent.
