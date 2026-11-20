# Ascend NPU Device Backend

`lmcache/v1/platform/devices/npu/` wires Ascend NPU (`torch.npu` from
`torch_npu`) into the platform device registry. The MP native core (host
bindings, recorders, pinned allocation and the Ascend C block-transfer kernel)
is built in-tree as `lmcache.npu_ops`; the external `lmcache-ascend` plugin
keeps SP and transfer-channel extensions and remains an optional activation
hook for hardware-specific patches.

| Module | Responsibility |
|---|---|
| `__init__.py` | `NpuDeviceSpec` registry entry |
| `ops.py` | Curated facade over `lmcache.npu_ops`: exports ops plus `PageBufferShapeDesc`, filtering format enums and torch-fallback re-exports |
| `ipc_wrapper.py` | `NpuIPCWrapper`: plane-aggregating KV IPC wrapper over torch_npu storage IPC (`_share_npu_` / `_new_shared_npu`); one wrapper per layer, registered form preserved across the wire |
| `event_ipc.py` | `NpuEventIPCBackend`: shared CUDA-style event ABI adapter with device-pinned export |
| `cache_context.py` | `NpuCacheContext`: device pointer tables, `_TempNpuBuffer` staging, transfer stream |
| `device_ops.py` | `NpuDeviceOps`: imports `lmcache_ascend` as an optional activation hook, binds `lmcache.npu_ops`; stream-ordered completion/event recording; torch `copy_`-based memcpy |
| `pin_memory.py` | Host-memory pinning via CANN `acl` / `libascendcl` |
