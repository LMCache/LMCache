# Ascend NPU Device Backend

`lmcache/v1/platform/devices/npu/` wires Ascend NPU (`torch.npu` from
`torch_npu`) into the platform device registry. All NPU-specific kernels live
in the external `lmcache-ascend` plugin; this package only wires detection and
backend selection.

| Module | Responsibility |
|---|---|
| `__init__.py` | `NpuDeviceSpec` registry entry |
| `ipc_wrapper.py` | `NpuIPCWrapper`: plane-aggregating KV IPC wrapper over torch_npu storage IPC (`_share_npu_` / `_new_shared_npu`); one wrapper per layer, registered form preserved across the wire |
| `event_ipc.py` | `NpuEventIPCBackend`: shared CUDA-style event ABI adapter with device-pinned export |
| `cache_context.py` | `NpuCacheContext`: device pointer tables, `_TempNpuBuffer` staging, transfer stream |
| `device_ops.py` | `NpuDeviceOps`: binds the plugin's `lmcache_ascend.ops`; stream-ordered completion/event recording; torch `copy_`-based memcpy |
| `pin_memory.py` | Host-memory pinning via CANN `acl` / `libascendcl` |
