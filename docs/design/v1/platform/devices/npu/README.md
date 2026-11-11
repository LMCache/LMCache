# Ascend NPU Device Backend

`lmcache/v1/platform/devices/npu/` wires Ascend NPU (`torch.npu` from
`torch_npu`) into the platform device registry. The MP native core (host
bindings, recorders, pinned allocation and the Ascend C block-transfer kernel)
lives in `csrc/ascend/` and is built in-tree as `lmcache.npu_ops`; the external
`lmcache-ascend` plugin keeps SP and transfer-channel extensions and remains an
optional activation hook for hardware-specific patches.

| Module | Responsibility |
|---|---|
| `__init__.py` | `NpuDeviceSpec` registry entry |
| `ipc_wrapper.py` | `NpuIPCWrapper`: plane-aggregating KV IPC wrapper over torch_npu storage IPC (`_share_npu_` / `_new_shared_npu`); one wrapper per layer, registered form preserved across the wire |
| `event_ipc.py` | `NpuEventIPCBackend`: CUDA-style event ABI adapter with a runtime constructor check and device-pinned export |
| `cache_context.py` | `NpuCacheContext`: device pointer tables, `_TempNpuBuffer` staging, transfer stream |
| `device_ops.py` | `NpuDeviceOps`: imports `lmcache_ascend` as an optional activation hook, then binds `lmcache.npu_ops`; stream-ordered completion/event recording; torch `copy_`-based memcpy |
| `pin_memory.py` | Host-memory pinning via CANN `acl` / `libascendcl` |

`NpuEventIPCBackend.check_event_support(device)` checks the Python event ABI,
then attempts `Event(interprocess=True)` with the target device current.
torch_npu can expose the IPC API while its constructor rejects the installed
runtime. Construction failures retain the native exception as their cause and
include the target device. The temporary event is released without recording
or exporting it; the check does not test cross-process import or synchronization.
Callers validate during initialization or KV-cache registration and reuse the
backend for transfers. The check does not substitute ordinary events or cache
success across devices.

## Native module

`setup_extensions/build_profiles/ascend.py` builds `lmcache.npu_ops` from
`csrc/ascend/` with CMake and the CANN toolchain. The module registers only its
own ops and the staged-plan types (`StagingCopy`, `LaunchVar`, `BatchStep`).
`PageBufferShapeDesc` and `KernelGroupSpec` come from `csrc/kv_transfer_plan_types.h`
and are registered by `lmcache_native`, which the server uses to build its
plans; the module is therefore compiled against torch's bundled pybind11, the
headers `lmcache_native` is built with, so both share one pybind11 type
registry.
