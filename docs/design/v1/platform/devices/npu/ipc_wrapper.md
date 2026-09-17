# NPU Platform: Plane-Aggregating IPC Wrapper

`lmcache/v1/platform/devices/npu/ipc_wrapper.py` ships worker KV caches across the
multiprocess wire on Ascend NPUs. It is the platform's implementation of
`DeviceSpec.ipc_wrapper_cls` (the class `resolve_kv_wrapper_factory` returns
for `device_type="npu"`), and it moved here from the LMCache-Ascend plugin's
`lmcache_ascend/v1/multiprocess/custom_types.py` so the binding needs no
plugin-side patch.

## Why it does not subclass `CudaIPCWrapper`

It subclasses `DeviceIPCWrapper` directly because Ascend differs from CUDA in
the storage-sharing mechanism:

| Aspect | CUDA | Ascend |
|---|---|---|
| Storage sharing | `UntypedStorage._share_cuda_()` / `_new_shared_cuda` | `_share_npu_()` / `_new_shared_npu` |

Device identity needs no Ascend-specific handling: torch_npu populates a
per-chip UUID in `get_device_properties(i).uuid` (a 16-logical-device
dual-die host reports 16 distinct UUIDs, stable across processes), so the
base-class `_get_device_uuid` / `_discover_devices` apply unchanged.

## Multi-plane aggregation (the wire contract)

Engines may register a layer as **one tensor or a sequence of paged planes**:
vLLM-Ascend's model runner hands per-layer `(K, V)` pairs, MLA
`(latent, rope)` / DSA `(latent, rope, dsa)` tuples, arity-1 `(k,)` tuples,
and plane *lists* from `_adjust_kv_layout` / Mamba state tensors
(`vllm_ascend/worker/model_runner_v1.py::_reshape_kv_cache_tensors`).

The wrapper is **plane-aggregating**: one wrapper per registered layer value,
one `PlaneRecord` `(handle, dtype, shape, stride, storage_offset)` per plane
inside it. On the wire (`KVCache = list[DeviceIPCWrapper]`) the list element
count therefore equals the **layer** count, and `to_tensor()` restores the
registered form faithfully: the bare tensor, or the tuple of planes — a
1-element sequence reconstructs as a 1-tuple, not a collapsed bare tensor.
The server-side `normalize_and_discover_per_layer_formats` therefore sees
exactly the per-layer tensor-or-tuple entries the engine registered.

This exercises the documented **multi-plane exception** of
`DeviceIPCWrapper` (see its class docstring): the singular interface fields
are not populated; equality compares `_plane_records` plus the bare-vs-
sequence form flag instead. `NpuIPCWrapper` is currently the only
implementation of the exception; generic code must not assume
`to_tensor()` returns a bare tensor without checking the device.

Because the round-trip is lossless, worker-side format discovery
(`create_engine_group_infos_from_vllm`) needs no canonicalization of its
own: it passes the registered values to detection as-is (converting the
engine dict to a positional list only), so both sides classify identically
— arity-1 single-head tuples as the `NP == 1` case of
`NL_X_NP_X_NB_BS_ONE_HS`.

## Dispatch

`wrap_one_kv_cache` resolves the device from the value itself via
`gpu_connector.utils.get_device` (first-tensor descent), so a plane sequence
dispatches to its device's factory without any device-specific branch in
generic code. The import is function-local: the platform layer must not gain
a module-level dependency on the `gpu_connector` package (its `__init__`
pulls the heavy connector modules).

## Testing split

- CPU (upstream, `tests/v1/platform/devices/npu/test_npu_ipc_wrapper.py`): registry
  binding, `wrap()` arity bookkeeping (storage sharing monkeypatched),
  equality/hash, pickle round-trip.
- Device (upstream, same file, `requires_npu`-gated): real `_share_npu_` /
  `_new_shared_npu` spawn round-trip asserting bare-vs-1-tuple form
  fidelity. The LMCache-Ascend plugin keeps broader multi-plane spawn
  coverage in `tests/v1/multiprocess/test_custom_types.py`.
