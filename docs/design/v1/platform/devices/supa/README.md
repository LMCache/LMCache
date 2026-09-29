# Biren SUPA Device Backend

LMCache device backend for Biren GPUs (BR1xx, e.g. Biren166M) exposed
through Biren's SUPA runtime. It plugs into the unified device registry
described in [`../../device-plugin-architecture.md`](../../device-plugin-architecture.md)
and [`../../device_ops_design.md`](../../device_ops_design.md).

## Torch integration

Biren ships the `torch_br` package. Importing it bridges PyTorch's
`PrivateUse1` backend to SUPA:

```python
torch.utils.rename_privateuse1_backend("supa")
torch._register_device_module("supa", torch_br.supa)
```

After that, `torch.device("supa")` and `torch.supa` exist process-wide.
`torch.supa` mirrors the CUDA module surface: `is_available`,
`device_count`, `current_stream`, `synchronize`, `Stream`, `Event` (with
`ipc_handle()` / `from_ipc_handle()`), and memory statistics.

Because a bare `import torch` does **not** register the SUPA backend,
`SupaDeviceSpec.is_available()` imports `torch_br` when `torch.supa` is
absent. A missing runtime or driver is caught and reported as
"unavailable" so LMCache still starts on non-Biren hosts.

## DeviceSpec contract

| Property | Value |
| --- | --- |
| `device_type` | `"supa"` |
| `torch_module_name` | `"supa"` |
| `ops_cls` | `SupaDeviceOps` (torch baseline, no overrides) |
| `is_available()` | imports `torch_br` if needed, then `torch.supa.is_available()` |
| `is_handle_transfer_available()` | `False` |
| `ipc_wrapper_cls` / `event_ipc_backend` | `None` |
| `pin_memory_backend` | `None` (default no-op backend) |

The spec is auto-discovered from `lmcache/v1/platform/devices/supa`; no
manual registration is required. Operators can force it with
`LMCACHE_DEVICE_BACKEND=supa` or `DEVICE_TYPE=supa`.

## Capability surface

`SupaDeviceOps` inherits every op verbatim from `DeviceOps`. The
pure-Python torch baseline in `lmcache/v1/platform/torch_ops` is
device-agnostic (it never calls `torch.cuda.*` directly), so it runs on
SUPA through the `torch.supa` module resolved by `get_torch_device()`.

This backend currently targets the **engine-driven** multiprocess path.
`mp_transfer_mode=lmcache_driven` is not enabled because it needs a KV IPC
handle wrapper and a cross-process event IPC backend:

- SUPA `Event` exposes `ipc_handle()` / `from_ipc_handle()`, so an
  `event_ipc_backend` is feasible.
- Tensor IPC sharing (`_new_shared_filename` / shared-memory descriptors) is
  not available for SUPA storage yet, so a `DeviceIPCWrapper` would need a
  SUPA-specific handle or a staging copy.

Returning `is_handle_transfer_available() == False` makes the unsupported
path fail at its documented validation point instead of crashing later.

## Pin memory

`torch_br.supa` does not expose a host-pinning API, so the default
`PinMemoryBackend` (no-op, `is_pin_supported == False`) applies. Pinned host
staging can be added later if Biren exposes a registration primitive.

## Validation

Tests live in `tests/v1/platform/devices/supa/test_supa_support.py`:

- Contract and registry tests run on any host (no Biren hardware needed).
- A `@pytest.mark.supa` test verifies on Biren hardware that
  `torch.supa` is registered, `is_available()` is true, and a tensor created
  on `supa:0` reports `device.type == "supa"`.
