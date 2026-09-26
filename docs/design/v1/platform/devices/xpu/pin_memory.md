# XPU Host-Memory Registration

`XpuPinMemoryBackend` makes an existing CPU memory range available for direct
DMA by registering it with the current XPU SYCL context. It is intended for
memory LMCache does not obtain from PyTorch's pinned-memory allocator, such as
shared `mmap` regions used by lazy allocation.

The backend calls `lmcache.xpu_ops.xpu_host_register(ptr, size)` and must be
paired with `unpin_memory(ptr)` before the underlying allocation is released.
The native implementation uses
`sycl::ext::oneapi::experimental::prepare_for_device_copy` and
`release_from_device_copy`; registration applies to the current XPU context.

The `PinMemoryBackend` `flags` argument is accepted for cross-platform
compatibility but has no SYCL equivalent and is ignored. Missing native
extensions, unsupported SYCL runtimes, invalid ranges, and native failures
return `False`. Callers must retain their synchronous-copy fallback in each
case.

`XpuDeviceSpec` probes the native extension before selecting this backend. If
the backend cannot initialize, the specification returns `None` and the common
platform layer instantiates its no-op `PinMemoryBackend` fallback instead.
Direct construction of `XpuPinMemoryBackend` remains safe: an unavailable
extension makes `is_pin_supported` and registration operations return `False`.
