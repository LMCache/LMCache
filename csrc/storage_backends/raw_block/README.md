# LMCache SPDK I/O Engine

This directory contains the C++ implementation of the SPDK-based I/O engine for LMCache raw block storage. It provides zero-copy DMA read/write operations over NVMe-oF (NVMe over Fabrics) and local PCIe NVMe devices using the [SPDK](https://spdk.io/) and [DPDK](https://dpdk.org/) frameworks.

## Overview

The SPDK I/O engine (`liblmcache_spdk.so`) wraps the `SpdkIoEngineCore` C++ class and exposes a C-compatible ABI for Python FFI via [`spdk_ffi.py`](../../lmcache/v1/storage_backend/raw_block/spdk_ffi.py). It is one of the supported `io_engine` values for `RawBlockCore`:

- `posix` — synchronous `pread`/`pwrite` through the Rust raw block layer
- `io_uring` — direct io_uring path with optional NVMe passthrough
- **`spdk`** — SPDK/DPDK-based NVMe-oF or PCIe NVMe with DMA-safe memory

When `io_engine="spdk"` is set in the config, `RawBlockCore` routes all I/O through SPDK instead of the Rust `RawBlockDevice`.

## Architecture

```text
RawBlockCore (io_engine="spdk")
                |
                v
    SpdkIoEngineFFI (ctypes)
                |
                v
    liblmcache_spdk.so (C++ SpdkIoEngineCore)
                |
                v
    ┌───────────┴───────────┐
    v                       v
    NVMe-oF(TCP / RDMA) PCIe (local NVMe)
        |                       |
        v                       v
    NVMe-oF target      PCIe device (e.g., 0000:01:00.0)
```

### Thread Model

SPDK runs with **two pinned worker threads**:

| Thread | Purpose | Core Affinity |
|--------|---------|---------------|
| `io_worker` | Connects to NVMe device, processes I/O operations via `rte_ring` | `LMCACHE_IO_WORKER_CORE` or derived from core mask |
| `admin_worker` | Processes admin command completions (NVMe-oF only) | `LMCACHE_ADMIN_WORKER_CORE` or derived from core mask |

The cores are resolved in priority order:
1. Environment variables `LMCACHE_IO_WORKER_CORE` and `LMCACHE_ADMIN_WORKER_CORE`
2. DPDK core mask (`spdk_core_mask`) — I/O worker gets the highest core, admin gets the second-highest
3. Defaults: I/O worker = total_cores - 1, Admin worker = total_cores - 1

## I/O Engines

### NVMe-oF (TCP)

Connects to a remote NVMe-over-Fabrics target over TCP:

```python
ffi.launch_io_worker(
    transport_type="tcp",
    addr="192.168.1.1",
    port="1100",
    nqn="nqn.2016-06.io.spdk:cnode1"
)
```

### Local PCIe NVMe

Connects directly to a local NVMe device:

```python
ffi.launch_io_worker(
    transport_type="pcie",
    addr="0000:01:00.0"
)
```

PCIe mode does **not** launch the admin worker thread, as admin commands are handled synchronously.

## Zero-Copy Data Path

```text
LocalCPUBackend (hugepage-backed pinned tensor)
                 |
                 |  memoryview (no payload memcpy)
                 v
    RawBlockCore._write_spdk_buffers()
                 |
                 |  buffer registered via register_spdk_external_buffers()
                 v
    SpdkIoEngineFFI.spdk_write_external()
                 |
                 v
    spdk_dma_zmalloc (DMA pool for headers)
                 |
                 v
    NVMe device (via SPDK)
```

### External Memory Registration

For zero-copy writes, the LocalCPUBackend's hugepage-allocated buffer must be registered with SPDK:

```python
# In RawBlockCore
core.register_spdk_external_buffers(memory_allocator)

# In RustRawBlockBackend (MP mode)
core.register_spdk_external_buffers(
    self.local_cpu_backend.get_memory_allocator()
)
```

Unregistered buffers are copied through a temporary DMA allocation (`spdk_dma_zmalloc`) before the I/O operation, which is then freed after completion.

## DMA Buffer Pools

SPDK mode uses two pre-allocated DMA buffer pools to avoid per-operation allocations:

| Pool | Purpose | Size |
|------|---------|------|
| `HeaderBufferPool` | Metadata header writes | 32 x `block_align` |
| `CheckPointPayloadBufferPool` | Checkpoint payload reads | 2 x `_meta_payload_capacity()` |

Both pools use `spdk_dma_zmalloc` for cache-line-aligned, DMA-safe memory.

## Build

### Prerequisites

- SPDK built and installed (default: `/opt/spdk`)
- DPDK built and installed (default: `$SPDK_ROOT/dpdk`)
- CMake >= 3.14
- C++17 compiler (gcc/clang)
- isa-l and isa-l-crypto libraries (inside SPDK tree)

### Quick Build

```bash
cd csrc/storage_backends/raw_block
./build_spdk.sh
```

This builds `liblmcache_spdk.so` and copies it to `lmcache/v1/storage_backend/raw_block/`.

### Manual Build

```bash
cd csrc/storage_backends/raw_block
mkdir -p build_spdk && cd build_spdk
cmake .. \
    -DSPDK_ROOT=/path/to/spdk \
    -DDPDK_ROOT=/path/to/dpdk \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Script Options

```bash
# Specify paths as arguments
./build_spdk.sh /home/user/spdk /home/user/dpdk
```

### CMake Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPDK_ROOT` | `/opt/spdk` | SPDK installation directory |
| `DPDK_ROOT` | `$SPDK_ROOT/dpdk` | DPDK build directory |
| `LMCACHE_SPDK_INSTALL_PREFIX` | `/usr/local` | Install destination |
| `CMAKE_BUILD_TYPE` | `Release` | Build type |

## Python Configuration

SPDK mode is enabled via `io_engine="spdk"` in the config:

```python
config = RawBlockCoreConfig(
    device_path="",  # Not required for SPDK (device_path is optional when io_engine="spdk")
    slot_bytes=1048576,
    block_align=4096,
    header_bytes=4096,
    meta_total_bytes=268435456,
    io_engine="spdk",
    # SPDK-specific options
    spdk_transport_type="tcp",      # "pcie" or "tcp"
    spdk_target_ip="127.0.0.1",     # IP for TCP, PCIe addr for PCIe
    spdk_target_port="4420",        # Port (ignored for PCIe)
    spdk_target_nqn="nqn.2016-06.io.spdk:cnode1",  # NQN (ignored for PCIe)
    spdk_core_mask="",              # Hex core mask (e.g., "0x3f")
)
```

### Config Keys Reference

| Key | Default | Description |
|-----|---------|-------------|
| `io_engine` | `"posix"` | I/O engine: `"posix"`, `"io_uring"`, or `"spdk"` |
| `rust_raw_block.spdk_transport_type` | `"tcp"` | `"pcie"` or `"tcp"` |
| `rust_raw_block.spdk_target_ip` | `"127.0.0.1"` | TCP IP or PCIe address |
| `rust_raw_block.spdk_target_port` | `"4420"` | NVMe-oF target port |
| `rust_raw_block.spdk_target_nqn` | `"nqn.2016-06.io.spdk:cnode1"` | NVMe Qualified Name |
| `rust_raw_block.spdk_core_mask` | `""` | DPDK core mask (hex, e.g., `"0x3f"`) |

## Limitations

- Linux only (SPDK/DPDK requirement).
- Requires NVMe device (local PCIe or remote NVMe-oF target).
- Hugepages are **mandatory** for SPDK mode — `LocalCPUBackend` forces `use_hugepages=True` when `use_spdk=True` and aligns `cpu_size_bytes` to 2 MiB hugepage boundaries.
- `spdk_core_mask` controls which CPU cores SPDK pollers run on.
- I/O worker and admin worker thread affinity can be set via environment variables.
- The SPDK library is loaded from `lmcache/v1/storage_backend/raw_block/liblmcache_spdk.so` by default.
- NVMe-oF TCP mode requires network connectivity to the target.
- PCIe mode requires the device to be detached from the kernel driver (e.g., `vfio-pci`).

## Build Dependencies

The CMake build links the following libraries:

- SPDK static libraries (`libspdk_*.a`) with `--whole-archive`
- DPDK static libraries (`librte_*.a`) with `--whole-archive`
- isa-l (`libisal.a`) and isa-l-crypto (`libisal_crypto.a`)
- System libraries: `pthread`, `dl`, `rt`, `numa`, `keyutils`, `aio`, `uuid`, `fuse3`, `gcc_s`, `crypto`, `ssl`

Excluded SPDK libraries:
- `libspdk_ut.a` — Unit test framework
- `libspdk_ut_mock.a` — Unit test mocks
- `libspdk_fuse_dispatcher.a` — FUSE dispatcher (incompatible with FUSE 3)

