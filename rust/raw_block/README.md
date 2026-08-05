# LMCache Rust Raw Block I/O

This crate provides the low-level raw device I/O layer for LMCache via Rust +
PyO3. It is used by both:

- the legacy non-MP `RustRawBlockBackend`
- the MP `raw_block` L2 adapter (`RawBlockL2Adapter`) via `RawBlockCore`

The Rust crate intentionally stays narrow: it owns the raw device handle and
exposes blocking `pwrite_from_buffer` / `pread_into` primitives. Slotting,
checkpointing, recovery, and MP task orchestration all live in Python.

## I/O Engines

`RawBlockDevice` accepts `io_engine`:

- `posix` (default): synchronous Linux `pread` / `pwrite`.
- `io_uring`: direct Rust io_uring syscall path using the existing worker,
  batch, and `wait_iouring` machinery.
- `libblkio` (requires `blkio` cargo feature): delegates I/O to
  [libblkio](https://gitlab.com/libblkio/libblkio), which manages its own
  `io_uring` instance internally.

`use_iouring=True` remains accepted for backward compatibility. If `io_engine`
is explicitly set, it wins over the legacy flag.

## io_uring_cmd (NVMe Passthrough)

When `io_engine="io_uring"`, you can optionally enable `use_uring_cmd=True` to
use NVMe passthrough via the io_uring command interface for direct device access.

**io_uring_cmd notes:**

- Requires NVMe character device node (`/dev/ngXnY`) instead of the block device
  node (`/dev/nvmeXnY`) for direct NVMe passthrough command.
- Requires `io_engine="io_uring"` to be set.
- Supports `max_data_transfer_size` parameter to split large transfers into
  smaller chunks that fit within device limits.
- Requires `alignment` to be a multiple of the NVMe namespace LBA size. An
  incompatible value is rejected when the device opens.
- When `use_uring_cmd=True`, `use_odirect` is ignored for NVMe namespace
  character devices.
- SQE build failures are returned by `wait_iouring` after the worker releases
  the request's global and per-batch in-flight accounting.

## MP Mode Integration

In MP mode, the stack looks like this:

```text
StoreController / PrefetchController
                |
                v
        RawBlockL2Adapter
                |
                v
           RawBlockCore
                |
                v
         lmcache_rust_raw_block_io
                |
                v
         raw device / file
```

This split lets LMCache reuse the same on-device metadata and recovery model in
both non-MP and MP mode without duplicating the raw-block implementation.

## Zero-Copy Data Path

```text
LMCache LocalCPUBackend (aligned pinned CPU tensor)
                 |
                 |  Python buffer / memoryview (no payload memcpy)
                 v
RustRawBlockBackend (PyO3 boundary)
                 |
                 |  direct pointer path when O_DIRECT constraints are met
                 |  fallback: bounce only for unaligned tail/block
                 v
RawBlockDevice::pwrite_from_buffer / pread_into
                 |
                 v
Block device or file
```

## How To Compare Performance

To compare `local_disk` vs `rust_raw_block` on a real NVMe device:
- Run `local_disk` on an ext4 mount of the device.
- Unmount it.
- Run `rust_raw_block` directly on the raw block device.

Use the benchmark commands in:
- `benchmarks/storage_backend_io/README.md`

No fixed numbers are included here because results are host/device/workload dependent.

## Limitations

- Linux only (`pread` / `pwrite`, O_DIRECT semantics).
- O_DIRECT requires aligned offset, size, and user buffer address.
- io_uring backend requires Linux kernel 5.1+.

## libblkio I/O Engine

The crate can optionally link against
[libblkio](https://gitlab.com/libblkio/libblkio) to provide a
`libblkio` I/O engine for `RawBlockDevice`.  libblkio manages its own
`io_uring` instance internally, so the Python-side io_uring worker thread
is not used.

### When to use which engine

| Engine | Best for |
|--------|----------|
| `posix` (default) | Simple synchronous I/O; widest compatibility |
| `io_uring` | High-throughput NVMe I/O; full io_uring batching and fixed-buffer support via the Rust worker thread |
| `libblkio` | Environments that already depend on libblkio (e.g. NIXL integration); single-queue synchronous I/O via the libblkio `io_uring` driver |

### Prerequisites

Install `libblkio` development headers:

```bash
# Ubuntu/Debian
sudo apt-get install libblkio-dev

# Verify
pkg-config --exists blkio && echo "libblkio found"
```

### Building with the `blkio` feature

```bash
cd rust/raw_block
pip install maturin
maturin develop --release --features blkio
```

Without `--features blkio`, only `posix` and `io_uring` engines are
available; `io_engine="libblkio"` will raise `ValueError`.

### Cargo feature flag

```toml
[features]
default = []
blkio = []          # links against libblkio via pkg-config
```

`build.rs` uses `pkg-config` to locate libblkio when the `blkio`
feature is active.  If pkg-config is not available, it falls back to
`-lblkio` in the system library path.

### Usage

```python
from lmcache_rust_raw_block_io import RawBlockDevice

dev = RawBlockDevice(
    "/dev/nvme0n1",
    writable=True,
    use_odirect=True,
    alignment=4096,
    io_engine="libblkio",
)
print(dev.size_bytes())

data = bytearray(4096)
dev.pwrite_from_buffer(offset=0, data=data, payload_len=100, total_len=4096)

out = bytearray(4096)
dev.pread_into(offset=0, out=out, payload_len=100, total_len=4096)

dev.close()
```

When `io_engine="libblkio"`, `RawBlockDevice` supports the same
synchronous methods (`pwrite_from_buffer`, `pread_into`, `size_bytes`,
`close`) but does **not** support the async io_uring batch methods
(`batched_write`, `batched_read`, `wait_iouring`, `register_fixed_buffers`).

An optional `blkio_driver` parameter selects the libblkio driver (default:
`"io_uring"`).

### Selecting the engine from `RustRawBlockBackend`

Set `rust_raw_block.io_engine` in the plugin's `extra_config`:

```yaml
extra_config:
  rust_raw_block.device_path: "/dev/nvme0n1"
  rust_raw_block.io_engine: "libblkio"
  rust_raw_block.use_odirect: true
  # rust_raw_block.blkio_driver: "io_uring"   # optional, defaults to "io_uring"
```

### Testing

```bash
# All libblkio engine tests (smoke + integration; no device needed)
pytest -xvs tests/v1/storage_backend/test_blkio_block_device.py

# With O_DIRECT on a real block device or loopback
LMCACHE_BLKIO_TEST_DEVICE=/dev/loop0 \
    pytest -xvs tests/v1/storage_backend/test_blkio_block_device.py
```

| Test class | Count | Coverage |
|-----------|-------|---------|
| `TestBlkioBlockDeviceSmoke` | 9 | Open/close, read/write roundtrip, padding, error handling |
| `TestBlkioRawBlockBackendIntegration` | 4 | Put/get, batched get, eviction, checkpoint recovery |
| `TestBlkioBlockDeviceODirect` | 4 | O_DIRECT roundtrip, large buffer, padding, multi-offset |
| `TestBlkioRawBlockBackendODirect` | 1 | Full backend put/get with O_DIRECT |

## io_uring Dependencies

The io_uring backend requires specific kernel configuration and versions:

### Kernel Version

- **Minimum version**: Linux kernel 5.1+
- **Recommended version**: Linux kernel 5.19+ for full feature support

### Kernel Configuration

The following kernel configuration options must be enabled:

```
CONFIG_IO_URING=y
```

To check if io_uring is enabled on your system:

```bash
# Check kernel config
grep -i uring /boot/config-$(uname -r)

# Or check the presence of io_uring setup function in kernel's symbol table
grep io_uring_setup /proc/kallsyms
```

### Rust io-uring Crate

- **Crate version**: `io-uring = "0.7"`
- **Source**: [io-uring crate on crates.io](https://crates.io/crates/io-uring)

The crate provides safe Rust bindings to the Linux io_uring API and is included in the project's `Cargo.toml`:

```toml
[dependencies]
io-uring = "0.7"
```
- `alignment` and `block_align` must be powers of two, such as 4096.
- O_DIRECT requires aligned offsets and I/O lengths. `batched_write` rejects
  requests whose offset or `total_len` is not a multiple of the configured
  alignment; misaligned write buffers are copied through an aligned bounce
  buffer.

## Build

```bash
cd rust/raw_block
pip install maturin
maturin develop --release
```

## Minimal Usage

```python
from lmcache_rust_raw_block_io import RawBlockDevice

dev = RawBlockDevice("/dev/nvme0n1", True, use_odirect=True, alignment=4096)
dev.pwrite_from_buffer(offset=0, data=b"hello", total_len=4096)

buf = bytearray(4096)
dev.pread_into(offset=0, out=buf, payload_len=5, total_len=4096)
```

io_uring:

```python
dev = RawBlockDevice(
    "/dev/nvme0n1",
    True,
    use_odirect=True,
    alignment=4096,
    io_engine="io_uring",
    iouring_queue_depth=256,
)
```

io_uring with io_uring_cmd (NVMe passthrough):

```python
dev = RawBlockDevice(
    "/dev/ng0n1",  # Note: NVMe character device node
    True,
    use_odirect=False,
    alignment=4096,
    io_engine="io_uring",
    use_uring_cmd=True,
    iouring_queue_depth=256,
    max_data_transfer_size=131072,  # Optional: split large transfers
)
```

## FDP Notes

FDP status can be queried from Python when `use_uring_cmd=True`:

```python
status = dev.fetch_fdp_status()  # [(placement_id, ruh_id), ...]
```

For writes, omitting `placement_id` leaves the NVMe directive unset
(`dtype=0, dspec=0`). NVMe default writes use the RUH mapping associated with
Placement Identifier 0, so LMCache rejects explicit `placement_id=0` at the
`RawBlockCore` layer. The low-level status query reads the controller-reported
16-bit `NRUHSD` count.

## MP Adapter Example

To use the MP adapter from `lmcache server`, pass a `raw_block` L2 adapter
config:

```bash
lmcache server \
  --l1-size-gb 10 \
  --eviction-policy LRU \
  --l1-align-bytes 4096 \
  --l2-adapter '{
    "type": "raw_block",
    "device_path": "/dev/nvme0n1",
    "slot_bytes": 1048576,
    "block_align": 4096,
    "header_bytes": 4096,
    "meta_total_bytes": 268435456,
    "use_odirect": true,
    "io_engine": "io_uring",
    "num_store_workers": 2,
    "num_lookup_workers": 1,
    "num_load_workers": 4
  }'
```

Notes:

- `device_path` should point to an unmounted raw block device or a dedicated
  file used only by LMCache.
- For `use_uring_cmd=true`, `device_path` must use the NVMe character
  device node (e.g., `/dev/ng0n1`) instead of the block device node.
- `block_align` must be a power of two. `slot_bytes`, `header_bytes`, and
  `meta_total_bytes` must be multiples of `block_align`.
- With `use_odirect=true`, LMCache MP L1 alignment must be at least
  `block_align`.
- Restart recovery uses the metadata checkpoint region on the same device.
- Raw-block slot reclamation is driven by the shared/global L2 eviction
  controller or explicit `delete()` calls.
- `raw_block` remains the adapter type for all supported engines.
