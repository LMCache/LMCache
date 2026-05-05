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

`use_iouring=True` remains accepted for backward compatibility. If `io_engine`
is explicitly set, it wins over the legacy flag.

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
- With `use_odirect=true`, LMCache MP L1 alignment must be at least
  `block_align`.
- Restart recovery uses the metadata checkpoint region on the same device.
- Raw-block slot reclamation is driven by the shared/global L2 eviction
  controller or explicit `delete()` calls.
- `raw_block` remains the adapter type for both supported engines.
