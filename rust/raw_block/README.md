# LMCache Rust Raw Block I/O

This crate provides raw block I/O for LMCache via Rust + PyO3.

## What Changed vs `origin/dev`

1. `RustRawBlockBackend` can use aligned Python buffer memory directly in O_DIRECT paths (no extra Python-side payload copy on the fast path).
2. O_DIRECT tail handling uses a hybrid path:
   - direct write/read for aligned prefix
   - bounce buffer only for the final padded tail block
3. `LocalCPUBackend` alignment can be auto-driven by rust raw block config for O_DIRECT compatibility:
   - `rust_raw_block.block_align`
   - `rust_raw_block.align_local_cpu_allocator`
   - `local_cpu.pinned_align_bytes` (explicit override)
4. Benchmark harness reliability improvements:
   - skip `truncate()` for real block devices (`/dev/...`)
   - unique manifest per run (avoid stale-index reuse)
   - timeout guard for local disk completion waits (scales with `num_ops`)

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
- Synchronous I/O only (no async kernel interface, no `io_uring` in this crate).
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
