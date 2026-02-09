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
4. Optional async batch submission mode in python plugin:
   - `rust_raw_block.enable_async_batch_mode=true`
   - one future submission per batch (instead of one future per key)
5. Benchmark harness reliability improvements:
   - skip `truncate()` for real block devices (`/dev/...`)
   - unique manifest per run (avoid stale-index reuse)
   - timeout/fallback for local disk completion waits

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

## Performance Snapshot (O_DIRECT, 5-run median)

From `benchmarks/storage_backend_io/README.md`:

### Current vs `origin/dev` (median ops/sec)

| Concurrency | local_disk (`origin/dev`) | local_disk (`current`) | Delta | rust_raw_block (`origin/dev`) | rust_raw_block (`current`) | Delta |
|-------------|----------------------------|-------------------------|-------|-------------------------------|----------------------------|-------|
| 2           | 1012.60                    | 1017.26                 | +0.46% | 1913.52                       | 2604.94                    | +36.13% |
| 4           | 783.83                     | 831.70                  | +6.11% | 1659.36                       | 2839.63                    | +71.13% |
| 8           | 672.70                     | 669.05                  | -0.54% | 1793.30                       | 1792.14                    | -0.06% |

### Current branch: rust vs local_disk (median ops/sec)

| Concurrency | local_disk (`current`) | rust_raw_block (`current`) | Rust vs local_disk |
|-------------|-------------------------|----------------------------|--------------------|
| 2           | 1017.26                 | 2604.94                    | +156.08% |
| 4           | 831.70                  | 2839.63                    | +241.43% |
| 8           | 669.05                  | 1792.14                    | +167.86% |

Real block-device smoke (`/dev/nvme1n1p2`, current branch, O_DIRECT):
- local_disk: 2149.71 ops/sec
- rust_raw_block: 3542.56 ops/sec

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
