# LMCache Rust Raw Block I/O

This crate provides raw block device I/O operations for LMCache using Rust and PyO3.

## Current Limitations

- Linux only (uses `pread`/`pwrite`, `O_DIRECT` semantics).
- Works with a raw block device path (for example `/dev/nvmeXnY`) or a regular file (test/benchmark mode).
- `O_DIRECT` requires aligned offset, I/O size, and user buffer alignment. Misalignment falls back to non-`O_DIRECT` usage or fails with `EINVAL` depending on call path.
- This crate currently exposes synchronous I/O only.

## Building

```bash
cd rust/raw_block
pip install maturin
maturin develop --release
```

## Features

- Direct block device access with O_DIRECT support
- Synchronous `pread` / `pwrite` only (no `preadv`/`pwritev`)
- No async I/O; `py.allow_threads` releases the GIL but still blocks the OS thread

## Python <-> Rust Zero-Copy Path (for O_DIRECT)

```text
LocalCPUBackend (aligned/pinned CPU tensor memory)
            |
            |  (Python passes memory view / buffer reference)
            v
RustRawBlockBackend (Python plugin)
            |
            |  (PyO3 borrows underlying buffer; no payload memcpy in Python)
            v
Rust RawBlockDevice::pwrite_from_buffer()
            |
            |  (single payload write to device/file, aligned for O_DIRECT)
            v
Block device or file
```

Notes:

- "Zero-copy" here means no extra payload copy in Python between LMCache tensor memory and Rust I/O submission.
- Rust may still do bounded handling for alignment metadata/header and OS/kernel still performs device-level I/O work.
- End-to-end gain depends on storage media latency/bandwidth and queue depth.

## Usage

```python
from lmcache_rust_raw_block_io import RawBlockDevice

# Open device (path, writable, use_odirect=False, alignment=4096)
dev = RawBlockDevice("/dev/nvme0n1", True, use_odirect=True)

# Write data
dev.pwrite_from_buffer(offset=0, data=b"hello", total_len=4096)

# Read data
buf = bytearray(4096)
dev.pread_into(offset=0, out=buf, payload_len=5, total_len=4096)
```

## LMCache allocator alignment knobs (for O_DIRECT zero-copy path)

When using the Rust raw block storage plugin from LMCache v1,
`LocalCPUBackend` can auto-align pinned allocations to improve direct
O_DIRECT compatibility for zero-copy submission.

`extra_config` keys:

- `rust_raw_block.use_odirect` (bool, default: `False`)
- `rust_raw_block.block_align` (int, default: `4096`)
- `rust_raw_block.align_local_cpu_allocator` (bool, default: `True`)
- `local_cpu.pinned_align_bytes` (optional int, explicit override)

Behavior:

- If `rust_raw_block.use_odirect=True` and `rust_raw_block.device_path` is set,
  LMCache auto-uses `rust_raw_block.block_align` for `LocalCPUBackend`
  pinned allocations (unless disabled by
  `rust_raw_block.align_local_cpu_allocator=False`).
- `local_cpu.pinned_align_bytes` has highest priority and overrides auto mode.
- Alignment values must be positive power-of-two values.

Note:

- This applies only when `LocalCPUBackend` builds its own allocator.
  If you inject a custom allocator (for tests/benchmarks), those objects follow
  that allocator's alignment behavior.
