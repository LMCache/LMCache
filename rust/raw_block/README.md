# LMCache Rust Raw Block I/O

This crate provides raw block device I/O operations for LMCache using Rust and PyO3.

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
- `local_cpu.enable_experimental_lazy_allocator` (bool, default: `False`)
- `local_cpu.lazy_init_size_gb` (float, optional, only for experimental lazy mode)

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
