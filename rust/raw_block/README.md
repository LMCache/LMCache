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
