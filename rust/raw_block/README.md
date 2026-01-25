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
- Batch read/write operations (preadv/pwritev)
- Optional asynchronous I/O via multi-process mode

## Usage

```python
from lmcache_rust_raw_block_io import RawBlockDevice

# Open device
dev = RawBlockDevice("/dev/nvme0n1", use_odirect=True)

# Write data
dev.pwrite(offset=0, data=b"hello", total_len=4096)

# Read data
buf = bytearray(4096)
dev.pread_into(offset=0, buf=buf, length=5, total_len=4096)
```

