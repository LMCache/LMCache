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
5. **io_uring support**: Added asynchronous I/O backend using Linux io_uring API:
   - Dedicated worker thread drives the io_uring submission/completion loop
   - Batch write, read support to reduce syscall overhead and improve throughput
   - Fixed buffer registration for true zero-copy I/O operations

## Zero-Copy Data Path

### Synchronous Path (pread/pwrite)

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

### Asynchronous Path (io_uring)

```text
LMCache LocalCPUBackend (aligned pinned CPU tensor)
                 |
                 |  Python buffer / memoryview (no payload memcpy)
                 v
RustRawBlockBackend (PyO3 boundary)
                 |
                 |  enqueue to worker thread queue
                 v
io_uring worker thread (batching & submission)
                 |
                 |  direct pointer path when O_DIRECT constraints are met
                 |  and fixed buffer path for true zero-copy
                 v
io_uring submission queue (kernel)
                 |
                 v
Block device or file
```

```text
Python Thread(s)              Worker Thread (io_uring)
===============               =======================
batched_write() /  --push-->   [queue]
batched_read()                [worker loop]
    |                               |
    |                               v
    |                         process queue
    |                               |
    v                               v
[IoCompletion]  <--signal--   submit to kernel
    |                               |
wait_iouring() GIL-release    completions
    |                               |
    v                               v
wait() non-blocking           wake up waiters
```

### Fixed Buffer Zero-Copy (io_uring)

io_uring with registered fixed buffers:
- Buffers are pre-registered with the kernel via `register_fixed_buffers()`
- Eliminates memory copies between user and kernel space
- Avoids kernel pin/unpin overhead on each I/O operation
- Particularly beneficial for repeated I/O operations on the same buffers

## How To Compare Performance

To compare `local_disk` vs `rust_raw_block` on a real NVMe device:
- Run `local_disk` on an ext4 mount of the device.
- Unmount it.
- Run `rust_raw_block` directly on the raw block device.

Use the benchmark commands in:
- `benchmarks/storage_backend_io/README.md`

No fixed numbers are included here because results are host/device/workload dependent.

## Limitations

- Linux only (`pread` / `pwrite`, O_DIRECT semantics, io_uring).
- O_DIRECT requires aligned offset, size, and user buffer address.
- io_uring backend requires Linux kernel 5.1+.

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

## Build

```bash
cd rust/raw_block
pip install maturin
maturin develop --release
```

## Minimal Usage

### Synchronous I/O (pread/pwrite)

```python
from lmcache_rust_raw_block_io import RawBlockDevice

dev = RawBlockDevice("/dev/nvme0n1", True, use_odirect=True, alignment=4096)
dev.pwrite_from_buffer(offset=0, data=b"hello", total_len=4096)

buf = bytearray(4096)
dev.pread_into(offset=0, out=buf, payload_len=5, total_len=4096)
```

### Asynchronous I/O (io_uring)

```python
from lmcache_rust_raw_block_io import RawBlockDevice

# Create device with io_uring enabled
dev = RawBlockDevice(
    "/dev/nvme0n1",
    writable=True,
    use_odirect=True,
    alignment=4096,
    use_iouring=True
)

buf1 = bytearray(4096)
buf2 = bytearray(4096)
buffer_ptrs = [ctypes.addressof(ctypes.c_char.from_buffer(buf1)), ctypes.addressof(ctypes.c_char.from_buffer(buf2))]
buffer_sizes = [len(buf1), len(buf2)]
dev.register_fixed_buffers(buffer_ptrs, buffer_sizes)

offsets = [0, 4096]
buffers = [buf1, buf2]
lens = [4096, 4096]
# Batched write submit multiple writes at once
batch_id = dev.batched_write(offsets, buffers, lens)

# Wait for all in-flight I/O to complete
dev.wait_iouring(batch_id)

# Single read using io_uring
buf = bytearray(4096)
dev.read_uring(offset=0, data=buf, payload_len=5, total_len=4096)

# Batched read submit multiple reads at once
batch_id = dev.batched_read(offsets, buffers, lens)

# Wait for all in-flight I/O to complete
dev.wait_iouring(batch_id)

dev.close()
```
