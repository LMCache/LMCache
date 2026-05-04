# LMCache libblkio Storage Backend

This backend provides high-performance block device I/O for KV cache storage
using the [libblkio](https://gitlab.com/libblkio/libblkio) library. libblkio
offers a unified interface to various block device backends; this connector
uses the **io_uring** driver for asynchronous I/O.

The design mirrors the
[NIXL libblkio plugin](https://github.com/ai-dynamo/nixl) — each worker
thread gets its own `io_uring` instance via libblkio, and every I/O operation
follows a map → submit → complete → unmap cycle on the DRAM buffer.

## Dependencies

Install the libblkio development package:

```bash
# Ubuntu/Debian
sudo apt-get install libblkio-dev

# RHEL/CentOS/Fedora
sudo dnf install libblkio-devel

# From source
git clone https://gitlab.com/libblkio/libblkio.git
cd libblkio && meson setup build && ninja -C build && sudo ninja -C build install
```

Verify the install:

```bash
pkg-config --cflags --libs blkio
# Expected output: -I/usr/local/include -lblkio  (paths may vary)
```

## Architecture

```
Non-MP mode:
  CacheEngine → RemoteBackend → BlkioClient → LMCacheBlkioClient (C++)
                                  (asyncio)       ↓
                                            BlkioConnector
                                              ├─ worker 0: blkio(io_uring) → block device
                                              ├─ worker 1: blkio(io_uring) → block device
                                              └─ worker N: blkio(io_uring) → block device

MP mode:
  StoreController / PrefetchController
        ↓
  NativeConnectorL2Adapter (Python bridge)
    ├─ 3 eventfds (store, lookup, load)
    ├─ completion demux thread
    └─ client-side lock tracking
        ↓
  LMCacheBlkioClient (C++)
    └─ BlkioConnector → per-worker io_uring instances
```

### Key design decisions

- **Per-worker io_uring instances**: Each C++ worker thread creates its own
  `struct blkio*` handle.  No shared-queue contention — true parallelism.
- **Map/IO/Unmap per operation**: Each read or write maps the DRAM buffer
  with `blkio_map_mem_region`, submits via `blkioq_read`/`blkioq_write`,
  waits for completion, then unmaps.  This matches the NIXL
  `registerBlkioBuf` → `postXfer` → `unregisterBlkioBufs` pattern.
- **Offset-based keys**: The last `@`-delimited field of the key string
  is a hex-encoded byte offset on the block device.  The Python layer is
  responsible for slot allocation and metadata tracking.
- **O_DIRECT support**: Enabled by passing `direct_io=true` (default).

## Files

| File | Purpose |
|------|---------|
| `csrc/storage_backends/blkio/connector.h` | `BlkioWorkerConn` + `BlkioConnector` class (inherits `ConnectorBase<BlkioWorkerConn>`) |
| `csrc/storage_backends/blkio/connector.cpp` | Implementation: `create_connection`, `do_single_get/set/exists`, `map_do_io_unmap` |
| `csrc/storage_backends/blkio/pybind.cpp` | Pybind11 module `lmcache_blkio` exposing `LMCacheBlkioClient` |
| `lmcache/v1/storage_backend/native_clients/blkio_client.py` | Python async client (`BlkioClient`) wrapping the C++ connector |
| `native_connector_l2_adapter.py` (appended) | `BlkioL2AdapterConfig` + factory for MP mode |

## Building

The blkio extension is built automatically with the rest of LMCache:

```bash
# Standard build (requires torch pre-installed)
pip install -e . --no-build-isolation

# Source-only (skip all C/CUDA extensions)
NO_CUDA_EXT=1 pip install -e .
```

The build links against `-lblkio`.  If libblkio is installed in a
non-standard location, set `CFLAGS` and `LDFLAGS`:

```bash
CFLAGS="-I/path/to/include" LDFLAGS="-L/path/to/lib" pip install -e . --no-build-isolation
```

## Configuration

### Non-MP mode (Python client)

```python
from lmcache.v1.storage_backend.native_clients.blkio_client import BlkioClient

client = BlkioClient(
    device_path="/dev/nvme0n1",   # block device path
    num_workers=4,                 # io_uring instances (default 4)
    direct_io=True,                # O_DIRECT (default True)
)
```

### MP mode (L2 adapter via CLI)

```bash
--l2-adapter '{
    "type": "blkio",
    "device_path": "/dev/nvme0n1",
    "num_workers": 4,
    "direct_io": true
}'
```

### Configuration parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_path` | str | *(required)* | Path to the block device (e.g. `/dev/nvme0n1`, `/dev/loop0`) |
| `num_workers` | int | 4 | Number of C++ worker threads.  Each gets its own io_uring instance |
| `direct_io` | bool | true | Enable `O_DIRECT` to bypass the page cache |

## Testing

### Test files

| File | Type | Needs C++ ext? | Needs device? |
|------|------|----------------|---------------|
| `tests/v1/distributed/test_blkio_l2_adapter.py` | Unit (pytest) | No | No |
| `tests/v1/storage_backend/test_blkio_connector.py` | Integration (pytest) | Yes | Auto-provisioned |

### Running unit tests

These test the `BlkioL2AdapterConfig` parsing, validation, and registry
wiring.  No block device or C++ extension required:

```bash
pytest -xvs tests/v1/distributed/test_blkio_l2_adapter.py
```

### Running integration tests

These exercise the full C++ → libblkio → io_uring → kernel path.
The test fixture auto-provisions a test device using this priority:

1. **`LMCACHE_BLKIO_TEST_DEVICE` env var** — use a real block device
2. **Auto-created loopback device** — requires root (`losetup`)
3. **Sparse temp file** — always available (no `O_DIRECT`)

```bash
# Automatic device provisioning (temp file fallback)
pytest -xvs tests/v1/storage_backend/test_blkio_connector.py

# With a real block device (best for performance validation)
sudo LMCACHE_BLKIO_TEST_DEVICE=/dev/loop0 \
    pytest -xvs tests/v1/storage_backend/test_blkio_connector.py
```

### Setting up a loopback device for testing

```bash
# Create a 64 MB backing file and attach it as a loop device
sudo dd if=/dev/zero of=/tmp/blkio_test.img bs=1M count=64
sudo losetup -f --show /tmp/blkio_test.img
# Note the loop device path (e.g. /dev/loop0)

# Run tests
sudo LMCACHE_BLKIO_TEST_DEVICE=/dev/loop0 \
    pytest -xvs tests/v1/storage_backend/test_blkio_connector.py

# Cleanup
sudo losetup -d /dev/loop0
rm /tmp/blkio_test.img
```

### Running with Docker

Docker may block `io_uring` syscalls by default.  Add them to the
seccomp profile:

```bash
# Download default seccomp profile
wget -O seccomp.json \
    https://raw.githubusercontent.com/moby/moby/master/profiles/seccomp/default.json

# Add to the "syscalls"."names" array in seccomp.json:
#   "io_uring_setup", "io_uring_enter", "io_uring_register"

# Run container with the updated profile
docker run --security-opt seccomp=seccomp.json \
    --device /dev/loop0:/dev/loop0 \
    -it <image>
```

### What the integration tests verify

| Test | Description |
|------|-------------|
| `test_construct_and_close` | Connector can be created and cleanly shut down |
| `test_event_fd_is_valid` | `event_fd()` returns a valid file descriptor |
| `test_write_read_verify` | Write 4 KB of `0xAB`, read back, verify `memcmp` |
| `test_write_read_distinct_patterns` | Write `0x55`, overwrite buffer with `0xAA`, read back confirms `0x55` on device |
| `test_batch_write_read` | Batch write/read 4 blocks with different fill patterns |
| `test_multiple_workers` | Verified with 1, 2, and 4 worker threads |
| `test_sync_set_get_roundtrip` | Python `BlkioClient` sync set → get roundtrip |

## Current Limitations

1. **io_uring only** — libblkio supports `virtio-blk-vhost-user` and
   `virtio-blk-vhost-vdpa` but only `io_uring` is wired up.
2. **No native existence tracking** — `do_single_exists` always returns
   `false`.  The Python layer must track which offsets have been written.
3. **Single device** — the connector opens one block device.  For
   multi-device setups, create multiple connectors.

## Rust Alternative: `BlkioBlockDevice`

libblkio is also available as an I/O backend for the **Rust raw block**
storage plugin (`RustRawBlockBackend`).  The Rust `BlkioBlockDevice`
class lives in the `lmcache_rust_raw_block_io` crate and is enabled via
the `blkio` cargo feature flag:

```bash
cd rust/raw_block
maturin develop --release --features blkio
```

Then set `rust_raw_block.io_backend: "libblkio"` in `extra_config` to
route all I/O through libblkio's io_uring driver instead of the default
Rust pread/pwrite path.

See `rust/raw_block/README.md` for full details, usage examples, and
test instructions.
