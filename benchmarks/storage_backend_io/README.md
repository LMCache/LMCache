# Storage Backend I/O Benchmark

This microbenchmark compares **LocalDiskBackend** vs **RustRawBlockBackend** under high write-concurrency.
For **RustRawBlockBackend** this also supports write and read performance testing for **posix** and **io_uring** backends.

## What It Measures

### Write Benchmark
- Total time to submit and complete `num_ops` write (put) operations
- Effective ops/sec under concurrent submission

### Read Benchmark (`--backend rust_raw_block_read`)
- Write phase: Time to write `num_ops` memory objects to the raw block device
- Read phase: Time to read back all `num_ops` objects using concurrent batched blocking reads
- Data integrity verification (optional with `--verify-integrity`)
- Separate metrics for write and read performance

## Usage

```bash
# Both backends (local disk + raw block)
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 512 \
  --concurrency 32 \
  --backend both \
  --local-disk-dir /tmp/lmcache_local_disk_bench \
  --max-local-disk-gb 2 \
  --raw-device /dev/nvme0n1 \
  --raw-odirect \
  --output-json /tmp/storage_backend_io.json
```

### Notes

- If `--raw-device` is not provided, the benchmark creates `raw_block.bin` in the same `--local-disk-dir` so both backends use the same filesystem.
- This is safe but **not** representative of true raw block performance.
- If `--raw-device` points to a real block device (`/dev/...`), the benchmark does not call `truncate()` on that path.
- `--raw-odirect` should only be used with a real block device that supports O_DIRECT.
- When `--local-disk-odirect` is enabled, the benchmark allocates **page-aligned** buffers to avoid EINVAL from O_DIRECT.
- Local disk backend uses its internal worker pool; completion is tracked via callbacks.
- Read benchmark (`rust_raw_block_read`) performs a write phase followed by a read phase, measuring both separately.
- Use `--verify-integrity` with read benchmark to ensure data correctness (compares read data with original written data).

## How To Compare On Real NVMe

Use the same physical device for both tests:
- local_disk on an ext4 mount
- rust_raw_block on the raw block device (unmounted)

Example parameters:
- `num_ops=65536`
- `concurrency=4`
- `--local-disk-odirect`
- `--raw-odirect`

### 1) Run local_disk on ext4

```bash
# WARNING: mkfs will erase the target device.
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /mnt/local_disk_mount
sudo mount -t ext4 /dev/nvme1n1 /mnt/local_disk_mount
sudo chown "$USER":"$USER" /mnt/local_disk_mount

python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 65536 \
  --concurrency 4 \
  --backend local_disk \
  --local-disk-dir /mnt/local_disk_mount/lmcache_local_disk_bench \
  --max-local-disk-gb 120 \
  --local-disk-odirect \
  --output-json /tmp/local_disk_ext4.json
```

### 2) Run rust_raw_block on raw device

```bash
sudo umount /mnt/local_disk_mount

python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 65536 \
  --concurrency 4 \
  --backend rust_raw_block \
  --raw-device /dev/nvme1n1 \
  --raw-odirect \
  --output-json /tmp/rust_raw_block_raw.json
```

### 3) Compute comparison

```bash
python - <<'PY'
import json

with open("/tmp/local_disk_ext4.json") as f:
    local = json.load(f)[0]["ops_per_sec"]
with open("/tmp/rust_raw_block_raw.json") as f:
    rust = json.load(f)[0]["ops_per_sec"]

print(f"local_disk ops/sec: {local:.2f}")
print(f"rust_raw_block ops/sec: {rust:.2f}")
print(f"rust vs local: {(rust / local - 1.0) * 100.0:+.2f}%")
PY
```

## Use io_uring on Real NVMe

### Compare posix vs io_uring write and read performance

```bash
# Run posix write & read benchmark. To enable data integrity `--verify-integrity`
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 1024 \
  --concurrency 4 \
  --backend rust_raw_block_read \
  --raw-device /dev/nvme1n1 \
  --chunk-size 256 \
  --alignement 4096 \
  --raw-odirect \
  --output-json /tmp/rust_raw_block_read_posix.json

# Run io_uring write & read benchmark. To enable data integrity `--verify-integrity`
python benchmarks/storage_backend_io/storage_backend_io_benchmark.py \
  --num-ops 1024 \
  --concurrency 4 \
  --backend rust_raw_block_read \
  --raw-device /dev/nvme1n1 \
  --raw-odirect \
  --chunk-size 256 \
  --alignement 4096 \
  --use-uring \
  --output-json /tmp/rust_raw_block_read_uring.json

# Compute comparison
python - <<'PY'
import json

with open("/tmp/rust_raw_block_read_posix.json") as f:
    posix = json.load(f)[0]
with open("/tmp/rust_raw_block_read_uring.json") as f:
    uring = json.load(f)[0]

print(f"posix read ops/sec: {posix['read_ops_per_sec']:.2f}")
print(f"uring read ops/sec: {uring['read_ops_per_sec']:.2f}")
print(f"uring vs posix read: {(uring['read_ops_per_sec'] / posix['read_ops_per_sec'] - 1.0) * 100.0:+.2f}%")
print(f"posix write ops/sec: {posix['write_ops_per_sec']:.2f}")
print(f"uring write ops/sec: {uring['write_ops_per_sec']:.2f}")
print(f"uring vs posix write: {(uring['write_ops_per_sec'] / posix['write_ops_per_sec'] - 1.0) * 100.0:+.2f}%")
PY
```
### Notes

- There is a limit on the number of fixed buffers that can be registered. For unprivileged users its 16384.
- Buffer registration and de-registration is time consuming.

## Output

The script prints a summary and optionally writes JSON results if `--output-json` is provided.
```

