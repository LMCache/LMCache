# Storage Backend I/O Benchmark

This microbenchmark compares **LocalDiskBackend** vs **RustRawBlockBackend** under high write-concurrency.

## What It Measures

- Total time to submit and complete `num_ops` write (put) operations
- Effective ops/sec under concurrent submission

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
- Rust raw block benchmark uses a unique manifest path per run to avoid stale-index reuse between runs.

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

## Output

The script prints a summary and optionally writes JSON results if `--output-json` is provided.
