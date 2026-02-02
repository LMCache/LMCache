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

- If `--raw-device` is not provided, the benchmark uses a temporary file. This is safe but **not** representative of true raw block performance.
- `--raw-odirect` should only be used with a real block device that supports O_DIRECT.
- Local disk backend uses its internal worker pool; completion is tracked via callbacks.

## Sample Results (2026-02-02)

Run parameters:
- num_ops: 512
- concurrency: 32
- local disk dir: `/tmp/lmcache_local_disk_bench`
- raw device: temp file (no `--raw-device` provided)
- O_DIRECT: disabled

| Backend         | Elapsed (s) | Ops/sec |
|-----------------|-------------|---------|
| local_disk      | 0.258       | 1985.24 |
| rust_raw_block  | 0.123       | 4167.19 |

> Results are machine- and device-dependent. Use real block devices and O_DIRECT for production-grade comparison.

## Output

The script prints a summary and optionally writes JSON results if `--output-json` is provided.
