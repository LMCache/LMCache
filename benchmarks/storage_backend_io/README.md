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

## Sample Results (2026-02-09, O_DIRECT, 5 runs each)

Method:
- Compare `current` branch vs `origin/dev`.
- `num_ops=4096`, `concurrency in {2,4,8}`.
- `--local-disk-odirect` enabled.
- `--raw-odirect` enabled.
- Raw path is a file on `/mnt/local_disk_mount` for apples-to-apples same-filesystem comparison.
- Table uses median ops/sec across 5 runs.

### Current vs origin/dev (median ops/sec)

| Concurrency | local_disk (`origin/dev`) | local_disk (`current`) | Delta | rust_raw_block (`origin/dev`) | rust_raw_block (`current`) | Delta |
|-------------|----------------------------|-------------------------|-------|-------------------------------|----------------------------|-------|
| 2           | 1012.60                    | 1017.26                 | +0.46% | 1913.52                       | 2604.94                    | +36.13% |
| 4           | 783.83                     | 831.70                  | +6.11% | 1659.36                       | 2839.63                    | +71.13% |
| 8           | 672.70                     | 669.05                  | -0.54% | 1793.30                       | 1792.14                    | -0.06% |

### Current branch (rust_raw_block vs local_disk, median ops/sec)

| Concurrency | local_disk (`current`) | rust_raw_block (`current`) | Rust vs local_disk |
|-------------|-------------------------|----------------------------|--------------------|
| 2           | 1017.26                 | 2604.94                    | +156.08% |
| 4           | 831.70                  | 2839.63                    | +241.43% |
| 8           | 669.05                  | 1792.14                    | +167.86% |

Interpretation:
- LocalDiskBackend stays near baseline; changes are small.
- RustRawBlockBackend shows clear improvement at low-mid concurrency in this setup.
- At concurrency 8, branch-to-branch rust throughput is effectively unchanged in this sample.

### Real block-device smoke (current branch only)

Single-run sanity check with raw block device:
- `num_ops=1024`
- `concurrency=4`
- Local disk path: `/mnt/local_disk_mount/lmcache_local_disk_bench_smoke` (`--local-disk-odirect`)
- Rust raw path: `/dev/nvme1n1p2` (`--raw-odirect`)

| Backend        | Ops/sec |
|----------------|---------|
| local_disk     | 2149.71 |
| rust_raw_block | 3542.56 |

> Results are host/device dependent. Re-run on your target hardware and queue-depth profile before concluding production impact.

## Output

The script prints a summary and optionally writes JSON results if `--output-json` is provided.
