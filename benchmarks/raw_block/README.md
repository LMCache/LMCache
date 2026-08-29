# raw_block io_uring_cmd load — end-to-end A/B

A reproducible serving A/B for the `raw_block` io_uring_cmd load-path changes,
measured on a real vLLM + LMCache MP stack (raw_block L2 on an NVMe passthrough
char device, `/dev/ng*`) rather than a storage microbenchmark — so the numbers
reflect user-facing **TTFT** and **throughput**, not just device bandwidth.

## Result

Llama-3.1-8B, 1× H100, Gen5 NVMe passthrough. Baseline arm = per-chunk
`read_uring()` (QD~1); fixed arm = `batched_read()` + one `wait_iouring()` per
object. KV genuinely resident on the device (guards below prove it).

### Realistic default (`mq_timeout=10s`, recompute fallback on)
What a deployed user sees — vLLM abandons a load that misses the window and
recomputes on the GPU instead.

| metric              | baseline (QD~1) | batched (QD256) | speedup |
|---------------------|----------------:|----------------:|--------:|
| C=1 TTFT p50        |        3154 ms  |         779 ms  |  4.0×   |
| C=1 TTFT p99        |        3960 ms  |         813 ms  |  4.9×   |
| C=4 TTFT p50        |        7008 ms  |        1476 ms  |  4.7×   |
| C=4 throughput      |      0.52 req/s |      2.60 req/s |  5.0×   |
| external hit rate   |         47–56 % |         73–83 % |    —    |

### Isolated storage load (`mq_timeout=90s`, no recompute rescue)
Loads *wait* for storage, so baseline TTFT is the full QD~1 restore time.

| metric              | baseline (QD~1) | batched (QD256) | speedup |
|---------------------|----------------:|----------------:|--------:|
| C=1 TTFT p50        |       11885 ms  |         763 ms  | 15.6×   |
| C=4 TTFT p50        |       20248 ms  |        1375 ms  | 14.7×   |
| C=4 throughput      |      0.19 req/s |      2.76 req/s | 14.5×   |

Batched TTFT is stable across both configs (779→763 ms) — it loads fast enough
to never hit the timeout. Baseline swings 3154→11885 ms with the timeout,
proving its slowness *is* the QD~1 storage load. The slow path also loses the
cache: its loads time out into recompute, so the external-cache hit rate falls
to 47–56 % vs 73–83 %.

## Running it

```bash
# prepare two core.py variants to compare (see ab_arm.sh header), then:
DEVICE=/dev/ng1n1 NVME_CTRL=/dev/nvme1 MODEL=meta-llama/Llama-3.1-8B-Instruct \
  MQ_TIMEOUT=10 ./ab_arm.sh stock
MQ_TIMEOUT=10 ./ab_arm.sh batched
# repeat with MQ_TIMEOUT=90 to isolate the storage load from recompute.
```

`ab_arm.sh` brings up one arm end to end and emits the validity guards.
`ab_drive.py` is the TTFT driver: it warms N distinct long prefixes to L2, then
measures by cycling them round-robin. **Paths and geometry are host-specific and
exposed as overridable environment variables** — adapt them to your setup.

## Why a naive serving A/B is invalid — the four confounds

Each of these silently collapses the two arms to *equal* (or produces garbage),
and each is instrumented out here. If you skip the guards you will "measure" no
difference and wrongly conclude the change does nothing.

1. **L1 DRAM-cache dilution.** The first load promotes KV into the L1 memory
   tier; repeat requests hit L1, not raw_block. A single repeated prefix always
   caches. → Use *N distinct* prefixes with a working set ≫ L1 and cycle them
   round-robin so every revisit misses L1. *Guard:* tier split shows `0 L1`.

2. **Staging-pool starvation.** `--eviction-policy noop` with a small L1 fills
   the shared DRAM pool and never evicts, so store/load allocations fail
   (`Failed to batched allocate … no memory`) and vLLM recomputes everything
   (external hit ~0 %). → Use `--eviction-policy LRU` and size L1 for concurrent
   staging while keeping it ≪ working set. *Guard:* 0 alloc failures.

3. **Recompute masking.** At the default `mq_timeout`, the baseline's slow loads
   time out and fall back to GPU recompute, which *caps* baseline TTFT and hides
   the true storage penalty. → Sweep `mq_timeout` (10 s realistic, 90 s to
   isolate storage) and report both. *Guard:* external hit rate.

4. **iostat is blind to io_uring_cmd.** Block-layer counters (`/proc/diskstats`,
   `iostat`) do **not** count NVMe passthrough on `/dev/ng`, so they read ~0
   even under a heavy load. → Use the controller's own `Data Units Read`
   (`nvme smart-log -o json`), which counts passthrough. *Guard:* device GB read.

## Metrics reported
TTFT p50/p99 at C=1 (latency) and C≥4 (saturation); throughput (req/s);
external prefix-cache hit rate (loads served from LMCache vs recompute);
tier split (no L1 dilution); alloc failures (no pool starvation); device GB
read via the NVMe controller counter (device-served, passthrough-aware).
