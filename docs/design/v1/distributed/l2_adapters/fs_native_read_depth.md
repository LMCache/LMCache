# Read depth in the native filesystem connector

How many reads the `fs_native` connector keeps outstanding against storage,
why that used to equal `num_workers`, and why the bound is now in bytes.

## The defect

`ConnectorBase::choose_num_tiles` splits a batch into `min(num_workers, n)`
tiles, and the default `do_batch_get` reads its tile one blocking
`do_single_get` at a time. So reads in flight equal `num_workers`, a CPU
sizing knob whose default is four. Measured on an 8x NVMe RAID0 array whose
ceiling is ~54 GB/s (fio, O_DIRECT), with a discarded warmup pass, the arms
interleaved, and within-arm drift under 0.5% on every row:

| object size (`chunk_size`) | `num_workers=4`, the default | read pool, defaults | ratio |
|---|---|---|---|
| 1.5 MiB (64) | 10.9 GB/s | 48.1 GB/s | 4.41x |
| 3 MiB (128) | 19.1 | 48.4 | 2.53x |
| 6 MiB (256, LMCache's default) | 30.2 | 50.6 | 1.67x |
| 24 MiB (1024) | 50.5 | 51.7 | 1.02x |
| 192 MiB (8192) | 54.5 | 53.4 | 0.98x |

End to end, at `chunk_size` 256 with 100 concurrent 120k-token requests
served from L2 through vLLM, the warm round went from 24.05 s to 14.49 s,
1.66x. On tmpfs the pool reads 86 GB/s at 6 MiB objects against 15 for the
default.

## Why the bound is in bytes, not objects

The alternative is a larger `num_workers`. Throughput is set by the bytes
outstanding against the device, and an object is `chunk_size x
bytes_per_token_per_rank`, so a count of objects means something different
at every `chunk_size`: 64 objects is 384 MiB at `chunk_size` 256 and 12 GiB
at 8192. Same controls as above, `num_read_workers=256` so the budget binds:

| reads in flight bounded by | 1.5 MiB objects | 192 MiB objects | worst case |
|---|---|---|---|
| `num_workers=4` | 11.0 GB/s (78% below best) | **53.9** | 78% |
| `num_workers=256` | 49.3 | 50.4 (6.5% below) | 6.5% |
| byte budget, 1536 MiB | **51.0** | 53.5 (0.8% below) | **0.8%** |

The budget crosses over on its own because `budget / object_size` falls as
objects grow: at 1.5 MiB it allows 1024 objects so the thread count binds,
at 192 MiB it allows 8 so the budget binds. The operator never has to know
the object size, which only exists at dispatch.

`num_workers=64`, the value tuned for 6 MiB objects, ties the pool in an
isolated 6 MiB benchmark (54.0 against 54.1 GB/s, separate run) and reads
6.7% behind it at 192 MiB (49.2 against 52.4), where it holds 12 GiB in
flight. End to end it is 7.4% slower (15.56 and 15.60 s against 14.49 s).
Both hold at most 64 reads in flight there, so that gap is not in bytes
outstanding and its cause is not established.

## Choosing the default

The right figure is a property of the storage. What makes a default possible
is that the cost is asymmetric: too small is expensive, too large is nearly
free. Each column is one storage condition, normalised to the best pinned
budget in that column; 6 MiB objects, `num_read_workers=256`, four workers:

| budget | NVMe array | single NVMe | 8 ms latency | 32 ms latency |
|---|---|---|---|---|
| 96 MiB | 86.6% | **100.0%** | | |
| 192 MiB | 93.8% | 98.7% | | |
| 384 MiB | 98.9% | 94.8% | 65.4% | 31.6% |
| 768 MiB | **100.0%** | 91.0% | 97.6% | 57.5% |
| **1536 MiB** | 99.0% | 89.3% | 98.9% | 99.8% |
| 3072 MiB | 99.2% | 89.5% | **100.0%** | 98.7% |
| 6144 MiB | | | 99.9% | **100.0%** |

The latency columns inject a fixed per-read delay to stand in for
network-attached storage; they rule out the smaller constants. The worst
case for an oversized budget is the single NVMe giving up 10.7%; for an
undersized one it is 68% at 32 ms. **1536 MiB is the default** as the value
whose worst column is best. A single slow device is the case most worth
pinning a smaller value for.

Three properties bound what the budget can do:

1. **Dispatch is quantised in whole objects.** A 192 MiB object is admitted
   as one 192 MiB step.
2. **The thread pool is a separate ceiling.** Bytes in flight never exceed
   `num_read_workers x object_size`, whatever the budget says.
3. **Two reads are always admitted.** The budget never holds fewer than
   two objects in flight, so objects larger than half of it still overlap.
   With 1536 MiB, 1 GiB objects read 39.4 GB/s one at a time and 49.2 two
   at a time; a third adds 1 GB/s and doubles per-read latency.

## Configuration

| field | meaning |
|---|---|
| `num_read_workers` | reader threads, and so the maximum reads in flight; default `256`; `0` keeps the legacy path where depth equals `num_workers` |
| `read_max_bytes_in_flight` | bytes outstanding against the device; `0` selects the 1536 MiB default when `num_read_workers` is positive |

The budget bounds dispatched bytes connector-wide, but the reader threads
are what hold reads against the device, so at 6 MiB objects the full
1536 MiB budget binds only from 256 readers, which is why that is the
default. At `num_read_workers=64` the pool holds 384 MiB, the same as
`num_workers=64` does on the legacy path.

## Where the fix lives

`FSConnector` owns the reader threads, the queues and the budget. When the
read pool is enabled, it overrides `submit_batch_get()` and
`submit_batch_exists()` to enqueue batches directly onto the pool without
going through `ConnectorBase`'s shared worker queue, and the reader finishing
the last key of a batch emits its completion. Lookups run ahead of reads
without consuming the read byte budget, and reads are popped one object at a
time as the budget has room, so concurrent small batches stream across the
entire pool and foreground lookups or reads no longer queue behind background
writes or deletes on `num_workers`. `ConnectorBase` only exposes input
validation and completion publishing to derived classes: the cost of a reader
is backend-specific, a `redis` reader would be one more socket and that path
needs pipelining instead, and `aerospike` shares one client with its own
thread pool.
