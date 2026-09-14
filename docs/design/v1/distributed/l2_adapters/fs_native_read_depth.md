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
at 8192. Same controls as above, `read_io_depth=256` so the budget binds:

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
budget in that column; 6 MiB objects, `read_io_depth=256`, four workers:

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

1. **Dispatch is quantised in whole objects.** Four workers and 192 MiB
   objects cannot dispatch less than 768 MiB.
2. **The thread pool is a separate ceiling.** Bytes in flight never exceed
   `read_io_depth x object_size`, whatever the budget says.
3. **A worker blocks on its own tile.** Objects are dispatched one at a
   time as the budget has room, so a large tile keeps the depth full, but
   a batch smaller than the depth holds only its own objects in flight:
   four workers and 16-object batches keep at most 64 objects in flight
   however deep the pool is. Against `num_workers=64`, which streams
   continuously, this measured between a tie and 12% behind on the same
   array, depending on per-read latency at the time; batches of a few
   hundred objects do not show it.

## Configuration

| field | meaning |
|---|---|
| `read_io_depth` | reader threads, and so the maximum reads in flight; `0` keeps the legacy path where depth equals `num_workers` |
| `read_max_bytes_in_flight` | bytes outstanding against the device; `0` selects the 1536 MiB default when `read_io_depth` is positive |

The budget bounds dispatched bytes connector-wide, but the reader threads
are what hold reads against the device, so at 6 MiB objects the full
1536 MiB default needs `read_io_depth=256` before the budget rather than the
thread count binds. At `read_io_depth=64` the pool holds 384 MiB, the same as
`num_workers=64` does on the legacy path.

## Where the fix lives

`FSConnector` owns the reader threads, the queue and the budget. It
overrides `do_batch_get()` to dispatch a tile's objects one at a time as the
budget has room and wait for all of them, and `choose_num_tiles()` to keep a
GET batch as one tile, the shape `MooncakeConnector` already uses; split
across workers first, a 16-object batch on four workers would keep four
reads in flight whatever the depth. `ConnectorBase` is unchanged:
the cost of a reader is backend-specific, a `redis` reader would be one more
socket and that path needs pipelining instead, and `aerospike` shares one
client with its own thread pool.
