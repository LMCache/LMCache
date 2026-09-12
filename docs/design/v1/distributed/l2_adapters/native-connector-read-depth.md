# Read depth in the native connectors

How many reads a native connector keeps outstanding against storage, why the
answer used to be wrong for every backend, and what each backend needs in
order to fix it.

## The defect

`ConnectorBase` splits a batch into tiles and hands each tile to one worker
thread:

```cpp
size_t choose_num_tiles(Op op, size_t num_items) const {
  return std::min<size_t>(worker_count_for_op(op), num_items);
}
```

and the default `do_batch_get` walks its tile's keys one at a time, calling
the blocking `do_single_get`. So for every backend that does not override
both methods:

> **reads in flight == `num_workers`**

That is a resource-sizing parameter, chosen for how much CPU the connector
should use orchestrating batches. It is not a storage queue depth, and
nothing makes the two equal. `redis` and `aerospike` inherit both methods
unchanged; `mooncake` overrides both and has its own depth. The
filesystem connector used to inherit them too.

On an array of several devices the default of four is far below what the
hardware needs to reach its read bandwidth. Measured on eight NVMe drives in
RAID0 whose ceiling is ~54 GB/s, with 6 MiB objects:

| reads in flight | delivered |
|---|---|
| 4 (`num_workers=4`, the default) | 31 GB/s |
| ~40 | 48 GB/s |

The bytes were always reachable. The code was not asking for enough of them
at once.

## Why the budget is in bytes, not objects

Throughput is set by the bytes outstanding against the device, not by the
number of objects. An object here is `chunk_size x
bytes_per_token_per_rank`, so the same object count means very different
things at different `chunk_size` settings: on the hardware above, a depth
of 64 objects is 384 MiB in flight at LMCache's default `chunk_size` of 256
and 12 GiB at `chunk_size` 8192. The first is about right; the second is far
past the point where more queueing helps:

| object size | default | 64 objects in flight | measured byte budget |
|---|---|---|---|
| 6 MiB | 31.3 | 48 | **48.1** |
| 24 MiB | 41.0 | ~50 | **51.3** |
| 192 MiB | 54.1 | 49 (**-8.7%**) | **53.6** |

No fixed setting is right everywhere: four workers wins at 192 MiB and loses
42% at 6 MiB; a fixed 64-object depth wins at 6 MiB and loses 8.7% at
192 MiB. A budget in bytes removes the object-size dependence; measuring it
removes the hardware dependence.

## Choosing the budget

The right figure is a property of the storage, and no single number is best
everywhere. What makes a default possible is that the cost is steeply
asymmetric: too small is expensive, too large is nearly free.

Each column is one storage condition, normalised to the best pinned budget
in that column. 6 MiB objects, `read_io_depth=256`, four workers, three
passes over a 16 GiB corpus:

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
network-attached storage. They are what rules out the smaller constants: at
32 ms, 768 MiB delivers 58% of what 1536 MiB does and 384 MiB delivers 32%.

Being too large is not free, but it is much cheaper. The worst case
measured for an oversized budget is the single NVMe, which gives up 10.7%
between 96 MiB and 1536 MiB; the worst case for an undersized one is 68% at
32 ms. On the array, every budget from 384 MiB to 3 GiB lands within 1.1%
of the best. Most of that asymmetry is structural: the budget is a
throttle, not an allocator. Buffers are allocated by the caller for the
whole batch either way, the thread pool is fixed at construction, and a
saturated device delivers its maximum however deep the queue behind it is,
so raising the budget usually cannot add work and therefore cannot add
queueing. A single device is where that stops holding, and it is the
column where the default costs the most.

**1536 MiB is therefore the default**: not the winner of any column, but the
value whose worst column is best. A deployment that knows its storage
should pin its own value, and a single slow device is the case most worth
pinning.

Two properties of the budget are worth stating because they bound what it
can do:

1. **Dispatch is quantised in whole objects.** With four workers and 192 MiB
   objects the smallest realizable dispatch is already 768 MiB, so the
   budget cannot express a smaller figure at that object size.
2. **The thread pool is a separate ceiling.** Bytes in flight can never
   exceed `read_io_depth * object_size`, whatever the budget says.

## Configuration

| field | meaning |
|---|---|
| `read_io_depth` | reader threads, and so the maximum reads in flight; `0` keeps the legacy path where depth equals `num_workers` |
| `read_max_bytes_in_flight` | bytes outstanding against the store; `0` selects the 1536 MiB default |

`read_io_depth` has to be large enough for the byte budget to be the
constraint that binds: whichever of the two limits is smaller wins, and
property 2 above is what makes that easy to get wrong. At the default
`chunk_size` of 256, whose objects are 6 MiB, the default budget's
384 MiB per-worker share needs a `read_io_depth` of 64 to be reachable.

Each reader thread holds one connection for the connector's lifetime, and
for the filesystem connector one open file at a time, so `read_io_depth` is
also the ceiling on both.

## Where the fix lives

`ConnectorBase` owns the reader threads, the grouping and the byte budget,
and executes a group by calling the backend's unmodified `do_single_get()`
on threads that each hold their own `ConnectionType` from the existing
`create_connection()`. Nothing about it is filesystem-specific, so every
backend gains real depth by setting `depth`, with no backend-specific code.

Per-thread connections are how the base already works: each worker thread
creates its own connection at the top of its loop, so this introduces no
new concurrency contract. Sharing one connection across reader threads
would, and several client libraries are not thread-safe per connection.

The cost differs by backend and has to be checked before enabling one.
`redis` opens a socket per connection, so `depth` is `depth` more sockets.
`aerospike` does not: its `create_connection()` copies a pointer to one
shared client whose own thread pool is sized from `num_workers`, which
would need to follow `depth` instead. `mooncake` overrides `do_batch_get()`
and is unaffected either way.
