# Read depth in the native filesystem connector

How many reads the native filesystem connector keeps outstanding against
storage, why the answer used to be `num_workers`, and how the connector now
sizes it in bytes.

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
unchanged; `mooncake` overrides both and has its own depth. The filesystem
connector used to inherit them too, and it is the one this document fixes.

On an array of several devices the default of four is far below what the
hardware needs to reach its read bandwidth. Measured on eight NVMe drives in
RAID0 whose ceiling is ~54 GB/s (fio, O_DIRECT), with a discarded warmup
pass and the two arms interleaved so that drive settling cannot be mistaken
for an effect; within-arm drift is under 0.5% on every row shown:

| object size (`chunk_size`) | `num_workers=4`, the default | read pool, defaults | ratio |
|---|---|---|---|
| 1.5 MiB (64) | 10.9 GB/s | 48.1 GB/s | 4.41x |
| 3 MiB (128) | 19.1 | 48.4 | 2.53x |
| 6 MiB (256, LMCache's default) | 30.2 | 50.6 | 1.67x |
| 24 MiB (1024) | 50.5 | 51.7 | 1.02x |
| 192 MiB (8192) | 54.5 | 53.4 | 0.98x |

At 12 MiB and 48 MiB the two arms tie within drift. End to end, at
`chunk_size` 256 and 100 concurrent requests of 120k tokens served from L2
through vLLM, the warm round went from 24.05 s to 14.49 s, 1.66x. On tmpfs,
where a read is a memcpy, the pool reads 86 GB/s at 6 MiB objects against 15
for the default, and ties `num_workers=64` at both 6 MiB and 1.5 MiB.

The bytes were always reachable. The code was not asking for enough of them
at once.

## Why the budget is in bytes, not objects

The alternative is to raise `num_workers`, and on this array a well-chosen
value reaches the same ceiling as the pool in an isolated read benchmark: at
6 MiB objects, `num_workers=64` reads 54.0 GB/s against the pool's 54.1. End
to end it does not keep up: the same `num_workers=64` completes the vLLM
round above in 15.56 and 15.60 s against the pool's 14.49 s, 7.4% slower
with 0.3% between its two runs. The two settings differ in what they hold in
flight, 64 objects (384 MiB) against the pool's 1536 MiB budget; the
benchmark has nothing but disk latency per read, the serving path has more,
and 384 MiB is what a 54 GB/s array needs at 7 ms per read with no margin.
Beyond that, `num_workers` has no value that is right at more than one
object size, and the budget does.

Throughput is set by the bytes outstanding against the device, not by the
number of objects. An object here is `chunk_size x
bytes_per_token_per_rank`, so the same object count means very different
things at different `chunk_size` settings: on the hardware above, 64 objects
in flight is 384 MiB at LMCache's default `chunk_size` of 256 and 12 GiB at
`chunk_size` 8192. Same controls as above, `read_io_depth=256` so the budget
is what binds:

| reads in flight bounded by | 1.5 MiB objects | 192 MiB objects | worst case |
|---|---|---|---|
| `num_workers=4` | 11.0 GB/s (78% below best) | **53.9** | 78% |
| `num_workers=256` | 49.3 | 50.4 (6.5% below) | 6.5% |
| byte budget, 1536 MiB | **51.0** | 53.5 (0.8% below) | **0.8%** |

Four workers wins at 192 MiB and loses 78% at 1.5 MiB; 256 wins at 1.5 MiB
and loses 6.5% at 192 MiB. The value tuned for 6 MiB objects,
`num_workers=64`, reads 49.2 GB/s at 192 MiB against the budget's 52.4, 6.7%
behind with 1.1% between its own repeats, because 64 workers there hold
12 GiB in flight where the budget holds 1.5. The budget crosses over on its own because
`budget / object_size` falls as objects grow: at 1.5 MiB it allows 1024
objects so the thread count binds, at 192 MiB it allows 8 so the budget
binds. The operator never has to know the object size, which only exists at
dispatch. A budget in bytes removes the object-size dependence; measuring it
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
3. **A tile is read a group at a time, and each group drains before the
   next is dispatched.** A batch too small to fill the depth therefore
   holds fewer reads in flight than `read_io_depth`, and reads in flight
   fall to zero at the end of every group. With four workers and
   16-object batches the pool holds at most 64 objects and pays that
   barrier once per 16 objects; against a `num_workers=64` that streams
   continuously it measured between a tie and 12% behind on the same
   array on the same day, depending on per-read latency at the time.
   Batches of a few hundred objects, which is what a long prompt at the
   default `chunk_size` produces, dispatch 64 at a time and do not show
   it. Dispatching each object as the previous one completes, a sliding
   window instead of groups, would remove the barrier; it is not in this
   change.

## Configuration

| field | meaning |
|---|---|
| `read_io_depth` | reader threads, and so the maximum reads in flight; `0` keeps the legacy path where depth equals `num_workers` |
| `read_max_bytes_in_flight` | bytes outstanding against the store; `0` selects the 1536 MiB default |

`read_io_depth` has to be large enough for the byte budget to be the
constraint that binds: whichever of the two limits is smaller wins, and
property 2 above is what makes that easy to get wrong. The budget is shared
across the workers as equal per-worker shares, so at the default
`chunk_size` of 256, whose objects are 6 MiB, the default budget's 384 MiB
share per worker needs a `read_io_depth` of 64 to be reachable.

Each reader thread has one file open at a time, so `read_io_depth` is also
the ceiling on files the connector holds open for reads.

## Where the fix lives

`FSConnector` owns the reader threads, the grouping and the byte budget. It
overrides `do_batch_get()` to hand a tile to the pool a group at a time and
wait for each group, and `choose_num_tiles()` so that a GET batch is split
across workers only as far as leaves every tile at least `read_io_depth`
objects deep. Without the second override the base class would split a
16-object batch across four workers, each of which hands its own four
objects to the pool and blocks, and reads in flight would stay at
`num_workers` however large the depth was configured. The reads themselves
run the unmodified `do_single_get()` on reader threads that each hold their
own `WorkerFSConn` from the existing `create_connection()`. `ConnectorBase`
is unchanged.

This is the same shape as `MooncakeConnector`, which overrides the same two
methods because its backend already parallelises a batch internally. The
pool is not in the base class because the cost of a reader thread is
backend-specific and the fix each backend needs is different: a `redis`
connection is a socket, so depth would be that many more sockets per rank,
and what a strictly send-then-receive protocol needs is pipelining, which is
a change to `do_single_get` rather than to how many of them run at once;
`aerospike` shares one client whose own thread pool is sized from
`num_workers`. A pool in the base would be right for the filesystem and
either costly or inert for the others.
