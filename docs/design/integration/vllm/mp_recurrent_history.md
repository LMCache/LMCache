# Recurrent history after an MP prefix restore

An align-mode recurrent cache can retain a state at the primary external-load
boundary but miss the earlier state required by the next local prefix lookup.
Speculative decoding can make the next usable local boundary earlier than that
external-load boundary. Repeating the same prompt then loads the same prefix
from CPU even though the previous transfer succeeded.

The opt-in `lmcache.mp.restore_recurrent_history` path restores complete earlier
CPU checkpoints as well as the ordinary final recurrent state. It never changes
the primary restore boundary or treats a later state as an earlier state.

## Flow and ownership

1. The scheduler finds already-local complete recurrent checkpoints and pins
   them while allocating destinations, so allocation cannot evict these states.
2. The existing server-module envelope carries an aligned prefix key and the
   already-local positions. The server resolves every earlier chunk boundary
   across all recurrent object groups and TP ranks, deduplicates object keys,
   and reserves one L1 read lock per unique key. A checkpoint missing any object
   is excluded; reads needed only by excluded checkpoints are released.
3. The scheduler allocates separate private blocks for the entire returned
   checkpoint set. These blocks are not the current request's active recurrent
   state. If the set does not fit, the whole optional plan is canceled.
4. Each worker runs its normal retrieve first, then copies its historical
   objects with the registered cache context's `transfer_object_group` method.
   Destinations must not alias each other, primary-load blocks, or null blocks.
5. Each rank fences its transfer stream before recording completion. CPU source
   reads are shared until all required ranks finish. No canceled or failed plan
   is published. A failed stream fence retains its source and destination
   references; a control timeout does not prove completion.
6. At receive completion, the connector registers each copied state with native
   `BlockPool.cache_full_blocks`, using its actual prefix position and group.
   It then releases private references and acknowledges the server reservation.

The reservation table distinguishes pending, running, canceled, and completed
copies. A server-generation prefix prevents an unknown receipt after server
restart from being mistaken for a completed transfer. Periodic server reaping
cancels abandoned plans, while scheduler polling retries cleanup after control
timeouts. Reaping cannot manufacture a device fence. L1 read TTL remains the
existing storage-layer contract: expired sources cannot be submitted or
published, and their old locks must not release a later reader's reservation.

## Compatibility and scope

The initial integration is disabled by default and supports align mode,
TP=2/DP=1 without PP or context parallelism, one MP server and one L1 manager,
and equal hash, scheduler-page, and LMCache chunk sizes. Unsupported connector
configurations fail before use. Multi-L1 servers return no optional plan.
Only checkpoints already present in L1 are candidates: this change neither
creates missing historical states nor loads them from L2. There is no change
to the primary receive result when the optional history copy fails.

The extra fence is intentionally synchronous. The first restore can be slower;
subsequent local reuse may amortize that cost. The previously measured Ascend
prototype results are evidence for the approach, not measurements of a new PR
revision. Concurrency, throughput, and new hardware configurations need separate
validation. A future asynchronous optimization must retain these ownership and
publication conditions.

## Validation

The regression tests cover complete/incomplete and shared-object reservations,
per-rank completion and cancellation, sparse prefix publication with rollback,
server-generation receipt isolation, and retrying canceled cleanup after an RPC
timeout. The target test harness must additionally verify actual copies and
correct model outputs after eviction, followed by local reuse on the same NPU
pair. Run the standard tests and target preflight before model loading.
