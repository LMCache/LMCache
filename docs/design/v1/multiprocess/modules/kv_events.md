# Committed CPU residency

`KVEventModule` publishes a chunk only after every required `(kv_rank,
object_group_id)` has a committed readable L1 copy. `CommittedKVResidency`
tracks L1 owners per `ObjectKey`. The registration's object groups and shard
replication describe the required set: one object per distinct DCP shard in
each TP-replicated group, or one per rank for an unreplicated group. Residency
and lookup share `get_kv_worker_ids` to enumerate those keyed workers within
the key's encoded world size. Neither subscriber count nor the number of ranks
observed so far proves completeness.

All projection and announcement changes run under the module's event lock.
`L1_WRITE_FINISHED` and `L1_WRITE_FINISHED_AND_READ_RESERVED` add committed
owners. `L1_KEYS_EVICTED` removes only the named L1 owner. Reservations, token
bindings, and GPU STORE submission do not establish residency. Missing or empty
registered geometry cannot establish readiness. The registry returns the group IDs and their shard
replication under one lock. Each chunk is evaluated with one registration
even if a worker is reaped or registers a new layout during an eviction batch.

For each `(model_name, world_size, cache_salt, identity_tags, chunk_hash)`:

| Change | Published event |
| --- | --- |
| All required objects become readable and tokens are known | Stored |
| Duplicate write or another readable copy, with Stored still retained | None |
| One L1 copy disappears while another holds the same object | None |
| The final copy of any required object disappears | Removed, once |
| The missing object is committed again | Stored |

Different L1 managers may supply different required objects. Distinct models,
world sizes, salts, identity tags, and object groups cannot satisfy each other's
requirements. Completeness probes preserve the incoming key's identity, so a
pinned revision or other `lmcache.tag.*` value does not hide a committed chunk.
Owner entries disappear when their last copy is evicted. Announcement state
disappears when a chunk loses readiness. The existing token cache and event log
remain bounded by their existing limits.

A new commit refreshes a complete chunk whose Stored record aged out of the
bounded replay log. This preserves re-announcement for a lagging subscriber;
duplicate commits within the retained history remain suppressed. A truncated
log still reports loss, and no full resident-state snapshot is provided.

EventBus loss invalidates ownership and announcement state and appends a loss
marker. Recovery reads the drop count before checking whether the queue is
empty. If a producer refilled the queue during loss notification, the module
discards queued events through an observed empty boundary, including the final
event that establishes that boundary. Any overflow after that observation has
a newer count and triggers another reset. A surviving queued completion cannot
reconstruct readiness across a dropped eviction. Every required object must
subsequently report a committed copy before another Stored is emitted.

## Token bindings

A Stored record carries the chunk's tokens and parent hash, so a complete chunk
is announced only if `MP_TOKENS` has bound its hash. Two operations publish
that binding, each before its writes can finish:

| Operation | Chunks it binds | Condition |
| --- | --- | --- |
| LMCache-driven STORE | the complete chunks it is about to write | always |
| LOOKUP | the range it submits to prefetch | the load policy accepted the request and the server has an L2 adapter |

LOOKUP binds because its prefetch can commit chunks that this server's engine
never stored: objects loaded from a shared L2, or from a P2P peer attached as
an L2 adapter. Such a copy is readable in the local L1 and is announced like a
stored one. A binding is not residency: a lookup that misses, or a load that
completes only some of a chunk's required objects, publishes no record.

A temporary prefetch copy (`skip_l1`, which `--enable-p2p-write` selects) is
announced when admitted and removed when its reader releases it.

A chunk that commits without a binding is counted in `unbound_stores` and is
not announced. This still applies to an engine-driven STORE, a qstore plugin
store, a P2P peer write on the receiving server, a warm prefetch
(`POST /cache/prefetches`) and a load submitted by the CacheBlend lookup.

This implements the residency step of [issue #5932](https://github.com/LMCache/LMCache/issues/5932).
The wire format, chunk hashes, token bindings, subscriptions, and engine
publication hooks remain the existing contract. Native block identities and
idle engine publication are separate work.
The legacy wire does not distinguish salts, identity tags, or parallel setups
for a shared model; separate ownership scopes prevent mixed-shard completeness
but do not solve
router identity collisions. The supported routing profile uses one unsalted,
uniformly tagged parallel setup per model and full-attention object groups.

## Native binding lifetime

An LMCache-driven STORE can carry one immutable `EventBinding` per complete
chunk. Each descriptor supplies the registered profile, ordered native block
hashes and the first block's parent. Its token payload is the corresponding
slice of the request's packed token bytes. Registration admits a single
full-attention address space. The first engine sender and native event output
are later steps in #5932. The engine sender should include its initialized
hash-seed fingerprint in the profile ID so overlapping workers with different
hash chains are rejected at registration.

A salted request can continue storing KV with empty bindings. With the event
channel enabled, a STORE that supplies bindings must be unsalted. Admission
checks the whole batch before updating the request session or reserving L1.
It reuses rolling hashes for matching packed prefix bytes, hashes the missing
chunks once and passes the resolved keys to the copy path. Rejection leaves
the session and its hash cache untouched.

Overlapping workers registered for one model and world size share a binding
lifetime. Within that lifetime a descriptor stays live while a STORE still
references it or any rank/group has a readable L1 owner. The final worker's
unregister or reap retires those descriptors even if L1 stays warm. A new
registration can bind the warm prefix to new engine IDs. A late release from
the retired registration cannot release the replacement.

A STORE with no reservations releases
only its own reference on return. A STORE with reservations keeps its reference
until its stream-ordered host callback runs. That callback releases the binding
after `finish_write` queues the L1 notifications, or after `abort_write` drops
the reservations from a failed copy. A retry sees the original descriptor until
then. Replay still uses the existing chunk records. Event loss drops
pending and retained bindings together with the residency projection.
During loss recovery a valid STORE proceeds without acquiring a binding or
queuing a release. The bus records an empty boundary after dispatching the
last popped event, including events with no KV subscriber. Native admission
resumes after that boundary and any subsequent overflow is handled separately.
