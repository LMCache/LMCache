# Committed CPU residency

`KVEventModule` publishes a chunk only after every required `(kv_rank,
object_group_id)` has a committed readable L1 copy. `CommittedKVResidency`
tracks owner tags per `ObjectKey`. The registered layout supplies object groups;
the key's encoded world size supplies the complete required rank set. Neither
subscriber count nor the number of ranks observed so far proves completeness.

All projection and announcement changes run under the module's event lock.
`L1_WRITE_FINISHED` and `L1_WRITE_FINISHED_AND_READ_RESERVED` add committed
owners. `L1_KEYS_EVICTED` removes only the named L1 owner. Reservations, token
bindings, and GPU STORE submission do not establish residency. Missing or empty
registered geometry cannot establish readiness.

For each `(model_name, world_size, cache_salt, chunk_hash)`:

| Change | Published event |
| --- | --- |
| All required objects become readable and tokens are known | Stored |
| Duplicate write or another readable copy, with Stored still retained | None |
| One L1 copy disappears while another holds the same object | None |
| The final copy of any required object disappears | Removed, once |
| The missing object is committed again | Stored |

Different L1 managers may supply different required objects. Distinct models,
world sizes, salts, and object groups cannot satisfy each other's requirements.
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
The wire format, chunk hashes, token bindings, elected worker subscription, and
engine publication hooks remain the existing contract. Native block identities,
scheduler-side subscription, and idle engine publication belong to later steps.
The legacy wire does not distinguish salts or parallel setups for a shared model;
separate ownership scopes prevent mixed-shard completeness but do not solve
router identity collisions. The supported routing profile uses one unsalted
parallel setup per model and full-attention object groups.
