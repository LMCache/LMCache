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
registered geometry cannot establish readiness. If registration disappears
between the group and replication lookups, the chunk is treated as incomplete
so a batch of evictions can still withdraw all affected announcements.

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

This implements the residency step of [issue #5932](https://github.com/LMCache/LMCache/issues/5932).
The wire format, chunk hashes, token bindings, subscriptions, and engine
publication hooks remain the existing contract. Native block identities and
idle engine publication are separate work.
The legacy wire does not distinguish salts, identity tags, or parallel setups
for a shared model; separate ownership scopes prevent mixed-shard completeness
but do not solve
router identity collisions. The supported routing profile uses one unsalted,
uniformly tagged parallel setup per model and full-attention object groups.
