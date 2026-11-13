# Committed CPU residency

`KVEventModule` publishes a chunk only after every required `(kv_rank,
object_group_id)` has a committed readable L1 copy. `CommittedKVResidency`
tracks owner tags per `ObjectKey`. The registered layout supplies object groups;
the key's encoded world size supplies the complete required rank set. Neither
subscriber count nor the number of ranks observed so far proves completeness.

## Recurrent-state groups

Object groups that the registration labels `recurrent` (Mamba / GDN state) are
not required. In vLLM's `mamba_cache_mode="align"`, a recurrent object exists
only at checkpoint chunks: vLLM snapshots the state at the end of a scheduler
step. If recurrent groups are required, only checkpoint chunks are announced.
Their parents are never announced, and a router that walks `(parent, block)`
edges from the root credits nothing.

With the rule, every chunk whose attention, sliding-window, and aux groups are
committed is announced, so a hybrid prefix forms one chain from the root. This
matches vLLM's own `OffloadingConnector`: attention groups carry the payload,
and recurrent groups emit only placeholders (no tokens, `block_size=0`) that
routers do not index. A registration with no non-recurrent group announces
nothing.

Trade-off: a router can credit attention chunks after the last recurrent
checkpoint. The servable hit ends at the last checkpoint at or before the
credited prefix, and lookup stays exact. The difference costs recompute on the
chosen engine, not correctness. It is at most the chunks after the last
checkpoint, which is usually one scheduler step. An evicted checkpoint whose
attention chunks stay resident emits no Removed, so the credit can exceed the
hit until those chunks are evicted too.

```
chunk:        c0      c1      c2      c3
attention:    yes     yes     yes     yes
recurrent:    -       -       yes     -
announced:    c0 <-   c1 <-   c2 <-   c3     (parent chain from the root)
servable hit: 3 chunks (last checkpoint c2); a router may credit 4
```

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

This implements the residency step of [issue #5932](https://github.com/LMCache/LMCache/issues/5932).
The wire format, chunk hashes, token bindings, elected worker subscription, and
engine publication hooks remain the existing contract. Native block identities,
scheduler-side subscription, and idle engine publication belong to later steps.
The legacy wire does not distinguish salts or parallel setups for a shared model;
separate ownership scopes prevent mixed-shard completeness but do not solve
router identity collisions. The supported routing profile uses one unsalted
parallel setup per model and full-attention object groups.
