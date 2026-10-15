# Committed CPU residency

`CommittedKVResidency` is the pure state behind native CPU KV events. It joins
immutable native bindings with the configured KV ranks/object groups and the
set of L1 owners holding a committed readable copy of each object. It returns
Stored/Removed transitions; it does not know the EventBus, RPC records, replay,
storage manager, or serving engine.

The owner serializes calls. `register_profile` pins already-validated identity
semantics and required geometry. `bind` validates descriptor conflicts and retains
in-flight references without announcing availability. A later `refresh` joins
those bindings at their ordered position among storage notifications. `commit`
and `evict` track distinct owner tags, so duplicates and redundant copies cannot
create duplicate announcements or premature removals.

`finish` releases one transfer's pending references after its commit notifications
have been applied. A failed transfer cannot establish residency; a live or partial
object retains its descriptor independently of request cleanup. Returned transitions
own immutable descriptors, including parent identities.

`reset` discards every residency, binding, and pending fact while retaining profile
configuration. New bindings alone cannot restore old availability after loss.
`KVEventModule` retains token/layout/storage validation, its existing condition
lock, the EventBus loss barrier, and the bounded replay log. It translates the
tracker's ordered transitions into wire records while holding that lock.
