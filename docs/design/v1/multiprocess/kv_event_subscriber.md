# MP KV event subscriber

The common receiver introduced in [PR #6074](https://github.com/LMCache/LMCache/pull/6074)
now consumes the versioned native-binding stream. Its current ownership,
registration, replay, and publication contracts are documented in
[native CPU KV events](native_kv_events.md).

`start()` owns stream/thread startup, `drain()` returns ordered batches and CPU
invalidations, and `close()` cancels and joins without closing the borrowed
client. Registration and heartbeat stay with the engine. The receive queue is
bounded and preserves an invalidation path on overflow. The engine thread owns
the native contribution ledger and publisher; receiver callbacks no longer
perform engine event conversion. Event-enabled operation never switches to
local completed-store reporting.
