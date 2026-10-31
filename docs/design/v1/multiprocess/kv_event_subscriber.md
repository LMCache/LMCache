# MP scheduler KV event subscription

When vLLM KV events are enabled, `LMCacheMPSchedulerAdapter` owns one
`MPKVEventSubscriber` per configured MP server. `LMCacheMPConnector.take_events()`
drains the scheduler directly and converts its CPU events to vLLM events.
Workers have no event subscription, event buffer, local store fallback, or
worker-metadata event carry. Lazy-offload store completion metadata is unchanged.

The current server chunk-hash records and chunk-sized output are preserved.
Native identities, committed residency, and SGLang integration are separate
steps of [issue #5932](https://github.com/LMCache/LMCache/issues/5932). Publication
while idle additionally requires vLLM to invoke `take_events()` while idle.

## Ownership and delivery

| Owner | State and responsibility |
| --- | --- |
| `MPKVEventSubscriber` | Borrowed client, owned stream/thread, stop signal, pending raw updates |
| `LMCacheMPSchedulerAdapter` | Client lifetime, heartbeat, per-server incarnation and announced CPU hashes, event conversion, shared ordered buffer, metrics |
| `LMCacheMPConnector` | Direct scheduler drain and vLLM event wrapping |
| vLLM | Drain opportunities and router-facing event publisher |

1. Scheduler construction creates every subscriber and starts heartbeats and
   reception. No worker inference, lookup, or first drain is required.
2. Each subscription uses `(instance_id, model_name, 0, 1024)`. The scheduler
   generates its own UUID, masked to a nonnegative 63-bit integer to fit the
   existing signed-int64 RPC field. Every server heartbeat uses that identity.
   Two schedulers in one process have separate identities; worker registration
   and worker shutdown do not own the scheduler subscription.
3. A receiver queues an unchanged batch and invokes the adapter callback.
   The callback drains and converts it under the scheduler's event lock.
4. `get_kv_events()` swaps out the converted buffer under that same lock.
   `take_events()` wraps the result without a worker output hop or TP merge.

Each server retains its own incarnation and announced hashes; a loss or restart
withdraws only that server's announcements. Ordering is preserved within each
stream and each batch. Different servers have no shared sequence; callbacks
serialize their arrival under the event lock. This retains legacy per-server
reporting, not a cross-server residency decision.

The notification callback is the sole raw-update consumer. It runs outside the
subscriber lock, drains before returning, and must never call `close()` on its
own subscriber. The receiver waits for conversion, so it cannot build a second
raw backlog while conversion is blocked. Buffer and generated metrics update
without a model step; drain metrics update when the scheduler consumes events.
The existing metric names and label keys remain; `worker_id` now contains the
scheduler subscription identity rather than an elected worker rank.

## Lifecycle and failures

`MPKVEventSubscriber` remains engine-neutral and does not own its request client.
`start()` starts once; terminal receivers are not restarted. `drain()` returns
pending raw batches and stream failures in order. `close()` cancels and joins
reception and is idempotent, including before startup. An in-flight subscription
call must return before it can be cancelled; no connection timeout is added.

Scheduler shutdown closes and joins subscribers without holding the event
lock, then stops heartbeats, then closes clients. An in-flight callback can
finish before its borrowed client closes. With events disabled there are no
subscriptions; heartbeat still starts on the first lookup with no instance ID.

| Input | Subscriber action | Scheduler action |
| --- | --- | --- |
| Enabled batch | Deliver unchanged | Interpret incarnation/loss and convert CPU records |
| `enabled=False` | Deliver once and stop | Withdraw the server's CPU placements; no local store fallback |
| `ConnectionError` or stream end | Deliver closure, wait one interruptible second, resubscribe from zero | Withdraw that server's placements before replay |
| Other subscription/receive error | Log, deliver terminal closure, stop | Withdraw that server's placements; caching continues |
| Owner closes subscriber | Cancel/join with no synthetic failure | Stop heartbeat, then close client |

Model-filtered sequence gaps are allowed. The first batch may be trimmed;
subsequent loss or incarnation changes invalidate known CPU hashes. Replaying
from zero retains the existing bounded-history limit. Invalidation emits CPU
`BlockRemoved`, never `AllBlocksCleared`, so GPU placements remain intact.

## Validation

`tests/v1/multiprocess/test_kv_event_subscriber.py` covers the engine-neutral
receiver, cancellation, callback ownership, reconnect delay, and borrowed client
lifetime. `tests/v1/test_vllm_mp_adapter.py` covers scheduler startup and identity,
ordered conversion and metrics, per-server resets, and failures without local
store fallback. `tests/v1/test_mp_connector_kv_roles.py` checks direct and real
`MultiConnector` drains without worker execution and preserves lazy-offload
completion behavior. Native identities and residency are not part of these tests.
