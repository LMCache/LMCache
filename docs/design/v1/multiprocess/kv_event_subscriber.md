# MP KV event subscriber extraction

The vLLM MP worker receives CPU cache changes through a transport-neutral
`RequestClient`, but its receiver loop lives inside the vLLM adapter. Moving
that loop into `MPKVEventSubscriber` makes reception reusable without moving
registration, heartbeat, event conversion, or publication out of the engine.

This extraction preserves the current chunk-hash event contract. Native
bindings, committed-residency decisions, SGLang activation, and a change to
the existing failure policy are separate work under
[issue #5932](https://github.com/LMCache/LMCache/issues/5932).

## Ownership and delivery

| Owner | State and responsibility |
| --- | --- |
| `MPKVEventSubscriber` | One borrowed client reference, one owned stream/thread, stop signal, and pending raw updates |
| `LMCacheMPWorkerAdapter` | Client lifetime, registration, heartbeat, election, incarnation/loss interpretation, announced CPU hashes, converted event buffer, metrics, and failure policy |
| `LMCacheMPConnector` | Existing direct hook or worker-metadata bridge, scheduler aggregation, and native event wrapping |
| vLLM | Event drain/publication opportunities and router-facing publisher |

The sequence is:

1. The adapter constructs a subscriber only for the elected rank. Other ranks
   remain in server-source mode and do not report their own stores.
2. The first `get_kv_events()` starts the existing heartbeat and the subscriber.
3. The receiver subscribes with `(instance_id, model_name, 0, 1024)`, queues the
   unchanged batch, and invokes the adapter's notification callback.
4. The callback drains raw updates and applies the existing adapter conversion
   and failure policy under the adapter lock. Metrics update on notification
   without waiting for another model step, as they did before extraction.
5. A later `get_kv_events()` returns the converted events in order. The existing
   connector chooses one delivery path per step, including under MultiConnector.

The notification callback runs on the receiver thread and must drain before
returning. Reception waits for it, so it cannot build an additional raw-update
backlog while conversion is blocked. It must not call `close()`, which joins
that same thread. The callback is needed to preserve the adapter's immediate
buffer metrics and transition to completed-store reporting on terminal failure;
moving conversion to the next engine step would change those behaviors.

## Subscriber API and lifecycle

`MPKVEventSubscriber(req_client, instance_id, model_name, on_update)` accepts
the existing attachment identity and a callback receiving the subscriber.
The class imports neither vLLM nor SGLang and does not construct hashes or
engine-native events.

- `start()` starts reception once. Repeated calls do not create another thread
  or restart terminal reception. Calling it after `close()` raises `RuntimeError`.
- `drain()` removes and returns pending `KVEventBatch | KVEventStreamClosed`
  updates in source order. The callback is the sole consumer; an extra drain
  with no new data returns an empty list.
- `close()` sets the stop signal, cancels the active stream, and joins the
  receiver. It is idempotent and also valid before `start()`. It does not close
  the borrowed client or stop the engine heartbeat. A subscription call already
  in progress must return before its stream can be cancelled; this extraction
  does not add transport connection timeouts.

The subscriber lock protects thread startup, active-stream assignment, and the
pending list. Notifications and stream cancellation run outside that lock.
The adapter callback takes the adapter lock before calling `drain()`. Shutdown
must therefore call subscriber `close()` without holding the adapter lock, then
stop heartbeat, unregister transfer state, and close the client in the existing
order. Shutdown waits for any in-flight notification to finish.

## Failure and replay behavior

| Input | Subscriber action | Existing adapter action |
| --- | --- | --- |
| Enabled batch | Deliver the batch unchanged | Interpret incarnation/loss, deduplicate advertised hashes, and convert CPU records |
| `enabled=False` batch | Deliver once and stop | Withdraw CPU placements and resume completed-store reporting |
| `ConnectionError` or ordinary stream end | Deliver `KVEventStreamClosed(error=None)`, wait one interruptible second, then subscribe from cursor zero | Withdraw CPU placements before replay; remain in server-source mode |
| Other receive/subscription exception, or exception during batch notification | Log the exception, deliver `KVEventStreamClosed(error=str(exc))`, and stop | Withdraw CPU placements and resume completed-store reporting |
| Owner closes subscriber | Cancel/join; no synthetic failure update | Continue ordinary adapter shutdown |

The callback must handle both transient and terminal stream-closure updates
without raising. A failure in either handler ends the receiver thread; the
subscriber cannot repair engine state. This preserves the existing receive-loop
boundary rather than adding a second engine recovery policy.

Batch incarnation and explicit loss are passed through to the adapter. A
model-filtered stream may have numeric sequence gaps; the receiver does not
reinterpret them. Reconnecting from zero preserves the existing conservative
replay behavior and its retained-history limit. CPU invalidation does not emit
`AllBlocksCleared` or alter GPU placements. Publication still depends on vLLM
draining its connector; this refactor adds no idle publisher.

## Implementation plan

1. Add `lmcache/v1/multiprocess/kv_event_subscriber.py` with the lifecycle and
   failure records above. Adapt the receiver ownership pattern from #5898, using
   the current four-argument subscription contract. Keep the one-second
   interruptible reconnect wait and 1024-record batch limit.
2. Update `lmcache/integration/vllm/vllm_multi_process_adapter.py` to own an
   optional subscriber instead of a stream, receiver thread, and stop event.
   `_resolve_kv_event_source()` constructs it only for the elected rank;
   `_drain_kv_event_updates()` handles raw updates on notification. Leave batch
   conversion, hash state, metrics, and the local-store failure policy in place.
3. Add `tests/v1/multiprocess/test_kv_event_subscriber.py`. Exercise raw payload
   and store/remove/store order, incarnation/loss pass-through, start/close
   idempotence, terminal/disabled behavior, reconnect delay, cancellation of a
   blocked read, shutdown during subscription, callback failures, and borrowed
   client ownership through real `MessagingStream` objects and a fake client
   boundary. Synchronize read entry and observe `start()` rejecting a closed
   subscriber before releasing an in-flight subscription. Join terminal
   receivers before testing restart rejection, and hold a notification callback
   open to prove `close()` waits for it. Verify import without loading either
   inference engine.
4. Run the unchanged adapter regressions in
   `tests/v1/test_vllm_mp_adapter.py` for immediate metrics, CPU invalidation,
   source election, store completion, and failure handling. Run
   `tests/v1/test_mp_connector_kv_roles.py` with vLLM installed to cover both
   #5524 delivery paths, TP aggregation, and ordered transitions. A skipped
   connector module is not passing delivery coverage.

No server module, wire schema, transfer-context interface, connector event
bridge, configuration option, or dependency changes are required. Rollback is
the inverse extraction; there is no stored-state or protocol migration. Run
applicable pre-commit hooks and the documentation build before submitting.
