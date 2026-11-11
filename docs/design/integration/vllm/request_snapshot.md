# MP connector request snapshots

`lmcache.mp.request_snapshot=true` reports observed active requests to server
EventBus subscribers. The default (`false`) skips collection and reporting.
Reporting does not change scheduling or trigger transfers; prefetch policies
are a separate follow-up.

```text
on_new_request        -> retain Request by request_id
build_connector_meta  -> copy current fields -> report_request_snapshot RPC
request_finished      -> remove Request

ManagementModule.report_request_snapshot -> EventBus (MP_REQUEST_SNAPSHOT)
```

Live Request references provide current fields, including for resumable requests,
independently of eager prefetch. Arrival only registers the request; each metadata
build copies the observed set into immutable tuples. No scheduler queues are read.

[`RequestSnapshot` and `RequestSnapshotEntry`](../../../../lmcache/v1/multiprocess/custom_types.py)
define the fields. Priority, arrival time, and status are hints; entry order does
not guarantee execution order. Raw token IDs omit cache salt and multimodal key
transformations and do not imply stored KV. Consumers must use the existing
cache-key path.

The adapter submits through ZMQ or gRPC without waiting for a response. Reports
can queue behind a slow receiver. The server publishes `metadata={"snapshot": snapshot}`
to `EventType.MP_REQUEST_SNAPSHOT` subscribers via the
[EventBus](../../v1/mp_observability/event-bus.md), without retaining a latest
snapshot or exposing a query API.

Delivery is best-effort: disabled observability or EventBus overload can drop
snapshots. Consumers should reject older `sequence` values per `scheduler_id`
and expire stale state. An empty snapshot clears the previous set; a final
snapshot on exit is not guaranteed.

Copying and transmission cost scales with active requests' total token count each
step. This requires vLLM's `on_new_request` callback and a server supporting the RPC.
