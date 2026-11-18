# MP connector request snapshots

`lmcache.mp.request_snapshot=true` reports observed active requests to server
EventBus subscribers. The default (`false`) skips collection and reporting.
Reporting does not change scheduling or trigger transfers; prefetch policies
are a separate follow-up.

```text
on_new_request        -> retain Request by request_id
build_connector_meta  -> check interval -> copy fields -> report_request_snapshot RPC
request_finished      -> remove Request

ManagementModule.report_request_snapshot -> EventBus (MP_REQUEST_SNAPSHOT)
```

Live Request references provide current fields, including for resumable requests,
independently of eager prefetch. Arrival only registers the request; reports
copy the observed set into immutable tuples. No scheduler queues are read.

`lmcache.mp.request_snapshot_interval_ms` is checked before copying tokens.
The first report and a transition to empty bypass the interval. Reports run only
on scheduler steps, so short-lived requests may be missed.

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

The interval reduces frequency, not per-report copying or per-server serialization
cost, which scale with total reported tokens. This requires vLLM's `on_new_request`
callback and a server supporting the RPC.
