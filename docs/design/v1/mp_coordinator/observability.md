# MP Coordinator Observability

The coordinator only knows what the mp servers' events tell it. If events are
lost, late or half-applied, it keeps answering, and answers wrong.
Observability exists to catch that.

Code: `lmcache/v1/mp_coordinator/observability.py` (metric names and
registration), `http_apis/metrics_api.py` (`GET /metrics`).

## 1. Design principles, goals and non-goals

**Principles**

- **Watch the event path first.** Is the coordinator's view complete? How old
  is it? Are the loops that act on it still running? Request rates come after.
- **One source per fact.** Each component keeps a `stats()` snapshot. Metrics
  and the `/status` page both read it, so they never disagree.
- **Plain OpenTelemetry metrics.** No event bus: that was built for the GPU
  server's hot path, and it isn't running in the coordinator.
- **Few labels.** Only fixed lists of values. `instance_id` only on gauges,
  which disappear when the server leaves. Never keys or hashes.
- **Cheap to read.** A metric reads counts that are updated on every change. It
  never walks a whole index while holding its lock, so a scrape cannot stall
  ingest or lookups.

**Goals**

- See lost or late events, per mp server.
- See within minutes when a loop dies, a controller fails to start, or the
  views drift apart.
- Let an operator see why from one page, without reading logs.

**Non-goals**

- Per-key or per-request metrics.
- A health check that fails when degraded (k8s would restart the coordinator
  and wipe its state).
- A readiness check that waits for warm-up (it would block the events needed to
  warm up).
- Tracing spans, for now (section 6).
- Changing the mp server's event bus.

## 2. Architecture

No new process or pipeline: each existing part keeps counts about itself, and
one small layer exposes them. Parts marked *(new)* are what observability adds
to the fault-tolerance architecture.

```mermaid
flowchart TB
    MP["MP servers<br/>send cache events and heartbeats<br/><i>(new) count dropped events in each batch</i>"]
    T["Kafka topic (or POST /events)<br/><i>(new) lag and partitions watched</i>"]
    subgraph C["Coordinator"]
        G["Event gate<br/>dedup, fence, flag gaps<br/><i>(new) counts loss and freshness per server</i>"]
        V["Views<br/>key directory, usage, registry<br/><i>(new) report size, lookups, failures</i>"]
        K["Controllers<br/>eviction, prefetch, pins<br/>built-in or loaded from a package<br/><i>(new) report work, failures, up/down</i>"]
        L["Background loops<br/>health, checkpoint, eviction<br/><i>(new) record last success, survive errors</i>"]
        O["(new) Metrics and /status<br/>read every part's stats()"]
    end
    P["Prometheus or OTLP collector"]
    OP["Operator / lmcache query"]
    CK["Checkpoint file<br/><i>(new) age, size, write failures</i>"]
    MD["Metadata file<br/><i>(new) write failures counted</i>"]
    MP --> T --> G
    G --> V
    G --> K
    K -- "delete, prefetch" --> MP
    O -- "/metrics" --> P
    O -- "/status" --> OP
    C --> CK
    C --> MD
```

- **MP servers** add their running count of dropped events to every batch, so
  the coordinator knows how much was lost, not only that something was.
- **Event gate** counts loss, duplicates and freshness per server.
- **Views and controllers** report what they hold, what they do and what fails.
- **Background loops** record when they last succeeded and no longer die on an
  error.
- **Metrics and /status** read every part's `stats()` snapshot. `/metrics`
  feeds Prometheus or an OTLP collector; `/status` is for a person or
  `lmcache query`.

## 3. Metrics

Metrics are grouped by the question they answer. Every coordinator metric
starts with `lmcache_coordinator.`, left out in the tables; mp-server metrics
start with `lmcache_mp.`.

A name says what it measures: `<part>.<what>_<unit>`.

- A counter names the thing counted (`event_batches_received`); Prometheus adds
  `_total`.
- A gauge names the current amount (`keys_tracked`).
- The unit ends the name: `_seconds`, `_bytes`, `_ratio`, `_timestamp_seconds`.
- `server_` in a name means one series per mp server (`instance_id` label).

The two existing directory gauges become `key_directory.placements` and
`key_directory.placement_bytes`; the old names stay for one release.

### Is the view complete?

| Metric | Type | Labels | Tells you |
| --- | --- | --- | --- |
| `ingest.event_batches_received` | counter | `result` = applied, duplicate, stale | Batches received from mp servers, and what the gate did with them |
| `ingest.event_batches_missing` | counter | | Batches that never arrived: every skipped `seq` counts |
| `ingest.events_dropped_by_servers` | counter | | Events the mp servers say they threw away before sending |
| `ingest.server_event_batches_missing` | gauge | `instance_id` | Missing batches for one server's current run |
| `ingest.server_events_dropped` | gauge | `instance_id` | Dropped events for one server's current run |
| `ingest.server_view_incomplete` | gauge | `instance_id` | 1 while the coordinator knows it is missing part of this server's cache |
| `ingest.batch_apply_failures` | counter | `consumer`, `op` | A view or controller failed to apply a batch, so they now disagree. Page on this |

Fleet-wide counters drive the alerts; per-server gauges show which server.
Per-server values are gauges because a counter series lives as long as the
process, while servers come and go.

Two kinds of loss, counted separately:

- **Never arrived.** A skipped `seq`. A stream's first batch after the
  coordinator starts tracking it counts nothing: earlier batches were sent
  before it was listening.
- **Thrown away by the server.** Each batch carries the server's running count
  of dropped events (`dropped_events`): a full Kafka buffer, a failed delivery,
  a failed HTTP post. The coordinator adds up the increases.

`server_view_incomplete` clears when the server's slice is rebuilt: the server
restarts (its L1 is fenced) or leaves (deregister or heartbeat timeout). A
Kafka replay cannot fill the hole: the missing batches never reached the topic,
and applying them late could apply a store after its delete. After the
coordinator starts without a checkpoint it is 1 for every server until each
one restarts, since the coordinator missed their history: it is a
view-completeness gauge, not an alert.

### How old is it?

| Metric | Type | Labels | Tells you |
| --- | --- | --- | --- |
| `ingest.server_event_delivery_delay_seconds` | gauge | `instance_id` | For each server's last applied batch, time from sent to applied (across the two hosts' clocks) |
| `ingest.server_seconds_since_last_event` | gauge | `instance_id` | Server heartbeats but has stopped sending events |
| `kafka.unread_records` | gauge | `partition` | Records waiting to be read. Kafka's own lag tools can't see this: the coordinator never commits offsets |
| `kafka.partitions_assigned` | gauge | | Fewer than the topic has means another coordinator took some |
| `kafka.records_read` | counter | `result` = applied, undecodable, failed | Bad records being skipped |

### Are the loops alive?

| Metric | Type | Labels | Tells you |
| --- | --- | --- | --- |
| `background_loop.last_success_timestamp_seconds` | gauge | `loop` = health, checkpoint, eviction | When each loop last finished a pass |
| `background_loop.interval_seconds` | gauge | `loop` | Configured period, for alerts |
| `background_loop.passes` | counter | `loop`, `result` = ok, failed | Passes that failed |
| `controller.up` | gauge | `controller` | 1 up, 0 failed to start |
| `event_loop.blocked_seconds` | counter | | Total time the request thread was stuck (ingest and checkpoints can stall heartbeats); its rate is the fraction of time the loop was stalled |

Every loop runs on one runner (`BackgroundLoops`): a pass that raises is logged
and counted, and the loop keeps going. Without that, a loop that raised died
silently, and a metric alone would only report the death.

### Views: what the coordinator holds

Every view gets two generic metrics, then one row per thing it holds or serves.
A view added later gets the generic ones for free and adds its own from its
`stats()`.

| View | Metric | Type | Labels | Tells you | Phase |
| --- | --- | --- | --- | --- | --- |
| Every view | `ingest.batches_delivered`, `ingest.batch_apply_time_seconds` | counter | `consumer` | Batches handed to it and time spent applying them; time ÷ batches is the average apply time, which finds the slow view (blend hashing) | P0 |
| Every view | `ingest.batch_apply_failures` | counter | `consumer`, `op` | Failed to apply a batch (above) | P0 |
| Instance registry | `registry.servers_registered` | gauge | | Number of mp servers in the fleet | P1 |
| Instance registry | `registry.server_seconds_since_heartbeat` | gauge | `instance_id` | Time since each server's last heartbeat | P0 |
| Instance registry | `registry.server_membership_changes` | counter | `change` = registered, re_registered, deregistered, timed_out, unknown_heartbeat | Churn and flapping | P1 |
| Key directory | `key_directory.keys_tracked` | gauge | | Keys with at least one placement | P1 |
| Key directory | `key_directory.placements`, `key_directory.placement_bytes` | gauge | `tier` | Placements and their bytes per tier. A placement is one place a key is stored: L1 on one server, or one L2 backend. The two existing gauges, renamed | P0 |
| Key directory | `key_directory.server_l1_keys` | gauge | `instance_id` | L1 keys each server reported; a sudden drop to 0 is a fence | P1 |
| Blend index (if enabled) | `blend_index.chunks_indexed`, `blend_index.unique_chunk_contents`, `blend_index.namespaces` | gauge | | Size of the fragment index | P1 |
| Lookups (prefix and non-prefix) | `key_directory.lookups` | counter | `lookup_type` = prefix, non_prefix | Lookup requests served | P1 |
| Lookups (prefix and non-prefix) | `key_directory.lookup_chunks_requested`, `key_directory.lookup_chunks_found` | counter | `lookup_type` | Chunks asked for and chunks found; found ÷ requested is the hit rate | P1 |
| Lookups (prefix and non-prefix) | `key_directory.lookup_chunks_found_per_request` | histogram | `lookup_type` | How much each lookup found | P1 |
| Usage | `usage.bytes_stored` | gauge | `tier` | Fleet bytes per tier | P1 |
| Usage + server config | `usage.server_capacity_used_ratio` | gauge | `instance_id`, `tier` | Each server's usage against its declared capacity; absent when none declared | P1 |

Prefix lookups (`/directory/lookup`) and non-prefix lookups
(`/directory/blend-lookup`) share one set of lookup metrics, told apart by
`lookup_type`. Both take tokens and return the chunks the fleet holds, so a
chunk is the unit for both; a lookup by keys counts each key as one chunk.
The two lookup endpoints count each request they serve through
`KeyDirectory.record_lookup`; counting inside `lookup()` would also count the
coordinator's own lookups (eviction routing, deletes).

**Histograms.** OpenTelemetry has no instrument read at scrape time for a
histogram, so a component keeps a `Histogram` (fixed bucket bounds, counts,
sum, count) in its `stats()`, and `observability.py` exports it as
`<name>_bucket` (cumulative, labelled `le`), `<name>_sum` and `<name>_count`
counters. `histogram_quantile` works on them as on a native histogram; in
Prometheus each carries the counter's `_total` suffix.

**Requirement: keep these counts as you go (P0).** The directory gauges call
`KeyDirectory.stats()` on every scrape, and it holds the directory's and the
blend index's locks while it runs. Walking every key or the whole index there
would block ingest and lookups for a time that grows with the cache. So:

- The key directory reads its placement total from the per-tier totals it
  keeps on each add and remove.
- The blend index keeps running counts of chunks and of claims per namespace (a
  namespace drops out when its last claim goes); contents is the size of its
  table.
- `stats()` itself is therefore cheap, and gauges and `/status` both read it.

### Controllers: what the coordinator does

Controllers can be built in or loaded from a package
(`extra_config.controller_packages`). Each one's `controller.up` says whether
it started; prefetch reports its own work below.

A controller that runs its own loop starts it with
`runtime.background_loops.start(...)` (from `ControllerRuntime`), named after
its role (`eviction`). It is then reported with the `background_loop.*`
metrics and gets the same catch-and-continue as the coordinator's own loops.

| Role | Metric | Type | Labels | Tells you | Phase |
| --- | --- | --- | --- | --- | --- |
| Every controller | `controller.up` | gauge | `controller` (class name) | 0 if it failed to start (above) | P0 |
| Prefetch | `prefetch.requests_submitted` | counter | `result` = submitted, noop, unknown_server, server_error | Prefetches asked for, and why some never started: too short to fill a chunk, server not registered, server refused or unreachable | P1 |
| Prefetch | `prefetch.chunks_requested_per_request` | histogram | | How big each prefetch is | P1 |
| Prefetch | `prefetch.submit_duration_seconds` | histogram | | Time for the mp server to accept a prefetch | P1 |
| Prefetch | `prefetch.requests_in_flight` | gauge | `instance_id` | Prefetches submitted to each server and not yet seen completed | P1 |
| Prefetch | `prefetch.completion_duration_seconds` | histogram | | Submit to the first poll that sees it completed. Includes the caller's poll interval, so it is an upper bound | P1 |
| Prefetch | `prefetch.keys_requested`, `prefetch.keys_found` | counter | | Keys asked to warm, and keys found in L2 and loaded into L1. Found ÷ requested is the prefetch hit rate | P1 |
| Prefetch | `prefetch.status_polls` | counter | `result` = pending, completed, unknown_request, server_error | Polling load. `unknown_request` means the server lost the job (restart) or it was already collected | P1 |
| Prefetch | `prefetch.requests_abandoned` | counter | | Submitted but never polled to completion within 10 minutes: the caller gave up, or the server left | P1 |

The metadata rule: if saving a pin or quota fails, the endpoint undoes the
change and returns 503. A controller loaded from a package that serves
`/quota` or `/cache/pins` needs the same.

**Prefetch tracking.** The coordinator only relays prefetches today, so it
cannot time them. To measure duration, in-flight and abandoned prefetches, it
keeps a small in-memory table: `(instance_id, request_id)` → submit time. An
entry leaves on the first completed poll, a 404, or after 10 minutes (counted
as abandoned; a server that left ends up here, since its prefetches are never
polled to completion). Keys requested and found come from the completed
poll. The table is as big as the
number of prefetches in flight and is not checkpointed; a coordinator restart
forgets in-flight prefetches without counting them.

Whether prefetched keys were actually used before eviction is not visible here.
The mp server sees that, and can report it as `lmcache_mp.prefetch_keys_used`.

### Persistence (P1)

| Metric | Type | Labels | Tells you |
| --- | --- | --- | --- |
| `checkpoint_file.last_write_timestamp_seconds` | gauge | | When the last good checkpoint was written |
| `checkpoint_file.writes` | counter | `result` = ok, quiesce_timeout, error | Failed checkpoints |
| `checkpoint_file.write_duration_seconds` | histogram | `phase` = pause_ingest, copy, write | `copy` is how long ingest was paused |
| `checkpoint_file.size_bytes` | gauge | | Checkpoint size |
| `checkpoint_file.restored_on_start` | gauge | `outcome` = cold, restored, ignored | 1 for what happened at startup |
| `metadata_file.writes` | counter | `result` = ok, error | Pin and quota changes not saved |

`save_checkpoint` reports each write's result, phase timings and size, and
`load_checkpoint` what startup did; a `CheckpointMonitor` keeps them for the
metrics. A failed periodic write also fails that pass of the `checkpoint`
background loop, so the loop metrics and `/status` show it.

A failed metadata write undoes the change and returns 503, so the caller knows
and can retry safely.

### HTTP (P2)

| Metric | Type | Labels |
| --- | --- | --- |
| `http.requests_received` | counter | `route`, `method`, `status_class` |
| `http.request_duration_seconds` | histogram | `route`, `method` |
| `http.outbound_requests_sent` | counter | `call` = eviction_delete, prefetch_submit, prefetch_status, cache_delete; `result` |
| `http.outbound_request_duration_seconds` | histogram | `call` |

`route` is the route template (`/instances/{instance_id}/heartbeat`), never the
raw path.

### Status page

`GET /status` returns one JSON snapshot: source and Kafka positions,
per-server stream state, controller state, loop ages and last errors,
checkpoint state. It always returns 200; `/healthz` stays a plain liveness
check.

## 4. Alerts

Eleven alerts cover the three questions plus the basics. **Page** means wake
someone up: the coordinator is giving wrong answers or has stopped acting.
**Ticket** means look at it today. Thresholds are starting points; they ship as
an example Prometheus rules file. Names leave out the prefix, as in section 3.

| Alert | Fires when | Severity | Needs | First check |
| --- | --- | --- | --- | --- |
| Coordinator down | `up{job="lmcache-coordinator"} == 0` for 2 min | Page | exists | Pod status and restart logs |
| Views disagree | `increase(ingest_batch_apply_failures_total[10m]) > 0` | Page | P0 | `/status` consumers, then the traceback in logs. The views need a rebuild |
| Loop stopped | `time() - background_loop_last_success_timestamp_seconds > 3 * background_loop_interval_seconds` | Page | P0 | `/status` loops, `last_error` |
| Controller down | `controller_up == 0` | Page | P0 | Startup logs; eviction or prefetch is not running |
| Kafka partitions split | `kafka_partitions_assigned` below the topic's partition count for 5 min | Page | P0 | A second coordinator in the same consumer group |
| Events being lost | `rate(ingest_event_batches_missing_total[5m]) > 0 or rate(ingest_events_dropped_by_servers_total[5m]) > 0` for 10 min | Ticket | P0 | `ingest_server_event_batches_missing` and `ingest_server_events_dropped` show which server |
| Server silent | `ingest_server_seconds_since_last_event > 120` while `registry_server_seconds_since_heartbeat < 30` | Ticket | P0 | The server's event flush (it rides the L1 eviction tick) |
| Kafka falling behind | `kafka_unread_records` rising for 10 min and above 10,000 | Ticket | P0 | `event_loop_blocked_seconds`, `ingest_batch_apply_time_seconds_total` ÷ `ingest_batches_delivered_total` |
| Event loop stalled | `rate(event_loop_blocked_seconds_total[5m]) > 0.1` (stalled more than 10% of the time) | Ticket | P0 | `checkpoint_file_write_duration_seconds`, `/events` load |
| Checkpoint stale | `time() - checkpoint_file_last_write_timestamp_seconds > 600` | Ticket | P1 | `/status` checkpoint `last_error`, disk |
| Pin or quota not saved | `increase(metadata_file_writes_total{result="error"}[1h]) > 0` | Ticket | P1 | Disk at the metadata path (callers already got a 503) |

Right after a cold start the coordinator has missed every server's history.
`event_batches_missing` does not count a stream's first batch, so "Events being
lost" stays quiet then.

## 5. Logging

Logs say why; metrics say how often.

- Any failure that has a metric logs a WARNING the first time, then at most
  once a minute with a running count. Today a bad Kafka record or a failed
  eviction delete logs every time, which floods the log just when someone is
  reading it.
- Inside `except` blocks, use `logger.exception` so the traceback is kept.
- Keep the existing INFO lines for rare events: registration, timeouts,
  eviction plans, checkpoint restore.

## 6. Tracing

No tracing spans for now. Coordinator work is either one request in, one
response out, or a timer loop; a route latency histogram shows the same thing.

The one cross-process question, "how long until a stored entry shows up in the
directory?", is answered on the coordinator by
`ingest.server_event_delivery_delay_seconds` (send to applied). Time an event
waits on the mp server before it is sent is not measured.

When blend lookups carry trace context (the `traceparent` work in the mp
observability design), the coordinator joins as a server span through FastAPI
instrumentation. Nothing here blocks that.

## 7. Execution plan

Three steps, each one or more small PRs. Tests read public state only:
`stats()`, `/status`, and metrics from a test-only meter.

| Step | Scope | Tests |
| --- | --- | --- |
| P0 | `lmcache_coordinator.` names and the rename of the two existing gauges; `dropped_events` on each batch, counted by the mp server; ingest counters and per-server gauges; consumer errors and apply time; running totals in the key directory and blend index so gauges never walk them; Kafka lag, partitions, bad records; loop watchdog and the catch-and-continue fix; controller up; event-loop blocked time; event delivery delay; `/status`; example alert rules | `test_event_gate.py`, `test_event_broadcaster.py`, `test_kafka_event_source.py`, `test_kafka_event_sink.py`, `test_cache_events.py`, `test_health.py`, `test_observability.py`, `test_metrics_api.py` |
| P1 | View and controller metrics (registry, directory, blend, usage, prefetch); checkpoint metrics; if P0 shows a blocked event loop (`event_loop.blocked_seconds` rate above 0.05, i.e. stalled more than 5% of the time), move `POST /events` ingest to a worker thread; metadata write failures undo the change and return 503 | `persistence/` tests, `test_eviction_controller.py`, `test_quota_api.py`, `test_registry.py` |
| P2 | HTTP server and outbound metrics; tracing once trace context exists | `test_metrics_api.py`, `test_directory_api.py`, `test_cache_api.py` |

Components keep plain running totals and `stats()` snapshots and create no
instruments. A component with metrics is a `MetricSource` (it has `stats()`)
and calls `register_metric_source(self)` (from `metric_sources.py`) as the
last line of its `__init__`; every metric is defined in `observability.py`,
one list per component type, read from that component's `stats()`. Which
controllers started is a component too, `ControllerStatus`, which the
lifespan marks as it enters each controller. When metrics are on, the app
calls `register_coordinator_metrics()` once, after building everything: it
takes the registered components and creates their instruments. So the app
names no component for metrics.

```python
class EventGate:
    def __init__(self, broadcaster, quiesce) -> None:
        ...
        register_metric_source(self)

# observability.py
def _specs_for(component: object) -> list[_MetricSpec]:
    match component:
        case EventGate():
            return _event_gate_specs(component)
        ...
```

Registration is keyed by class, since a coordinator process builds one of each;
a later component of the same class replaces the earlier. Adding a component
with metrics means one `register_metric_source(self)` line and one `case` with
its metric list.
`register_coordinator_metrics` takes an optional `meter`: production passes
nothing and gets the global one; tests pass a private meter so readings don't
leak between tests, and clear the registered components before each test.
