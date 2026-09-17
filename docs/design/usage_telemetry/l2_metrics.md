# L2 Usage Telemetry

Status: draft. Sibling track of
[continuous_metrics.md](continuous_metrics.md) (same reporter
infrastructure, receiver, and schema rules); covers the L2 storage
layer of the MP cache server. Supersedes the L2 half of that doc's
"eviction counts" roadmap row (L1 evictions stay there).

## Goals

1. **Connector attribution** (primary): which L2 adapter types
   (`l2_name`, e.g. `"dax"`, `"nixl_store"`) are used in the fleet, for
   how long, and how many bytes they move.
2. **Hit rate**: L2 prefetch lookup hits vs. lookups (raw
   numerator/denominator; this is a *contiguous-prefix* hit count, so
   dashboards should label it "prefix hit rate").
3. **Occupancy**: how full L2 runs vs. configured capacity — are
   deployments capacity-bound or over-provisioned?

Non-goals: per-adapter-*instance* and per-tenant detail (operator
observability covers those with per-salt attributes), and
reconfiguration-op tracking — `L2_RECONFIGURED` / `L2_ADAPTER_ADDED` /
`L2_ADAPTER_REMOVED` events were considered and discarded: connector
attribution is served better by traffic counters plus presence
sampling, which is also crash-correct (no open add/remove intervals).

## Design

No new events anywhere: all traffic events already carry `l2_name`
(`L2_STORE_COMPLETED`, `L2_LOAD_TASK_SUBMITTED`), and adapter presence
is probed from `StorageManager` at flush time, which is automatically
correct across `/reconfigure` and runtime adapter add/remove.

**`L2ConnectorUsageMessage`** *(implemented)* — one message per
*active adapter type* per interval (`ENDPOINT = "l2-usage"`; type count
is bounded by the adapter registry, so cardinality is safe; `l2_name`
is an Influx tag):

| Field | Source |
|---|---|
| `l2_name` | probe / event metadata |
| `active_seconds` | presence probe: ≈ flush interval while the type is active (sub-interval add/remove rounds to one interval) |
| `interval_stored_bytes` | `L2_STORE_COMPLETED.bytes_transferred` |
| `interval_store_succeeded_keys` / `interval_store_failed_keys` | `L2_STORE_COMPLETED` |
| `interval_load_submitted_keys` / `interval_load_submitted_bytes` | `L2_LOAD_TASK_SUBMITTED` (*requested*; completion events carry no counts) |
| `bytes_used` / `capacity_bytes` / `unbounded_adapters` | occupancy probe (below); `bytes_used = -1` when the probe was unavailable for the type |

Duration = `SUM(active_seconds) GROUP BY l2_name`; volume =
`SUM(interval_stored_bytes) GROUP BY l2_name`; adoption = distinct
`machine_id` per `l2_name`. An idle-but-configured adapter still
reports presence (zero traffic), so duration works without load.

This is the group-by case that does not fit the fixed-field
`MetricSpec` reporter: `L2ConnectorUsageReporter`
(`usage_telemetry/l2_usage.py`) keeps per-`l2_name` counter dicts
(drain thread) and, at flush, emits one message per type present in the
probe or the counters (all sharing that flush's `sequence_number`).

**Occupancy rides the same probe** rather than a separate `GaugeSpec`
mechanism: the flush-time probe wraps the new public
`StorageManager.get_l2_usages_by_type()` (per-adapter `AdapterUsage`
snapshots grouped by type; failing adapters skipped) and aggregates per
type — `capacity_bytes` sums only capacity-bounded adapters
(`total_capacity_bytes > 0`) and `unbounded_adapters` counts the rest,
so the backend knows when used/capacity is a meaningful ratio.
Per-type occupancy subsumes the machine-level gauge (backend sums over
types); `GaugeSpec` is deferred until a gauge that is not per-type
appears.

**`L2UsageMessage`** — machine-level counters with no per-type source
(request-level events carry no `l2_name`), via the existing `MetricSpec`
reporter (same endpoint; `message_type` discriminates):

| Field | Source |
|---|---|
| `interval_l2_lookup_keys` | `L2_PREFETCH_LOOKUP_SUBMITTED.key_count` |
| `interval_l2_lookup_hit_keys` | `L2_PREFETCH_LOOKUP_COMPLETED.prefix_hit_count` |
| `interval_l2_loaded_keys` / `interval_l2_load_failed_keys` | `L2_PREFETCH_LOAD_COMPLETED` |
| `interval_l2_prefetch_failures_{l1_oom,not_found,other}` | `L2_PREFETCH_FAILED.reason` (conditional extract returning `None`; `_other` absorbs new reasons, e.g. planned `serde_failure`) |
| `interval_l2_evicted_keys` | `L2_KEYS_EVICTED.key_count` (policy eviction; hotplug deletions bypass this event, verified) |

This requires generalizing `MPContinuousUsageReporter` with a
`message_cls: type[UsageMessage]` parameter (default
`ContinuousContextMessage`, behavior-preserving); the exact-cover
validation runs against `fields(message_cls)` minus
`{sequence_number, uptime_seconds}`. One reporter instance per message
type, each with its own flush thread and `sequence_number` space.

## Privacy

Only counts, bytes, seconds, and factory type names (`l2_name`) leave
the machine. Excluded: `bytes_by_cache_salt` (tenant identifiers),
device paths, adapter indices, `ObjectKey`s. Opt-out, identity,
no-throw guards, and drop-not-retry semantics inherited unchanged.

## PR plan

| PR | Content |
|---|---|
| `l2-connector-attribution` | **(implemented)** `L2ConnectorUsageMessage` incl. per-type occupancy + `L2ConnectorUsageReporter` + `StorageManager.get_l2_usages_by_type()` |
| `l2-server-counters` | reporter `message_cls` generalization + `L2UsageMessage` counters (hit rate, failures, policy evictions) |
| `influx-ingest` | backend, out of repo: register both message types + `l2-usage` endpoint (unknown `message_type`s are dropped silently) |

`l2-server-counters` specs will live in `metric_specs.py`; the connector
subscriber is `l2_usage.py`; both stay un-exported from the package root.
Testing follows the existing pattern: recording senders, fake events on
a real EventBus, validation errors, probe-failure → sentinel.

## Open decisions

1. Keys vs. tokens on the wire (lean: keys; backend converts via the
   one-shot chunk-size join).
2. Influx tag/field mapping for the new endpoint — confirm with the
   ingest owner.
