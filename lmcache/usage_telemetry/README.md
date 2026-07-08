# Anonymous Usage Telemetry (`lmcache/usage_telemetry/`)

Phone-home usage statistics: what LMCache deployments look like and how much
caching they do. This is **not** operator observability — Prometheus/OTel
metrics for operators live in `lmcache/v1/mp_observability/` (MP mode) and
`lmcache/observability.py` (single-process mode). The two systems share event
sources in places but have different consumers (LMCache maintainers vs.
deployment operators), different transports, and independent opt-outs.

## Package layout

| Module | Contents |
|---|---|
| `identity.py` | Opt-out gate (`is_usage_tracking_enabled`), `UsageIdentity` (session/machine ids) |
| `transport.py` | `UsageMessageSender` (HTTP), payload building, `USAGE_SCHEMA_VERSION` |
| `env_probe.py` | `EnvMessage` and hardware/platform/cloud detection |
| `one_shot.py` | `UsageContextBase`, single-process `UsageContext`, `InitializeUsageContext` |
| `continuous.py` | `ContinuousUsageContext` interval counters and lifespan histogram |
| `mp.py` | MP server `MPUsageContext`, `MPServerMessage`, `InitializeMPUsageContext` |

`lmcache/usage_context.py` remains as a backward-compatibility shim
re-exporting the pre-package public names.

## Reporting paths

| Path | Class | Cadence | Producer |
|---|---|---|---|
| One-shot | `UsageContext` | once at engine startup | `LMCacheEngine.__init__` (single-process) |
| One-shot | `MPUsageContext` | once at server startup | `run_cache_server` (MP mode) |
| Continuous | `ContinuousUsageContext` | every `LMCACHE_USAGE_TRACK_INTERVAL` s (default 600) | `LMCacheStatsLogger.log_worker` (single-process) |

One-shot reporters share `UsageContextBase`, which owns identity, transport,
and optional local logging; subclasses only define which messages to send
(`_collect_messages`). MP-mode continuous reporting (an EventBus subscriber
feeding the same endpoints) is planned but not yet implemented.

## Message catalog

All messages POST to `LMCACHE_USAGE_TRACK_URL` (default
`http://stats.lmcache.ai:8080`).

| `message_type` | Endpoint | Contents |
|---|---|---|
| `EnvMessage` | `/context` | cloud provider, CPU/GPU/memory, install source |
| `EngineMessage` | `/context` | single-process engine config + model/kv metadata |
| `MPServerMessage` | `/context` | MP server config: chunk size, transfer mode, L1 medium/size, L2 adapter types, policies |
| `MetadataMessage` | `/context` | start time, uptime (single-process only) |
| `ContinuousContextMessage` | `/cache-usage` | interval hit/stored tokens, stored KV bytes |
| `CacheLifespanMessage` | `/cache-lifespan` | cache-entry lifespan histogram |

`MPServerMessage` carries no model information because vLLM instances register
with the MP server only after startup; model names ride the continuous
messages instead.

## Identity and association contract

Every payload — one-shot or continuous — is stamped with:

- `session_id`: random UUID minted once per process. The join key: the
  `/context` rows keyed by a `session_id` describe the deployment that
  produced all other rows with that `session_id`.
- `machine_id`: random UUID persisted at `~/.config/lmcache/machine_id`.
  Groups sessions from the same machine across restarts (deployment-level
  dedup, restart cadence). Empty string when the file cannot be created.
- `schema_version`: integer, bump `USAGE_SCHEMA_VERSION` whenever a field
  changes meaning so backend analysis can partition by schema.

Continuous messages additionally carry `sequence_number` (monotonic per
session). Send failures drop the interval's data rather than retrying —
telemetry must never accumulate unbounded state — so sequence gaps are the
backend's signal that intervals were lost rather than idle.

Privacy rules:

- Identifiers are random UUIDs, never derived from hardware (no MAC address,
  hostname, or `/etc/machine-id`).
- The MP server's `instance_id` is deliberately excluded from payloads: it is
  operator-settable (`--instance-id`) and may contain identifying strings.
  It stays in the operator-facing OTel `service.instance.id` only.

## Opt-out semantics

`is_usage_tracking_enabled()` is the single gate, checked by every path:

1. `LMCACHE_TRACK_USAGE=false` (LMCache-specific),
2. `DO_NOT_TRACK` in `1`/`true`/`yes` (cross-tool convention),
3. presence of `~/.config/lmcache/do_not_track`.

When disabled, the factory functions return `None`, the continuous reporter
no-ops, and no state files (`machine_id`) are created.

## Threading contract

- One-shot reports run on a daemon thread spawned by the factory functions
  (`InitializeUsageContext`, `InitializeMPUsageContext`); startup must never
  block on the stats server. `report_once()` itself is synchronous so tests
  can call it directly.
- Single-process continuous flushes happen inline on the stats-logger loop
  (`LMCacheStatsLogger.log_worker`), which tolerates the 5 s send timeout.
- Rule for the planned MP continuous subscriber: EventBus subscriber
  callbacks run on the bus's single drain thread and must only increment
  in-memory counters; sending must happen on a dedicated flush thread.
  A blocking send on the drain thread would stall all operator metrics,
  logging, and tracing.

## Transport and testing

`UsageMessageSender` is the injectable transport boundary: it POSTs JSON with
a 5 s timeout and swallows every failure at debug log level. Tests inject a
recording stub into the usage contexts (see `tests/test_usage_telemetry.py`);
no test may hit the network. The test suite globally disables tracking via an
autouse fixture in `tests/conftest.py` so no other test phones home.

## Extension guide

To add a new message type:

1. Define a dataclass with flat, JSON-serializable fields (the stats backend
   stores flat key-value payloads; join lists into comma-separated strings).
2. Send it through `build_usage_payload` so identity and schema stamping
   stay uniform.
3. Coordinate with the stats-server owner: unknown `message_type` values are
   dropped silently on the backend.
4. Bump `USAGE_SCHEMA_VERSION` only when an *existing* field changes meaning;
   adding a message type or field is not a schema bump.
