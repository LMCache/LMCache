# Continuous Usage Metrics for MP Mode

Status: draft. Extends the init-time telemetry (see [README.md](README.md))
with runtime metrics. Receiver is InfluxDB, visualized in Grafana.

## Goals

1. **Parity**: MP mode emits the continuous metrics the single-process path
   already sends (`ContinuousContextMessage`).
2. **Fleet dashboards**: high-level Grafana insights, e.g. total KV volume
   per week, hit rate over time.
3. **Hybrid-model attribution**: KV volume split by attention architecture
   (full / full+SWA / full+linear / DSA).
4. **Reuse insights**: idealized (infinite-storage) hit rate, chunk
   lifecycle time, and reuse patterns (bursty-short vs. sustained-long).

## Parity baseline: continuous metrics non-MP mode sends today

Flushed by `LMCacheStatsLogger.log_worker` via `ContinuousUsageContext`
every `LMCACHE_USAGE_TRACK_INTERVAL` (default 600 s). All payloads also
carry the common header (`session_id`, `machine_id`, `schema_version`,
`deployment_mode`) and `sequence_number`.

| Message (endpoint) | Field | Meaning |
|---|---|---|
| `ContinuousContextMessage` (`/cache-usage`) | `interval_num_hit_tokens` | tokens served from LMCache this interval |
| | `interval_num_stored_tokens` | tokens stored this interval |
| | `interval_stored_kv_size` | stored bytes, *estimated* as kv-bytes-per-token × stored tokens |
| `CacheLifespanMessage` (`/cache-lifespan`) | `cache_lifespan_histogram` | store→reuse gap per reused chunk, in **minutes**; fixed buckets 0–5000 min (~3.5 days) |

## Metric sources (MP mode)

All runtime metrics come from an EventBus subscriber. Drain-thread callbacks
only increment in-memory state; a dedicated flush thread sends every
`LMCACHE_USAGE_TRACK_INTERVAL` (default 600 s). Registered in
`run_cache_server`, gated by `is_usage_tracking_enabled()`.

| Metric | Event source |
|---|---|
| retrieved (hit) tokens | `MP_RETRIEVE_END` |
| requested / lookup-hit tokens (goal-4 denominators) | `MP_LOOKUP_PREFETCH_END` |
| stored tokens / stored bytes (exact) | `MP_STORE_END` |
| eviction counts | `L1_KEYS_EVICTED`, `L2_KEYS_EVICTED` |
| chunk identity stream (goal 4) | `MP_LOOKUP` (`chunk_hashes`) |
| attention architecture (goal 3) | KV-cache registration (`AttnWindowDesc`) |

The non-MP lifespan histogram gets no direct MP equivalent: its ~3.5-day
bucket ceiling and store→reuse definition cannot capture the goal-4
patterns (first-reuse→last-reuse spans up to a month); the chunk tracker
supersedes it. MP's `interval_stored_kv_size` uses exact bytes from
`MP_STORE_END` rather than the non-MP estimate. For parity,
`interval_num_hit_tokens` means *retrieved* tokens (the non-MP
definition), so it is sourced from `MP_RETRIEVE_END`, not lookup hits;
engine-driven transfer coverage must be verified during implementation.

**Message organization**: MP reuses the shared message *types*
(`ContinuousContextMessage` etc.) but POSTs to mode-specific endpoints —
every MP payload goes under the `/mp/` prefix (`/mp/context`,
`/mp/cache-usage`). Rationale: MP and non-MP code paths are fully
separated (`usage_telemetry/mp/` vs `usage_telemetry/non_mp/`) because
the non-MP path is scheduled for removal within months; separate
endpoints let the backend drop the legacy handlers with it. The ingest
layer should still write both endpoints into one Influx measurement per
`message_type` (with the `deployment_mode` tag) so fleet-wide panels
never union across measurements. MP-only additions (evictions,
per-model counters) are new message types in later PRs, not new fields
on the shared class.

**Data availability**: `MP_LOOKUP` emission is `has_subscribers`-gated —
zero cost until a telemetry subscriber registers. KV-cache registration
has no EventBus event today; PR 2 adds one (or hooks the layout
registry). Counters ride the EventBus, so `--disable-observability`
zeroes them; the interval heartbeat itself continues (init-time messages
unaffected).

## InfluxDB schema rules

Locked in once, documented per field in `messages.py`:

- **Tags** (indexed, low-cardinality only): `deployment_mode`,
  `message_type`, `model_name`, `attn_arch`, `lmcache_version`.
  `session_id` is a **field**, never a tag (unbounded series growth);
  `machine_id` is a tag only while fleet size stays ~10^4.
- **Interval deltas, not cumulative counters**: weekly volume is
  `SUM(interval_stored_kv_size) GROUP BY time(1w)`; restart-safe.
- **Histograms** stay dict fields on the wire; ingest explodes each bucket
  into a point with an `le=<bound>` tag (Grafana heatmap format).
- Continuous messages carry `sequence_number` (gaps = lost intervals) and
  `uptime_seconds`. Send numerators, denominators, and sample rates —
  never precomputed ratios.

## Hybrid-architecture attribution

At KV-cache registration, derive `attn_arch` from
`AttnWindowDesc.num_chunks_in_sw`: all `-1` → `full`; windows `> 1` chunk →
`full+swa`; windows `== 1` chunk → `full+linear` (mamba/GDN state);
`use_mla` reported as its own bit. Reported via the re-landed
`MPInstanceMessage` and stamped as a tag on per-model interval counters, so
"KV volume by architecture" needs no query-time joins.

**Known gap**: DSA (sparse attention) is invisible in the KV window layout.
Exact classification needs the connector to pass the HF `model_type` in the
registration payload — an additive protocol field, done as its own PR.

## Chunk reuse tracker (goal 4)

A deterministic hash-sample of chunks (`hash % R == 0`, default `R=64`) is
tracked in a bounded table: `first_seen`, `first_reuse`, `last_access`,
`reuse_count`, `was_stored`. Chunk hashes never leave the process; only
bucketed aggregates are sent.

- **Ideal hit rate**: among sampled lookup accesses, the fraction whose
  chunk was seen before — vs. actual hit rate, the gap is capacity+policy
  miss headroom. Horizon = tracker retention; restarts reset it (reported).
- **Lifespan**: `last_access - first_reuse`, emitted when an entry retires
  (idle > TTL, default ~3 days). Log buckets from seconds to ~1 month.
  Cap-forced retirements are counted separately (they bias lifespan down).
- **Reuse pattern**: 2D log-bucketed histogram `reuse_count x lifespan`
  emitted at retirement — separates multi-turn-burst / daily-sustained /
  shared-prefix populations — plus a 1D inter-reuse-gap histogram.

Knobs (env, conservative defaults): sample denominator `R`, idle TTL,
table cap.

## PR plan

| PR | Content |
|---|---|
| 1 | **(done)** MP continuous subscriber + flush thread: parity counters + `uptime_seconds` (evictions moved to PR 2) |
| 2 | `MPInstanceMessage` re-land + `attn_arch` + per-model/arch tags on counters |
| 3 | `ChunkReuseTracker` pure class + unit tests (no wiring) |
| 4 | Tracker subscriber wiring + `ReusePatternMessage` |
| 5 | Connector passes `model_type` at registration (DSA, exact arch) |
| — | Backend (parallel, out of repo): JSON→Influx ingest mapping, starter dashboard |

## Open decisions

1. ~~Per-model tags in PR 1 vs. PR 2~~ — resolved: PR 1 is strict parity
   (hit/stored tokens, stored bytes, `uptime_seconds`); evictions and
   per-model tags land in PR 2 as new message types.
2. Confirm tag/field split against the actual Influx ingest.
3. New metrics stay MP-only; the tracker is mode-agnostic if the
   single-process path ever needs it.
4. PR 2 registration signal: new `MP_KV_REGISTERED` EventType vs. direct
   layout-registry hook (lean: the event, now that multiple consumers are
   plausible).
