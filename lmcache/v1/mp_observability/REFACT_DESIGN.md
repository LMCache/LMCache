# MP Observability Refactor: Unified Event Bus Design

**Status:** Proposal / RFC
**Author:** Roy
**Date:** 2026-03-16

---

## TL;DR

The `mp_observability` module has two separate systems (Listener/Logger for Prometheus
metrics, and TelemetryController for event tracing) that both do the same thing: record
when operations start/end and react to them. This proposal **unifies them into a single
EventBus with pub/sub dispatch**, where producers call `bus.publish(Event(...))` and
subscribers (metrics, logging, tracing) register callbacks for the event types they care
about. We adopt **OpenTelemetry** as the instrumentation API so metrics, traces, and logs
flow through one vendor-neutral SDK — existing Prometheus dashboards continue to work via
the OTel Prometheus exporter. The migration is broken into **4 PRs**: (1) EventBus core +
OTel dependency, (2) L1 + StorageManager migration, (3) MP Server telemetry migration,
(4) cleanup. PRs 2 and 3 can land in parallel.

---

## 1. Motivation

The `mp_observability` module currently has two parallel systems that do fundamentally
the same thing — record the start and/or end of operations and react to them:

**System A — Listener/Logger (stats + Prometheus)**
- Producers (`L1Manager`, `StorageManager`) iterate `_registered_listeners` and call
  typed callback methods (e.g., `on_l1_keys_read_finished(keys)`).
- Consumers (`L1ManagerStatsLogger`, `StorageManagerStatsLogger`) implement Listener ABCs,
  accumulate stats into dataclass buckets (`L1Stats`, `StorageManagerStats`), and
  periodically flush to Prometheus via `PrometheusController`.
- Callbacks are **synchronous** — called inline from business logic under a `_stats_lock`.
- Uses `prometheus_client` directly.

**System B — Telemetry (event queue + processors)**
- Producers (call sites in `server.py`) manually construct `TelemetryEvent` objects
  with START/END types and a `session_id` for correlation, then call `log_telemetry()`.
- `TelemetryController` queues events in a deque and drains them in a background thread,
  dispatching to registered `TelemetryProcessor` instances (currently only `LoggingProcessor`).
- Events are **asynchronous** — queue + background drain thread.
- Uses string-based event names (`"store"`, `"retrieve"`, `"lookup_and_prefetch"`).

**Problems:**
1. Adding a new observable operation requires touching both systems — defining a Listener
   ABC method *and* adding telemetry event emission.
2. The same "event" concept is modeled twice: typed interface methods vs. generic event
   objects with string names.
3. Two separate controller singletons (`PrometheusController`, `TelemetryController`),
   two drain/flush threads, two registration patterns.
4. The Listener ABC approach forces consumers to implement all methods even when they
   only care about a subset (e.g., `on_l1_keys_reserved_read` is a no-op in the stats
   logger but must still be defined).
5. Direct `prometheus_client` usage couples the codebase to a single metrics backend,
   making it harder to adopt OpenTelemetry or export to other systems.

---

## 2. Proposed Design: Event Bus with Pub/Sub

Replace both systems with a single **Event Bus** that producers publish to and consumers
subscribe to, using **OpenTelemetry** as the underlying instrumentation API.

### 2.1 Architecture Overview

```
Producers (L1Manager, StorageManager, MPServer)
    │
    │  bus.publish(Event(...))
    ▼
EventBus  (single deque + drain thread)
    │
    │  dispatches by EventType
    │
    ├──► L1MetricsSubscriber        → OTel counter.add(...)
    ├──► SMMetricsSubscriber        → OTel counter.add(...)
    ├──► MPServerMetricsSubscriber  → OTel counter.add(...), histogram.record(...)
    ├──► MPServerSpanSubscriber     → OTel span start/end (explicit timestamps)
    ├──► L1LoggingSubscriber        → logger.debug(...)
    ├──► SMLoggingSubscriber        → logger.debug(...)
    └──► ...

OpenTelemetry SDK  (configured once at startup)
    │
    ├──► PrometheusMetricReader   → /metrics endpoint (backward compatible)
    ├──► OTLPMetricExporter       → Datadog / Grafana / etc. (optional)
    └──► BatchSpanProcessor       → Jaeger / Tempo (for traces from span subscriber)
```

### 2.2 Event Model

A single `Event` dataclass replaces both the Listener callback signatures and
`TelemetryEvent`:

```python
class EventType(Enum):
    # L1 Manager events
    L1_READ_RESERVED     = "l1.read.reserved"
    L1_READ_FINISHED     = "l1.read.finished"
    L1_WRITE_RESERVED    = "l1.write.reserved"
    L1_WRITE_FINISHED    = "l1.write.finished"
    L1_KEYS_EVICTED      = "l1.keys.evicted"

    # StorageManager events
    SM_READ_PREFETCHED          = "sm.read.prefetched"
    SM_READ_PREFETCHED_FINISHED = "sm.read.prefetched_finished"
    SM_WRITE_RESERVED           = "sm.write.reserved"
    SM_WRITE_FINISHED           = "sm.write.finished"

    # MP Server request-level events
    MP_STORE_START              = "mp.store.start"
    MP_STORE_END                = "mp.store.end"
    MP_RETRIEVE_START           = "mp.retrieve.start"
    MP_RETRIEVE_END             = "mp.retrieve.end"
    MP_LOOKUP_PREFETCH_START    = "mp.lookup_prefetch.start"
    MP_LOOKUP_PREFETCH_END      = "mp.lookup_prefetch.end"

    # L2 (placeholder — add members when L2 interface is finalized)


@dataclass
class Event:
    event_type: EventType
    timestamp: float = 0.0            # stamped by EventBus.publish() at call time
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""              # for correlating start/end pairs
```

**Timestamp semantics:** `timestamp` is set by `EventBus.publish()` at the moment it is
called — **not** when the drain thread processes the event. This is critical because:

- Subscribers (span subscriber, latency metrics) use the timestamp for duration
  computation. Using drain-time would introduce arbitrary queue delay into latency
  measurements.
- For MP server events, `publish()` is invoked from **CUDA host callbacks**
  (`cupy_stream.launch_host_func`). The CUDA runtime calls the host function when the
  GPU stream reaches that point, so `time.time()` inside `publish()` captures the
  GPU-accurate moment — not the earlier moment when the Python line was scheduled.
- For L1Manager/StorageManager events, `publish()` is called directly from Python,
  so the timestamp reflects the actual operation time.

All event-specific data lives in `metadata`. See [Section 2.7](#27-metadata-contracts)
for per-event-type metadata contracts.

### 2.3 Event Bus

```python
EventCallback = Callable[[Event], None]

class EventBus:
    def __init__(self, config: EventBusConfig):
        self._subscribers: dict[EventType, list[EventCallback]] = defaultdict(list)
        self._queue: collections.deque[Event] = collections.deque(maxlen=config.max_queue_size)
        self._wake = threading.Event()
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Register a callback for a specific event type."""
        self._subscribers[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """Submit an event (hot path — non-blocking)."""
        event.timestamp = time.time()
        self._queue.append(event)
        self._wake.set()

    def start(self) -> None:
        """Start the background drain thread."""
        ...

    def stop(self) -> None:
        """Stop the drain thread and flush remaining events."""
        ...

    def _drain_all(self) -> None:
        """Pop all queued events and dispatch to subscribers."""
        while True:
            try:
                event = self._queue.popleft()
            except IndexError:
                break
            for cb in self._subscribers.get(event.event_type, []):
                try:
                    cb(event)
                except Exception:
                    logger.exception(...)
```

Replaces both `PrometheusController` and `TelemetryController` — one queue, one thread,
one dispatch loop.

### 2.4 Subscriber ABC

```python
class EventSubscriber(ABC):
    """Base class for per-component event subscribers."""

    @abstractmethod
    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return event_type → callback mapping.

        Called once during register(). The EventBus wires these up.
        """
        ...

    def register(self, bus: EventBus) -> None:
        """Subscribe all declared handlers to the bus."""
        for event_type, callback in self.get_subscriptions().items():
            bus.subscribe(event_type, callback)

    def shutdown(self) -> None:
        """Optional cleanup hook. Called on EventBus.stop()."""
        pass
```

### 2.5 Per-Component Subscribers (Metrics)

Each component gets its own subscriber. Subscribers use the **OpenTelemetry Metrics API**
instead of `prometheus_client` directly.

```python
from opentelemetry import metrics

class L1MetricsSubscriber(EventSubscriber):
    def __init__(self):
        meter = metrics.get_meter("lmcache.l1")
        self._read_counter = meter.create_counter(
            "lmcache_mp.l1_read_keys",
            description="Total keys read from L1",
        )
        self._write_counter = meter.create_counter(
            "lmcache_mp.l1_write_keys",
            description="Total keys written to L1",
        )
        self._evicted_counter = meter.create_counter(
            "lmcache_mp.l1_evicted_keys",
            description="Total keys evicted from L1",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_finished(self, event: Event) -> None:
        self._read_counter.add(len(event.metadata["keys"]))

    def _on_write_finished(self, event: Event) -> None:
        self._write_counter.add(len(event.metadata["keys"]))

    def _on_evicted(self, event: Event) -> None:
        self._evicted_counter.add(len(event.metadata["keys"]))


class SMMetricsSubscriber(EventSubscriber):
    """Metrics subscriber for StorageManager events."""
    # Same pattern — create OTel counters, subscribe to SM_* events
    ...
```

### 2.6 Per-Component Subscribers (Logging)

```python
class L1LoggingSubscriber(EventSubscriber):
    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.L1_READ_FINISHED: self._on_read_finished,
            EventType.L1_WRITE_FINISHED: self._on_write_finished,
            EventType.L1_KEYS_EVICTED: self._on_evicted,
        }

    def _on_read_finished(self, event: Event) -> None:
        logger.debug("L1 read finished: %d keys", len(event.metadata["keys"]))

    def _on_write_finished(self, event: Event) -> None:
        logger.debug("L1 write finished: %d keys", len(event.metadata["keys"]))

    def _on_evicted(self, event: Event) -> None:
        logger.debug("L1 eviction: %d keys", len(event.metadata["keys"]))
```

### 2.7 Metadata Contracts

Each `EventType` has a documented metadata schema. This serves as the contract between
producers and subscribers. Subscribers can rely on these keys being present.

| EventType | Metadata keys | Types |
|---|---|---|
| `L1_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_READ_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_KEYS_EVICTED` | `keys` | `list[ObjectKey]` |
| `SM_READ_PREFETCHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_READ_PREFETCHED_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_RESERVED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `MP_STORE_START` | `session_id`, `device` | `str`, `str` |
| `MP_STORE_END` | `session_id`, `stored_count` | `str`, `int` |
| `MP_RETRIEVE_START` | `session_id`, `device` | `str`, `str` |
| `MP_RETRIEVE_END` | `session_id`, `retrieved_count` | `str`, `int` |
| `MP_LOOKUP_PREFETCH_START` | `session_id` | `str` |
| `MP_LOOKUP_PREFETCH_END` | `session_id`, `found_count` | `str`, `int` |

### 2.8 OTel Spans via Explicit Start/End

OTel's idiomatic `with tracer.start_as_current_span(...)` pattern requires start and end
in the **same lexical scope**. This does not apply to any of our observable operations:

- **`store` / `retrieve`:** START and END are fired from **separate CUDA host callbacks**
  (`cupy_stream.launch_host_func`) — the GPU stream schedules them asynchronously.
- **`lookup_and_prefetch`:** START is in `lookup()`, END is in `query_prefetch_status()`
  — two entirely separate RPC methods called at different times by the client.
- **L1Manager:** `reserve_read()` and `finish_read()` are separate method calls invoked
  at different points by `StorageManager`.
- **StorageManager:** `reserve_write()` and `finish_write()` are separate method calls.

However, OTel **does** support explicit span management with caller-provided timestamps.
A subscriber can create a span on START, stash it by `session_id`, and end it on END —
preserving GPU-accurate timing while producing real OTel spans:

```python
class MPServerSpanSubscriber(EventSubscriber):
    """Creates OTel spans from START/END event pairs."""

    def __init__(self):
        self._tracer = trace.get_tracer("lmcache.mp_server")
        self._pending: dict[str, trace.Span] = {}

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_STORE_START: self._on_store_start,
            EventType.MP_STORE_END: self._on_store_end,
            # ... same for retrieve, lookup_prefetch
        }

    def _on_store_start(self, event: Event) -> None:
        span = self._tracer.start_span(
            "mp.store",
            start_time=_sec_to_ns(event.timestamp),
        )
        span.set_attribute("device", event.metadata.get("device", ""))
        self._pending[event.session_id] = span

    def _on_store_end(self, event: Event) -> None:
        span = self._pending.pop(event.session_id, None)
        if span is not None:
            span.set_attribute("stored_count", event.metadata.get("stored_count", 0))
            span.end(end_time=_sec_to_ns(event.timestamp))
```

This approach:
- **Preserves GPU-accurate timing** — timestamps originate from CUDA host callbacks,
  passed through the EventBus, and forwarded to the OTel span as explicit start/end times.
- **Produces real OTel spans** — exportable to Jaeger, Tempo, Datadog, or any
  OTLP-compatible backend.
- **Uses the EventBus as the delivery mechanism** — the subscriber receives the same
  events as the metrics and logging subscribers. No separate instrumentation path.

### 2.9 OTel SDK Initialization

Configured once at startup. The choice of exporters determines where data goes:

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from prometheus_client import start_http_server

def init_observability(config):
    # Metrics → Prometheus /metrics endpoint (backward compatible)
    reader = PrometheusMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    start_http_server(config.prometheus_port)

    # Optional: traces → OTLP
    # trace.set_tracer_provider(TracerProvider(...))
```

Existing Prometheus dashboards and alerts continue to work — the `/metrics` endpoint
format is unchanged.

### 2.10 Startup Wiring

```python
def init_mp_observability(bus: EventBus, config: ObservabilityConfig):
    """Register all subscribers with the event bus."""
    subscribers = []

    if config.metrics_enabled:
        subscribers.extend([
            L1MetricsSubscriber(),
            SMMetricsSubscriber(),
            MPServerMetricsSubscriber(),
        ])

    if config.logging_enabled:
        subscribers.extend([
            L1LoggingSubscriber(),
            SMLoggingSubscriber(),
            MPServerLoggingSubscriber(),
        ])

    for sub in subscribers:
        sub.register(bus)

    bus.start()
```

---

## 3. Implementation Plan (PR Breakdown)

### Important constraint: L1ManagerListener has non-observability consumers

`StoreController` registers a `StoreListener(L1ManagerListener)` to get notified when
L1 writes finish (it signals an eventfd to wake its background loop).
`EvictionPolicy(L1ManagerListener)` tracks key creation/access/deletion for LRU eviction.

These are **business logic**, not observability. The L1Manager listener pattern cannot be
fully removed — only the observability listener (`L1ManagerStatsLogger`) migrates to the
EventBus. `StorageManagerListener`, by contrast, is only used by
`StorageManagerStatsLogger`, so StorageManager can fully migrate.

The strategy for L1Manager: add a **bridge listener** that implements `L1ManagerListener`
and publishes each callback as an `Event` to the EventBus. This replaces
`L1ManagerStatsLogger` in L1Manager's listener list. StoreListener and EvictionPolicy
remain as direct listeners.

### PR 1: Core infrastructure + OpenTelemetry dependency

**Files added:**
- `mp_observability/event.py` — `EventType` enum, `Event` dataclass
- `mp_observability/event_bus.py` — `EventBus` class, `EventSubscriber` ABC, singleton
  (`get_event_bus()`, `init_event_bus()`)
- `mp_observability/otel_init.py` — `init_otel_metrics(config)` helper that sets up
  `MeterProvider` + `PrometheusMetricReader`, and optionally `TracerProvider`
- `tests/v1/mp_observability/test_event_bus.py` — unit tests for EventBus (subscribe,
  publish, drain, queue overflow, start/stop)

**Files modified:**
- `pyproject.toml` / `setup.cfg` — add `opentelemetry-api`, `opentelemetry-sdk`,
  `opentelemetry-exporter-prometheus`
- `mp_observability/config.py` — add `ObservabilityConfig` (extends existing
  `PrometheusConfig` with `metrics_enabled`, `logging_enabled`, `tracing_enabled`)

**Why first:** Pure additions with no risk to existing behavior. Establishes the
foundation (EventBus + OTel) that all subsequent PRs depend on.

---

### PR 2: L1 + StorageManager observability migration

**Files added:**
- `mp_observability/bridge.py` — `L1EventBusBridge(L1ManagerListener)` that publishes
  events to the EventBus:
  ```python
  class L1EventBusBridge(L1ManagerListener):
      def __init__(self, bus: EventBus):
          self._bus = bus

      def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
          self._bus.publish(Event(
              event_type=EventType.L1_READ_FINISHED,
              metadata={"keys": keys},
          ))
      # ... same for other L1 callbacks
  ```
- `mp_observability/subscribers/l1_metrics.py` — `L1MetricsSubscriber` (OTel counters)
- `mp_observability/subscribers/l1_logging.py` — `L1LoggingSubscriber`
- `mp_observability/subscribers/sm_metrics.py` — `SMMetricsSubscriber`
- `mp_observability/subscribers/sm_logging.py` — `SMLoggingSubscriber`

**Files modified:**
- `distributed/l1_manager.py` — in `__init__`, replace `L1ManagerStatsLogger` with
  `L1EventBusBridge`:
  ```python
  # Before:
  l1_stats_logger = L1ManagerStatsLogger()
  self.register_listener(l1_stats_logger)
  get_prometheus_controller().register_logger(l1_stats_logger)

  # After:
  bridge = L1EventBusBridge(get_event_bus())
  self.register_listener(bridge)
  ```
  StoreListener and EvictionPolicy registrations unchanged.
- `distributed/storage_manager.py` — replace listener iteration with `bus.publish()`:
  ```python
  # Before:
  for listener in self._registered_listeners:
      listener.on_sm_reserved_write(successful_keys, failed_keys)

  # After:
  get_event_bus().publish(Event(
      event_type=EventType.SM_WRITE_RESERVED,
      metadata={"succeeded_keys": successful_keys, "failed_keys": failed_keys},
  ))
  ```
  Remove `_registered_listeners`, `register_listener()`, and `StorageManagerStatsLogger`
  import/registration.
- `distributed/internal_api.py` — remove `StorageManagerListener`

**Files removed:**
- `mp_observability/stats/l1_stats.py`
- `mp_observability/logger/l1_stats_logger.py`
- `mp_observability/stats/storage_manager_stats.py`
- `mp_observability/logger/storage_manager_stats_logger.py`

**Note:** L1 uses a bridge (StoreListener and EvictionPolicy need the listener pattern).
StorageManager migrates fully (no non-observability listeners).

**Verification:** Run existing tests + compare Prometheus metric output before/after
to confirm parity for both L1 and SM counters.

---

### PR 3: MP Server telemetry migration

**Files added:**
- `mp_observability/subscribers/mp_server_metrics.py` — `MPServerMetricsSubscriber`
- `mp_observability/subscribers/mp_server_logging.py` — `MPServerLoggingSubscriber`
- `mp_observability/subscribers/mp_server_spans.py` — `MPServerSpanSubscriber`
  (OTel explicit spans from START/END pairs)

**Files modified:**
- `multiprocess/server.py` — replace `log_telemetry(make_start_event(...))` /
  `log_telemetry(make_end_event(...))` with `get_event_bus().publish(Event(...))`.
  Remove `TelemetryController` imports and `is_enabled()` guards (the EventBus handles
  disabled state via empty subscriber lists).
- `multiprocess/blend_server.py` — remove `init_telemetry_controller()` call
- `multiprocess/blend_server_v2.py` — remove `init_telemetry_controller()` call

**Files removed (entire telemetry/ subdirectory):**
- `telemetry/__init__.py`
- `telemetry/config.py`
- `telemetry/controller.py`
- `telemetry/event.py`
- `telemetry/processors/__init__.py`
- `telemetry/processors/base.py`
- `telemetry/processors/logging_processor.py`

---

### PR 4: Final cleanup

**Files removed:**
- `mp_observability/prometheus_controller.py` — no more PrometheusLogger consumers
- `mp_observability/logger/prometheus_logger.py` — ABC no longer needed
- `mp_observability/logger/l2_stats_logger.py` — placeholder, replace with future
  L2 EventBus subscriber when L2 is finalized
- `mp_observability/stats/mp_server_stats.py` — empty
- `mp_observability/stats/vllm_integrator_stats.py` — empty
- `mp_observability/logger/integrator_stats_logger.py` — empty
- `mp_observability/logger/mp_server_logger.py` — empty

**Files modified:**
- `multiprocess/server.py`, `blend_server.py`, `blend_server_v2.py` — replace
  `init_prometheus_controller(config)` + `init_telemetry_controller(config)` with
  unified `init_mp_observability(config)` call
- `distributed/internal_api.py` — remove `EventListener` base class. Keep
  `L1ManagerListener` (still used by StoreListener, EvictionPolicy). Remove
  `L2ManagerListener` placeholder.
- `mp_observability/LOGGER_GUIDE.md` — rewrite for new subscriber pattern
- `mp_observability/METRICS.md` — update to reflect new architecture
- `pyproject.toml` — remove `prometheus_client` direct dependency (it remains as a
  transitive dep via `opentelemetry-exporter-prometheus`)

---

### PR dependency graph

```
PR 1 (EventBus core + OTel dependency)
  │
  ├──► PR 2 (L1 + SM migration)
  │
  └──► PR 3 (MPServer migration)     ← can be parallel with PR 2
       │
       ▼
     PR 4 (cleanup)                   ← depends on PR 2 + 3 both merged
```

PRs 2 and 3 are independent of each other (they touch different producers and
different subscriber files) and can be reviewed/merged in parallel after PR 1 lands.
PR 4 depends on both completing.

---

## Appendix

### A. Migration Map

#### A.1 Files Removed

| File | Reason |
|---|---|
| `stats/l1_stats.py` | No intermediate accumulation — counters update directly in callbacks |
| `stats/storage_manager_stats.py` | Same |
| `stats/mp_server_stats.py` | Empty, unused |
| `stats/vllm_integrator_stats.py` | Empty, unused |
| `logger/l1_stats_logger.py` | Replaced by `L1MetricsSubscriber` + `L1LoggingSubscriber` |
| `logger/l2_stats_logger.py` | Replaced by future `L2MetricsSubscriber` |
| `logger/storage_manager_stats_logger.py` | Replaced by `SMMetricsSubscriber` + `SMLoggingSubscriber` |
| `logger/integrator_stats_logger.py` | Empty, unused |
| `logger/mp_server_logger.py` | Empty, unused |
| `logger/prometheus_logger.py` | ABC replaced by OTel Metrics API |
| `prometheus_controller.py` | Merged into `EventBus` |
| `telemetry/controller.py` | Merged into `EventBus` |
| `telemetry/event.py` | Replaced by unified `Event` + `EventType` |
| `telemetry/config.py` | Merged into unified `ObservabilityConfig` |
| `telemetry/processors/base.py` | Replaced by `EventSubscriber` ABC |
| `telemetry/processors/logging_processor.py` | Replaced by per-component logging subscribers |

#### A.2 Files Modified

| File | Change |
|---|---|
| `distributed/internal_api.py` | Remove `StorageManagerListener`, `L2ManagerListener`, `EventListener`. **Keep `L1ManagerListener`** (used by StoreListener, EvictionPolicy). |
| `distributed/l1_manager.py` | Replace `L1ManagerStatsLogger` with `L1EventBusBridge` in listener list. Listener iteration stays (for StoreListener, EvictionPolicy). |
| `distributed/storage_manager.py` | Replace listener iteration with `bus.publish(Event(...))`. Remove `_registered_listeners`. |
| `multiprocess/server.py` | Replace `log_telemetry(make_start_event(...))` with `bus.publish(Event(...))` |
| Server startup code | Replace `init_prometheus_controller()` + `init_telemetry_controller()` with `init_mp_observability()` |

#### A.3 Files Added

| File | Purpose |
|---|---|
| `event.py` | `EventType` enum + `Event` dataclass |
| `event_bus.py` | `EventBus` class + `EventSubscriber` ABC + singleton management |
| `bridge.py` | `L1EventBusBridge(L1ManagerListener)` — adapts L1 listener callbacks to EventBus events |
| `subscribers/l1_metrics.py` | `L1MetricsSubscriber` |
| `subscribers/l1_logging.py` | `L1LoggingSubscriber` |
| `subscribers/sm_metrics.py` | `SMMetricsSubscriber` |
| `subscribers/sm_logging.py` | `SMLoggingSubscriber` |
| `subscribers/mp_server_metrics.py` | `MPServerMetricsSubscriber` |
| `subscribers/mp_server_logging.py` | `MPServerLoggingSubscriber` |
| `subscribers/mp_server_spans.py` | `MPServerSpanSubscriber` (OTel spans from START/END pairs) |
| `otel_init.py` | OTel SDK initialization helper |
| `config.py` (rewrite) | Unified `ObservabilityConfig` |

#### A.4 Dependencies

| Add | Remove |
|---|---|
| `opentelemetry-api` | `prometheus_client` (eventually — see [Appendix B.1](#b1-phased-vs-big-bang-migration)) |
| `opentelemetry-sdk` | |
| `opentelemetry-exporter-prometheus` | |

---

### B. Design Options and Tradeoffs

#### B.1 Phased vs. Big-Bang Migration

**Option A — Big bang:** Replace everything in one PR. All Listener ABCs, stats
dataclasses, both controllers, and `prometheus_client` usage are removed at once.
- **Pro:** Clean cut, no transitional glue code.
- **Con:** Large diff, higher review burden, harder to bisect regressions.

**Option B — Phased:**
1. Introduce `EventBus`, `Event`, `EventType`, `EventSubscriber` ABC alongside existing code.
2. Create new subscribers that use OTel. Wire them up in parallel with existing loggers.
3. Migrate producers one component at a time (L1 → SM → MPServer).
4. Remove old code once all producers are migrated and verified.
- **Pro:** Incremental, each step is independently testable.
- **Con:** Temporary duplication during migration.

**Recommendation:** Option B. The phased approach lets us verify metric parity at each step.

#### B.2 Sync vs. Async Event Dispatch

**Option A — Async only (queue + drain thread):**
All events go through the queue. Subscribers never run on the producer's thread.
- **Pro:** Producers never block on subscriber work. Clean separation.
- **Con:** Slight delay before metrics/logs update. During high load, queue could back up.

**Option B — Sync dispatch (no queue):**
`publish()` calls subscribers inline on the producer's thread.
- **Pro:** Zero latency, no queue management.
- **Con:** Slow subscribers block producers. Subscriber exceptions can crash producer code.

**Option C — Hybrid (configurable per-subscriber):**
Each subscriber declares sync or async preference.
- **Pro:** Flexibility.
- **Con:** Complexity — two code paths, harder to reason about ordering.

**Recommendation:** Option A (async only). The current Prometheus system already tolerates
batched/delayed updates (periodic flush at 10s intervals). The current telemetry system
is already async. An async-only bus is simpler and keeps the hot path (L1Manager,
StorageManager) fast.

#### B.3 OTel Span Creation Strategy

All observable operations have their start and end in **different function calls** (separate
CUDA host callbacks, separate RPC methods, separate L1Manager/StorageManager calls). OTel's
context-manager span pattern (`with tracer.start_as_current_span(...)`) is not viable for
any of them.

**Option A — Subscriber-managed explicit spans (proposed in Section 2.8):**
A span subscriber creates spans on START events, stashes them by `session_id`, and ends
them on END events, passing the original event timestamps to OTel.
- **Pro:** Produces real OTel spans with GPU-accurate timing. All span lifecycle logic is
  in one subscriber, not scattered across producers. Exportable to any OTLP backend.
- **Con:** Manual `session_id` → span mapping. Orphaned spans if an END event is lost
  (needs a TTL cleanup or periodic sweep). No automatic parent-child nesting.

**Option B — No OTel spans; metrics-only:**
Use OTel only for the Metrics API (counters, histograms). Compute latency in a metrics
subscriber that correlates START/END timestamps. Don't produce spans at all.
- **Pro:** Simpler — no span lifecycle management, no orphan cleanup. Fewer OTel
  dependencies (`opentelemetry-sdk` tracing components not needed).
- **Con:** No trace visualization in Jaeger/Tempo. Latency data only visible as
  histogram distributions in Prometheus, not as individual request traces.

**Recommendation:** Option A. The span subscriber is straightforward and the orphan
cleanup is a minor concern (a periodic sweep of `_pending` entries older than N seconds).
The ability to visualize individual request traces is valuable for debugging latency
issues in production.

#### B.4 OpenTelemetry vs. Keeping `prometheus_client` Directly

**Option A — OTel Metrics API + Prometheus exporter:**
- **Pro:** Vendor-neutral. Can add OTLP/Datadog/etc. exporters with config changes only,
  no code changes. Industry standard. Trace + metrics in one SDK.
- **Con:** New dependency (~3 packages). Slightly more boilerplate for SDK init. Team
  needs to learn OTel concepts.

**Option B — Keep `prometheus_client` directly:**
- **Pro:** No new dependencies. Team already familiar with it.
- **Con:** Locked to Prometheus. No tracing support. Adding a second backend requires
  a second metrics library.

**Option C — Abstract backend interface:**
Define our own `MetricsBackend` ABC with `emit_counter()`, `emit_histogram()`, etc.
Implement for Prometheus and optionally OTel.
- **Pro:** Full control, no external dependency on OTel API stability.
- **Con:** Reinventing a wheel that OTel already provides. Maintenance burden.

**Recommendation:** Option A. OTel is the CNCF standard, widely adopted, and the
Prometheus exporter means existing dashboards and alerts are unaffected. The dependency
cost is low relative to the flexibility gained.

#### B.5 Metadata Typing: `dict[str, Any]` vs. Typed Payloads

**Option A — `dict[str, Any]` (proposed):**
- **Pro:** Simple, flexible, easy to extend.
- **Con:** No compile-time type checking on metadata keys. Typos in key names are
  silent bugs. Subscribers must cast values.

**Option B — Typed payload dataclasses per event:**
```python
@dataclass
class L1ReadFinishedPayload:
    keys: list[ObjectKey]

@dataclass
class Event:
    event_type: EventType
    payload: L1ReadFinishedPayload | SMReadPrefetchedPayload | ...
```
- **Pro:** Full type safety. IDE autocomplete. Mypy catches mismatches.
- **Con:** Many small dataclasses. Adding a new event type requires a new class.
  Union type grows with each event.

**Option C — TypedDict per event type (middle ground):**
```python
class L1ReadFinishedMeta(TypedDict):
    keys: list[ObjectKey]
```
Metadata contracts documented via TypedDict but not enforced at runtime.
- **Pro:** Type hints in IDE, no runtime overhead, lighter than full dataclasses.
- **Con:** Still requires discipline to use the right TypedDict.

**Recommendation:** Start with Option A for simplicity. The metadata contracts table
in Section 2.7 serves as documentation. If metadata bugs become a problem, upgrade
to Option C (TypedDict) — it's a backward-compatible addition.

#### B.6 One Subscriber per Component vs. Merged Metrics+Logging

**Option A — Separate subscribers (proposed):**
`L1MetricsSubscriber` and `L1LoggingSubscriber` are independent classes.
- **Pro:** Single responsibility. Can enable/disable metrics and logging independently.
  Metrics subscribers can be tested without log assertions and vice versa.
- **Con:** More files. Some duplication in `get_subscriptions()` (both subscribe to
  the same event types).

**Option B — Merged subscriber per component:**
`L1Subscriber` handles both metrics and logging in the same callback.
- **Pro:** Fewer files. One place to see everything that happens on an L1 event.
- **Con:** Can't disable logging without disabling metrics. Harder to test in isolation.
  Callback methods mix concerns.

**Recommendation:** Option A. The EventBus natively supports multiple callbacks per
event type, so there's no awkwardness. The file count increase is modest (2 files per
component instead of 1) and each file stays small and focused.

---

### C. Open Questions

1. **Metric name migration:** The current Prometheus metrics use `:` as separator
   (e.g., `lmcache_mp:l1_read_keys`). OTel convention uses `.` (e.g.,
   `lmcache_mp.l1_read_keys`). The Prometheus exporter auto-converts `.` to `_`.
   Do we want to preserve the exact current metric names for dashboard compatibility,
   or adopt the OTel convention and update dashboards?

2. **L2 events:** `L2ManagerListener` is currently a placeholder. Should we define
   L2 event types now (based on expected interface) or wait until L2 is finalized?

3. **VLLMIntegrator / MPServer events:** `VLLMIntegratorStats` and `MPServerStats`
   are currently empty. Are there planned metrics for these components that should
   influence the event type design?

4. **Queue overflow policy:** The current telemetry system silently drops events when
   the queue is full (tail-drop with rate-limited warning). Should the unified bus
   keep this behavior, or should we use a bounded deque that drops the oldest events
   instead?

5. **Thread safety model:** The current Listener callbacks fire under `L1Manager`'s
   internal lock. With async dispatch (events queued, not processed inline), should
   producers still hold their lock while calling `bus.publish()`, or can `publish()`
   be called lock-free since it's just a deque append?
