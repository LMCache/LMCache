# MP Observability

Event-driven observability for LMCache's multiprocess (MP) mode, built on
[OpenTelemetry](https://opentelemetry.io/).

For the full design rationale and migration plan, see [REFACT_DESIGN.md](REFACT_DESIGN.md).
For the current list of metrics, see [METRICS.md](METRICS.md).

---

## Architecture

```
Producers (L1Manager, StorageManager, MPServer)
    │
    │  event_bus.publish(Event(...))
    ▼
EventBus  (async queue + drain thread)
    │
    ├──► L1MetricsSubscriber   → OTel counter.add(...)
    ├──► SMMetricsSubscriber   → OTel counter.add(...)
    ├──► L1LoggingSubscriber   → logger.debug(...)
    └──► SMLoggingSubscriber   → logger.debug(...)

OTel SDK  (configured at startup)
    │
    ├──► OTLP push (production)       → OTel collector → Prometheus / Grafana / etc.
    └──► Prometheus pull (dev/debug)   → /metrics on configured port
```

## Metrics export modes

Controlled by the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable:

| Env var | Mode | How to query |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT=http://host:4317` | OTLP push | Query the OTel collector's Prometheus exporter |
| *(not set)* | Prometheus pull fallback | `curl http://localhost:<prometheus-port>/metrics` |

> **Note:** OTel counters only appear on `/metrics` after the first increment.
> If you see only Python runtime metrics, trigger a store/retrieve first.

---

## How to Add a New Event and Subscriber

### Step 1 — Define the event type

Add a new member to `EventType` in `event.py`:

```python
class EventType(Enum):
    # ... existing events ...

    # My new component events
    MY_COMPONENT_OPERATION = "my_component.operation"
```

### Step 2 — Publish the event from the producer

In your component (e.g., a manager class), publish to the EventBus:

```python
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

class MyComponent:
    def __init__(self):
        self._event_bus = get_event_bus()

    def do_operation(self, keys):
        # ... business logic ...

        self._event_bus.publish(Event(
            event_type=EventType.MY_COMPONENT_OPERATION,
            metadata={"keys": keys},
        ))
```

### Step 3 — Create a metrics subscriber

Create `mp_observability/subscribers/my_component_metrics.py`:

```python
from opentelemetry import metrics
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


class MyComponentMetricsSubscriber(EventSubscriber):
    def __init__(self):
        meter = metrics.get_meter("lmcache.my_component")
        self._op_counter = meter.create_counter(
            "lmcache_mp.my_component_operations",
            description="Total operations on my component",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MY_COMPONENT_OPERATION: self._on_operation,
        }

    def _on_operation(self, event: Event) -> None:
        self._op_counter.add(len(event.metadata["keys"]))
```

### Step 4 — Register the subscriber at startup

In the server startup function (e.g., `server.py`), add:

```python
from lmcache.v1.mp_observability.subscribers.my_component_metrics import (
    MyComponentMetricsSubscriber,
)

bus.register_subscriber(MyComponentMetricsSubscriber())
```

That's it. The subscriber will receive events asynchronously via the EventBus drain
thread and update OTel counters, which are exported via whichever mode is configured.

---

## Design rules

| Rule | Reason |
|---|---|
| Create meters and counters in `__init__()`, not at module level | `MeterProvider` must be set before `get_meter()` is called. Module-level calls happen at import time, before setup. |
| Prefix OTel metric names with `lmcache_mp.` | Keeps the MP namespace separate from `lmcache.` (the single-process engine namespace). |
| Use `metadata: dict[str, Any]` for event payloads | Flexible, no coupling between producers and subscribers. See metadata contracts in [METRICS.md](METRICS.md). |
| Separate metrics and logging subscribers | Single responsibility. Can enable/disable independently. |
| Store `self._event_bus = get_event_bus()` in `__init__` | Avoids calling the singleton getter on every publish. |
