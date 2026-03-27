# Telemetry Event System — Design

## Overview

The telemetry system provides per-event tracing for LMCache's multiprocess (MP)
mode. It complements the existing Prometheus-based metrics (aggregated
counters/histograms) by capturing individual operation events that downstream
processors can use to reconstruct spans, build traces, or feed into
observability backends like OpenTelemetry.

The system uses a **START/END event model with session IDs**. A caller emits a
`START` event when an operation begins and an `END` event when it completes.
Both events share the same `session_id`, allowing processors to correlate them
into a span. This two-event design is necessary because start and end happen at
different call sites in LMCache's async submit/check/complete patterns.

## Architecture

```
 Call site                     TelemetryController              Processors
 ─────────                     ───────────────────              ──────────
   log_telemetry(event) ──►  deque.append(event)
                                     │
                              drain thread (daemon)
                                     │
                              deque.popleft() ──────►  proc.on_new_event(event)
                                                       proc.on_new_event(event)
                                                       ...
```

**Hot path** (`log_telemetry` / `controller.log`): Lock-free append to a
`collections.deque`. CPython's GIL guarantees atomic `append()`/`popleft()`.
A `threading.Event` wakes the drain thread immediately.

**Drain thread**: A single daemon thread waits on the wake signal (with a 0.1 s
timeout as a safety net), pops all queued events, and dispatches each to every
registered processor. Exceptions in one processor do not block others.

**Backpressure**: When the queue reaches `max_queue_size` (default 10,000),
new events are silently discarded (tail-drop) with a rate-limited WARNING log
(at most once per second).

## Module Layout

```
lmcache/v1/mp_observability/telemetry/
├── DESIGN.md                      # this file
├── __init__.py                    # public API re-exports
├── event.py                       # EventType enum + TelemetryEvent dataclass
├── config.py                      # TelemetryConfig + argparse helpers
├── controller.py                  # TelemetryController + singleton + factory
└── processors/
    ├── __init__.py                # imports submodules to trigger registration
    ├── base.py                    # TelemetryProcessor ABC, TelemetryProcessorConfig ABC, registry
    └── logging_processor.py       # built-in LoggingProcessor + LoggingProcessorConfig
```

## Key Components

### `TelemetryEvent` (event.py)

```python
@dataclass
class TelemetryEvent:
    name: str                                        # e.g. "mp.store", "mp.lookup"
    event_type: EventType                            # START or END
    session_id: str                                  # correlates START/END pairs
    metadata: dict[str, str | int | float | bool]    # flat, OTel-compatible
    timestamp: float = 0.0                           # set by controller.log()
```

**Timestamp** is stamped by `TelemetryController.log()` using `time.time()`
(wall-clock), not at event construction time. This ensures the timestamp
reflects when the event was actually ingested, which matters for GPU callback
call sites where event objects are created earlier but logged later via
`launch_host_func`.

**Metadata values** are restricted to `str | int | float | bool` for
compatibility with OpenTelemetry attribute types.

### `TelemetryProcessor` (processors/base.py)

Abstract base class that all processors implement:

```python
class TelemetryProcessor(ABC):
    @abstractmethod
    def on_new_event(self, event: TelemetryEvent) -> None: ...

    @abstractmethod
    def shutdown(self) -> None: ...
```

- `on_new_event` is called from the drain thread — implementations should be
  fast or offload heavy work.
- `shutdown` is called once when the controller stops, for resource cleanup.
- Exceptions are caught and logged; a failing processor never blocks others.

### `TelemetryProcessorConfig` (processors/base.py)

Abstract base class for processor configuration, following the same pattern as
L2 adapter configs (`lmcache/v1/distributed/l2_adapters/config.py`):

```python
class TelemetryProcessorConfig(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> Self: ...

    @classmethod
    @abstractmethod
    def help(cls) -> str: ...
```

Each config class is registered at import time via
`register_telemetry_processor_type(name, config_cls)`, mapping a string type
name (used in CLI JSON specs) to the config class.

### `TelemetryController` (controller.py)

Manages the event queue, drain thread, and processor dispatch. Key methods:

| Method | Description |
|--------|-------------|
| `log(event)` | Non-blocking hot path. Appends to deque or discards on overflow. |
| `register_processor(proc)` | Thread-safe registration (protected by lock). |
| `start()` | Launches the daemon drain thread. No-op when disabled. |
| `stop()` | Signals stop, joins thread, final drain, shuts down processors. |

### Singleton & Convenience Functions (controller.py)

```python
init_telemetry_controller(config)   # replace global singleton, create & register processors
get_telemetry_controller()          # return current singleton
log_telemetry(event)                # convenience — logs to the global controller
```

The default singleton is disabled (no thread, `log()` is a no-op) until
`init_telemetry_controller` is called with `enabled=True`.

### Configuration & CLI (config.py)

```python
@dataclass
class TelemetryConfig:
    enabled: bool = False
    max_queue_size: int = 10000
    processor_configs: list[TelemetryProcessorConfig] = field(default_factory=list)
```

CLI arguments (composable argparse pattern):

```
--enable-telemetry                   Enable the telemetry system
--telemetry-max-queue-size N         Queue capacity before tail-drop (default 10000)
--telemetry-processor JSON           Processor spec, repeatable
```

Processor specs are JSON objects with a `"type"` field that maps to a
registered processor type name, plus any type-specific fields:

```bash
--telemetry-processor '{"type": "logging"}'
--telemetry-processor '{"type": "logging", "log_level": "INFO"}'
```

## How to Implement a New Telemetry Processor

Follow these steps to add a new processor (e.g., an OpenTelemetry exporter).

### 1. Create the processor module

Create a new file under `processors/`, e.g.
`processors/otel_processor.py`.

### 2. Define the config class

Subclass `TelemetryProcessorConfig` and implement `from_dict` and `help`:

```python
# processors/otel_processor.py

from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessorConfig,
    register_telemetry_processor_type,
)


class OtelProcessorConfig(TelemetryProcessorConfig):
    """Config for the OpenTelemetry processor."""

    def __init__(self, endpoint: str = "http://localhost:4317"):
        self.endpoint = endpoint

    @classmethod
    def from_dict(cls, d: dict) -> "OtelProcessorConfig":
        return cls(endpoint=d.get("endpoint", "http://localhost:4317"))

    @classmethod
    def help(cls) -> str:
        return (
            "OpenTelemetry processor: exports telemetry events as OTel spans.\n"
            "Fields:\n"
            '  "endpoint" (str): OTel collector endpoint. '
            'Default: "http://localhost:4317"\n'
            'Example: \'{"type": "otel", "endpoint": "http://collector:4317"}\''
        )


register_telemetry_processor_type("otel", OtelProcessorConfig)
```

Key points:
- `from_dict` receives the full JSON dict (including the `"type"` key).
  Extract any type-specific fields you need.
- Call `register_telemetry_processor_type` at module level so the type name
  is available as soon as the module is imported.

### 3. Define the processor class

Subclass `TelemetryProcessor` and implement `on_new_event` and `shutdown`:

```python
# processors/otel_processor.py (continued)

from lmcache.v1.mp_observability.telemetry.processors.base import (
    TelemetryProcessor,
)
from lmcache.v1.mp_observability.telemetry.event import (
    EventType,
    TelemetryEvent,
)


class OtelProcessor(TelemetryProcessor):
    """Processor that exports events as OpenTelemetry spans."""

    def __init__(self, config: OtelProcessorConfig):
        self._endpoint = config.endpoint
        # Initialize OTel SDK resources here...

    def on_new_event(self, event: TelemetryEvent) -> None:
        # Called from the drain thread for every event.
        # Reconstruct spans from correlated START/END pairs,
        # then export via OTel SDK.
        pass

    def shutdown(self) -> None:
        # Flush any pending spans and release resources.
        pass
```

`on_new_event` is called from a single drain thread, so the processor does
not need to be thread-safe internally. However, it should be fast — if
export is slow, consider buffering and flushing in batches.

### 4. Register the import

Add an import to `processors/__init__.py` so the module is loaded (and the
type is registered) at startup:

```python
# processors/__init__.py

from lmcache.v1.mp_observability.telemetry.processors import otel_processor  # noqa: F401
```

### 5. Add the factory branch

In `controller.py`, add an `isinstance` branch in `create_processors`:

```python
from lmcache.v1.mp_observability.telemetry.processors.otel_processor import (
    OtelProcessor,
    OtelProcessorConfig,
)

def create_processors(config: TelemetryConfig) -> list[TelemetryProcessor]:
    processors: list[TelemetryProcessor] = []
    for proc_config in config.processor_configs:
        if isinstance(proc_config, LoggingProcessorConfig):
            processors.append(LoggingProcessor(proc_config))
        elif isinstance(proc_config, OtelProcessorConfig):
            processors.append(OtelProcessor(proc_config))
        else:
            raise ValueError(...)
    return processors
```

### 6. Add tests

Create `tests/v1/mp_observability/telemetry/test_otel_processor.py` with
tests for:
- Config parsing: `from_dict` with defaults and custom values
- `help()` returns a non-empty string
- `on_new_event` processes events correctly
- `shutdown` cleans up without errors

### 7. Use it from the CLI

```bash
python -m lmcache.v1.multiprocess.server \
    --enable-telemetry \
    --telemetry-processor '{"type": "otel", "endpoint": "http://collector:4317"}'
```

## Design Decisions

| Decision | Rationale |
|---|---|
| Timestamp set at log time, not construction | Creation time != log time in async code (GPU callbacks). `controller.log()` stamps `time.time()`. |
| `collections.deque` without lock | CPython GIL makes `append()`/`popleft()` atomic. Single consumer thread. |
| `threading.Event` wake signal | Immediate drain on new events, unlike timer-based polling. |
| 0.1 s timeout on wake wait | Safety net so drain thread checks stop flag even with no events. |
| Tail-drop backpressure | Discard newest events when queue full; rate-limited WARNING (1/sec). |
| Default disabled singleton | Safe no-op — no thread, no allocations until explicitly enabled. |
| Config-based `isinstance` dispatch | Follows L2 adapter pattern; avoids string-based dispatch in the factory. |
| Narrow public API | Call sites depend only on `log_telemetry()`. Internal classes are not exported. |
| Processor exception isolation | A crashing processor never blocks event delivery to other processors. |
