# Plugin L2 Adapter Design

## Overview

The **Plugin L2 Adapter** framework allows third-party developers to extend
LMCache with custom L2 storage backends **without modifying any LMCache source
code**. A plugin is simply a Python module that implements `L2AdapterInterface`
and is loaded at runtime via the `PluginL2AdapterConfig` mechanism.

This is the recommended way to integrate external storage systems (e.g.
NitroFS, custom distributed caches) into LMCache's MP-mode L2 pipeline.

---

## Key Components

### `PluginL2AdapterConfig`

Config class registered under the type name `"plugin"`. Fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `module_path` | `str` | yes | Dotted Python import path of the module containing the adapter class. |
| `class_name` | `str` | yes | Name of the class inside `module_path` that implements `L2AdapterInterface`. |
| `adapter_params` | `dict` | no | Arbitrary keyword arguments forwarded to the adapter class constructor. |

Defined in `plugin_l2_adapter.py` and self-registered at import time via:
```python
register_l2_adapter_type("plugin", PluginL2AdapterConfig)
register_l2_adapter_factory("plugin", _create_plugin_adapter)
```

### `_create_plugin_adapter`

Factory function that:
1. Calls `importlib.import_module(config.module_path)` to load the user module.
2. Retrieves `config.class_name` from the module via `getattr`.
3. Validates it is a subclass of `L2AdapterInterface`.
4. Instantiates it with `adapter_cls(**kwargs, **config.adapter_params)`.

Any framework-level kwargs (e.g. `l1_memory_desc`) are passed through as
keyword arguments so the plugin can optionally consume or ignore them.

---

## Loading Flow

```
CLI / config JSON
  │
  ▼
parse_args_to_l2_adapters_config()
  │  JSON: {"type": "plugin",
  │         "module_path": "my_plugin",
  │         "class_name": "MyL2Adapter",
  │         "adapter_params": {...}}
  │
  ▼
PluginL2AdapterConfig.from_dict(d)
  │  validates module_path, class_name, adapter_params
  │
  ▼
create_l2_adapter_from_registry(config, **kwargs)
  │  looks up factory for "plugin"
  │
  ▼
_create_plugin_adapter(config, **kwargs)
  │
  ├─ importlib.import_module(config.module_path)
  ├─ getattr(module, config.class_name)
  ├─ issubclass check against L2AdapterInterface
  └─ adapter_cls(**kwargs, **config.adapter_params)
      │
      ▼
  L2AdapterInterface instance (ready for use)
```

---


## Plugin Contract

A plugin adapter class **must**:

1. **Subclass `L2AdapterInterface`** from `lmcache.v1.distributed.l2_adapters.base`.
2. **Implement all abstract methods**: store, lookup & lock, load, close,
   and all three event-fd getters.
3. **Provide three distinct event fds** (store / lookup / load). The
   controllers build `fd → adapter` maps; duplicates will misroute events.
4. **Be thread-safe**: the `StoreController` and `PrefetchController`
   call adapter methods from different threads concurrently.
5. **Accept `**kwargs` in `__init__`** to stay forward-compatible with new
   framework-level arguments.

A plugin adapter class **should**:

1. Create its own asyncio event loop and background thread if it needs
   async I/O (the framework does **not** provide a loop to L2 adapters,
   unlike the old Connector-based architecture).
2. Use `os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)` for the three
   event fds.
3. Clean up all resources (event fds, threads, connections) in `close()`.

---

## Threading Model (Plugin Side)

Since the framework does **not** provide an event loop to L2 adapters
(unlike the old non-MP `ConnectorContext.loop`), plugins that need async
I/O must manage their own:

```
Plugin.__init__()
  ├─ self._loop = asyncio.new_event_loop()
  └─ self._thread = Thread(target=run_loop, daemon=True)

Caller threads (StoreController / PrefetchController)
  │
  ├─ submit_store_task()    → run_coroutine_threadsafe(...)
  ├─ submit_lookup_task()   → call_soon_threadsafe(...)
  └─ submit_load_task()     → run_coroutine_threadsafe(...)
  │
  ▼
Plugin background thread (event loop)
  │
  ├─ Executes store/load coroutines
  ├─ Writes to eventfd on completion
  └─ Accesses shared state under lock
```

This pattern is identical to the one used by `MockL2Adapter` and
`NixlStoreL2Adapter`.

---


## Example: Minimal Plugin

### 1. Implement the Adapter

```python
# my_plugin/adapter.py
import asyncio, os, threading
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface, L2TaskId,
)

class MyL2Adapter(L2AdapterInterface):
    def __init__(self, host="localhost", **_kw):
        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        # ... set up connection to `host`, background thread, etc.

    # implement all abstract methods ...
```

### 2. Configure via JSON

```json
{
  "type": "plugin",
  "module_path": "my_plugin.adapter",
  "class_name": "MyL2Adapter",
  "adapter_params": {
    "host": "10.0.0.1"
  }
}
```

### 3. Launch

```bash
# via CLI
--l2-adapter '{"type":"plugin","module_path":"my_plugin.adapter","class_name":"MyL2Adapter","adapter_params":{"host":"10.0.0.1"}}'

# or via pytest (for testing)
cfg = PluginL2AdapterConfig.from_dict({...})
adapter = create_l2_adapter_from_registry(cfg)
```

---

## Reference Implementation

See `examples/lmc_external_l2_adapter/` for a complete, pip-installable
example plugin (`InMemoryL2Adapter`) that demonstrates:

- FIFO eviction with configurable capacity.
- Simulated bandwidth delay for realistic testing.
- Background asyncio event loop with proper shutdown.
- Full test suite covering store, lookup, load, batch operations,
  and eviction behavior.

---

## Self-Registration Mechanism

The `plugin_l2_adapter.py` module follows the same self-registration
pattern as all other adapters in the package:

```
__init__.py
  └─ pkgutil.iter_modules → importlib.import_module
       └─ plugin_l2_adapter.py (auto-discovered)
            ├─ register_l2_adapter_type("plugin", PluginL2AdapterConfig)
            └─ register_l2_adapter_factory("plugin", _create_plugin_adapter)
```

No changes to existing codes are needed when this module
is present in the `l2_adapters/` package directory.
