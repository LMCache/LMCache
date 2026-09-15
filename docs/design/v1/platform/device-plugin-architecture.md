# External Device Plugin Architecture

## Goal

Hardware vendors can integrate a new LMCache device backend from an
independently versioned wheel. LMCache discovers the wheel at runtime, so the
vendor can ship fixes and native binaries without adding vendor code or a
device-name list to the LMCache repository.

This is an alternative to the existing in-tree model, not a replacement for
it. Vendors can still contribute a backend under
`lmcache/v1/platform/<device>/` when joint maintenance and the LMCache release
cadence are preferable. Both models implement the same interfaces and merge
into the same runtime registry.

The plugin boundary is a `DeviceSpec` subclass. Existing dispatch paths then
resolve operations, IPC wrappers, event IPC, pinning, and cache contexts from
that same object; plugins do not need separate registration hooks for each
capability.

## Package Contract

An external distribution publishes an entry point in the
`lmcache.device_plugins` group:

```toml
[project.entry-points."lmcache.device_plugins"]
foo = "lmcache_foo.device:FooDeviceSpec"
```

The loaded object must satisfy all of these rules:

1. It is a concrete subclass of
   `lmcache.v1.platform.base.device_spec.DeviceSpec`, not an instance or
   factory.
2. It has a no-argument constructor.
3. Its `device_type` and `backend_name` are non-empty lowercase strings. The
   entry-point name equals `backend_name`; `device_type` may be shared by
   multiple backends when torch exposes them through the same device category.
4. Its import and constructor do not require the device to be present.
   Hardware probing belongs in `is_available()`.

The entry-point module loads while `lmcache.v1.platform` is initializing. It
must import interfaces from `lmcache.v1.platform.base` and lazy-load heavier
modules from properties such as `ops_cls`, `ipc_wrapper_cls`, and
`event_ipc_backend`. This keeps discovery free of native-library requirements
and avoids platform initialization cycles.

## Discovery and Dispatch

```text
first platform access
        |
        v
scan lmcache.v1.platform subpackages ------> built-in DeviceSpec instances
        |
        v
read importlib.metadata entry points ------> external DeviceSpec instances
        |                                    group: lmcache.device_plugins
        v
index by unique backend_name, then group by device_type
        |
        +--> LMCACHE_DEVICE_BACKEND=<name> exact selection, when set
        |
        +--> DEVICE_TYPE=<type> limits selection to one torch category
        |
        +--> otherwise scan categories and select their only available backend
        |
        v
one cached DeviceSpec instance
        |
        +--> torch device module
        +--> DeviceOps singleton
        +--> IPC wrapper / event IPC / pinning / cache context
```

`_build_backend_registry()` owns the process-wide `DeviceSpec` instances,
indexed by `backend_name`. `_build_device_registry()` groups those same
instances by `device_type`. Both device detection and backend resolution use
these registries. This is important because `DeviceSpec` caches its `DeviceOps`
and pin-memory backend instances; constructing separate specs would create
different capability objects for detection and use.

The registry and detected device are process-lifetime caches. Installing or
removing a plugin requires restarting every LMCache and serving-engine process.

## Ordering, Conflicts, and Failure Isolation

Built-in device specs are registered before external specs. External entry
points are sorted by `(name, value)` before loading. If multiple specs claim the
same `backend_name`, the first valid entry wins and later entries are logged and
ignored. This prevents an external wheel from silently replacing a bundled
backend with the same identity.

Multiple backends may intentionally share one `device_type`. If exactly one of
them reports `is_available()`, LMCache selects it automatically. If more than
one reports available, LMCache raises an actionable error instead of choosing
by registration order; operators can select the intended implementation with
`LMCACHE_DEVICE_BACKEND`. CUDA and ROCm are the built-in example: both use
torch's `cuda` device type, while mutually exclusive runtime checks select the
`cuda` or `rocm` backend without an environment variable.

Each external entry point has an independent failure boundary. Load,
validation, and construction exceptions produce a warning and skip only that
plugin. This keeps CLI and CPU fallback behavior available when an optional
vendor runtime or shared library is absent. Exceptions from `is_available()`
remain the plugin's responsibility: implementations should catch vendor probe
errors and return `False`.

## Compatibility and Trust

`DeviceSpec` and `DeviceOps` are Python interfaces imported from LMCache, so a
plugin wheel must declare and test an appropriate LMCache dependency range.
Native extensions remain owned and versioned by the vendor wheel and should be
loaded lazily by `DeviceOps.ensure_native()`.

Python entry points execute installed package code in the LMCache process.
They are an extensibility mechanism, not a sandbox; deployment environments
must trust installed device-plugin distributions.

## Non-goals

- Hot installation or registry refresh in a running process.
- Silently overriding an existing `backend_name`.
- Priority-based automatic selection when multiple implementations for one
  `device_type` report available simultaneously.
- Guaranteeing ABI compatibility for native extensions outside the version
  range declared by the plugin.
