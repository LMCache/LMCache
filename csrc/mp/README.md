# LMCache Native MP

This directory contains the native C++ implementation for the LMCache
multiprocess server path.

The native server keeps the existing LMCache MP ZMQ/msgpack protocol shape so
vLLM can continue using the Python `MessageQueueClient` and connector code. The
C++ side owns the controller handlers, tiered cache metadata, optional
filesystem L2 persistence, and optional CUDA IPC KV byte movement.

## Build

Build the no-CUDA native server from the repository root:

```bash
cmake -S . -B build-native -DLMCACHE_BUILD_NATIVE_MP=ON
cmake --build build-native --target lmcache-mp-server-native
```

Build the CUDA-capable native server:

```bash
cmake -S . -B build-native-cuda \
  -DLMCACHE_BUILD_NATIVE_MP=ON \
  -DLMCACHE_ENABLE_CUDA=ON
cmake --build build-native-cuda --target lmcache-mp-server-native
```

The Python helper package under `csrc/mp/python/lmcache_mp_cpp` is packaged as
part of LMCache and can also build the shared library on import from a source
checkout.

## Run

Launch the default no-CUDA native server:

```bash
lmcache server --native --l1-size-gb 1 --eviction-policy LRU
```

Launch the CUDA-capable native server:

```bash
lmcache server --native-cuda --l1-size-gb 1 --eviction-policy LRU
```

Force the Python MP server path:

```bash
lmcache server --python --l1-size-gb 1 --eviction-policy LRU
```

Environment switches are also available:

- `LMCACHE_MP_NATIVE=1` selects native no-CUDA mode.
- `LMCACHE_MP_NATIVE_CUDA=1` selects native CUDA mode.
- `LMCACHE_MP_NATIVE_BINARY=/path/to/lmcache-mp-server-native` selects an
  explicit no-CUDA binary.
- `LMCACHE_MP_NATIVE_CUDA_BINARY=/path/to/lmcache-mp-server-native-cuda`
  selects an explicit CUDA binary.

CUDA mode is explicit. A normal native launch does not silently choose a CUDA
binary, and a CUDA launch does not fall back to a no-CUDA binary.

## vLLM Connector Options

The default connector remains compatible with the existing Python MP path.
Native-specific options are opt-in through `kv_connector_extra_config`:

- `lmcache.mp.raw_cuda_ipc`: send raw CUDA IPC tensor handles for the native
  CUDA transfer path.

The scheduler lookup path is asynchronous: `LOOKUP` submits work and
`QUERY_PREFETCH_STATUS` polls for the result on later scheduler steps.

## Supported Native Scope

The current native server supports:

- MP protocol envelope handling for the default KV operations.
- Native `STORE`, `LOOKUP`, `QUERY_PREFETCH_STATUS`, `RETRIEVE`,
  `FREE_LOOKUP_LOCKS`, `END_SESSION`, and `CLEAR` handling.
- Native protocol extensions for hash-key lookup, store, retrieve, and
  free-lock requests.
- C++ DRAM/disk byte cache with LRU spill, promotion, pin/lock protection, and
  clear semantics compatible with MP force clear.
- Request-priority lanes so status/control and retrieve work are not blocked
  behind store work.
- Native object-key compatibility helpers for Python LMCache keys, chunk
  hashes, KV rank expansion, and cache salt handling.
- CUDA IPC transfer support when built with `LMCACHE_ENABLE_CUDA=ON`.
- CUDA hot-cache support behind the explicit native CUDA mode.
- Filesystem L2 adapter support through local filesystem paths.
- HTTP status, healthcheck, clear-cache, loglevel, threads, quota, metrics, and
  related management endpoints needed by the native MP server.

## Intentionally Out Of Scope

Use the Python MP server path for server modes that the native server rejects,
including:

- Non-`LRU` native eviction policies.
- Non-`blake3` hash algorithms.
- Runtime plugins and storage plugin configuration.
- Remote storage backends other than the native filesystem L2 adapter.
- Blend-engine execution beyond schema-compatible safe responses.
- Python EventBus, OpenTelemetry, standalone trace recording, and standalone
  Prometheus-port options.

Unsupported native options fail before server startup instead of being silently
ignored.

## Protocol Schema

The Python and native request constants must stay aligned. Keep
`docs/protocol_schema.md` updated when adding or removing native protocol
requests, and keep additions append-only unless the branch has not been
released yet.

## Focused Checks

Useful local checks for this directory:

```bash
uv run pytest tests/v1/multiprocess/test_custom_types.py
uv run pytest tests/v1/multiprocess/test_free_locks.py
uv run pytest tests/cli/commands/test_server.py
```

CUDA-specific checks are skipped automatically on hosts without CUDA support.
