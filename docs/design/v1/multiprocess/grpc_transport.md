# Multiprocess gRPC Transport Design

## 1. Motivation

LMCache multiprocess mode uses an IPC boundary between engine workers
(vLLM, SGLang, TRT-LLM, or tests) and the LMCache cache server. The boundary
needs to carry control-plane calls such as `Lookup`, `Store`, `Retrieve`,
`Ping`, and P2P discovery calls, while still preserving LMCache Python domain
objects such as `IPCCacheServerKey`, `KVCache`, `DeviceIPCWrapper`, and
`MemoryLayoutDesc`.

The current transport is built around typed unary gRPC methods. The `.proto`
descriptor defines the service, method, request, and response schema. Python
code then adds only the LMCache-specific pieces that protobuf cannot know:
how protobuf messages map to Python domain objects, how a handler should be
scheduled, and which backend implementation owns the actual behavior.

The design goal is that a caller writes:

```python
future = client.lookup(key, tp_size)
```

instead of constructing an envelope like:

```python
future = client.submit_request(RequestType.LOOKUP, [key, tp_size])
```

The server side follows the same principle: each proto service has a Python
implementation class with methods named exactly like the generated gRPC RPCs,
for example `EngineServiceImpl.Lookup` and `P2PServiceImpl.P2PLookupAndLock`.

## 2. Architecture

```text
Engine worker
  |
  | client.lookup(key, tp_size)
  v
MultiprocessGrpcClient
  |
  | TypedRpcSpec.python_to_request(...)
  v
Generated gRPC stub: EngineServiceStub.Lookup.future(...)
  |
  | unary gRPC over localhost TCP or unix socket
  v
MultiprocessGrpcServer
  |
  | generated gRPC runtime calls _GrpcServicer.Lookup(...)
  v
_GrpcServicer._dispatch("Lookup", request, context)
  |
  | TypedRpcSpec.request_to_python(...)
  | HandlerType decides direct vs thread-pool execution
  v
EngineServiceImpl.Lookup(key, tp_size)
  |
  v
EngineLookupService.lookup(key, tp_size)
```

The main source files are:

| File | Role |
|---|---|
| `lmcache/v1/multiprocess/transport/grpc_impl/proto/lmcache_mq.proto` | Source of truth for gRPC services, methods, requests, and responses. |
| `lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py` | Local stub generation helper. Generated `*_pb2.py` files are not checked into Git. |
| `lmcache/v1/multiprocess/protocol.py` | Derives `RpcMethod` tokens and client method names from the protobuf descriptor. |
| `lmcache/v1/multiprocess/transport/grpc_impl/typed_rpc.py` | Maps protobuf messages to Python payload/response types and records scheduling metadata. |
| `lmcache/v1/multiprocess/mq.py` | Implements `MultiprocessGrpcClient`, `MultiprocessGrpcServer`, request encoding/decoding, dispatch, and thread-pool assignment. |
| `lmcache/v1/multiprocess/services/rpc_services.py` | Contains the Python service implementation classes that match generated gRPC service surfaces. |
| `lmcache/v1/multiprocess/services/*.py` | Backend service logic, such as lookup, management, transfer, blend, and P2P controller behavior. |
| `lmcache/v1/multiprocess/server.py` | Builds concrete service implementations and registers them on the gRPC server. |

There are two deliberately separate layers on the server:

1. `_GrpcServicer` is transport glue. It owns protobuf decoding, handler
   scheduling, protobuf encoding, and gRPC status mapping. It has no cache
   business logic.
2. `EngineServiceImpl`, `ControllerServiceImpl`, `P2PServiceImpl`, and the
   Blend service implementation classes are the gRPC service implementations.
   They expose the proto method names and delegate to narrower backend services
   only when the behavior is large enough to warrant a separate class.

This keeps the runtime standard from the caller's point of view: a proto
service method maps to a same-named Python method and to a same-named gRPC
method on the wire.

## 3. End-to-End Flow

### 3.1 Server Startup

Server startup follows this sequence:

1. `run_cache_server()` creates `MPCacheServerContext`, which holds shared
   runtime state such as `StorageManager`, `TokenHasher`, `SessionManager`,
   `EventBus`, layout descriptors, and transfer context state.
2. `_build_rpc_services()` constructs backend services such as
   `EngineLookupService`, `ManagementService`, `LMCacheDrivenTransferService`,
   `EngineDrivenTransferService`, `QStoreService`, `P2PController`, and the
   selected Blend implementation.
3. `_build_rpc_services()` wraps those backends in concrete gRPC service
   implementations such as `EngineServiceImpl`, `ControllerServiceImpl`, and
   `P2PServiceImpl`.
4. `run_cache_server()` registers each generated service explicitly:

   ```python
   server.add_service("EngineService", rpc_services.engine_service)
   server.add_service("ControllerService", rpc_services.controller_service)
   server.add_service("DebugService", rpc_services.debug_service)
   server.add_service("ObservabilityService", rpc_services.observability_service)
   server.add_service("P2PService", rpc_services.p2p_service)
   ```

   Optional services, such as `BlendService`, are registered only when their
   feature is enabled.

5. `MultiprocessGrpcServer.add_service()` looks up the service in the proto
   descriptor and requires the implementation object to provide every method
   in that service with the exact CamelCase proto method name.
6. `server.assign_thread_pools()` assigns all blocking handlers either to a
   normal thread pool or to a per-client affinity pool.
7. `server.start()` creates a real `grpc.server(...)`, attaches generated
   service registrations from the descriptor, binds the configured address,
   and starts serving.

### 3.2 Client Call

The client exposes one Python method per proto RPC. Method names are the
snake_case form of the protobuf method name, so:

| Proto method | Client method |
|---|---|
| `Lookup` | `client.lookup(...)` |
| `QueryPrefetchStatus` | `client.query_prefetch_status(...)` |
| `Store` | `client.store(...)` |
| `P2PLookupAndLock` | `client.p2p_lookup_and_lock(...)` |
| `CbUnifiedLookup` | `client.cb_unified_lookup(...)` |

A call to `client.lookup(key, tp_size)` does the following:

1. The installed client method resolves to `RpcMethod.Lookup`.
2. `MultiprocessGrpcClient._call_rpc()` retrieves the `TypedRpcSpec`.
3. `TypedRpcSpec.python_to_request(key, tp_size)` builds a `LookupRequest`.
4. The generated gRPC stub sends `EngineServiceStub.Lookup.future(...)` with
   `wait_for_ready=True`. Calls submitted while the daemon is starting remain
   pending until the server becomes reachable.
5. The returned `grpc.Future` completion callback decodes the protobuf response
   through `TypedRpcSpec.response_to_python(...)` and completes an LMCache
   `MessagingFuture`.

The caller only sees the LMCache future:

```python
future = client.lookup(key, tp_size)
future.result(timeout=5.0)
```

### 3.3 Server Dispatch

When gRPC receives the request:

1. The generated gRPC runtime calls the same-named method on `_GrpcServicer`,
   such as `_GrpcServicer.Lookup(request, context)`.
2. The thin method calls `_dispatch("Lookup", request, context)`.
3. `_dispatch()` uses the `TypedRpcSpec` to decode protobuf into Python
   payloads.
4. `_BlockingHandler.run()` invokes the registered implementation method:
   either directly for `HandlerType.SYNC` or through a thread pool for
   `HandlerType.BLOCKING`.
5. The implementation method runs the actual RPC behavior. For lookup, that is
   `EngineServiceImpl.Lookup(...) -> EngineLookupService.lookup(...)`.
6. `_dispatch()` converts the Python return value back into the protobuf
   response.

If an implementation method raises `NotImplementedError`, the transport maps it
to gRPC `UNIMPLEMENTED`. This is used for feature-gated methods: the proto can
still describe the full surface, while a server that did not enable a feature
returns a standard gRPC status instead of silently accepting the call.

### 3.4 Scheduling and Affinity

Each RPC has a Python-side scheduling contract in `typed_rpc.py`:

```python
"Lookup": _PythonRpcContract(
    (IPCCacheServerKey, int),
    None,
    HandlerType.BLOCKING,
)
```

`HandlerType.SYNC` methods run in the gRPC worker thread and should be fast.
`HandlerType.BLOCKING` methods are handed off to a dedicated server-side
thread pool. Methods that interact with per-client GPU or event state can set
`requires_client_affinity=True`; the server then routes calls from the same
client metadata key to a stable affinity worker.

This scheduling metadata is intentionally outside the proto. It is not wire
schema; it is a Python runtime policy for how LMCache executes a method after
the request has already been decoded.

## 4. Why This Is Better Than the Old ZMQ Envelope

The old path used a custom message-queue protocol: the client sent a request
type plus a list of serialized payload frames, the server decoded the request
type, selected a Python handler, and later matched a response back to the
client future. That worked, but it mixed transport, routing, schema, and
business dispatch into one custom envelope.

The gRPC design improves the boundary in several concrete ways:

| Area | Old ZMQ-style envelope | Current gRPC design |
|---|---|---|
| Method identity | Runtime `request_type` token in the payload. | Method identity is the gRPC service/method path. |
| Payload schema | Python-side payload list; correctness depends on sender and receiver agreeing on ordering. | Request and response messages are declared in `.proto`. |
| Client API | `submit_request(request_type, [payloads])`. | `client.lookup(key, tp_size)` and other named methods. |
| Server API | Generic handler registration and historical service-name declarations. | `server.add_service("EngineService", EngineServiceImpl(...))`. |
| Service boundary | Mostly implicit Python grouping. | Explicit proto services with matching Python implementation classes. |
| Validation | Mismatches can surface late at runtime. | `typed_rpc.py` verifies descriptor, `RpcMethod`, and Python contracts align at import time. |
| Error model | Custom response/error handling. | Standard gRPC status codes such as `UNIMPLEMENTED`. |
| Startup behavior | Custom polling-loop and socket behavior. | gRPC futures with `wait_for_ready=True`. |
| Tooling | LMCache-specific wire format. | Standard protobuf/gRPC tooling and generated stubs. |
| Extensibility | New request types required touching custom routing conventions. | New methods follow proto -> contract -> implementation -> registration. |

The remaining `typed_rpc.py` layer is not a second protocol. It is a codec and
scheduling table. The wire protocol is still the proto descriptor; the table
only explains how wire messages become LMCache Python objects.

## 5. Review Concerns and Design Responses

This design addresses the main maintainability concerns raised during the
earlier gRPC transport review.

### 5.1 Conversion Code Size

The transport does not use per-RPC Python conversion wrappers. The proto
descriptor and `_PYTHON_RPC_CONTRACTS` are compiled once into `TypedRpcSpec`
objects. Reusable structural codecs handle dataclasses, `msgspec.Struct`,
`TypedDict`, lists, tuples, maps, optional fields, enums, `torch.dtype`,
`torch.Size`, and selected LMCache IPC wrapper types.

This means adding an ordinary RPC is not a new conversion-function exercise.
The developer declares the wire messages in proto, declares the Python
payload/response types once in `_PythonRpcContract`, and the shared codec
compiler validates the mapping at import time.

### 5.2 `mq.py` Ownership

`mq.py` owns transport control flow only:

- client channel creation and unary future submission,
- server handler registration,
- protobuf request decode / response encode dispatch,
- thread-pool and affinity execution,
- gRPC server lifecycle.

Business behavior lives in service implementation classes and backend services.
Python/protobuf value conversion lives in `typed_rpc.py`. This prevents `mq.py`
from becoming a mix of networking, schema conversion, and cache logic.

### 5.3 Proto Organization

The proto is kept as one checked-in source-of-truth file, but it is organized by
explicit gRPC services. Splitting the schema into many proto files is possible
later, but the current single-file layout avoids cross-file import churn while
the transport is still being reviewed. The important API boundaries are the
service declarations: `EngineService`, `ControllerService`, `DebugService`,
`ObservabilityService`, `P2PService`, and `BlendService`.

### 5.4 Pickle Avoidance

New RPCs should not use pickle for regular request or response fields.
Opaque strategy-owned dictionaries, such as engine-driven transfer context
metadata and `LayoutHints`, are encoded with msgspec msgpack bytes. Concrete
protobuf messages are preferred whenever the structure is stable enough to
describe in the schema.

The remaining `DeviceIPCWrapper` pickle path is a separate polymorphic
IPC-handle compatibility boundary: the receiver needs the concrete wrapper
subclass to reconstruct device memory correctly. Replacing that with an
explicit protobuf `oneof` is a separable follow-up and should not be copied by
new RPCs.

### 5.5 ZMQ Removal Risk

The transport intentionally makes gRPC the single mp-mode wire protocol. Keeping
ZMQ and gRPC live in parallel would keep two dispatch stacks, two failure
models, and the old request-type envelope alive during the most sensitive part
of the migration.

Risk is reduced in other ways:

- legacy `tcp://` and `ipc://` configuration strings still parse as gRPC
  loopback TCP and Unix socket targets, so most operator config does not change;
- gRPC `wait_for_ready=True` preserves the old startup behavior where requests
  submitted during daemon startup remain pending;
- missing feature implementations return standard gRPC `UNIMPLEMENTED`;
- typed protocol tests exercise every descriptor-derived method, request arity,
  encode/decode path, and selected real client/server round trips;
- the design document defines a small mechanical process for adding future
  methods without reintroducing custom request envelopes.

## 6. Adding a New RPC

Adding a new RPC to an existing service is a small mechanical change. The
important rule is that the proto method name, Python contract key, `_GrpcServicer`
forwarding method, and service implementation method all use the same CamelCase
name.

### 6.1 Choose the Service

Pick the service that owns the behavior:

| Behavior | Service |
|---|---|
| Engine cache operations such as lookup, store, retrieve, registration | `EngineService` |
| Server control-plane operations such as ping, clear, configuration queries | `ControllerService` |
| Debug-only calls | `DebugService` |
| Observability/event ingestion | `ObservabilityService` |
| Peer-to-peer lookup/lock operations | `P2PService` |
| CacheBlend calls | `BlendService` |

Create a new service only when the new methods form a separate stable API
surface. Otherwise, add the method to the existing service that already owns
the lifecycle and state.

### 6.2 Update the Proto

Add request/response messages and the `rpc` entry in
`transport/grpc_impl/proto/lmcache_mq.proto`.

Guidelines:

- Put request/response messages near the method or service they support.
- Use stable field numbers; never reuse a removed field number.
- Use `optional` when Python needs `None`.
- Use `repeated` for Python `list[...]`.
- Use `map<...>` for Python dictionaries when the value can be described by
  proto fields.
- Prefer concrete messages over opaque bytes. For intentionally opaque
  strategy-owned dictionaries, use msgspec msgpack bytes rather than pickle.
  `DeviceIPCWrapper` is the separate polymorphic IPC-handle exception today;
  replacing that with an explicit `oneof` is a follow-up to the transport
  cleanup, not the default pattern for new RPCs.

After changing the proto, regenerate local stubs for validation:

```bash
pip install -r requirements/proto.txt
python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
```

The generated `lmcache_mq_pb2.py` and `lmcache_mq_pb2_grpc.py` files are local
build artifacts and are not checked into Git.

### 6.3 Add the Python Contract

Add one `_PythonRpcContract` entry in `typed_rpc.py`:

```python
"GetServerVersion": _PythonRpcContract(
    (),
    str,
    HandlerType.SYNC,
)
```

This tells LMCache:

- how many Python payloads the client method takes,
- which Python response type the future returns,
- whether the handler is `SYNC` or `BLOCKING`,
- whether the handler needs per-client affinity.

Importing `typed_rpc.py` validates that:

- every proto method has exactly one Python contract,
- there are no extra Python contracts for methods missing from proto,
- request/response fields can be encoded to and decoded from the declared
  Python types.

### 6.4 Add the Transport Forwarder

Add a same-named thin method on `_GrpcServicer` in `mq.py`:

```python
def GetServerVersion(self, request: Any, context: "grpc.ServicerContext") -> Any:
    return self._dispatch("GetServerVersion", request, context)
```

This method is the entry point that the generated gRPC runtime calls. It should
contain no business logic.

### 6.5 Implement the Service Method

Implement the same CamelCase method in the Python service implementation class
that corresponds to the proto service.

For an existing service:

```python
class ControllerServiceImpl:
    def GetServerVersion(self) -> str:
        """Return the running LMCache server version."""
        return self._management.get_server_version()
```

For a new service, create a new implementation class in `services/rpc_services.py`
and keep it close to the generated service surface:

```python
class MaintenanceServiceImpl:
    """Implementation of the generated ``MaintenanceService`` RPC surface."""

    def __init__(self, version: str) -> None:
        self._version = version

    def GetServerVersion(self) -> str:
        """Return the running LMCache server version."""
        return self._version
```

If the implementation needs shared state, inject a backend service or
`MPCacheServerContext` through the constructor. The gRPC implementation class
should describe the RPC surface; larger reusable behavior can live in a
narrower backend service.

### 6.6 Register the Service

For a new service, add it to `_BuiltRpcServices`, construct it in
`_build_rpc_services()`, and register it from `run_cache_server()`:

```python
server.add_service("MaintenanceService", rpc_services.maintenance_service)
```

`add_service()` checks the proto descriptor. If `MaintenanceService` declares
three RPC methods, the implementation object must provide all three same-named
CamelCase methods. Missing methods fail server startup with a clear `TypeError`.

### 6.7 Call It From the Client

The client method appears automatically from the descriptor-derived
`RpcMethod` list:

```python
future = client.get_server_version()
version = future.result(timeout=5.0)
```

No caller should pass `request_type` or a positional payload list. The Python
method name is the lower-case snake_case form of the operation name.

### 6.8 Add Tests

At minimum, add tests for:

- the `TypedRpcSpec` codec round trip,
- the client method round trip over a real local gRPC server,
- `add_service()` coverage when a new service is introduced,
- server startup/build behavior when the method is feature-gated.

Useful commands:

```bash
pytest -q tests/v1/multiprocess/test_typed_rpc.py
pytest -q tests/v1/multiprocess/test_mq.py
pytest -q tests/v1/multiprocess
```

Run pre-commit before opening or updating a PR:

```bash
SKIP=rust-fmt,rust-clippy pre-commit run --all-files
```

On Linux with a working Rust toolchain, the Rust hooks can be left enabled.

## 7. Complete Example: Add `MaintenanceService.GetServerVersion`

This example adds a small new service to show the full path. It is intentionally
simple: no payloads, scalar string response, no blocking work.

### 7.1 Proto

Add messages and a service to `lmcache_mq.proto`:

```proto
message GetServerVersionRequest {}

message GetServerVersionResponse {
  string version = 1;
}

service MaintenanceService {
  rpc GetServerVersion(GetServerVersionRequest)
      returns (GetServerVersionResponse);
}
```

Regenerate stubs locally:

```bash
python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
```

### 7.2 Python Contract

Add the contract to `_PYTHON_RPC_CONTRACTS`:

```python
"GetServerVersion": _PythonRpcContract(
    (),
    str,
    HandlerType.SYNC,
)
```

Because `GetServerVersionRequest` has no fields, the client method takes no
payloads. Because `GetServerVersionResponse` has one string field, the future
returns `str`.

### 7.3 Transport Forwarder

Add the gRPC entry point in `_GrpcServicer`:

```python
def GetServerVersion(self, request: Any, context: "grpc.ServicerContext") -> Any:
    return self._dispatch("GetServerVersion", request, context)
```

This is the only transport-level code needed for the new method.

### 7.4 Service Implementation

Add the implementation class:

```python
class MaintenanceServiceImpl:
    """Implementation of the generated ``MaintenanceService`` RPC surface."""

    def __init__(self, version: str) -> None:
        self._version = version

    def GetServerVersion(self) -> str:
        """Return the running LMCache server version."""
        return self._version
```

### 7.5 Server Registration

Extend `_BuiltRpcServices`:

```python
@dataclass(frozen=True)
class _BuiltRpcServices:
    maintenance_service: MaintenanceServiceImpl
    ...
```

Construct it in `_build_rpc_services()`:

```python
maintenance_service = MaintenanceServiceImpl(version=lmcache.__version__)
```

Return it and register it:

```python
server.add_service("MaintenanceService", rpc_services.maintenance_service)
```

No routing table or request-type switch is needed. The proto descriptor tells
`add_service()` which methods belong to `MaintenanceService`; the implementation
provides the same method names.

### 7.6 Client Usage

The client call is direct:

```python
future = client.get_server_version()
version = future.result(timeout=5.0)
```

That is the intended developer experience: adding a new RPC creates a named
function-like client method and a same-named server method.

### 7.7 Minimal Round-Trip Test

```python
def test_get_server_version_roundtrip() -> None:
    port = _find_free_port()
    server_url = f"grpc://127.0.0.1:{port}"

    server = MultiprocessGrpcServer(server_url)
    server.add_service(
        "MaintenanceService",
        MaintenanceServiceImpl(version="0.0.test"),
    )
    server.assign_thread_pools(max_cpu_workers=1, max_gpu_workers=1)
    server.start()

    client = MultiprocessGrpcClient(server_url)
    try:
        assert client.get_server_version().result(timeout=5.0) == "0.0.test"
    finally:
        client.close()
        server.close()
```

The exact server URL helper may differ by test fixture, but the important part
is the shape: register the service implementation, start the gRPC server, call
the generated client method, and assert the Python response value.

## 8. Design Rules

- The `.proto` file is the wire-protocol source of truth.
- Generated stubs are build artifacts, not reviewed source.
- Python service implementation method names must match proto method names
  exactly.
- Client code calls named methods such as `client.store(...)`; it must not pass
  request-type tokens and payload lists.
- `typed_rpc.py` contains Python value conversion and scheduling policy only.
- `_GrpcServicer` contains transport glue only.
- Backend service files contain business logic and lifecycle ownership.
- Feature-gated methods should either not register their service or raise
  `NotImplementedError` from the implementation method so callers receive
  gRPC `UNIMPLEMENTED`.
