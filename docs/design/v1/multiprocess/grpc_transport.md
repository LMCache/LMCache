# Multiprocess gRPC Transport Design

## Motivation

LMCache multiprocess mode uses an IPC boundary between engine workers
(vLLM, SGLang, TRT-LLM, or tests) and the LMCache cache server. The boundary
carries control-plane RPCs such as `Lookup`, `Store`, `Retrieve`, `Ping`, and
P2P discovery while preserving LMCache Python domain objects such as
`IPCCacheServerKey`, `KVCache`, `DeviceIPCWrapper`, and `MemoryLayoutDesc`.

The transport is now built around concrete unary gRPC methods. The `.proto`
descriptor defines every service, method, request, and response. There is no
second per-RPC Python contract table. Python code derives routing and message
classes from the descriptor, and derives LMCache value conversion from the
concrete service implementation method annotations.

The client-facing goal is:

```python
future = client.lookup(key, tp_size)
```

instead of a custom envelope such as:

```python
future = client.submit_request(RequestType.LOOKUP, [key, tp_size])
```

The server-facing goal is that each proto service maps to a Python
implementation class with same-named methods, for example
`EngineServiceImpl.Lookup` and `P2PServiceImpl.P2PLookupAndLock`.

## Architecture

```text
Engine worker
  |
  | client.lookup(key, tp_size)
  v
MultiprocessGrpcClient
  |
  | proto descriptor + call args
  | encode_request_from_call(...) -> LookupRequest
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
  | handler annotation decoder
  | HandlerType decides direct vs thread-pool execution
  v
EngineServiceImpl.Lookup(key, tp_size)
  |
  v
EngineLookupService.lookup(key, tp_size)
```

Main source files:

| File | Role |
|---|---|
| `lmcache/v1/multiprocess/transport/grpc_impl/proto/lmcache_mq.proto` | Source of truth for gRPC services, methods, requests, and responses. |
| `lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py` | Local stub generation helper. Generated `*_pb2.py` files are not checked into Git. |
| `lmcache/v1/multiprocess/protocol.py` | Derives `RpcMethod` tokens and client method names from the protobuf descriptor. Also provides the `@grpc_method` scheduling decorator. |
| `lmcache/v1/multiprocess/transport/grpc_impl/proto_codec.py` | Generic protobuf/Python value conversion. It has no per-RPC registry. |
| `lmcache/v1/multiprocess/mq.py` | Implements the gRPC client, server, dispatch, futures, and thread-pool assignment. |
| `lmcache/v1/multiprocess/services/rpc_services.py` | Concrete Python implementations of generated gRPC service surfaces. |
| `lmcache/v1/multiprocess/services/*.py` | Backend service logic, such as lookup, management, transfer, blend, and P2P controller behavior. |
| `lmcache/v1/multiprocess/server.py` | Builds concrete service implementations and registers them on the gRPC server. |

There are two server layers:

1. `_GrpcServicer` is transport glue. It owns gRPC method entry points,
   protobuf request decoding, scheduling, protobuf response encoding, and gRPC
   status mapping. It has no cache business logic.
2. `EngineServiceImpl`, `ControllerServiceImpl`, `P2PServiceImpl`, and
   `BlendServiceImpl` are the generated service implementations. Their public
   methods are named exactly like the proto RPCs and delegate to narrower
   backend services when the behavior is large.

## End-to-End Flow

### Server Startup

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
   descriptor, verifies that the implementation object has every same-named
   RPC method, and registers those methods.
6. For each method, `add_service()` compiles request decoding from the method
   annotations and response encoding from the method return annotation.
7. `server.assign_thread_pools(...)` assigns worker pools for methods marked
   `HandlerType.BLOCKING`.
8. `server.start()` creates the `grpc.Server`, registers generated service
   bindings, binds the TCP or Unix socket endpoint, and starts serving.

### Client Call

The client exposes one Python method per proto RPC. Method names are the
snake_case form of the protobuf method name:

| Proto method | Client method |
|---|---|
| `Lookup` | `client.lookup(...)` |
| `QueryPrefetchStatus` | `client.query_prefetch_status(...)` |
| `Store` | `client.store(...)` |
| `P2PLookupAndLock` | `client.p2p_lookup_and_lock(...)` |
| `CbUnifiedLookup` | `client.cb_unified_lookup(...)` |

A call to `client.lookup(key, tp_size)` does the following:

1. The installed client method resolves to `RpcMethod.Lookup`.
2. `MultiprocessGrpcClient._call_rpc()` gets the generated `LookupRequest`
   class from the proto descriptor.
3. `encode_request_from_call(...)` maps positional args or keyword fields into
   the protobuf request. For `Lookup`, `key` becomes `LookupRequest.key` and
   `tp_size` becomes `LookupRequest.tp_size`.
4. The generated gRPC stub sends `EngineServiceStub.Lookup.future(...)` with
   `wait_for_ready=True`. Calls submitted while the daemon is starting remain
   pending until the server becomes reachable.
5. The completion callback decodes the protobuf response into the Python shape
   expected by current callers and completes an LMCache `MessagingFuture`.

The caller only sees the LMCache future:

```python
future = client.lookup(key, tp_size)
future.result(timeout=5.0)
```

### Server Dispatch

When gRPC receives a request:

1. The generated gRPC runtime calls the same-named method on `_GrpcServicer`,
   such as `_GrpcServicer.Lookup(request, context)`.
2. The thin method calls `_dispatch("Lookup", request, context)`.
3. `_dispatch()` finds the registered `GrpcRequestHandler` for that method.
4. The handler's request decoder maps protobuf fields to the concrete service
   method signature. For example, `LookupRequest.key` and
   `LookupRequest.tp_size` become `EngineServiceImpl.Lookup(key, tp_size)`.
5. The handler executes either in the gRPC worker thread for
   `HandlerType.SYNC` or in a dedicated thread pool for
   `HandlerType.BLOCKING`.
6. The handler's response encoder maps the Python return value back into the
   generated protobuf response.

If a service implementation raises `NotImplementedError`, the transport maps
it to gRPC `UNIMPLEMENTED`. This is used for feature-gated methods: the proto
can describe the full surface, while a server that did not enable a feature
returns a standard gRPC status.

### Scheduling and Affinity

Scheduling is local to the service implementation method:

```python
@grpc_method(HandlerType.BLOCKING)
def Lookup(self, key: IPCCacheServerKey, tp_size: int) -> None:
    ...


@grpc_method(HandlerType.BLOCKING, requires_client_affinity=True)
def Store(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    gpu_block_ids: list[list[int]],
    event_ipc_handle: bytes,
) -> tuple[bytes, bool]:
    ...
```

Methods without a decorator default to `HandlerType.SYNC`. Methods that
interact with per-client GPU or event state set
`requires_client_affinity=True`; the server then routes calls from the same
client metadata key to a stable affinity worker.

Scheduling metadata is not wire schema and is intentionally not in the proto.
It is runtime policy for how LMCache executes a method after gRPC has routed
the request.

## Why This Is Better Than the Old ZMQ Envelope

The old path used a custom message-queue protocol: the client sent a request
type plus serialized payload frames, the server decoded the request type,
selected a Python handler, and later matched a response back to the client
future. That mixed transport, routing, schema, and business dispatch into one
custom envelope.

The gRPC design improves the boundary in concrete ways:

| Area | Old ZMQ-style envelope | Current gRPC design |
|---|---|---|
| Method identity | Runtime `request_type` token in the payload. | Method identity is the gRPC service/method path. |
| Payload schema | Python-side payload list; correctness depends on sender and receiver agreeing on ordering. | Request and response messages are declared in `.proto`. |
| Client API | `submit_request(request_type, [payloads])`. | `client.lookup(key, tp_size)` and other named methods. |
| Server API | Generic request dispatch with request-type switching. | `server.add_service("EngineService", EngineServiceImpl(...))`. |
| Service boundary | Mostly implicit Python grouping. | Explicit proto services with matching Python implementation classes. |
| Validation | Mismatches can surface late at runtime. | Descriptor-derived method tokens and handler annotation codecs are checked during registration and tests. |
| Error model | Custom response/error handling. | Standard gRPC status codes such as `UNIMPLEMENTED`. |
| Startup behavior | Custom polling-loop and socket behavior. | gRPC futures with `wait_for_ready=True`. |
| Tooling | LMCache-specific wire format. | Standard protobuf/gRPC tooling and generated stubs. |
| Extensibility | New request types required touching custom routing conventions. | New methods follow proto -> implementation -> registration. |

This also removes the old `typed_rpc.py` maintenance cost. There is no manual
`_PythonRpcContract` row to update when adding a field or method. The proto
descriptor and the concrete implementation method are the contract.

## Adding A New RPC

Use a new `EngineService.GetServerVersion` unary RPC as an example.

### 1. Update the Proto

Add request/response messages and the RPC to
`transport/grpc_impl/proto/lmcache_mq.proto`:

```proto
message GetServerVersionRequest {}

message GetServerVersionResponse {
  string version = 1;
}

service EngineService {
  rpc GetServerVersion(GetServerVersionRequest)
      returns (GetServerVersionResponse);
}
```

Prefer concrete protobuf fields:

- Use `optional` only when Python `None` is a real semantic value.
- Use `repeated` for Python `list[...]`.
- Use `map<...>` for dictionaries when the value can be described by proto
  fields.
- Prefer concrete messages over opaque bytes.

Regenerate local stubs for validation:

```bash
pip install -r requirements/proto.txt
python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
```

The generated `lmcache_mq_pb2.py` and `lmcache_mq_pb2_grpc.py` files are local
build artifacts and are not checked into Git.

### 2. Implement the Service Method

Add a same-named method to the matching Python service implementation:

```python
class EngineServiceImpl:
    def GetServerVersion(self) -> str:
        """Return the cache server build version."""
        return lmcache_version
```

If the method blocks on storage, transfer, or coordination work, mark it:

```python
class EngineServiceImpl:
    @grpc_method(HandlerType.BLOCKING)
    def GetServerVersion(self) -> str:
        """Return the cache server build version."""
        return self._metadata.get_server_version()
```

No transport registry entry is needed. During `server.add_service(...)`, the
transport sees `EngineService.GetServerVersion` in the descriptor, finds
`EngineServiceImpl.GetServerVersion`, and compiles the protobuf response
encoder from the return annotation `str`.

### 3. Call It From the Client

The client method is installed automatically from the proto method name:

```python
future = client.get_server_version()
version = future.result(timeout=5.0)
```

For request fields, pass positional args in proto field order or keyword args
by field name:

```python
future = client.some_rpc(request_id="req-1", timeout=2.0)
response_value = future.result(timeout=5.0)
```

### 4. Add Tests

At minimum, add a focused gRPC roundtrip test:

```python
def test_get_server_version_roundtrip() -> None:
    def handler() -> str:
        return "dev"

    server = MultiprocessGrpcServer(server_url)
    server.add_handler(RPC.GetServerVersion, handler)
    server.start()

    client = MultiprocessGrpcClient(server_url)
    assert client.get_server_version().result(timeout=5.0) == "dev"
```

Also add a service-level test if the implementation delegates to backend logic.

## Adding A New Service

Use a new `DemoService.Demo` RPC as an example.

### 1. Update the Proto

```proto
message DemoRequest {
  string probe_token = 1;
}

message DemoResponse {
  string echoed_probe_token = 1;
}

service DemoService {
  rpc Demo(DemoRequest) returns (DemoResponse);
}
```

### 2. Implement the Python Service Class

```python
class DemoServiceImpl:
    def Demo(self, probe_token: str) -> str:
        """Return the demo probe token for RPC wiring validation."""
        return f"demo:{probe_token}"
```

### 3. Register the Service

Wire it in `server.py`:

```python
demo_service = DemoServiceImpl()
server.add_service("DemoService", rpc_services.demo_service)
```

`MultiprocessGrpcServer.start()` registers the generated
`DemoServiceServicer` automatically because service names come from the proto
descriptor.

### 4. Call It

```python
future = client.demo("request-token")
echoed = future.result(timeout=5.0)
assert echoed == "demo:request-token"
```

The resulting change is mechanical: proto message/method, Python service
implementation, server registration, and a focused call-site/test. There is no
global RPC contract table to update.

## Notes

- Generated protobuf stubs are local artifacts. After switching branches that
  modify `lmcache_mq.proto`, regenerate stubs before running tests.
- The public `RpcMethod` namespace is descriptor-derived. Adding a proto method
  automatically gives the client a snake_case method with the same semantic
  name.
- `proto_codec.py` is intentionally generic. If a new RPC requires adding
  RPC-specific logic there, first consider whether the request/response schema
  should be expressed more directly in protobuf.
