# Multiprocess Request Transport

## Motivation

Multiprocess RPC metadata used to be repeated in `RequestType`,
`ProtocolDefinition`, the `RequestClient` API, transport adapters, protobuf
services, and handler decorators. A new RPC could compile while one of those
registries was missing or inconsistent.

The transport boundary now has one Python contract and transport-owned wire
schemas:

```text
MP integration / SDK / benchmark
              |
              v
     RequestClientFactory  -- selects by URL scheme
              |
              v
         RequestClient     -- typed RPC contract
          /       \
         v         v
  ZMQ adapter   gRPC adapter
```

## Shared RPC contract

Methods marked with `@rpc_method` on `RequestClient` are the Python source of
truth. The method name is the stable operation name, its parameters define the
ordered payload types, and `MessagingFuture[T]` defines the response type.
`get_rpc_specs()` discovers these contracts at startup. Low-level ZMQ sockets,
polling, multipart frames, msgspec codecs, and worker-pool dispatch stay in
`zmq_impl/mq.py`.

Business modules use `@request_handler` only for scheduling. The operation
defaults to the handler name; `operation=` is for legacy method names.
Discovery validates handler annotations against the shared contract before
either server starts.

## Transport implementations

`RequestClientFactory` selects an implementation by endpoint scheme:

| Scheme | Implementation |
|---|---|
| no scheme, `tcp`, `ipc`, `inproc` | ZMQ |
| `grpc`, `grpc+unix` | gRPC |

The gRPC adapter discovers generated service methods from protobuf descriptors.
Each snake-case descriptor method must match one shared RPC operation. Its
codec combines the protobuf message schema with the Python payload and response
types. Types needing a non-structural representation register an explicit
message codec under `grpc_impl/codecs/`.

The ZMQ adapter installs client methods from the same RPC specifications. IDs
1–33 remain frozen for compatibility with deployed clients and servers. New
operations use their stable string name on the wire and must not be added to
the legacy ID table. The server accepts both encodings.

The server follows the same boundary. `server.py` builds transport-neutral
engine modules and passes them to `create_request_server()`, which returns the
`RequestServer` protocol implemented by either `MessageQueueServer` or
`GrpcMultiprocessServer`. Shared runtime code starts and closes only that
protocol; concrete server classes are accessed only inside their transport
packages and implementation-level tests.

### gRPC server modes

`--grpc-server-mode` picks one of two gRPC servers that serve the same
handlers:

| Mode | Receives requests on | Handler runs on |
|---|---|---|
| `sync` (default) | `grpc_server_workers` threads | the receiving thread, or the client's affinity worker |
| `aio` | one `grpc.aio` event loop thread | a pool picked by handler type |

In `sync` mode each in-flight call holds a gRPC thread until its handler
returns, and every call also passes through grpcio's Python polling thread. In `aio` mode one event loop receives every call and only the
handlers take threads.

Handler types mean the same in both modes: `SERIALIZED` handlers run one at a
time, at most `max_cpu_workers` normal `BLOCKING` handlers run at once, and
affinity handlers stay on the worker bound to their client. Each
server-streaming call is read on its own thread, so idle subscriptions never
occupy a handler worker. In `aio` mode, the `grpc_server_workers` threads only
run out-of-tree services that register plain functions instead of coroutines.

Both servers subclass `GrpcMultiprocessServerBase` in `grpc_impl/server_base.py`,
which owns the affinity pool, port binding, service registration and the close
sequence. `GrpcMultiprocessServer` (sync) and `GrpcAioMultiprocessServer` only
provide the gRPC server object, the servicer for each generated service, how
normal handlers are bounded (a semaphore on the sync server, a thread pool on
the aio event loop), and how the server starts and stops.
`build_grpc_request_server` in `grpc_impl/server_builder.py` picks one of them
from `grpc_server_mode` and returns it as a `GrpcMultiprocessServerBase`.

## Adding an RPC

An ordinary RPC requires three changes:

1. Add one typed `@rpc_method` to `RequestClient`.
2. Add the protobuf request, response, and service method.
3. Add a same-named `@request_handler` method to a business module.

No request enum, Python protocol-definition registry, manual ZMQ client method,
or gRPC service registration list is required. Add a custom gRPC message codec
only when the structural codec cannot represent the Python type.

## Adding a transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.

## KV event subscriptions

The CPU/L1 channel reuses the event bus and the existing vLLM publisher:

```text
MP_TOKENS + L1 write/eviction -> KVEventModule's bounded log
    -> subscribe_kv_events(instance, model, cursor, limit)
    -> ZMQ / gRPC push -> MP worker buffer -> vLLM -> router
```

One rank per server subscribes. Completed server writes are the store source;
a successful store request can skip chunks it could not reserve. Stores
require known tokens and parent hashes. Unknown bindings are counted and
skipped, and the binding cache is bounded.

Batches carry an incarnation, cursor, loss flag and ordered records.
Restart, lost records or disconnect withdraw announced CPU placements.
Disconnects reopen the subscription; other failures fall back to own completed
stores. The buffer gauge and generated, drained, batch and resync counters
expose delivery progress.

Both transports use `MessagingStream` and the same bounded replay log. Readers
wait on a condition notified by event-bus callbacks, including dropped events.
gRPC uses a server stream; ZMQ retains the request ID and grants one batch of
credit per acknowledgment. Slow subscribers do not create unbounded server
queues; if the shared log overruns their cursor, they receive a loss marker.
Stream readers do not occupy the ordinary request executor. Cancellation,
server shutdown and the existing worker reaper release subscriptions.

`modules/kv_events.py` owns the log, token bindings, subscriptions and status.
It uses the existing `InstanceLivenessTarget` hooks for PING refresh and cleanup;
`ManagementModule` retains the shared reaper. Subscription expiry only closes
that stream; it does not reap the worker's cache registrations.

The connector retains `KVEventAggregator` and an ordered batch buffer for
store/remove/store transitions. Workers receive pushes while idle; vLLM still
drains and publishes during engine steps, so idle routing entries can remain
stale until stepping resumes.

Both transports advertise `kv_event_stream`. Enable the event
bus and vLLM KV events, and match hash algorithms and chunk sizes. Workers
subscribe automatically to capable servers. See
`docs/source/mp/configuration.rst` for the server's log-size/disable control.
L2 events, access events and salted hash interoperability are outside scope.
