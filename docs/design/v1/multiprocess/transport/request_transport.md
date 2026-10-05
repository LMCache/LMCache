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

### Worker affinity lifecycle

Handlers marked `requires_client_affinity` are routed by the RPC's integer
`instance_id`, not the connection identity. The payload position comes from the
shared RPC signature, including operations with a different argument order.
Reconnecting the same instance therefore preserves its worker binding in both
ZMQ and gRPC.

The concrete ZMQ and gRPC servers opt into the existing `InstanceLivenessTarget`
state mirror; the general `RequestServer` contract remains start/close only.
Server construction attaches the transport to the management reaper.
`drop_instance_state` retires the instance's affinity key. The transports also
retire that key after successful KV, engine-driven KV, or Q unregister RPCs.

The pool serializes submission and retirement with one lock. Retirement does not
cancel outstanding work: running and queued tasks, including new submissions for
a reactivated instance, retain their original worker until that key drains.
The next submission can establish a new binding. New keys use the least-bound
worker, preferring reclaimed slots over sharing a worker with another bound key.
If restarts precede reaping, a surviving key may already share a worker.
After capacity is reclaimed, that key can move to a free slot on its next
submission, but only after its running and queued handlers have drained.
Dedicated bindings stay in place. True oversubscription still shares workers
and emits the existing once-per-pool warning.

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
