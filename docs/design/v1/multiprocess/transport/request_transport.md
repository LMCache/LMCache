# Multiprocess Request Transport

## Motivation

MP clients previously constructed `MessageQueueClient` directly and submitted
an integer request identifier with a positional payload list. This coupled
every caller to ZMQ and made adding another request transport an
application-wide change.

The request transport is now split into a transport-neutral API and
transport-specific implementations:

```text
MP integration / SDK / benchmark
              |
              v
     RequestClientFactory  -- selects by URL scheme
              |
              v
         RequestClient     -- named request methods
          /       \
         v         v
   ZMQ client   gRPC client
          \       /
           v     v
     Python request/response messages
               |
               v
       business request handlers
```

## Design

`RequestClient` defines named methods such as `lookup()`, `store()`, and
`retrieve()`. Each call is represented internally by one Python request class
and one Python response class from the domain modules under `rpc_messages/`.
These classes are the transport-neutral RPC contract consumed by business
request handlers.

`RequestClientFactory` normalizes an endpoint and selects an implementation by
scheme:

| Scheme | Implementation |
|---|---|
| no scheme, `tcp`, `ipc`, `inproc` | ZMQ |
| `grpc`, `grpc+unix` | gRPC |

A bare `host:port` endpoint is normalized to `tcp://host:port`. Invalid or
unknown schemes fail before a client is created. The server selects the matching
implementation through `--transport zmq` or `--transport grpc`.

This abstraction covers MP request RPCs only. It does not select the mechanism
used to move KV data between an engine worker and the server.

### Transport boundaries

Both ZMQ and gRPC serialize the same complete Python request and response
messages with the shared MessagePack representation. ZMQ sends the RPC route
as an ASCII frame and places each encoded message in one additional frame.
gRPC installs descriptor-derived generic method handlers whose
serializer/deserializer operates directly on the Python message class. There
is no protobuf object in the request path and no protobuf/Python conversion
layer.

The generated protobuf modules provide service and method descriptors only.
Every RPC declares the same empty `TransportPayload` placeholder; gRPC uses it
only to preserve the named route. Protobuf is not the payload data model.
Consequently this internal Python transport is not wire-compatible with an
independently generated protobuf client. This is intentional: domain-local
Python messages, shared with ZMQ, are the canonical contract rather than a
second protobuf object model.

Server-side binding and scheduling are separate from serialization. A business
handler is named `handle_<operation>` and uses the transport-neutral
`@request_handler` annotation only for its `HandlerType` and client-affinity
requirement. Both ZMQ and gRPC derive the same operation route from the
handler/method name. A handler receives exactly one Python request message and
returns exactly one Python response message. Server startup validates those
annotations against the local RPC message registration.

Adding an RPC therefore requires a route-only protobuf method declaration, a
locally registered Python request/response pair in the owning `rpc_messages/`
domain module, and a matching `handle_<operation>` business handler. Changing
payload fields changes only the Python pair. No integer request enum,
`ProtocolDefinition`, protobuf conversion, adapter registry, or per-RPC
serialization definition is required.

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
