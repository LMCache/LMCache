# Multiprocess Request Transport

## Motivation

MP clients previously constructed `MessageQueueClient` directly and submitted a
`RequestType` with a positional payload list. This coupled every caller to ZMQ
and made adding another request transport an application-wide change.

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
   ZMQ facade   gRPC client
         |
         v
 MessageQueueClient
```

## Design

`RequestClient` defines named methods such as `lookup()`, `store()`, and
`retrieve()`. The ZMQ facade translates each method back to the existing
`RequestType`, payload order, and response type, so this refactor does not
change the ZMQ wire protocol.

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

### gRPC codecs

gRPC keeps protobuf as its wire format. During initialization, each generated
gRPC method is mapped to a transport-neutral `RequestType`. Its method codec
combines the protobuf descriptor, which defines the wire schema, with the
payload and response types exposed by `ProtocolDefinition`, which define the
corresponding Python contract. Both the client and server use the resulting
read-only registry.

Most dataclasses and containers use the structural codec. Types that need a
non-structural representation register an explicit message codec in
`grpc_impl/codecs/`, organized by protobuf message domain; types shared across
domains register in the common codec module. Missing protocol definitions and
duplicate registrations fail while the method codec registry is initialized.

Server-side binding and scheduling are separate from serialization. Business
module methods use the transport-neutral `@request_handler` annotation to
declare their `RequestType`, `HandlerType`, and client-affinity requirement.
Both ZMQ and gRPC discover this metadata. When gRPC registers the modules, it
also validates each handler's parameter and return annotations against the
Python contract used to compile that method's codec.

Adding an RPC therefore requires a protobuf method, a matching `RequestType`
and `ProtocolDefinition`, and an annotated business-module handler. A custom
message codec is needed only when the structural codec cannot represent the
Python type directly.

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
