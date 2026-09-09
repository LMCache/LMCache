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

gRPC keeps protobuf as its wire format. At startup, the transport combines each
generated RPC descriptor with the existing transport-neutral
`ProtocolDefinition` and compiles one request/response codec for that method.
Both the client and server use this read-only registry, so adding an RPC requires
one protocol definition rather than response-type checks in a central decoder.

Most dataclasses and containers use the structural codec. Types that need a
non-structural representation register a small codec next to the service that
owns the protobuf message; shared types register in the common codec module.
Missing protocol definitions, duplicate registrations, and handler annotation
mismatches fail while the transport is initialized.

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
