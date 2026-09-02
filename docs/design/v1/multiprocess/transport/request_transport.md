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

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
