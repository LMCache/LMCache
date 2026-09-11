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
and one Python response class from `rpc_messages.py`. These classes are the
transport-neutral RPC contract consumed by business request handlers.

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

ZMQ serializes one complete Python request message into one request frame and
one complete Python response message into one response frame. It no longer
serializes an RPC as a positional list of independently typed payload frames.

gRPC keeps protobuf as its wire format. Each generated method binding maps its
protobuf input and output classes to the same Python request and response
classes used by ZMQ. Protobuf conversion therefore exists only at the gRPC
client/server boundary. Small leaf adapters remain for values whose native
Python representation is intentionally different from protobuf, such as
`DeviceIPCWrapper` and `torch.Size`.

Server-side binding and scheduling are separate from serialization. Business
module methods use the transport-neutral `@request_handler` annotation to
declare their `RequestType`, `HandlerType`, and client-affinity requirement.
Both ZMQ and gRPC discover this metadata. A handler receives exactly one Python
request message and returns exactly one Python response message. Server startup
validates those annotations against the shared RPC message registry.

Adding an RPC therefore requires a protobuf method, a matching `RequestType`
and `ProtocolDefinition`, a Python request/response pair in `rpc_messages.py`,
and an annotated business-module handler. No per-RPC serialization definition
is required by either transport.

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
