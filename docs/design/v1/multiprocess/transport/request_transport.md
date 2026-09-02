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

## Protobuf contracts

The planned gRPC transport keeps its wire contracts under
`grpc_impl/protos/`. These `.proto` files are the source of truth for request
and response messages; they do not enable the gRPC runtime by themselves.

Python protobuf modules are generated into `grpc_impl/_proto_gen/` during a
package build. Most generated files remain ignored, while the type stubs used
by handwritten adapters are tracked so static analysis also works from a
source checkout. Regenerate the bindings after changing a schema with:

```bash
pip install -r requirements/proto.txt
python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
```

The generator cleans stale outputs, compiles every schema, rewrites generated
imports to use the package-qualified path, and verifies that all generated
Python modules import successfully.

## Extending the transport

A new transport implements the `RequestClient` contract inside its own
subdirectory and adds its scheme mapping to the factory. Application code must
continue to depend only on `RequestClientFactory` and named request methods;
transport-specific serialization and connection management stay behind that
boundary.
