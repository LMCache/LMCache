# Multiprocess Protocol Helpers

The gRPC protocol is defined by
`lmcache/v1/multiprocess/transport/grpc_impl/proto/lmcache_mq.proto`.
`protocol.py` derives `RpcMethod` and the `RPC` namespace from that protobuf
descriptor, and `transport/grpc_impl/typed_rpc.py` maps protobuf fields to
LMCache Python domain objects.

This directory is no longer a protocol-definition registry. It only contains
small Python helper types that are still shared by service implementations:

```text
protocols/
├── README.md
├── __init__.py
├── base.py      # HandlerType execution modes
└── engine.py    # Engine-driven response dataclasses
```

## Adding a gRPC Method

1. Add the request/response messages and unary method to
   `transport/grpc_impl/proto/lmcache_mq.proto`.
2. Regenerate protobuf stubs:

   ```bash
   pip install -r requirements/proto.txt
   python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
   ```

3. Add the Python value mapping and handler scheduling mode to
   `transport/grpc_impl/typed_rpc.py`.
4. Implement the handler method on the service object that declares the
   matching `GRPC_SERVICE_NAMES` entry.

The server registers generated gRPC services directly. It does not collect
per-module request-name definitions before starting.
