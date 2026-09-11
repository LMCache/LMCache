# Multiprocess RPC contracts

Multiprocess requests have three independent pieces:

- The lower-snake-case operation name is the shared ZMQ and gRPC route. ZMQ
  sends it as an ASCII frame; gRPC derives it from its method name.
- The domain modules under `rpc_messages/` define and locally register the
  transport-neutral Python request and response pair for each operation.
- A business handler named `handle_<operation>` uses `@request_handler` to
  declare scheduling mode and optional client affinity.

There is no separate protocol-definition registry. Handler scheduling belongs
to the handler annotation, while request and response structure belongs to the
Python message pair.

## Adding an RPC

1. Add the Python request and response dataclasses and register the pair beside
   its domain message definitions under `rpc_messages/`.
2. Add the route-only protobuf service method using `TransportPayload`.
3. Add the matching business handler:

   ```python
   @request_handler(HandlerType.BLOCKING)
   def handle_your_new_op(
       self, request: YourNewOpRequest
   ) -> YourNewOpResponse:
       ...
   ```

At server startup, each annotated handler is checked against its locally
registered Python request and response classes. The gRPC method registry also
checks that every generated service method resolves to the same pair.
