# Multiprocess RPC contracts

Multiprocess requests have three independent pieces:

- `RequestType` in `base.py` is the stable ZMQ wire identifier. Existing values
  must never be reordered or reused.
- The domain modules under `rpc_messages/` define and locally register the
  transport-neutral Python request and response pair for every canonical
  `RequestType`.
- `@request_handler` on a business method declares its `RequestType`, scheduling
  mode, and optional client-affinity requirement.

There is no separate protocol-definition registry. Handler scheduling belongs
to the handler annotation, while request and response structure belongs to the
Python message pair.

## Adding an RPC

1. Append its stable identifier to `RequestType`.
2. Add the Python request and response dataclasses and register the pair beside
   its domain message definitions under `rpc_messages/`.
3. Add the route-only protobuf service method using `TransportPayload`.
4. Annotate the business handler:

   ```python
   @request_handler(RequestType.YOUR_NEW_OP, HandlerType.BLOCKING)
   def handle_your_new_op(
       self, request: YourNewOpRequest
   ) -> YourNewOpResponse:
       ...
   ```

At import, the Python message registry is checked against every canonical
`RequestType`. At server startup, each annotated handler is checked against its
registered Python request and response classes. The gRPC method registry also
checks that every generated service method resolves to the same pair.
