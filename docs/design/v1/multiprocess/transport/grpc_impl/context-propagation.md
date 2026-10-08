# W3C context across gRPC workers

With `LMCACHE_MP_TRACE_CONTEXT=1`, the client captures only `traceparent` and
`tracestate` for each call, before gRPC schedules it. Client affinity metadata
is kept unchanged. This works for keyed requests and keyless session controls.

The server reads the per-call metadata, then attaches an isolated context in
the thread that executes the handler. Sync handlers keep their existing lock;
blocking handlers keep their normal or affinity pool. The original context is
restored in `finally`, including handler failures and cancellation exceptions.
No SDK, exporter, or second provider is configured by the transport.

`encoded_trace_context` is an optional protobuf map encoding for the shared
`IPCCacheServerKey` field. Existing protobuf peers ignore the new field. The
gRPC worker's parent comes from RPC metadata, rather than cache identity.

The CPU regression tests use spawned processes and real loopback ZMQ/gRPC
connections. They check sampled and unsampled parents and a later keyless
session end. They do not validate GPU execution or remote cancellation of a
running native handler.
