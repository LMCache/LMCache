# LMC server connection recovery

`LMCServerConnector` implements the existing `lm://` PUT, GET and EXIST protocol.
It keeps the same constructor and wire format. This addresses the permanently
dead socket described in [#3565](https://github.com/LMCache/LMCache/issues/3565).
The earlier, closed [#3569](https://github.com/LMCache/LMCache/pull/3569) proposed
bounded reconnect and replay; this implementation also handles cancellation,
receive ownership and unread bodies.

## Request and connection ownership

The existing asyncio lock covers one entire request, including response headers,
GET bodies, connection replacement and retry backoff. Other requests wait and
reuse the replacement connection. A failed connection is closed before any
request can reuse it.

Connection errors allow at most three attempts per public RPC, with 0.1 and
0.2 second delays before the second and third attempts. Reconnect uses a fresh
nonblocking socket and an awaited connection with a five second timeout. Failed
or cancelled connection attempts close that temporary socket. The replacement
then uses the existing blocking receive mode. Initial construction and ordinary
reads/writes retain their existing blocking behavior; this is not an end-to-end
request deadline or a general network-stall remedy.

After the retry budget is exhausted the last error propagates, leaving the old
socket closed. A later request can establish a new connection. A completed
`close()` is terminal: later GET, PUT and EXIST calls raise `RuntimeError`.
`exists_sync()` retains its existing exception-to-`False` behavior.

## Framing and memory ownership

Headers are read to their full fixed size before deserialization. Early EOF in
a header or GET body is a connection failure. A failed GET releases its newly
allocated `MemoryObj`; a successful GET transfers that reference to its caller.
If allocation returns `None`, GET retains its existing `None` result and closes
the socket, discarding the unread body before the next request. Invalid metadata
and other non-connection errors propagate without replay, also discarding the
possibly incomplete stream. Cancellation is never converted into a retry.

The connector does not change PUT source ownership. Its instrumentation wrapper
continues to release the reference it owns. Retrying PUT sends the same key and
bytes again. The server overwrites the value for that key, but the protocol has
no PUT acknowledgement: successful local sending does not establish durability,
exactly-once delivery, or that a restarted server retained an earlier value.

The server must only publish complete PUT bodies. An old connection can observe
EOF after a replacement connection has already completed a retry. Publishing
that incomplete body's `None` would overwrite the successful retry. The paired
server guard discards only `None`, retaining the existing empty-body behavior.
It changes neither the wire format nor storage backend APIs.

## Validation scope

The regression suite uses real loopback TCP and the public constructor and RPCs,
with tiny CPU `TensorMemoryObj` payloads. It checks normal byte-for-byte round
trips, dropped replies, incomplete bodies, source replay, allocation refusal,
terminal close, failed reconnects and cancellation. Faults that cannot be made
deterministic using a tiny TCP payload are injected only at socket I/O boundaries.
These checks do not cover Kubernetes rollout, model serving, throughput, or
multi-process writers racing to replace the same key.

A separate two-connection test runs the actual server handlers and backend. It
confirms a complete PUT using a following EXIST reply before terminating the old
partial PUT, then checks both stored bytes and a real GET response. This covers
the interrupted replay publication ordering, rather than arbitrary competing
complete writes from independent clients.
