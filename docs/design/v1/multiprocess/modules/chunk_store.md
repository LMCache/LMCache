# Chunk-store plugin

The plugin factory in `chunk_store.py` is loaded through `--server-module` from
the pluggable MP server interface. It requires the existing
`LMCacheDrivenTransferModule` and shares its worker registrations and streams.
It owns no additional device resources. The ordinary store RPC is unchanged.

`store_with_chunk_events(key, instance_id, block_ids, producer_event_handle)`
returns `(terminal_handle, [(chunk_handle, start, end), ...], success)`.
The typed `RequestClient` method and `ChunkStoreService` protobuf register the
contract on the existing ZMQ/gRPC transport; the dynamically loaded module
provides the handler. No legacy integer operation ID is added.

The plugin validates the complete range and per-kernel-group block counts before
submitting work. It splits raw block IDs at each group's chunk stride and calls
the existing store once per chunk, preserving the full token prefix, salt and
request metadata. Downsampling, null blocks, storage reservation, observability
and commits remain the ordinary transfer module's responsibility.

Each returned event follows all object-group copies for its chunk. The terminal
handle is the last submitted chunk's event. Completion permits source-buffer
reuse; publication and cache residency follow ordinary store semantics. A failed
chunk stops further submission and retains any available completion handle.
Earlier successful chunks may already be committed; the batch is not atomic.
An empty, valid range succeeds without device work.

Event handles use the existing backend's retention contract. In particular, the
default backend retains a bounded ring of exports; arbitrary delayed imports
after that ring turns over are not guaranteed. This plugin does not add event
leases or change backend lifetime behavior. Clients must obey the same handle
usage constraints as the ordinary store path.

Tests cover actual factory loading, opt-in registration, group geometry,
metadata preservation, rejection before submission, partial failure, both RPC
transports and cross-process GPU source reuse followed by exact KV retrieval.
The implementation issues one ordinary store per chunk, so throughput and CPU
submission overhead need measurement before making performance claims.
