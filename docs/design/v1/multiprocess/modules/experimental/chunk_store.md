# Chunk store completion and event ownership

`ChunkStoreModule` is an opt-in engine module enabled with `--enable chunk_store`.
It shares registered KV contexts and transfer helpers with the ordinary transfer
module. It owns its RPC handlers and completion-event leases. The ordinary
`store` handler keeps its existing transfer path.

## RPC contract

The typed `@rpc_method` declarations on `RequestClient` define both transports:

```
store_with_chunk_events(key, instance_id, block_ids, producer_event_handle)
  -> (terminal_event_handle, [(chunk_event_handle, start, end), ...], success,
      lease_id)
release_chunk_store_events(instance_id, lease_id) -> None
```

ZMQ routes these operations by name. gRPC discovers `ChunkStoreService` from its
generated descriptors. Neither operation consumes a legacy integer operation ID.
The worker checks the existing `get_experimental` RPC before the first chunk
store. If `chunk_store` is absent it submits an ordinary store instead. This
fallback requires a server supporting the existing module-discovery RPC; it does
not send an unknown chunk-store request and wait for a timeout.

## Source safety and storage visibility

All object groups reserve their storage before chunk transfers are enqueued.
Each chunk event follows the copies for every object group of that chunk on the
registered transfer stream. A completed chunk permits reuse of its source GPU
blocks; it does not independently publish cached objects. A successful operation
enqueues one `finish_write` for the reserved keys. On a transfer failure,
`abort_write` discards only this writer's staging after queued copies complete.
Previously resident objects remain untouched. Storage may decline individual
reservations, just as with ordinary store; success means no fatal transfer error,
not that every requested object was reserved.

Ranges use absolute, end-exclusive token offsets. Staging and per-chunk slicing
share `kept_blocks_per_chunk`; both the downsampled host IDs (direct-copy path)
and GPU IDs (kernel path) use that stride. Null-block masks are computed before
downsampling with the configured marker.

## Event lifetime

Exported bytes do not own a CUDA/HIP event. The backend's bounded exported-event
ring cannot protect a slow client once newer exports evict its events.

```
server: reserve lease -> retain events -> record/export -> return handles
client: import -> poll ranges -> observe terminal completion
client: cache remaining ranges -> drop imported events -> release RPC
server: drain terminal event -> remove lease
```

The server retains terminal and chunk event objects until the release RPC. It
does not free them when the response is sent or merely when its GPU completes.
The release is scoped to the worker and idempotent. At 1024 outstanding leases
per worker, new stores are rejected before device submission; live leases are
never evicted to make room.

`ChunkEventDeviceMessagingFuture` implements the host-polled `MessagingFuture`
interface. Its `query`, `wait`, and `result` observe terminal completion;
`take_completed_ranges` is nonblocking and drains each range once. It owns the
imported handles exclusively and does not expose `wait_on_stream` or raw event
objects. This avoids acknowledging a lease while a caller is still arranging
GPU waits. Ordinary `DeviceMessagingFuture.wait_on_stream` is unaffected.
After terminal completion, repeated queries/results and undrained ranges use
cached state. Transfer-context unregister and close drain release
acknowledgements before destroying registrations or closing the request client.
Failed release RPCs retain their lease IDs for an idempotent retry on the next
drain; they do not change the store's data-path result or discard other leases.
An errored raw store RPC has no imported handles or known lease ID. Cleanup
retires that future so unregister can reclaim any lease whose response was lost;
the request's result still raises its original error. Event-import failures
remain tracked because their response already identified a lease.

Explicit unregister requires the caller to drain futures first and retires any
remaining worker leases. The liveness reaper also retires a dead worker's leases;
a reaped worker must discard its old futures before registering again. A lost
response/release remains held until worker cleanup rather than being reclaimed
by an arbitrary event timeout. Server shutdown stops transport before cleanup.

## Validation

- `test_chunk_store.py`: delayed import after GC and 2100 unrelated exports,
  at-most-once drains, concurrent drains, terminal caching, release ownership,
  backpressure, worker cleanup, partial failure, shared host/device geometry,
  and ZMQ/gRPC store/release round trips.
- `test_event_ipc_handle_path.py`: worker capability fallback and integration.
- `test_grpc_transport.py`: generated descriptor/handler coverage.
- `test_chunk_store_gpu.py`: real CUDA/ROCm processes, delayed event import
  after 2100 exports, and exact KV round trips over both transports.
- `test_distributed_storage_manager.py`: aborted staging never becomes readable
  and leaves resident objects intact.

The optional path performs one transfer call per chunk and object group plus
event record/export work. GPU throughput and first-release latency must be
measured on the deployment backend; no performance improvement is implied by
the unit tests.
