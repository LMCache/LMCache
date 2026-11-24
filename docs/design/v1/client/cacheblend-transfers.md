# CacheBlend worker transfers

`TransferContext` exposes `register_blend_rope` and `submit_blend_retrieve` so
engine adapters use the same registration identity, IPC wrapper/fd channel,
and event backend as ordinary KV transfers. These are optional capabilities:
`LMCacheDrivenTransferContext` implements them; engine-driven contexts raise
`NotImplementedError` before sending an RPC.

The caller first registers paged KV, then registers rotary tables and waits
for their acknowledgement. Tables remain caller-owned until unregistration.
Group mappings use the engine-group IDs from KV registration, including the
rotation windows and per-group head dimensions. An empty table list supports
NoPE models. A failed registration does not enable blend retrieval.
`unregister` waits asynchronously for rotary-table teardown before unregistering
paged KV; its returned future covers both acknowledgements. This also cleans up
a registration whose reply failed, since the server may have imported its tables.

After lookup and allocation, the adapter can submit selected matches:

```python
producer = context.create_recorded_event()
future = context.submit_blend_retrieve(key, matches, block_tables, producer)
ready = future.wait_on_stream(forward_stream, timeout=60)
```

The future retains the producer event and imports the server's completion
event through the registered platform backend. Stream waiting orders model
execution without synchronizing the device on the host. The caller retains
the destination allocation and future until completion and ends the session
only after outstanding retrieves have finished.

The returned bool preserves the server contract: `False` requires recomputation.
`True` also covers already-applied matches and deferral when all matches are
beyond the allocated blocks. A scheduler must not interpret it as evidence that
unallocated KV positions were populated. Partial allocation that intersects
only part of a match fails without scatter.

This interface does not select matches, schedule selective recomputation, or
advertise support for every engine layout. Those remain adapter and server
capabilities; separate rotary K/value components require matching metadata
support on both sides.
