# DeepInfer LMCache MP client

DeepInfer integrates through `lmcache.integration.deepinfer`. Its scheduler and
worker roles share one `LMCacheMPConnector`, which selects the matching adapter
from its `role`. Lifecycle handling, cache mapping, and transfer
specifications are all maintained in that directory. The integration depends
only on LMCache's engine-neutral multiprocess core; it does not import another
serving-engine integration. The existing multiprocess server, hashing, lookup,
transfer, and sparse recurrent-state behavior remain shared, so no
DeepInfer-specific server module or object type is needed.

The adapter discovers cache geometry from the live DeepInfer cache objects:

- locally owned `ica_cache` and `ca_cache` tensors are registered as one full
  attention engine group and share DeepInfer block IDs;
- shared layer aliases are deduplicated by tensor identity;
- the source-token coverage of each page comes from DeepInfer's configured
  block size, while physical compressed slots come from each tensor shape;
- a bounded full runtime-state snapshot pool is registered as a recurrent
  group. Historical chunks use null block ID 0 and only the final chunk points
  at the immutable snapshot slot owned by that transfer.

The last rule makes a prefix recoverable only when full CA/ICA data and an
endpoint state snapshot exist on every rank. Start the MP server with
`--separate-object-groups`; this lets the recurrent group remain sparse without
weakening the full-attention prefix requirement.

DeepInfer allocates its KV pool with `cuMemAlloc`, outside PyTorch's caching
allocator. Set `isolated_ipc` in the connector and start the MP server with
`--isolated-ipc`; this selects LMCache's raw CUDA IPC wrapper instead of
`UntypedStorage._share_cuda_()`. The setting must match on both sides because it
also selects the event IPC implementation. No DeepInfer-specific server module
is required.

DeepInfer packs and restores state through its public
`select_prefix_state_to_store(...)` and
`load_full_state_to_cache(...)` helpers. The full transfer format includes
stored SWA, YoCo consumer SWA, and canonical boundary compression state rather
than the smaller persistent-disk state format. This avoids hard-coding layer
counts or assuming that raw ring-buffer rows are portable snapshots.

Stores are asynchronous. Before submitting a store, the worker copies the
request's mutable runtime state into a free GPU snapshot slot on the model
stream. The MP transfer event orders D2H after that copy, and the slot is not
returned to the pool until the store future completes. This permits later
decode steps and later stores for the same request to update the live state
without waiting for an earlier D2H.

DeepInfer separately pins every CA/ICA cache block exposed to an outstanding
store. Worker connectors report `finished_sending` only after their local D2H
queues drain; the scheduler releases the pins after every context-parallel rank
reports completion. Request completion therefore marks connector bookkeeping
finished but does not synchronously wait for storage.

Example DeepInfer configuration:

```python
kv_transfer_config = {
    "connector": "LMCacheMPConnector",
    "connector_module_path": (
        "lmcache.integration.deepinfer.lmcache_mp_connector"
    ),
    "server_url": "tcp://127.0.0.1:5555",
    "transfer_mode": "lmcache_driven",
    "isolated_ipc": True,
    "mq_timeout": 300.0,
    "max_inflight_loads": 1,
    "state_snapshot_pool_size": 4,
}
```

`connector_module_path` selects this out-of-tree implementation instead of a
connector with the same name in DeepInfer's built-in registry.

`state_snapshot_pool_size` must be greater than one and bounds concurrent state
snapshots per worker. If all slots are busy, that store opportunity is skipped
without blocking inference; a later prefix store resumes from the last
contiguously completed chunk. `max_inflight_loads` also reserves that many
disjoint state staging slots for retrieves, so incoming and outgoing transfers
cannot overwrite each other.

The server chunk size must be a positive multiple of the DeepInfer block size.
The current DeepInfer model uses 128-token blocks, so a 128-token MP chunk is
the simplest configuration.
